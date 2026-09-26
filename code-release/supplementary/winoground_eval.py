"""
Standalone Winoground runner — IST / STI / STIT, reformulated as yes/no QA.

Winoground (facebook/winoground, 400 examples) pairs two images with two captions
where caption_j matches image_j. Native Winoground is an image-text *matching* task;
to fit the same generate→judge→ordering harness as NaturalBench / POPE we reformulate
each example into 4 yes/no questions:

    (image_i, caption_j) -> "Does this caption correctly describe the image?"
                            gt = Yes if i == j else No

So every example yields 4 balanced pairs (2 Yes, 2 No), mirroring NaturalBench's
paired structure. The same IST/STI/STIT ordering (driven by model_manager's order
string) is applied, so the prompt-ordering comparison transfers head-to-head.

Kept FULLY SEPARATE from NaturalBench/POPE pipelines — imports only two stateless,
model-agnostic helpers (yes/no parser + shared suffix). Reuses model loading + the
ordering machinery from model_manager.

Run (resumable; checkpoints every --checkpoint-every):
    CUDA_VISIBLE_DEVICES=0 python winoground_eval.py --model qwen3-vl-8b --order IST,STI,STIT
    CUDA_VISIBLE_DEVICES=1 python winoground_eval.py --model gemma-3-27b --order IST,STI,STIT

Output (separate dir; one file per model+order):
    winoground/results/<model>__<order>__results.json
"""

import argparse
import json
import os
import time

from PIL import Image

from constants import SYSTEM_MESSAGE
from naturalbench_eval import _first_yes_no, YESNO_SUFFIX

MODEL_CHOICES = ["gemma-3-27b", "gemma-3-12b", "gemma-3-4b",
                 "qwen3-vl-8b", "qwen2.5-vl-7b", "llava-1.5"]
# the 4 (image_idx, caption_idx) pairs per example; gt = yes iff i == j
PAIRS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _pred_yes_no(raw):
    p = _first_yes_no(raw)
    if p is not None:
        return p
    low = raw.lower()
    return "no" if ("no" in low or "not" in low) else "yes"


def _question(caption):
    return (f'Does this caption correctly describe the image? '
            f'Caption: "{caption}"' + YESNO_SUFFIX)


def metrics(records):
    """Pair-level acc/precision/recall/F1/yes-ratio ('yes' = caption matches) plus
    group accuracy (an example is correct iff all 4 of its pairs are correct)."""
    tp = fp = tn = fn = yes_pred = correct = 0
    n = len(records)
    for r in records:
        gt, pred = r["gt"], r["pred"]
        if pred == "yes":
            yes_pred += 1
        if pred == gt:
            correct += 1
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # group accuracy: all 4 pairs of an example correct
    by_ex = {}
    for r in records:
        by_ex.setdefault(r["example_id"], []).append(r["correct"])
    full = [e for e in by_ex.values() if len(e) == 4]
    g_acc = sum(all(e) for e in full) / len(full) if full else 0.0
    return {
        "n": n,
        "acc": round(correct / n, 4) if n else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_pred / n, 4) if n else 0.0,
        "group_acc": round(g_acc, 4),
        "n_examples": len(by_ex),
    }


def _build_meta(model_name, order, system_prompt, max_tokens, records,
                order_tag=None, echo_scale=None, echo_which=None):
    return {
        "model": model_name,
        "order": order,
        "order_tag": order_tag or order,
        "echo_scale": echo_scale,
        "echo_which": echo_which,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "overall": metrics(records),
    }


def load_winoground_samples(max_examples=None):
    """Return a flat list of {example_id, pair_key, image_idx, caption_idx,
    image, caption, gt} from facebook/winoground (4 pairs per example)."""
    from datasets import load_dataset
    ds = load_dataset("facebook/winoground", split="test")
    samples = []
    for ei, row in enumerate(ds):
        if max_examples is not None and ei >= max_examples:
            break
        ex_id = row["id"]
        imgs = {0: row["image_0"], 1: row["image_1"]}
        caps = {0: row["caption_0"], 1: row["caption_1"]}
        for (i, j) in PAIRS:
            samples.append({
                "example_id": ex_id,
                "pair_key": f"{ex_id}_{i}_{j}",
                "image_idx": i,
                "caption_idx": j,
                "image": imgs[i],
                "caption": caps[j],
                "gt": "yes" if i == j else "no",
            })
    return samples


def run_winoground_experiment(mm, samples, order, system_prompt,
                              max_tokens=16, out_dir="./winoground/results",
                              checkpoint_every=200, log_every=100,
                              order_tag=None, echo_scale=None, echo_which=None):
    import torch

    tag = order_tag or order
    if echo_scale is not None:
        assert order.count("I") == 2, (
            f"echo_scale requires an order with exactly 2 'I' occurrences, got "
            f"order={order!r} ({order.count('I')} I's)")
        assert echo_which in ("first", "second"), \
            f"echo_which must be 'first' or 'second', got {echo_which!r}"

    records, done, results_path = [], set(), None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{mm.model_name}__{tag}__results.json")
        if os.path.exists(results_path):
            try:
                records = json.load(open(results_path))["results"]
                done = {r["pair_key"] for r in records}
                print(f"  [resume] {order}: {len(done)} pairs already done")
            except Exception:
                records, done = [], set()

    def _checkpoint():
        if results_path:
            meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records,
                               order_tag=tag, echo_scale=echo_scale, echo_which=echo_which)
            with open(results_path, "w") as f:
                json.dump({"meta": meta, "results": records}, f, indent=2)

    n = len(samples)
    t0 = time.time()
    for si, s in enumerate(samples):
        if s["pair_key"] in done:
            continue
        img = s["image"].convert("RGB")
        prompt = _question(s["caption"])

        if echo_scale is not None:
            w, h = img.size
            scaled = img.resize(
                (max(1, round(w * echo_scale)), max(1, round(h * echo_scale))),
                Image.LANCZOS)
            img_arg = [scaled, img] if echo_which == "first" else [img, scaled]
        else:
            img_arg = img

        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [prompt], img_arg, system_prompt=system_prompt, order=order,
        )
        with torch.inference_mode():
            out = mm.llm_model.generate(
                input_ids, do_sample=False, num_beams=1,
                max_new_tokens=max_tokens, use_cache=True, **kwargs,
            )
        gen = out[:, input_ids.shape[1]:]
        raw = mm.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        pred = _pred_yes_no(raw)

        records.append({
            "pair_key": s["pair_key"],
            "example_id": s["example_id"],
            "image_idx": s["image_idx"],
            "caption_idx": s["caption_idx"],
            "caption": s["caption"],
            "gt": s["gt"],
            "model_answer_raw": raw,
            "pred": pred,
            "correct": pred == s["gt"],
        })

        if (si + 1) % log_every == 0 or (si + 1) == n:
            m = metrics(records)
            rate = (len(records) - len(done)) / max(1e-9, time.time() - t0)
            print(f"  [{order}] [{si+1}/{n}] acc={m['acc']:.3f} g_acc={m['group_acc']:.3f} "
                  f"f1={m['f1']:.3f} yes={m['yes_ratio']:.3f} ({rate:.2f} q/s)", flush=True)
        if results_path and (si + 1) % checkpoint_every == 0:
            _checkpoint()

    meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records,
                       order_tag=tag, echo_scale=echo_scale, echo_which=echo_which)
    if out_dir:
        _checkpoint()
    return meta, records


def main():
    ap = argparse.ArgumentParser(description="Evaluate a VLM on Winoground (IST/STI/STIT).")
    ap.add_argument("--model", default="qwen3-vl-8b", choices=MODEL_CHOICES)
    ap.add_argument("--order", default="IST",
                    help="Comma-separated orderings, e.g. IST,STI,STIT")
    ap.add_argument("--max-examples", type=int, default=None, dest="max_examples",
                    help="Subsample N examples (default: all 400).")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--out-dir", default="./winoground/results", dest="out_dir")
    ap.add_argument("--checkpoint-every", type=int, default=200, dest="checkpoint_every")
    ap.add_argument("--echo-scale", type=float, default=None, dest="echo_scale",
                    help="Scale ONE of the two image occurrences (e.g. 0.5) in a "
                         "2-image order like SITIT; requires --echo-which.")
    ap.add_argument("--echo-which", default=None, dest="echo_which",
                    choices=["first", "second"],
                    help="Which occurrence --echo-scale applies to.")
    ap.add_argument("--tag-suffix", default="", dest="tag_suffix",
                    help="Suffix for the output filename, e.g. 'echo2half'.")
    args = ap.parse_args()

    if args.echo_scale is not None and not args.echo_which:
        ap.error("--echo-scale requires --echo-which {first,second}")
    if args.echo_scale is not None and not args.tag_suffix:
        frac = "half" if args.echo_scale == 0.5 else f"{args.echo_scale:g}x"
        occ = "1" if args.echo_which == "first" else "2"
        args.tag_suffix = f"echo{occ}{frac}"

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    orders = [o.strip().upper() for o in args.order.split(",") if o.strip()]
    print(f"[data] loading Winoground (max_examples={args.max_examples}) …", flush=True)
    samples = load_winoground_samples(max_examples=args.max_examples)
    print(f"[data] {len(samples)} pairs ({len(samples)//4} examples) under orders {orders}")
    print(f"[load] model = {args.model}")
    mm = ModelManager(args.model)

    for order in orders:
        tag = f"{order}_{args.tag_suffix}" if args.tag_suffix else order
        print(f"\n=== {tag} ===")
        meta, _ = run_winoground_experiment(
            mm, samples, order, SYSTEM_MESSAGE, max_tokens=args.max_tokens,
            out_dir=args.out_dir, checkpoint_every=args.checkpoint_every,
            order_tag=tag, echo_scale=args.echo_scale, echo_which=args.echo_which,
        )
        o = meta["overall"]
        print(f"  [done] {tag}: acc={o['acc']:.3f} group_acc={o['group_acc']:.3f} "
              f"f1={o['f1']:.3f}")


if __name__ == "__main__":
    main()
