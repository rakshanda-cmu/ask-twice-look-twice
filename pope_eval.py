"""
Standalone POPE (object-hallucination) runner — IST / STI / STIT.

POPE asks yes/no questions "Is there a <object> in the image?" with three negative-
sampling regimes (random / popular / adversarial). It directly measures object
hallucination, so it complements NaturalBench: same prompt-ordering manipulation
(IST / STI / STIT, driven by model_manager's order string), new failure mode.

Kept FULLY SEPARATE from the NaturalBench pipeline — this script does NOT import or
modify naturalbench_eval beyond two stateless, model-agnostic helpers (the yes/no
parser and the shared yes/no answer suffix). NaturalBench's script, results and tab
are untouched. Reuses model loading + the IST/STI/STIT ordering machinery from
model_manager (so the S/I/T positional semantics are identical to the NB runs).

Data: HuggingFace `lmms-lab/POPE` (test split, 9000 rows = 3000 per category, COCO),
images embedded — no separate COCO download needed.

Run (resumable; checkpoints every --checkpoint-every, skips done question_ids):
    CUDA_VISIBLE_DEVICES=1 python pope_eval.py --model gemma-3-27b --order IST,STI,STIT
    CUDA_VISIBLE_DEVICES=0 python pope_eval.py --model qwen3-vl-8b --order IST,STI,STIT

Output (separate dir; one file per model+order):
    pope/results/<model>__<order>__results.json
"""

import argparse
import json
import os
import time

from constants import SYSTEM_MESSAGE
# model-agnostic helpers shared with the Qwen/NB pipeline (no NB logic pulled in)
from naturalbench_eval import _first_yes_no, YESNO_SUFFIX

MODEL_CHOICES = ["gemma-3-27b", "gemma-3-12b", "gemma-3-4b",
                 "qwen3-vl-8b", "qwen2.5-vl-7b", "llava-1.5"]
CATEGORIES = ["random", "popular", "adversarial"]


def _pred_yes_no(raw):
    """Normalize a model reply to 'yes' / 'no'. Uses the shared first-token parser;
    falls back to POPE's convention (a 'no'/'not' anywhere → no, else yes)."""
    p = _first_yes_no(raw)
    if p is not None:
        return p
    low = raw.lower()
    return "no" if ("no" in low or "not" in low) else "yes"


def pope_metrics(records):
    """Accuracy / precision / recall / F1 / yes-ratio with 'yes' as the positive
    class (object present) — the standard POPE reporting."""
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
    return {
        "n": n,
        "acc": round(correct / n, 4) if n else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_pred / n, 4) if n else 0.0,
    }


def _build_meta(model_name, order, system_prompt, max_tokens, records):
    by_cat = {c: pope_metrics([r for r in records if r["category"] == c])
              for c in CATEGORIES}
    return {
        "model": model_name,
        "order": order,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "overall": pope_metrics(records),
        "by_category": by_cat,
    }


def load_pope_samples(max_per_category=None, cache_dir=None):
    """Return a list of {question_id, category, question, answer, image} from
    lmms-lab/POPE (test split). max_per_category subsamples each regime."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", cache_dir=cache_dir)
    seen = {c: 0 for c in CATEGORIES}
    samples = []
    for row in ds:
        cat = str(row["category"]).lower()
        if cat not in seen:
            continue
        if max_per_category is not None and seen[cat] >= max_per_category:
            continue
        seen[cat] += 1
        samples.append({
            "question_id": row["question_id"],
            "category": cat,
            "question": row["question"],
            "answer": str(row["answer"]).strip().lower(),
            "image": row["image"],          # PIL.Image (RGB-convertible)
        })
    return samples


def run_pope_experiment(mm, samples, order, system_prompt,
                        max_tokens=16, out_dir="./pope/results",
                        checkpoint_every=200, log_every=100):
    """Evaluate POPE `samples` under one ordering. Resumes from an existing results
    file in out_dir (skips done question_ids)."""
    import torch

    records, done, results_path = [], set(), None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{mm.model_name}__{order}__results.json")
        if os.path.exists(results_path):
            try:
                records = json.load(open(results_path))["results"]
                done = {r["question_id"] for r in records}
                print(f"  [resume] {order}: {len(done)} samples already done")
            except Exception:
                records, done = [], set()

    def _checkpoint():
        if results_path:
            meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records)
            with open(results_path, "w") as f:
                json.dump({"meta": meta, "results": records}, f, indent=2)

    n = len(samples)
    t0 = time.time()
    for si, s in enumerate(samples):
        if s["question_id"] in done:
            continue
        img = s["image"].convert("RGB")
        prompt = s["question"] + YESNO_SUFFIX

        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [prompt], img, system_prompt=system_prompt, order=order,
        )
        with torch.inference_mode():
            out = mm.llm_model.generate(
                input_ids, do_sample=False, num_beams=1,
                max_new_tokens=max_tokens, use_cache=True, **kwargs,
            )
        gen = out[:, input_ids.shape[1]:]              # decoder-only: slice prompt
        raw = mm.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        pred = _pred_yes_no(raw)

        records.append({
            "question_id": s["question_id"],
            "category": s["category"],
            "question": s["question"],
            "gt": s["answer"],
            "model_answer_raw": raw,
            "pred": pred,
            "correct": pred == s["answer"],
        })

        if (si + 1) % log_every == 0 or (si + 1) == n:
            m = pope_metrics(records)
            rate = (len(records) - len(done)) / max(1e-9, time.time() - t0)
            print(f"  [{order}] [{si+1}/{n}] acc={m['acc']:.3f} f1={m['f1']:.3f} "
                  f"yes={m['yes_ratio']:.3f} ({rate:.2f} q/s)", flush=True)
        if results_path and (si + 1) % checkpoint_every == 0:
            _checkpoint()

    meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records)
    if out_dir:
        _checkpoint()
    return meta, records


def main():
    ap = argparse.ArgumentParser(description="Evaluate a VLM on POPE (IST/STI/STIT).")
    ap.add_argument("--model", default="gemma-3-27b", choices=MODEL_CHOICES)
    ap.add_argument("--order", default="IST",
                    help="Comma-separated orderings, e.g. IST,STI,STIT")
    ap.add_argument("--max-per-category", type=int, default=None,
                    dest="max_per_category",
                    help="Subsample N per regime (default: all 3000 each = 9000).")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--out-dir", default="./pope/results", dest="out_dir")
    ap.add_argument("--checkpoint-every", type=int, default=200, dest="checkpoint_every")
    args = ap.parse_args()

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    orders = [o.strip().upper() for o in args.order.split(",") if o.strip()]
    print(f"[data] loading POPE (max_per_category={args.max_per_category}) …", flush=True)
    samples = load_pope_samples(max_per_category=args.max_per_category)
    by_cat = {c: sum(1 for s in samples if s["category"] == c) for c in CATEGORIES}
    print(f"[data] {len(samples)} samples {by_cat} under orders {orders}")
    print(f"[load] model = {args.model}")
    mm = ModelManager(args.model)

    for order in orders:
        print(f"\n=== {order} ===")
        meta, _ = run_pope_experiment(
            mm, samples, order, SYSTEM_MESSAGE, max_tokens=args.max_tokens,
            out_dir=args.out_dir, checkpoint_every=args.checkpoint_every,
        )
        o = meta["overall"]
        print(f"  [done] {order}: acc={o['acc']:.3f} f1={o['f1']:.3f} "
              f"yes_ratio={o['yes_ratio']:.3f}")


if __name__ == "__main__":
    main()
