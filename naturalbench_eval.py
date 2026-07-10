"""
Run a VLM over NaturalBench and record, per group, whether the model answered
each of the 4 (image, question) pairs correctly — for a given prompt *ordering*.

This module powers the two requested experiments:
    Experiment 1 — IST : Image · System · Task
    Experiment 2 — STI : System · Task · Image

For each experiment we write three JSON files:
    <model>__<order>__results.json   all groups + metrics (used by the UI)
    <model>__<order>__correct.json   groups the model got fully right (all 4 pairs)
    <model>__<order>__wrong.json     groups with >=1 wrong pair

Pairs per group (image i, question j):
    (0,0) (1,0) (0,1) (1,1)
Metrics (official NaturalBench):
    pair_acc : fraction of all 4N pairs correct
    q_acc    : a question is right iff BOTH images answered correctly  (2N units)
    i_acc    : an image is right iff BOTH questions answered correctly (2N units)
    g_acc    : a group is right iff ALL 4 pairs correct                (N units)

CLI:
    python naturalbench_eval.py --order IST --num-groups 100 --model qwen3-vl-8b
    python naturalbench_eval.py --order STI --num-groups 100 --model qwen3-vl-8b

Programmatic (used by the Streamlit UI):
    from naturalbench_eval import run_experiment
    meta = run_experiment(model_manager, groups, order="IST",
                          system_prompt=..., progress_cb=...)
"""

import argparse
import json
import os
import re
import time

from constants import SYSTEM_MESSAGE

# Pairs in a stable display order: (image_index, question_index)
PAIRS = [(0, 0), (1, 0), (0, 1), (1, 1)]

YESNO_SUFFIX = "\nAnswer the question using only Yes or No."
MC_SUFFIX = "\nAnswer with only the option letter (A or B)."


# ──────────────────────────────────────────────────────────────────────────────
#  Judging
# ──────────────────────────────────────────────────────────────────────────────

def _first_yes_no(text):
    """Return 'yes' / 'no' for the first yes/no token in text, else None."""
    t = text.lower()
    m = re.search(r"(?<!\w)(yes|no)(?!\w)", t)
    return m.group(1) if m else None


def _parse_mc_options(question_text):
    """
    Parse 'Option: A:To the right; B:To the left;' -> {'a': 'to the right', ...}.
    Returns {} if no options found.
    """
    opts = {}
    for letter, body in re.findall(r"([A-Da-d])\s*:\s*([^;]+)", question_text):
        opts[letter.lower()] = body.strip().lower().rstrip(".")
    return opts


def _first_letter(text):
    """First standalone A/B/C/D option letter in the model output, else None."""
    m = re.search(r"(?<!\w)([A-Da-d])(?!\w)", text)
    return m.group(1).lower() if m else None


def judge_pair(model_answer, gt_answer, question_type, question_text):
    """
    Return (correct: bool, model_pred: str) for one (image, question) pair.

    model_pred is the normalized prediction we extracted (e.g. 'yes', 'a'), or
    the raw answer if nothing could be parsed.
    """
    gt = gt_answer.strip().lower()

    if question_type == "yes_no":
        pred = _first_yes_no(model_answer)
        if pred is None:
            return False, model_answer.strip()
        return pred == gt, pred

    # multiple_choice: gold answer is a letter (e.g. 'A')
    pred = _first_letter(model_answer)
    if pred is None:
        # fall back to matching the option *text* contained in the reply
        opts = _parse_mc_options(question_text)
        low = model_answer.lower()
        for letter, body in opts.items():
            if body and body in low:
                pred = letter
                break
    if pred is None:
        return False, model_answer.strip()
    return pred == gt, pred


def answer_suffix(question_type):
    return MC_SUFFIX if question_type == "multiple_choice" else YESNO_SUFFIX


# Explicit chain-of-thought prompting (the model's native enable_thinking toggle
# is a no-op for this template, so we elicit reasoning via the prompt).
COT_YESNO = ("\nReason briefly about the image step by step, then on the final "
             "line write exactly 'Final answer: Yes' or 'Final answer: No'.")
COT_MC = ("\nReason briefly step by step, then on the final line write exactly "
          "'Final answer: A' or 'Final answer: B'.")


def cot_suffix(question_type):
    return COT_MC if question_type == "multiple_choice" else COT_YESNO


def extract_final_answer(text):
    """Pull the answer out of a CoT response: text after the last 'final answer:'
    / 'answer:' (or after a </think> block), else the whole text."""
    hits = list(re.finditer(r"final answer\s*:|answer\s*:", text, re.I))
    if hits:
        return text[hits[-1].end():].strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
#  Group helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gt_for(group, image_idx, question_idx):
    return group[f"image_{image_idx}_question_{question_idx}"]


def _question_for(group, question_idx):
    return group[f"question_{question_idx}"]


def summarize_group(pair_results):
    """
    pair_results: dict {(img,q): correct_bool}.
    Returns per-group metric flags.
    """
    c = pair_results
    q0 = c[(0, 0)] and c[(1, 0)]
    q1 = c[(0, 1)] and c[(1, 1)]
    i0 = c[(0, 0)] and c[(0, 1)]
    i1 = c[(1, 0)] and c[(1, 1)]
    g = all(c[p] for p in PAIRS)
    n_correct = sum(int(c[p]) for p in PAIRS)
    return {
        "q0_correct": q0, "q1_correct": q1,
        "i0_correct": i0, "i1_correct": i1,
        "group_correct": g, "num_pairs_correct": n_correct,
    }


def aggregate_metrics(group_records):
    """Compute pair/q/i/g accuracy over all evaluated groups."""
    n = len(group_records)
    if n == 0:
        return {"num_groups": 0, "pair_acc": 0, "q_acc": 0, "i_acc": 0, "g_acc": 0}
    pair_total = pair_ok = 0
    q_ok = i_ok = g_ok = 0
    for r in group_records:
        for p in r["pairs"]:
            pair_total += 1
            pair_ok += int(p["correct"])
        q_ok += int(r["q0_correct"]) + int(r["q1_correct"])
        i_ok += int(r["i0_correct"]) + int(r["i1_correct"])
        g_ok += int(r["group_correct"])
    return {
        "num_groups": n,
        "pair_acc": round(pair_ok / pair_total, 4),
        "q_acc": round(q_ok / (2 * n), 4),
        "i_acc": round(i_ok / (2 * n), 4),
        "g_acc": round(g_ok / n, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_groups(nb_dir="./naturalbench"):
    path = os.path.join(nb_dir, "groups.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python download_naturalbench.py` first."
        )
    with open(path) as f:
        return json.load(f)["groups"]


# ──────────────────────────────────────────────────────────────────────────────
#  Core experiment runner (shared by CLI + Streamlit UI)
# ──────────────────────────────────────────────────────────────────────────────

def order_tag(order, n_spaces):
    """
    File/label tag for an ordering + space count:
        n=0  -> 'IST'
        n=20 -> 'IST_space'    (kept for back-compat with existing result files)
        else -> 'IST_space{n}' (e.g. 'IST_space5')
    """
    if not n_spaces or n_spaces <= 0:
        return order
    if n_spaces == 20:
        return f"{order}_space"
    return f"{order}_space{n_spaces}"


def _full_tag(order, n_spaces, tag_suffix=""):
    tag = order_tag(order, n_spaces)
    return f"{tag}_{tag_suffix}" if tag_suffix else tag


def _build_meta(model_manager, order, system_prompt, max_tokens, n_spaces,
                records, tag_suffix="", resize=None, resize_mode="exact",
                image_copies=1, cue_mode=False, think=False):
    return {
        "model": model_manager.model_name,
        "order": order,
        "n_spaces": int(n_spaces or 0),
        "order_tag": _full_tag(order, n_spaces, tag_suffix),
        "resize": list(resize) if resize else None,
        "resize_mode": resize_mode if resize else None,
        "image_copies": int(image_copies),
        "cue_mode": bool(cue_mode),
        "think": bool(think),
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        **aggregate_metrics(records),
    }


def run_experiment(model_manager, groups, order, system_prompt,
                   nb_dir="./naturalbench", max_tokens=16,
                   progress_cb=None, log_every=10,
                   n_spaces=0, out_dir=None, checkpoint_every=50,
                   resize=None, resize_mode="exact", tag_suffix="", image_copies=1,
                   cue_mode=False, think=False):
    """
    Evaluate `groups` under one prompt `order` (e.g. 'IST' or 'STI').

    n_spaces>0 inserts that many space tokens (after Task for IST / after the
    image for STI) via prepare_inputs_with_spaces.
    resize=(w, h) resizes every image to that fixed size before the model (e.g.
    the dataset mean resolution, to control vision-token count); tag_suffix names
    the output files (e.g. 'meanres').

    If out_dir is given, the results file is checkpointed every
    `checkpoint_every` groups and the run RESUMES by skipping groups already
    present in it (survives kills).

    Returns (meta, group_records). `progress_cb(frac, msg)` is optional.
    """
    import torch
    from PIL import Image
    from model_manager import QWEN_MODELS, INTERNVL_MODELS

    is_qwen = model_manager.model_name in (QWEN_MODELS + INTERNVL_MODELS)
    tag = _full_tag(order, n_spaces, tag_suffix)

    records, done, results_path = [], set(), None
    if out_dir:
        results_path = os.path.join(
            out_dir, f"{model_manager.model_name}__{tag}__results.json")
        if os.path.exists(results_path):
            try:
                records = json.load(open(results_path))["results"]
                done = {r["index"] for r in records}
                print(f"  [resume] {tag}: {len(done)} groups already done")
            except Exception:
                records, done = [], set()

    def _checkpoint():
        if results_path:
            meta = _build_meta(model_manager, order, system_prompt,
                               max_tokens, n_spaces, records,
                               tag_suffix=tag_suffix, resize=resize, resize_mode=resize_mode, image_copies=image_copies, cue_mode=cue_mode, think=think)
            write_experiment_outputs(meta, records, out_dir)

    n = len(groups)
    t0 = time.time()

    for gi, g in enumerate(groups):
        if g["index"] in done:
            continue
        imgs = {
            0: Image.open(os.path.join(nb_dir, g["image_0"])).convert("RGB"),
            1: Image.open(os.path.join(nb_dir, g["image_1"])).convert("RGB"),
        }
        if resize:
            if str(resize_mode) == "cap":
                # downscale-only: shrink images LARGER than `resize` to fit it
                # (aspect preserved); leave smaller images untouched.
                capped = {}
                for k, v in imgs.items():
                    v = v.copy()
                    v.thumbnail(tuple(resize), Image.LANCZOS)
                    capped[k] = v
                imgs = capped
            else:
                imgs = {k: v.resize(tuple(resize), Image.LANCZOS)
                        for k, v in imgs.items()}
        qtype = g["question_type"]
        pair_correct = {}
        pair_list = []

        for (img_i, q_j) in PAIRS:
            question = _question_for(g, q_j)
            gt = _gt_for(g, img_i, q_j)
            prompt = question + (cot_suffix(qtype) if think else answer_suffix(qtype))
            # short-cue control (order STIT): repeat only the answer instruction
            # after the image, NOT the question content.
            task2 = answer_suffix(qtype).strip() if cue_mode else None

            if n_spaces and n_spaces > 0:
                _, input_ids, kwargs = model_manager.prepare_inputs_with_spaces(
                    [prompt], imgs[img_i], system_prompt=system_prompt,
                    order=order, n_spaces=n_spaces,
                )
            else:
                _, input_ids, kwargs = model_manager.prepare_inputs_from_pil(
                    [prompt], imgs[img_i], system_prompt=system_prompt, order=order,
                    image_copies=image_copies, task2_text=task2,
                    enable_thinking=think,
                )
            with torch.inference_mode():
                out = model_manager.llm_model.generate(
                    input_ids, do_sample=False, num_beams=1,
                    max_new_tokens=max_tokens, use_cache=True, **kwargs,
                )
            if is_qwen:
                gen = out[:, input_ids.shape[1]:]
                model_answer = model_manager.tokenizer.batch_decode(
                    gen, skip_special_tokens=True)[0].strip()
            else:
                model_answer = model_manager.tokenizer.batch_decode(
                    out, skip_special_tokens=True)[0].strip()
                if "ASSISTANT:" in model_answer:
                    model_answer = model_answer.split("ASSISTANT:")[-1].strip()

            # in think mode judge only the final answer (after the reasoning)
            judge_text = extract_final_answer(model_answer) if think else model_answer
            correct, pred = judge_pair(judge_text, gt, qtype, question)
            pair_correct[(img_i, q_j)] = correct
            pair_list.append({
                "image_index": img_i,
                "question_index": q_j,
                "image_file": g[f"image_{img_i}"],
                "question": question,
                "gt_answer": gt,
                "model_answer_raw": model_answer,
                "model_pred": pred,
                "correct": correct,
            })

        flags = summarize_group(pair_correct)
        records.append({
            "index": g["index"],
            "question_type": qtype,
            "source": g["source"],
            "image_0": g["image_0"],
            "image_1": g["image_1"],
            "question_0": g["question_0"],
            "question_1": g["question_1"],
            "pairs": pair_list,
            **flags,
        })

        done.add(g["index"])
        if progress_cb:
            progress_cb((gi + 1) / n, f"{tag}: group {gi+1}/{n}")
        if (gi + 1) % log_every == 0 or (gi + 1) == n:
            agg = aggregate_metrics(records)
            rate = (gi + 1) / max(1e-9, time.time() - t0)
            print(f"  [{tag}] [{gi+1}/{n}] "
                  f"g_acc={agg['g_acc']:.3f} q_acc={agg['q_acc']:.3f} "
                  f"i_acc={agg['i_acc']:.3f} pair_acc={agg['pair_acc']:.3f} "
                  f"({rate:.2f} grp/s)")
        if results_path and len(records) % checkpoint_every == 0:
            _checkpoint()

    meta = _build_meta(model_manager, order, system_prompt,
                       max_tokens, n_spaces, records,
                       tag_suffix=tag_suffix, resize=resize, resize_mode=resize_mode, image_copies=image_copies, cue_mode=cue_mode, think=think)
    if results_path:
        _checkpoint()
    return meta, records


def write_experiment_outputs(meta, records, out_dir):
    """Write results / correct / wrong JSONs; return the three paths."""
    os.makedirs(out_dir, exist_ok=True)
    base = f"{meta['model']}__{meta.get('order_tag', meta['order'])}"

    results_path = os.path.join(out_dir, f"{base}__results.json")
    correct_path = os.path.join(out_dir, f"{base}__correct.json")
    wrong_path = os.path.join(out_dir, f"{base}__wrong.json")

    correct = [r for r in records if r["group_correct"]]
    wrong = [r for r in records if not r["group_correct"]]

    with open(results_path, "w") as f:
        json.dump({"meta": meta, "results": records}, f, indent=2)
    with open(correct_path, "w") as f:
        json.dump({"meta": {**meta, "subset": "correct", "count": len(correct)},
                   "results": correct}, f, indent=2)
    with open(wrong_path, "w") as f:
        json.dump({"meta": {**meta, "subset": "wrong", "count": len(wrong)},
                   "results": wrong}, f, indent=2)
    return results_path, correct_path, wrong_path


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate a VLM on NaturalBench.")
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "qwen2.5-vl-7b", "llava-1.5", "internvl3-8b"])
    ap.add_argument("--order", default="IST",
                    help="Prompt order, e.g. IST or STI. Both experiments: 'IST,STI'.")
    ap.add_argument("--num-groups", type=int, default=100, dest="num_groups")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--nb-dir", default="./naturalbench", dest="nb_dir")
    ap.add_argument("--out-dir", default="./naturalbench/results", dest="out_dir")
    ap.add_argument("--system-prompt", default=None, dest="system_prompt",
                    help="System text (default: repo SYSTEM_MESSAGE).")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--n-spaces", type=int, default=0, dest="n_spaces",
                    help="Insert N space tokens (after Task for IST / after image "
                         "for STI). Use 20 for the spaces experiment.")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    dest="checkpoint_every",
                    help="Checkpoint the results file every N groups (resume-safe).")
    ap.add_argument("--resize", default=None,
                    help="Resize images to WxH before the model, e.g. 1233x1070 "
                         "(controls vision-token count).")
    ap.add_argument("--resize-mode", default="exact", dest="resize_mode",
                    choices=["exact", "cap"],
                    help="'exact' resizes every image to WxH; 'cap' downscales only "
                         "images larger than WxH (aspect preserved), leaving smaller "
                         "images untouched.")
    ap.add_argument("--image-copies", type=int, default=1, dest="image_copies",
                    help="Repeat the image N times at the image position (distance "
                         "sweep). N>1 auto-tags files 'copies{N}' unless --tag-suffix.")
    ap.add_argument("--cue-mode", action="store_true", dest="cue_mode",
                    help="Short-cue STIT control: with --order STIT, the post-image "
                         "repeat is only the answer instruction, not the question "
                         "(auto-tags 'cue').")
    ap.add_argument("--think", action="store_true", dest="think",
                    help="Enable Qwen3-VL thinking/CoT before the answer; the "
                         "reasoning trace is stripped before judging. Auto-tags "
                         "'think' and bumps max-tokens to 512.")
    ap.add_argument("--tag-suffix", default="", dest="tag_suffix",
                    help="Suffix for output filenames/labels, e.g. 'meanres'.")
    args = ap.parse_args()

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    # auto-tag the image-copy sweep when no explicit suffix is given
    if args.image_copies > 1 and not args.tag_suffix:
        args.tag_suffix = f"copies{args.image_copies}"
    if args.cue_mode and not args.tag_suffix:
        args.tag_suffix = "cue"
    if args.think:
        if not args.tag_suffix:
            args.tag_suffix = "think"
        if args.max_tokens < 256:        # need room for the reasoning trace
            args.max_tokens = 512

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    system_prompt = SYSTEM_MESSAGE if args.system_prompt is None else args.system_prompt
    orders = [o.strip().upper() for o in args.order.split(",") if o.strip()]

    groups = load_groups(args.nb_dir)[args.start: args.start + args.num_groups]
    print(f"[data] evaluating {len(groups)} groups under orders {orders}")

    print(f"[load] model = {args.model}")
    mm = ModelManager(args.model)

    if args.n_spaces > 0:
        print(f"[spaces] inserting {args.n_spaces} space tokens per pair")

    for order in orders:
        tag = order_tag(order, args.n_spaces)
        print(f"\n===== Experiment {tag} =====")
        meta, records = run_experiment(
            mm, groups, order=order, system_prompt=system_prompt,
            nb_dir=args.nb_dir, max_tokens=args.max_tokens,
            n_spaces=args.n_spaces, out_dir=args.out_dir,
            checkpoint_every=args.checkpoint_every,
            resize=resize, resize_mode=args.resize_mode, tag_suffix=args.tag_suffix,
            image_copies=args.image_copies, cue_mode=args.cue_mode,
            think=args.think,
        )
        paths = write_experiment_outputs(meta, records, args.out_dir)
        print(json.dumps(meta, indent=2))
        for p in paths:
            print("[saved]", p)


if __name__ == "__main__":
    main()
