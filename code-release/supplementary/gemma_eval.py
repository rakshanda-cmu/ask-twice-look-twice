"""
Standalone NaturalBench runner for Gemma 3 (multimodal) — IST / STI / STIT.

Kept SEPARATE from naturalbench_eval.py (the Qwen pipeline) on purpose: all
Gemma-specific run logic lives here, so the proven Qwen script is never touched.
Only model-agnostic helpers (judging, metrics, group loading, output writing) are
imported from naturalbench_eval.

Gemma's weights come via the gated repo google/gemma-3-27b-it; the 27B is sharded
across both GPUs (device_map="auto") by model_manager.load_gemma_model.

Run (resumable — checkpoints every 50 groups, skips groups already done):
    CUDA_VISIBLE_DEVICES=0,1 python gemma_eval.py \
        --model gemma-3-27b --order IST,STI,STIT --num-groups 1900

Output files (same schema/dir as the Qwen runs, distinct model prefix):
    naturalbench/results/gemma-3-27b__IST__results.json   (+ correct/wrong)
"""

import argparse
import json
import os
import time

from constants import SYSTEM_MESSAGE
# model-agnostic helpers shared with the Qwen pipeline (no Gemma logic there)
from naturalbench_eval import (
    PAIRS, answer_suffix, judge_pair, summarize_group, aggregate_metrics,
    load_groups, _question_for, _gt_for, _full_tag, order_tag,
    write_experiment_outputs,
)

GEMMA_CHOICES = ["gemma-3-27b", "gemma-3-12b", "gemma-3-4b"]


def _build_meta(model_name, order, system_prompt, max_tokens, records):
    return {
        "model": model_name,
        "order": order,
        "n_spaces": 0,
        "order_tag": _full_tag(order, 0, ""),
        "resize": None,
        "resize_mode": None,
        "image_copies": 1,
        "cue_mode": False,
        "think": False,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        **aggregate_metrics(records),
    }


def run_gemma_experiment(mm, groups, order, system_prompt,
                         nb_dir="./naturalbench", max_tokens=16,
                         out_dir=None, checkpoint_every=50, log_every=10):
    """Evaluate `groups` under one ordering for a Gemma model. Resumes from an
    existing results file in out_dir (skips done groups)."""
    import torch
    from PIL import Image

    tag = _full_tag(order, 0, "")
    records, done, results_path = [], set(), None
    if out_dir:
        results_path = os.path.join(out_dir, f"{mm.model_name}__{tag}__results.json")
        if os.path.exists(results_path):
            try:
                records = json.load(open(results_path))["results"]
                done = {r["index"] for r in records}
                print(f"  [resume] {tag}: {len(done)} groups already done")
            except Exception:
                records, done = [], set()

    def _checkpoint():
        if results_path:
            meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records)
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
        qtype = g["question_type"]
        pair_correct, pair_list = {}, []

        for (img_i, q_j) in PAIRS:
            question = _question_for(g, q_j)
            gt = _gt_for(g, img_i, q_j)
            prompt = question + answer_suffix(qtype)

            _, input_ids, kwargs = mm.prepare_inputs_from_pil(
                [prompt], imgs[img_i], system_prompt=system_prompt, order=order,
            )
            with torch.inference_mode():
                out = mm.llm_model.generate(
                    input_ids, do_sample=False, num_beams=1,
                    max_new_tokens=max_tokens, use_cache=True, **kwargs,
                )
            gen = out[:, input_ids.shape[1]:]              # decoder-only: slice prompt
            model_answer = mm.tokenizer.batch_decode(
                gen, skip_special_tokens=True)[0].strip()

            correct, pred = judge_pair(model_answer, gt, qtype, question)
            pair_correct[(img_i, q_j)] = correct
            pair_list.append({
                "image_index": img_i, "question_index": q_j,
                "image_file": g[f"image_{img_i}"], "question": question,
                "gt_answer": gt, "model_answer_raw": model_answer,
                "model_pred": pred, "correct": correct,
            })

        flags = summarize_group(pair_correct)
        records.append({
            "index": g["index"], "question_type": qtype,
            "question_0": g["question_0"], "question_1": g["question_1"],
            "image_0": g["image_0"], "image_1": g["image_1"],
            "pairs": pair_list, **flags,
        })

        if (gi + 1) % log_every == 0 or (gi + 1) == n:
            m = aggregate_metrics(records)
            rate = (len(records) - len(done)) / max(1e-9, time.time() - t0)
            print(f"  [{tag}] [{gi+1}/{n}] g_acc={m['g_acc']:.3f} "
                  f"q_acc={m['q_acc']:.3f} i_acc={m['i_acc']:.3f} "
                  f"pair_acc={m['pair_acc']:.3f} ({rate:.2f} grp/s)", flush=True)
        if results_path and (gi + 1) % checkpoint_every == 0:
            _checkpoint()

    meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records)
    if out_dir:
        write_experiment_outputs(meta, records, out_dir)
    return meta, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b", choices=GEMMA_CHOICES)
    ap.add_argument("--order", default="IST",
                    help="Comma-separated orderings, e.g. IST,STI,STIT")
    ap.add_argument("--num-groups", type=int, default=1900, dest="num_groups")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--nb-dir", default="./naturalbench", dest="nb_dir")
    ap.add_argument("--out-dir", default="./naturalbench/results", dest="out_dir")
    ap.add_argument("--checkpoint-every", type=int, default=50, dest="checkpoint_every")
    args = ap.parse_args()

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    orders = [o.strip().upper() for o in args.order.split(",") if o.strip()]
    groups = load_groups(args.nb_dir)[:args.num_groups]
    print(f"[data] evaluating {len(groups)} groups under orders {orders}")
    print(f"[load] model = {args.model}")
    mm = ModelManager(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    for order in orders:
        print(f"\n=== {order} ===")
        meta, _ = run_gemma_experiment(
            mm, groups, order, SYSTEM_MESSAGE, nb_dir=args.nb_dir,
            max_tokens=args.max_tokens, out_dir=args.out_dir,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"  [done] {order}: g_acc={meta['g_acc']:.3f} "
              f"pair_acc={meta['pair_acc']:.3f}")


if __name__ == "__main__":
    main()
