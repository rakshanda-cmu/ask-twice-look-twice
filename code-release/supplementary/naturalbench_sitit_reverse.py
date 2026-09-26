"""
SITIT-with-reversed-2nd-image ablation on NaturalBench (Qwen3-VL-8B) — SEPARATE,
additive runner. Layout:  S · Image · Task · **ReverseImage** · Task   (order SITIT,
2nd image block's patches reversed).

Reuses the validated reverse hooks from naturalbench_stiti_reverse.py (reverse the
2nd image's pooler embeds + DeepStack + M-RoPE positions together, so each patch
keeps its true content/position, just read back-to-front) — the exact mechanism used
for STITI-reverse, now applied to the SITIT ordering. No existing code changed.

Writes standard NaturalBench result files → shows in the NaturalBench ablations
table as the "SITIT_rev" column:
    naturalbench/results/qwen3-vl-8b__SITIT_rev__results.json

Run (resumable):  CUDA_VISIBLE_DEVICES=0 python naturalbench_sitit_reverse.py --num-groups 1900
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

from constants import SYSTEM_MESSAGE
from naturalbench_eval import (
    PAIRS, answer_suffix, judge_pair, summarize_group, aggregate_metrics,
    load_groups, _question_for, _gt_for, write_experiment_outputs,
)
from reverse_image_hooks import install_reverse_hooks, REVERSE

ORDER_TAG = "SITIT_rev"
ORDER = "SITIT"          # S·I·T·I·T — 2nd image block gets reversed by the hooks


def _build_meta(model_name, records, max_tokens):
    return {
        "model": model_name, "order": ORDER_TAG, "order_tag": ORDER_TAG,
        "n_spaces": 0, "resize": None, "resize_mode": None, "image_copies": 1,
        "cue_mode": False, "think": False, "reverse_second_image": True,
        "base_order": ORDER,
        "system_prompt": SYSTEM_MESSAGE, "max_tokens": max_tokens,
        **aggregate_metrics(records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
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
    setup_seeds(); disable_torch_init()

    groups = load_groups(args.nb_dir)[:args.num_groups]
    print(f"[data] SITIT-reverse (S·I·T·Ī·T) over {len(groups)} groups")
    mm = ModelManager(args.model)
    install_reverse_hooks(mm)
    REVERSE["on"] = True
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, f"{mm.model_name}__{ORDER_TAG}__results.json")
    records, done = [], set()
    if os.path.exists(results_path):
        try:
            records = json.load(open(results_path))["results"]
            done = {r["index"] for r in records}
            print(f"  [resume] {len(done)} groups already done")
        except Exception:
            records, done = [], set()

    def _checkpoint():
        write_experiment_outputs(_build_meta(mm.model_name, records, args.max_tokens),
                                 records, args.out_dir)

    n = len(groups)
    t0 = time.time()
    for gi, g in enumerate(groups):
        if g["index"] in done:
            continue
        imgs = {0: Image.open(os.path.join(args.nb_dir, g["image_0"])).convert("RGB"),
                1: Image.open(os.path.join(args.nb_dir, g["image_1"])).convert("RGB")}
        qtype = g["question_type"]
        pair_correct, pair_list = {}, []
        for (img_i, q_j) in PAIRS:
            question = _question_for(g, q_j)
            gt = _gt_for(g, img_i, q_j)
            prompt = question + answer_suffix(qtype)
            _, input_ids, kwargs = mm.prepare_inputs_from_pil(
                [prompt], imgs[img_i], system_prompt=SYSTEM_MESSAGE, order=ORDER)
            with torch.inference_mode():
                out = mm.llm_model.generate(input_ids, do_sample=False, num_beams=1,
                                            max_new_tokens=args.max_tokens,
                                            use_cache=True, **kwargs)
            model_answer = mm.tokenizer.batch_decode(
                out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
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
            "index": g["index"], "question_type": qtype, "source": g["source"],
            "image_0": g["image_0"], "image_1": g["image_1"],
            "question_0": g["question_0"], "question_1": g["question_1"],
            "pairs": pair_list, **flags,
        })
        if (gi + 1) % 10 == 0 or (gi + 1) == n:
            m = aggregate_metrics(records)
            rate = len(records) / max(1e-9, time.time() - t0)
            print(f"  [SITIT_rev] [{gi+1}/{n}] g_acc={m['g_acc']:.3f} q_acc={m['q_acc']:.3f} "
                  f"i_acc={m['i_acc']:.3f} pair_acc={m['pair_acc']:.3f} ({rate:.2f} grp/s)",
                  flush=True)
        if len(records) % args.checkpoint_every == 0:
            _checkpoint()

    _checkpoint()
    m = aggregate_metrics(records)
    print(f"  [done] SITIT_rev: g_acc={m['g_acc']:.3f} pair_acc={m['pair_acc']:.3f}")


if __name__ == "__main__":
    main()
