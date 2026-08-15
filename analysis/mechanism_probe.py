"""
Mechanistic probe for the STI/STIT/IST orderings.

For each (image, question) pair under each ordering we do ONE eager forward pass
on the prompt (ending at the answer position) and extract, per layer:

  • attention mass the answer position places on the QUESTION / IMAGE / SYSTEM
    token spans (mean over heads)  → "does the answer reach the question?"
  • logit-lens P(correct answer token) from the pre-answer hidden state
    → "answer emergence across layers"

Aggregated across pairs, this tests the top-down hypothesis:
  - STI: answer under-attends to the (far) question; answer emerges late/weakly.
  - STIT/IST: answer attends to the adjacent question; emerges earlier/stronger.

Run:
    CUDA_VISIBLE_DEVICES=0 python mechanism_probe.py --num-pairs 120
Outputs: naturalbench/probe/probe_results.json  (+ figures via make_probe_figs.py)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from core.model_manager import ModelManager
from core.utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hf_logging
from core.constants import SYSTEM_MESSAGE
from benchmarks.naturalbench_eval import answer_suffix, judge_pair
from ordering_variants.naturalbench_stiti_reverse import install_reverse_hooks, REVERSE

PROBE_DIR = "./naturalbench/probe"
ORDERS = ("STI", "SIT", "STIT", "SITIT", "SITIT_rev")


def _load(tag):
    p = f"naturalbench/results/qwen3-vl-8b__{tag}__results.json"
    return {g["index"]: g for g in json.load(open(p))["results"]}


def _pair_tuple(gi, pi):
    return (gi["index"], pi["image_index"], pi["question_index"],
            gi[f"image_{pi['image_index']}"], pi["question"], pi["gt_answer"])


def disagreement_pairs(limit):
    """yes/no pairs where STI (question-first) is wrong but SIT (question-last)
    is right (the diagnostic set; IST dropped)."""
    sit, sti = _load("SIT"), _load("STI")
    out = []
    for gi in sit.values():
        if gi["question_type"] != "yes_no":
            continue
        gs = sti.get(gi["index"])
        if not gs:
            continue
        bys = {(p["image_index"], p["question_index"]): p for p in gs["pairs"]}
        for pi in gi["pairs"]:
            ps = bys.get((pi["image_index"], pi["question_index"]))
            if ps and pi["correct"] and not ps["correct"]:
                out.append(_pair_tuple(gi, pi))
    return out[:limit]


def neutral_pairs(limit):
    """A deterministic, outcome-independent sample of yes/no pairs (clean set)."""
    sit = _load("SIT")
    out = []
    for gidx in sorted(sit):           # deterministic order
        gi = sit[gidx]
        if gi["question_type"] != "yes_no":
            continue
        for pi in gi["pairs"]:
            out.append(_pair_tuple(gi, pi))
    # even stride across the dataset for a representative sample
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def _find_span(ids, sub):
    """All start positions where list `sub` occurs contiguously in `ids`."""
    n, m = len(ids), len(sub)
    hits = []
    for i in range(n - m + 1):
        if ids[i:i + m] == sub:
            hits.append((i, i + m))
    return hits


@torch.no_grad()
def probe_pair(mm, img, question, gt, order):
    """Return per-layer dicts for one pair under one ordering, or None.

    order 'SITIT_rev' uses the SITIT layout with the 2nd image block reversed
    (patches, DeepStack features, and M-RoPE positions), via the shared reverse
    hooks; every other order runs stock (REVERSE off).
    """
    qtext = question + answer_suffix("yes_no")
    # STIT = full question repeated after the image (task2_text=None => second T
    # reuses the full question).
    reverse = order.endswith("_rev")
    base_order = "SITIT" if reverse else order
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [qtext], img, system_prompt=SYSTEM_MESSAGE, order=base_order)
    ids = input_ids[0].tolist()

    # spans: image (from model_manager), question (subsequence match)
    img_span = set(range(mm.img_start_idx, mm.img_end_idx))
    q_ids = mm.tokenizer(question, add_special_tokens=False).input_ids
    q_hits = _find_span(ids, q_ids) or _find_span(ids, q_ids[1:-1] or q_ids)
    q_pos = set(p for s, e in q_hits for p in range(s, e))
    if not q_pos:
        return None
    sys_ids = mm.tokenizer(SYSTEM_MESSAGE, add_special_tokens=False).input_ids
    sys_hits = _find_span(ids, sys_ids)
    sys_pos = set(p for s, e in sys_hits for p in range(s, e))

    REVERSE["on"] = reverse
    try:
        out = mm.llm_model(input_ids, output_hidden_states=True,
                           output_attentions=True, **kwargs)
    finally:
        REVERSE["on"] = False
    hs, att = out.hidden_states, out.attentions
    n_layers = len(att)

    # GT answer token id (single-token "Yes"/"No")
    gt_word = "Yes" if gt.strip().lower() == "yes" else "No"
    gt_id = mm.tokenizer(gt_word, add_special_tokens=False).input_ids[0]

    def mass(att_layer, pos):
        if not pos:
            return 0.0
        row = att_layer[0, :, -1, :].float().mean(0)  # mean over heads, last-token row
        return float(row[list(pos)].sum())

    a_q = [mass(att[l], q_pos) for l in range(n_layers)]
    a_img = [mass(att[l], img_span) for l in range(n_layers)]
    a_sys = [mass(att[l], sys_pos) for l in range(n_layers)]

    # answer emergence: logit-lens P(gt token) at the last position, per layer
    p_gt = []
    for l in range(1, len(hs)):           # skip embedding layer
        logits = mm.llm_model.lm_head(hs[l][0, -1].unsqueeze(0).to(
            mm.llm_model.lm_head.weight.dtype)).float()[0]
        p_gt.append(float(torch.softmax(logits, -1)[gt_id]))

    # final answer the model would give + correctness
    final_logits = mm.llm_model.lm_head(hs[-1][0, -1].unsqueeze(0).to(
        mm.llm_model.lm_head.weight.dtype)).float()[0]
    pred_tok = mm.tokenizer.decode([int(final_logits.argmax())]).strip()
    correct, _ = judge_pair(pred_tok, gt, "yes_no", question)
    return {"a_q": a_q, "a_img": a_img, "a_sys": a_sys, "p_gt": p_gt,
            "n_q": len(q_pos), "n_img": len(img_span), "correct": correct,
            "seq_len": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=120, dest="num_pairs")
    ap.add_argument("--set", default="disagreement", dest="pset",
                    choices=["disagreement", "neutral"],
                    help="'disagreement' = STI-wrong/IST-right (diagnostic); "
                         "'neutral' = outcome-independent sample (clean).")
    ap.add_argument("--img-cap", type=int, default=640, dest="img_cap",
                    help="thumbnail images to <= this longest side (bound seq).")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()
    os.makedirs(PROBE_DIR, exist_ok=True)

    pairs = (neutral_pairs(args.num_pairs) if args.pset == "neutral"
             else disagreement_pairs(args.num_pairs))
    print(f"[data] {len(pairs)} {args.pset} yes/no pairs")
    print("[load] qwen3-vl-8b (eager attention)")
    mm = ModelManager("qwen3-vl-8b", attn_implementation="eager")
    install_reverse_hooks(mm)   # active only when REVERSE['on'] (SITIT_rev)

    agg = {o: {"a_q": [], "a_img": [], "a_sys": [], "p_gt": [], "correct": []}
           for o in ORDERS}
    t0 = time.time()
    n_ok = 0
    for n, (g, i, q, imgrel, question, gt) in enumerate(pairs, 1):
        img = Image.open(os.path.join("naturalbench", imgrel)).convert("RGB")
        img.thumbnail((args.img_cap, args.img_cap), Image.LANCZOS)
        try:
            res = {o: probe_pair(mm, img, question, gt, o) for o in ORDERS}
        except Exception as e:
            print(f"  [{n}] err: {type(e).__name__} {str(e)[:80]}")
            continue
        if any(r is None for r in res.values()):
            continue
        n_ok += 1
        for o in ORDERS:
            for k in ("a_q", "a_img", "a_sys", "p_gt"):
                agg[o][k].append(res[o][k])
            agg[o]["correct"].append(res[o]["correct"])
        if n % 10 == 0:
            print(f"  [{n}/{len(pairs)}] ok={n_ok} ({(n)/(time.time()-t0):.2f}/s)")

    # aggregate (mean over examples, per layer)
    summary = {"n_pairs": n_ok, "pset": args.pset, "orders": {}}
    for o in ORDERS:
        d = agg[o]
        summary["orders"][o] = {
            "a_q": np.mean(d["a_q"], 0).tolist() if d["a_q"] else [],
            "a_img": np.mean(d["a_img"], 0).tolist() if d["a_img"] else [],
            "a_sys": np.mean(d["a_sys"], 0).tolist() if d["a_sys"] else [],
            "p_gt": np.mean(d["p_gt"], 0).tolist() if d["p_gt"] else [],
            "acc": float(np.mean(d["correct"])) if d["correct"] else 0.0,
        }
    json.dump(summary, open(os.path.join(PROBE_DIR, f"probe_{args.pset}.json"), "w"),
              indent=2)
    print(f"\n[done] {n_ok} pairs probed ({(time.time()-t0)/60:.1f} min)")
    for o in ORDERS:
        s = summary["orders"][o]
        if s["a_q"]:
            print(f"  {o}: max answer→question attn {max(s['a_q']):.3f} · "
                  f"final P(gt) {s['p_gt'][-1]:.3f} · acc {s['acc']:.2f}")


if __name__ == "__main__":
    main()
