"""
Causal attention-knockout for the question-first paradox (Qwen3-VL-8B, NaturalBench).

The read-out probe (mechanism_probe.py) is correlational: under STI the answer
position *attends less* to the (far) question. This script tests the causal
claim directly. We sever the answer region's ability to read the question by
adding a pre-softmax -inf bias to every (query in the answer region, key in the
question span) attention edge, at all text layers, then measure accuracy.

Prediction (necessity):
  - SIT (question adjacent, well attended): knocking out answer->question
    attention should DROP accuracy toward STI. The SIT advantage is *caused* by
    the answer reading the question.
  - STI (question far, already under-attended): the same knockout removes little
    that was being used, so the drop is small.
Specificity controls:
  - ko_image:  knock out answer->image attention (same machinery, image span).
  - ko_random: knock out answer->(random same-size text span) (span-size control).

Mechanism: transformers 5.4 dispatches attention through
ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, ...). We
register a wrapped eager function under the key "eager_ko" and point the text
attention modules at it; a module-level KO dict gates the intervention so the
same loaded model serves clean and knocked-out passes.

Run:
    CUDA_VISIBLE_DEVICES=0 python causal_knockout.py --num-pairs 200
Outputs: naturalbench/probe/knockout.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS, repeat_kv)

from core.model_manager import ModelManager
from core.utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hf_logging
from core.constants import SYSTEM_MESSAGE
from benchmarks.naturalbench_eval import answer_suffix, judge_pair
from analysis.mechanism_probe import neutral_pairs, _find_span, PROBE_DIR

ORDERS = ("STI", "SIT")
CONDS = ("clean", "ko_question", "ko_image", "ko_random")

# module-level intervention state, read by ko_eager_attention_forward
KO = {"on": False, "key_pos": None, "q_pos": None}
NEG = torch.finfo(torch.float32).min / 4  # large negative, safe under fp32 softmax


def ko_eager_attention_forward(module, query, key, value, attention_mask,
                               scaling, dropout=0.0, **kwargs):
    """eager attention + optional pre-softmax knockout of (q_pos -> key_pos) edges."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    if KO["on"] and KO["key_pos"] is not None and len(KO["key_pos"]):
        qp, kp = KO["q_pos"], KO["key_pos"]
        attn_weights[:, :, qp.unsqueeze(-1), kp.unsqueeze(0)] += NEG

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                               dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def install_knockout(mm):
    """Register the KO attention fn and route text self-attention through it."""
    ALL_ATTENTION_FUNCTIONS["eager_ko"] = ko_eager_attention_forward
    n = 0
    for mod in mm.llm_model.modules():
        if mod.__class__.__name__ == "Qwen3VLTextAttention":
            mod.config._attn_implementation = "eager_ko"
            n += 1
    print(f"[knockout] routed {n} text attention layers through eager_ko")


def _spans(mm, ids, question, seq_len, q_scope, rng):
    """Return (q_pos_tensor, key spans dict) for one prompt. q_pos = the answer
    region (query side); key spans are the token sets to sever (key side)."""
    dev = mm.llm_model.device
    img_span = list(range(mm.img_start_idx, mm.img_end_idx))
    q_ids = mm.tokenizer(question, add_special_tokens=False).input_ids
    q_hits = _find_span(ids, q_ids) or _find_span(ids, q_ids[1:-1] or q_ids)
    if not q_hits:
        return None, None
    q_pos_all = sorted({p for s, e in q_hits for p in range(s, e)})
    q_end = max(e for s, e in q_hits)

    # query side: the answer region. 'last' = only the answer position;
    # 'downstream' = every position after the question (fully severs re-reading).
    if q_scope == "last":
        qpos = [seq_len - 1]
    else:
        qpos = list(range(q_end, seq_len))
    qpos = torch.tensor(qpos, device=dev, dtype=torch.long)

    # random control: a contiguous non-image, non-question span of equal length,
    # drawn from the text tokens before the image (deterministic per pair).
    qn = len(q_pos_all)
    banned = set(img_span) | set(q_pos_all) | set(range(q_end, seq_len))
    lo_choices = [i for i in range(1, max(2, mm.img_start_idx - qn))
                  if not (set(range(i, i + qn)) & banned)]
    rand_span = (list(range(lo_choices[rng.randrange(len(lo_choices))],
                            lo_choices[rng.randrange(len(lo_choices))] + qn))
                 if lo_choices else [])

    keys = {
        "ko_question": torch.tensor(q_pos_all, device=dev, dtype=torch.long),
        "ko_image": torch.tensor(img_span, device=dev, dtype=torch.long),
        "ko_random": torch.tensor(rand_span, device=dev, dtype=torch.long),
    }
    return qpos, keys


@torch.no_grad()
def run_pair(mm, img, question, gt, order, q_scope, rng, yes_id, no_id):
    """Return {cond: (correct, p_corr2)} for one pair under one ordering."""
    qtext = question + answer_suffix("yes_no")
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [qtext], img, system_prompt=SYSTEM_MESSAGE, order=order)
    ids = input_ids[0].tolist()
    seq_len = len(ids)
    qpos, keys = _spans(mm, ids, question, seq_len, q_scope, rng)
    if qpos is None:
        return None

    gt_yes = gt.strip().lower() == "yes"
    corr_id, wrong_id = (yes_id, no_id) if gt_yes else (no_id, yes_id)

    out = {}
    for cond in CONDS:
        KO["on"] = cond != "clean"
        KO["q_pos"] = qpos
        KO["key_pos"] = None if cond == "clean" else keys[cond]
        try:
            o = mm.llm_model(input_ids, **kwargs)
        finally:
            KO["on"] = False
            KO["key_pos"] = None
        logits = o.logits[0, -1].float()
        pred = mm.tokenizer.decode([int(logits.argmax())]).strip()
        correct, _ = judge_pair(pred, gt, "yes_no", question)
        lc, lw = float(logits[corr_id]), float(logits[wrong_id])
        m = max(lc, lw)
        p2 = float(np.exp(lc - m) / (np.exp(lc - m) + np.exp(lw - m)))
        out[cond] = (bool(correct), p2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=200, dest="num_pairs")
    ap.add_argument("--img-cap", type=int, default=640, dest="img_cap")
    ap.add_argument("--q-scope", default="downstream", choices=["downstream", "last"],
                    dest="q_scope", help="query side of the severed edge: the whole "
                    "answer region ('downstream') or only the answer token ('last')")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()
    os.makedirs(PROBE_DIR, exist_ok=True)
    rng = random.Random(0)

    pairs = neutral_pairs(args.num_pairs)
    print(f"[data] {len(pairs)} neutral yes/no pairs; q_scope={args.q_scope}")
    print("[load] qwen3-vl-8b (eager)")
    mm = ModelManager("qwen3-vl-8b", attn_implementation="eager")
    install_knockout(mm)
    yes_id = mm.tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = mm.tokenizer("No", add_special_tokens=False).input_ids[0]

    agg = {o: {c: {"correct": [], "p2": []} for c in CONDS} for o in ORDERS}
    t0 = time.time()
    n_ok = 0
    for n, (g, i, q, imgrel, question, gt) in enumerate(pairs, 1):
        img = Image.open(os.path.join("naturalbench", imgrel)).convert("RGB")
        img.thumbnail((args.img_cap, args.img_cap), Image.LANCZOS)
        try:
            res = {o: run_pair(mm, img, question, gt, o, args.q_scope, rng,
                               yes_id, no_id) for o in ORDERS}
        except Exception as e:
            print(f"  [{n}] err: {type(e).__name__} {str(e)[:90]}")
            continue
        if any(r is None for r in res.values()):
            continue
        n_ok += 1
        for o in ORDERS:
            for c in CONDS:
                agg[o][c]["correct"].append(res[o][c][0])
                agg[o][c]["p2"].append(res[o][c][1])
        if n % 10 == 0:
            print(f"  [{n}/{len(pairs)}] ok={n_ok} ({n/(time.time()-t0):.2f}/s)")

    # paired bootstrap on the headline contrast: SIT clean -> SIT ko_question
    def boot_ci(a, b, iters=5000):
        a, b = np.array(a, float), np.array(b, float)
        d = a - b
        idx = np.random.RandomState(0).randint(0, len(d), (iters, len(d)))
        deltas = d[idx].mean(1)
        return float(d.mean()), float(np.percentile(deltas, 2.5)), \
            float(np.percentile(deltas, 97.5))

    summary = {"n_pairs": n_ok, "q_scope": args.q_scope, "orders": {}}
    for o in ORDERS:
        summary["orders"][o] = {
            c: {"acc": float(np.mean(agg[o][c]["correct"])),
                "p_corr2": float(np.mean(agg[o][c]["p2"])),
                "correct": [int(x) for x in agg[o][c]["correct"]]}
            for c in CONDS}
    d, lo, hi = boot_ci(agg["SIT"]["clean"]["correct"],
                        agg["SIT"]["ko_question"]["correct"])
    summary["sit_question_knockout"] = {"delta_acc": d, "ci95": [lo, hi]}
    json.dump(summary, open(os.path.join(PROBE_DIR, f"knockout_{args.q_scope}.json"), "w"), indent=2)
    print(f"  [SIT clean - SIT ko_question] delta acc {d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")

    print(f"\n[done] {n_ok} pairs ({(time.time()-t0)/60:.1f} min)")
    for o in ORDERS:
        s = summary["orders"][o]
        base = s["clean"]["acc"]
        print(f"  {o}: clean acc {base:.3f} | "
              + " ".join(f"{c.replace('ko_','-'):>9} {s[c]['acc']:.3f} "
                         f"(d{s[c]['acc']-base:+.3f})" for c in CONDS[1:]))


if __name__ == "__main__":
    main()
