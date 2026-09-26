#!/usr/bin/env python3
"""
Probe one scene for the new Fig. 3: read the logit-lens top-3 tokens at each
object's anchor patch under three conditions

    no question  (image only)          -> does each region name its own object?
    STI + q1     (question before img) -> do the same patches move toward q1?
    STI + q2     (question before img) -> ...and differently toward q2?

Prints a table so the steering can be checked before any figure is drawn.

    CUDA_VISIBLE_DEVICES=1 python fig3_probe.py --file COCO_val2014_000000397268.jpg \
        --q1 "..." --q2 "..."
"""
import argparse, json, os

import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager

IMGDIR = "datasets/COCO/val2014"
LONG_SIDE = 896


_EN_MASK = {}


def english_mask(mm):
    """Vocabulary mask keeping whole English words (>=3 ASCII letters).

    The lens is otherwise dominated by sub-word fragments and Chinese tokens that
    say nothing readable; restricting the read-out to English words is a display
    choice on the SAME hidden state, not a change to the model or the ordering,
    and it is applied identically to every condition in the figure.
    """
    key = id(mm.tokenizer)
    if key in _EN_MASK:
        return _EN_MASK[key]
    V = mm.llm_model.lm_head.weight.shape[0]
    keep = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        t = mm.tokenizer.convert_ids_to_tokens(i)
        if t is None:
            continue
        s = t.replace("Ġ", "").replace("▁", "")
        if len(s) >= 3 and s.isascii() and s.isalpha():
            keep[i] = True
    _EN_MASK[key] = keep
    return keep


def topk_patch_tokens(mm, pil, layer, question=None, order="I", system="", k=3,
                      english=False):
    q = [question] if question else [""]
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(q, pil, system_prompt=system, order=order)
    with torch.inference_mode():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
        n = mm.grid_h * mm.grid_w
        h = out.hidden_states[layer + 1][0, mm.img_start_idx: mm.img_start_idx + n]
        logits = mm.llm_model.lm_head(h).float()
        if english:
            m = english_mask(mm).to(logits.device)
            logits = logits.masked_fill(~m.unsqueeze(0), float("-inf"))
        probs = F.softmax(logits, dim=-1)
        p, idx = probs.topk(k, dim=-1)
    toks = [[mm.tokenizer.decode(int(i)).strip() for i in row] for row in idx.cpu()]
    return toks, p.cpu().numpy(), mm.grid_h, mm.grid_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--q1", required=True)
    ap.add_argument("--q2", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--layers", default=None,
                    help="comma-separated layers to sweep instead of --layer")
    ap.add_argument("--english", action="store_true",
                    help="restrict the lens read-out to whole English words")
    ap.add_argument("--ranked", default="fig3_anchor_ranked.json")
    ap.add_argument("--out", default="fig3_probe.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    rec = next(r for r in json.load(open(args.ranked)) if r["file"] == args.file)
    pil = Image.open(os.path.join(IMGDIR, args.file)).convert("RGB")
    s = LONG_SIDE / max(pil.size)
    pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)

    mm = ModelManager("qwen3-vl-8b")
    conds = [("no question", None, "I", ""),
             ("SIT + q1", args.q1, "SIT", SYSTEM_MESSAGE),
             ("SIT + q2", args.q2, "SIT", SYSTEM_MESSAGE),
             ("STI + q1", args.q1, "STI", SYSTEM_MESSAGE),
             ("STI + q2", args.q2, "STI", SYSTEM_MESSAGE)]

    layers = [int(x) for x in args.layers.split(",")] if args.layers else [args.layer]
    print(f"\nq1={args.q1!r}\nq2={args.q2!r}\n")
    dump = {}
    for L in layers:
        res = {}
        for name, q, order, sysm in conds:
            toks, p, gh, gw = topk_patch_tokens(mm, pil, L, q, order, sysm,
                                                english=args.english)
            res[name] = {"toks": toks, "p": p.tolist(), "gh": gh, "gw": gw}
        dump[str(L)] = res
        # how often does the decoded word change when only the question changes?
        n_patch = len(res["no question"]["toks"])
        chg = {}
        for tag in ("SIT", "STI"):
            a1 = res[f"{tag} + q1"]["toks"]; a2 = res[f"{tag} + q2"]["toks"]
            chg[tag] = sum(1 for i in range(n_patch) if a1[i][0] != a2[i][0]) / n_patch
        res["frac_changed"] = chg
        print(f"=== layer {L} ===   patches whose top word changes with the question: "
              f"image-first(SIT) {chg['SIT']*100:.1f}%   question-first(STI) {chg['STI']*100:.1f}%")
        hdr = f"{'region':14s} {'patch':8s} " + " ".join(f"{c[0]:26s}" for c in conds)
        print(hdr); print("-" * len(hdr))
        for a in rec["anchors"]:
            rr, cc = a["rc"]
            i = rr * res["no question"]["gw"] + cc
            cells = [f"{' / '.join(x[:7] for x in res[n]['toks'][i]):26s}" for n, *_ in conds]
            print(f"{a['name']:14s} {str((rr, cc)):8s} " + " ".join(cells))
        print()

    dump["anchors"] = rec["anchors"]
    dump["file"] = args.file
    dump["q1"], dump["q2"] = args.q1, args.q2
    dump["english"] = args.english
    json.dump(dump, open(args.out, "w"))
    print(f"[probe] -> {args.out}")


if __name__ == "__main__":
    main()
