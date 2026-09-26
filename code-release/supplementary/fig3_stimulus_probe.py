#!/usr/bin/env python3
"""
Probe the constructed stimulus: localization per object, and steering under two questions.

Anchors come from the composite's own layout file (exact pixel boxes), so each callout
is guaranteed to sit on one object.

    CUDA_VISIBLE_DEVICES=1 python fig3_stimulus_probe.py \
        --q1 "What does the cat look like?" --q2 "What does the pizza look like?"
"""
import argparse, json, os

import numpy as np
from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names, clean
from fig3_probe import topk_patch_tokens

LONG_SIDE = 896


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="paper/figs/fig3_stimulus.png")
    ap.add_argument("--layout", default="paper/figs/fig3_stimulus_layout.json")
    ap.add_argument("--q1", required=True)
    ap.add_argument("--q2", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default="fig3_probe_stimulus.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    lay = json.load(open(args.layout))
    W0, H0 = lay["size"]
    full = Image.open(args.image).convert("RGB")
    s = LONG_SIDE / max(full.size)
    pil = full.resize((int(full.width * s), int(full.height * s)), Image.LANCZOS) if s < 1 else full

    mm = ModelManager("qwen3-vl-8b")
    conds = [("no question", None, "I", ""),
             ("SIT + q1", args.q1, "SIT", SYSTEM_MESSAGE),
             ("SIT + q2", args.q2, "SIT", SYSTEM_MESSAGE),
             ("STI + q1", args.q1, "STI", SYSTEM_MESSAGE),
             ("STI + q2", args.q2, "STI", SYSTEM_MESSAGE)]
    res = {}
    for nm, q, order, sysm in conds:
        toks, p, gh, gw = topk_patch_tokens(mm, pil, args.layer, q, order, sysm, english=True)
        res[nm] = {"toks": toks, "p": p.tolist(), "gh": gh, "gw": gw}
    gh, gw = res["no question"]["gh"], res["no question"]["gw"]

    base = res["SIT + q1"]["toks"]
    anchors = []
    for nm, o in lay["objects"].items():
        x, y, w, h = o["bbox"]
        c0, c1 = int(x / W0 * gw), int(np.ceil((x + w) / W0 * gw))
        r0, r1 = int(y / H0 * gh), int(np.ceil((y + h) / H0 * gh))
        best = None
        for rr in range(max(0, r0), min(gh, r1)):
            for cc in range(max(0, c0), min(gw, c1)):
                t = base[rr * gw + cc]
                sc = 3.0 * names(nm, t) + sum(1 for x_ in t if clean(x_))
                if best is None or sc > best[0]:
                    best = (sc, [rr, cc], t)
        anchors.append({"name": nm, "rc": best[1], "toks": best[2],
                        "named": bool(names(nm, best[2]))})

    n = gh * gw
    chg = {t: sum(1 for i in range(n)
                  if res[f"{t} + q1"]["toks"][i][0] != res[f"{t} + q2"]["toks"][i][0]) / n
           for t in ("SIT", "STI")}
    res["frac_changed"] = chg

    print(f"[stimulus] grid {gh}x{gw} layer {args.layer}")
    print(f"[stimulus] image-first {chg['SIT']*100:.1f}%   question-first {chg['STI']*100:.1f}%")
    for a in anchors:
        i = a["rc"][0] * gw + a["rc"][1]
        print(f"   {a['name']:14s} {'OK ' if a['named'] else 'BAD'} @{str(a['rc']):9s} "
              f"img-first={' / '.join(res['SIT + q1']['toks'][i]):32s} "
              f"q1={' / '.join(res['STI + q1']['toks'][i]):32s} "
              f"q2={' / '.join(res['STI + q2']['toks'][i])}")

    json.dump({str(args.layer): res, "anchors": anchors, "file": os.path.basename(args.image),
               "image_path": args.image, "q1": args.q1, "q2": args.q2, "english": True},
              open(args.out, "w"))
    print(f"[stimulus] -> {args.out}")


if __name__ == "__main__":
    main()
