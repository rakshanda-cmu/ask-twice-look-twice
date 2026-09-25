#!/usr/bin/env python3
"""
Build a Fig. 3 probe whose callouts are CHOSEN PATCHES rather than object regions.

fig3_scene.py anchors each callout on a ground-truth object box, which is right for
the localization claim. This script instead takes the patches found by
fig3_perfect.py -- the ones that decode to A's vocabulary when A is asked about and
B's when B is asked about -- so the figure can show the semantic side of steering:
the same patch answering to whichever object the question names.

    CUDA_VISIBLE_DEVICES=1 python fig3_patches.py \
        --file COCO_val2014_000000246999.jpg --patches "0,24;6,21;20,3" \
        --q1 "What does the couch look like?" --q2 "What does the plant look like?"
"""
import argparse, json, os

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_probe import topk_patch_tokens
from fig3_rescan import IMGDIR, LONG_SIDE, LAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--patches", required=True, help='"r,c;r,c;..." patch coordinates')
    ap.add_argument("--labels", default=None, help="comma-separated callout labels")
    ap.add_argument("--q1", required=True)
    ap.add_argument("--q2", required=True)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out", default="fig3_probe_patches.json")
    ap.add_argument("--image-out", default="paper/figs/fig3_scene_patches.png",
                    dest="image_out")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pts = [[int(v) for v in p.split(",")] for p in args.patches.split(";")]
    labels = ([l.strip() for l in args.labels.split(",")] if args.labels
              else [f"patch {chr(65 + i)}" for i in range(len(pts))])

    full = Image.open(os.path.join(IMGDIR, args.file)).convert("RGB")
    os.makedirs(os.path.dirname(args.image_out), exist_ok=True)
    full.save(args.image_out)
    s = LONG_SIDE / max(full.size)
    pil = full.resize((int(full.width * s), int(full.height * s)), Image.LANCZOS)

    mm = ModelManager("qwen3-vl-8b")
    conds = [("no question", None, "I", ""),
             ("SIT + q1", args.q1, "SIT", SYSTEM_MESSAGE),
             ("SIT + q2", args.q2, "SIT", SYSTEM_MESSAGE),
             ("STI + q1", args.q1, "STI", SYSTEM_MESSAGE),
             ("STI + q2", args.q2, "STI", SYSTEM_MESSAGE)]
    res = {}
    for name, q, order, sysm in conds:
        toks, p, gh, gw = topk_patch_tokens(mm, pil, args.layer, q, order, sysm, english=True)
        res[name] = {"toks": toks, "p": p.tolist(), "gh": gh, "gw": gw}
    gh, gw = res["no question"]["gh"], res["no question"]["gw"]

    anchors = [{"name": lab, "rc": rc, "toks": res["SIT + q1"]["toks"][rc[0] * gw + rc[1]]}
               for lab, rc in zip(labels, pts)]

    n = gh * gw
    chg = {}
    for tag in ("SIT", "STI"):
        a1, a2 = res[f"{tag} + q1"]["toks"], res[f"{tag} + q2"]["toks"]
        chg[tag] = sum(1 for i in range(n) if a1[i][0] != a2[i][0]) / n
    res["frac_changed"] = chg

    print(f"[patches] {args.file} grid {gh}x{gw} layer {args.layer}")
    print(f"[patches] image-first {chg['SIT']*100:.1f}%   question-first {chg['STI']*100:.1f}%")
    for a in anchors:
        i = a["rc"][0] * gw + a["rc"][1]
        print(f"   {a['name']:9s} @{str(a['rc']):9s} "
              f"SIT={' / '.join(res['SIT + q1']['toks'][i]):30s} "
              f"q1={' / '.join(res['STI + q1']['toks'][i]):30s} "
              f"q2={' / '.join(res['STI + q2']['toks'][i])}")

    json.dump({str(args.layer): res, "anchors": anchors, "file": args.file,
               "image_path": args.image_out, "q1": args.q1, "q2": args.q2,
               "english": True}, open(args.out, "w"))
    print(f"[patches] -> {args.out}")


if __name__ == "__main__":
    main()
