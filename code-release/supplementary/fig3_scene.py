#!/usr/bin/env python3
"""
Prepare one chosen scene for Fig. 3: optionally crop it, re-find the anchor patch
for each object, run the four (ordering x question) conditions, and dump a probe
file that fig3_make.py can render.

Anchors are re-found on the CROPPED image rather than shifted from the uncropped
one, because cropping changes the patch grid the model actually sees.

    CUDA_VISIBLE_DEVICES=1 python fig3_scene.py \
        --file COCO_val2014_000000333772.jpg --crop-left 34 \
        --objects cat,tv,chair \
        --q1 "What does the cat look like?" \
        --q2 "What does the television look like?"
"""
import argparse, json, os

import numpy as np
from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names, clean
from fig3_probe import topk_patch_tokens

IMGDIR = "datasets/COCO/val2014"
ANN = "datasets/COCO/annotations_trainval2014/annotations/instances_val2014.json"
LONG_SIDE = 896
LAYER = 28


def gt_boxes(fname, wanted):
    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    iid = int(fname.split("_")[-1].split(".")[0])
    best = {}
    for a in d["annotations"]:
        if a["image_id"] != iid or a.get("iscrowd"):
            continue
        n = cats[a["category_id"]]
        if n in wanted and (n not in best or a["area"] > best[n]["area"]):
            best[n] = a
    return {n: best[n]["bbox"] for n in best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--objects", required=True, help="comma-separated COCO category names")
    ap.add_argument("--q1", required=True)
    ap.add_argument("--q2", required=True)
    ap.add_argument("--crop-left", type=int, default=0, dest="crop_left")
    ap.add_argument("--crop-right", type=int, default=0, dest="crop_right")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out", default="fig3_probe.json")
    ap.add_argument("--image-out", default="paper/figs/fig3_scene.png", dest="image_out")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    wanted = [w.strip() for w in args.objects.split(",")]
    boxes = gt_boxes(args.file, set(wanted))
    missing = [w for w in wanted if w not in boxes]
    if missing:
        raise SystemExit(f"no ground-truth box for {missing} in {args.file}")

    full = Image.open(os.path.join(IMGDIR, args.file)).convert("RGB")
    x0, x1 = args.crop_left, full.width - args.crop_right
    crop = full.crop((x0, 0, x1, full.height))
    W0, H0 = crop.size
    boxes = {n: [b[0] - x0, b[1], b[2], b[3]] for n, b in boxes.items()}

    os.makedirs(os.path.dirname(args.image_out), exist_ok=True)
    crop.save(args.image_out)

    s = LONG_SIDE / max(crop.size)
    pil = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
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

    # anchor per object: the patch inside its box whose top-3 best name it
    anchors = []
    base = res["SIT + q1"]["toks"]
    for n in wanted:
        x, y, w, h = boxes[n]
        c0, c1 = int(x / W0 * gw), int(np.ceil((x + w) / W0 * gw))
        r0, r1 = int(y / H0 * gh), int(np.ceil((y + h) / H0 * gh))
        best = None
        for rr in range(max(0, r0), min(gh, max(r0 + 1, r1))):
            for cc in range(max(0, c0), min(gw, max(c0 + 1, c1))):
                t = base[rr * gw + cc]
                sc = 3.0 * names(n, t) + sum(1 for x_ in t if clean(x_))
                if best is None or sc > best[0]:
                    best = (sc, [rr, cc], t)
        anchors.append({"name": n, "rc": best[1], "toks": best[2],
                        "cy": (best[1][0] + .5) / gh, "cx": (best[1][1] + .5) / gw})

    n_patch = gh * gw
    chg = {}
    for tag in ("SIT", "STI"):
        a1, a2 = res[f"{tag} + q1"]["toks"], res[f"{tag} + q2"]["toks"]
        chg[tag] = sum(1 for i in range(n_patch) if a1[i][0] != a2[i][0]) / n_patch
    res["frac_changed"] = chg

    print(f"[scene] {args.file} crop->{crop.size} grid {gh}x{gw} layer {args.layer}")
    print(f"[scene] image-first {chg['SIT']*100:.1f}%   question-first {chg['STI']*100:.1f}%")
    for a in anchors:
        rr, cc = a["rc"]
        i = rr * gw + cc
        print(f"   {a['name']:8s} @{str(a['rc']):9s} "
              f"SIT={' / '.join(res['SIT + q1']['toks'][i]):28s} "
              f"STIq1={' / '.join(res['STI + q1']['toks'][i]):28s} "
              f"STIq2={' / '.join(res['STI + q2']['toks'][i])}")

    dump = {str(args.layer): res, "anchors": anchors, "file": args.file,
            "image_path": args.image_out, "q1": args.q1, "q2": args.q2,
            "english": True, "crop": [x0, x1]}
    json.dump(dump, open(args.out, "w"))
    print(f"[scene] -> {args.out}  (scene image {args.image_out})")


if __name__ == "__main__":
    main()
