#!/usr/bin/env python3
"""
Verify Fig. 3 candidates under the EXACT conditions fig3_scene.py renders with.

The joint ranking picked anchors from an image-only pass; the renderer re-picks
them under image-first with the question present, and a ground-truth box can
contain a different object (a rider inside a motorcycle box, an earring inside a
chair box), so an anchor can name something other than its region. This script
loads the model once and, for each candidate, reproduces the renderer's anchor
choice and reports whether every region still names itself, plus the steering.

Only scenes where EVERY anchor names its own region are usable: that is the
claim the figure makes.

    CUDA_VISIBLE_DEVICES=1 python fig3_verify.py
"""
import argparse, json, os

import numpy as np
from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names, clean
from fig3_probe import topk_patch_tokens
from fig3_scene import gt_boxes
from fig3_rescan import phrase, IMGDIR, LONG_SIDE, LAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="fig3_joint.json")
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--out", default="fig3_verified.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pool = json.load(open(args.joint))[: args.num]
    mm = ModelManager("qwen3-vl-8b")

    rows = []
    for r in pool:
        wanted = list(dict.fromkeys(r["regions"]))
        boxes = gt_boxes(r["file"], set(wanted))
        if len(boxes) < len(wanted):
            continue
        full = Image.open(os.path.join(IMGDIR, r["file"])).convert("RGB")
        W0, H0 = full.size
        s = LONG_SIDE / max(full.size)
        pil = full.resize((int(full.width * s), int(full.height * s)), Image.LANCZOS)

        q1, q2 = r["q1"], r["q2"]
        sit1 = topk_patch_tokens(mm, pil, LAYER, q1, "SIT", SYSTEM_MESSAGE, english=True)
        toks, _, gh, gw = sit1
        sti1 = topk_patch_tokens(mm, pil, LAYER, q1, "STI", SYSTEM_MESSAGE, english=True)[0]
        sti2 = topk_patch_tokens(mm, pil, LAYER, q2, "STI", SYSTEM_MESSAGE, english=True)[0]

        anchors, allnamed = [], True
        for n in wanted:
            x, y, w, h = boxes[n]
            c0, c1 = int(x / W0 * gw), int(np.ceil((x + w) / W0 * gw))
            r0, r1 = int(y / H0 * gh), int(np.ceil((y + h) / H0 * gh))
            best = None
            for rr in range(max(0, r0), min(gh, max(r0 + 1, r1))):
                for cc in range(max(0, c0), min(gw, max(c0 + 1, c1))):
                    t = toks[rr * gw + cc]
                    sc = 3.0 * names(n, t) + sum(1 for x_ in t if clean(x_))
                    if best is None or sc > best[0]:
                        best = (sc, [rr, cc], t)
            named = names(n, best[2])
            allnamed &= bool(named)
            anchors.append({"name": n, "rc": best[1], "toks": best[2], "named": bool(named)})

        steered = sum(1 for i in range(gh * gw) if sti1[i][0] != sti2[i][0]) / (gh * gw)
        rows.append({"file": r["file"], "q1": q1, "q2": q2, "grid": [gh, gw],
                     "all_named": bool(allnamed), "steered": round(steered, 3),
                     "size": [W0, H0], "anchors": anchors})
        print(f"{r['file'][-10:]} {W0}x{H0} all_named={allnamed} steer={steered*100:.0f}%")
        for a in anchors:
            print(f"    {a['name']:14s} {'OK ' if a['named'] else 'BAD'} "
                  f"@{str(a['rc']):9s} {' / '.join(a['toks'])}")

    rows.sort(key=lambda x: (x["all_named"], x["steered"]), reverse=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print("\n===== usable (every region names itself), best steering first =====")
    for x in rows:
        if x["all_named"]:
            print(f"  {x['file']}  {x['size'][0]}x{x['size'][1]}  "
                  f"steer={x['steered']*100:.0f}%  "
                  f"regions={[a['name'] for a in x['anchors']]}")


if __name__ == "__main__":
    main()
