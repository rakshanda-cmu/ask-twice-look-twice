#!/usr/bin/env python3
"""
Rank Fig. 3 scenes on localization AND steering jointly.

fig3_rescan.py gated on localization (does each region's anchor name its object at
all?) and then sorted by steering. That treats a region whose top-3 is
"orange / Orange / citrus" the same as one where only the third token lands. Here
localization is graded instead:

    strength(region) = how many of its top-3 tokens under image-first name it (0..3)
    localization     = mean strength / 3, over ALL the scene's regions
    steering         = % of patches whose decoded word changes with the question

and scenes are ranked by localization + steering, both on 0..1, with a small bonus
for having four regions rather than three (more callouts = a stronger localization
demonstration). Scenes already used in the paper are excluded.

    CUDA_VISIBLE_DEVICES=1 python fig3_joint.py --top 12
"""
import argparse, json, os

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names
from fig3_probe import topk_patch_tokens
from fig3_rescan import phrase, IMGDIR, LONG_SIDE, LAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", default="fig3_rescan2.json")
    ap.add_argument("--num", type=int, default=45)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--exclude",
                    default="COCO_val2014_000000397268.jpg,"
                            "COCO_val2014_000000333772.jpg,"
                            "COCO_val2014_000000497855.jpg")
    ap.add_argument("--out", default="fig3_joint.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    skip = set(args.exclude.split(","))
    pool = [r for r in json.load(open(args.rescan))
            if r["file"] not in skip and r["localized"] == r["n_anchors"]][: args.num]
    print(f"[joint] {len(pool)} fully-localized scenes", flush=True)
    mm = ModelManager("qwen3-vl-8b")

    rows = []
    for k, r in enumerate(pool):
        gh, gw = r["grid"]
        pil = Image.open(os.path.join(IMGDIR, r["file"])).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            sit1 = topk_patch_tokens(mm, pil, LAYER, r["q1"], "SIT", SYSTEM_MESSAGE, english=True)[0]
            sti1 = topk_patch_tokens(mm, pil, LAYER, r["q1"], "STI", SYSTEM_MESSAGE, english=True)[0]
            sti2 = topk_patch_tokens(mm, pil, LAYER, r["q2"], "STI", SYSTEM_MESSAGE, english=True)[0]
        except Exception as e:
            print(f"  [skip] {r['file']}: {e}", flush=True)
            continue

        strengths = []
        for a in r["anchors"]:
            rr, cc = a["rc"]
            toks = sit1[rr * gw + cc]
            strengths.append(sum(1 for t in toks if names(a["name"], [t])))
        loc = sum(strengths) / (3.0 * len(strengths))
        steered = sum(1 for i in range(gh * gw) if sti1[i][0] != sti2[i][0]) / (gh * gw)
        bonus = 0.05 if len(r["anchors"]) >= 4 else 0.0
        joint = loc + steered + bonus

        rows.append({"file": r["file"], "A": r["A"], "B": r["B"],
                     "q1": r["q1"], "q2": r["q2"], "grid": [gh, gw],
                     "regions": [a["name"] for a in r["anchors"]],
                     "strengths": strengths, "loc": round(loc, 3),
                     "steered": round(steered, 3), "joint": round(joint, 3),
                     "anchors": r["anchors"]})
        print(f"[{k+1}/{len(pool)}] {r['file'][-10:]} loc={loc:.2f} "
              f"steer={steered*100:.0f}% joint={joint:.2f} "
              f"{list(zip([a['name'] for a in r['anchors']], strengths))}", flush=True)

    rows.sort(key=lambda x: -x["joint"])
    json.dump(rows, open(args.out, "w"), indent=2)
    print("\n===== best on localization AND steering =====")
    for x in rows[: args.top]:
        det = ", ".join(f"{n}:{s}/3" for n, s in zip(x["regions"], x["strengths"]))
        print(f"  {x['file']}  joint={x['joint']:.2f}  loc={x['loc']:.2f}  "
              f"steer={x['steered']*100:.0f}%  [{det}]  q1={x['A']} q2={x['B']}")


if __name__ == "__main__":
    main()
