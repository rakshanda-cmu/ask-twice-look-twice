#!/usr/bin/env python3
"""
Re-pick the Fig. 3 scene, scoring candidates on the figure's actual claim.

For each scene that survived fig3_anchors.py we build two questions that differ in
exactly ONE word -- "What does the {A} look like?" vs "What does the {B} look
like?", where A and B are the two most separated objects -- so any change in the
patch read-out is attributable to the object asked about and nothing else. Then:

  localization : how many object regions still name themselves under image-first
  frozen       : % of patches whose decoded word differs between q1 and q2 under
                 image-first (must be 0: the causal mask hides the question)
  steered      : the same % under question-first (want it large)

Ranks by steered% with localization as a gate, and prints the shortlist.

    CUDA_VISIBLE_DEVICES=1 python fig3_rescan.py --top 14
"""
import argparse, json, os

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names, clean
from fig3_probe import topk_patch_tokens

IMGDIR = "/data2/datasets/COCO/val2014"
LONG_SIDE = 896
LAYER = 28

PRETTY = {"potted plant": "plant", "dining table": "table", "tv": "television",
          "teddy bear": "teddy bear", "traffic light": "traffic light"}


def phrase(name):
    return PRETTY.get(name, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default="fig3_anchor_ranked.json")
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--exclude", default="COCO_val2014_000000397268.jpg")
    ap.add_argument("--out", default="fig3_rescan.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pool = [r for r in json.load(open(args.ranked))
            if r["file"] not in set(args.exclude.split(","))][: args.num]
    print(f"[rescan] {len(pool)} scenes", flush=True)
    mm = ModelManager("qwen3-vl-8b")

    rows = []
    for k, r in enumerate(pool):
        anchors = r["anchors"]
        gh, gw = r["grid"]
        # the two most separated anchors carry the two questions
        best = max(((i, j) for i in range(len(anchors)) for j in range(i + 1, len(anchors))),
                   key=lambda p: (abs(anchors[p[0]]["rc"][0] - anchors[p[1]]["rc"][0]) / gh) ** 2
                   + (abs(anchors[p[0]]["rc"][1] - anchors[p[1]]["rc"][1]) / gw) ** 2)
        A, B = anchors[best[0]]["name"], anchors[best[1]]["name"]
        if A == B:
            continue
        q1 = f"What does the {phrase(A)} look like?"
        q2 = f"What does the {phrase(B)} look like?"

        pil = Image.open(os.path.join(IMGDIR, r["file"])).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            out = {}
            for tag, q, order, sysm in (("sit1", q1, "SIT", SYSTEM_MESSAGE),
                                        ("sit2", q2, "SIT", SYSTEM_MESSAGE),
                                        ("sti1", q1, "STI", SYSTEM_MESSAGE),
                                        ("sti2", q2, "STI", SYSTEM_MESSAGE)):
                out[tag] = topk_patch_tokens(mm, pil, LAYER, q, order, sysm, english=True)[0]
        except Exception as e:
            print(f"  [skip] {r['file']}: {e}", flush=True)
            continue

        n = len(out["sit1"])
        frozen = sum(1 for i in range(n) if out["sit1"][i][0] != out["sit2"][i][0]) / n
        steered = sum(1 for i in range(n) if out["sti1"][i][0] != out["sti2"][i][0]) / n
        # localization under image-first: does each region still name its object?
        loc = 0
        for a in anchors:
            rr, cc = a["rc"]
            if names(a["name"], out["sit1"][rr * gw + cc]):
                loc += 1
        clean_frac = sum(1 for t in out["sit1"] if clean(t[0])) / n

        rows.append({"file": r["file"], "A": A, "B": B, "q1": q1, "q2": q2,
                     "grid": [gh, gw], "n_anchors": len(anchors), "localized": loc,
                     "frozen": round(frozen, 4), "steered": round(steered, 4),
                     "clean": round(clean_frac, 3), "anchors": anchors})
        print(f"[{k+1}/{len(pool)}] {r['file'][-10:]} loc={loc}/{len(anchors)} "
              f"frozen={frozen*100:.1f}% steered={steered*100:.1f}% | {A} vs {B}",
              flush=True)

    rows.sort(key=lambda x: (x["localized"] == x["n_anchors"], x["localized"],
                             x["steered"]), reverse=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print("\n===== shortlist (all regions localized, then most steering) =====")
    for x in rows[: args.top]:
        print(f"  {x['file']}  loc={x['localized']}/{x['n_anchors']}  "
              f"frozen={x['frozen']*100:.0f}%  steered={x['steered']*100:.0f}%  "
              f"| q1={x['A']}  q2={x['B']}")


if __name__ == "__main__":
    main()
