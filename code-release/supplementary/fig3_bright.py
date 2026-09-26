#!/usr/bin/env python3
"""
Search bright, real scenes where question-first steering is unmistakable.

The lesson from the earlier attempts: a generic question ("what does the X look
like?") cannot move a patch that already names X. A question with a SPECIFIC answer
can. So each candidate scene gets two such questions,

    q_action  "What is the {actor} doing?"      -> expect sitting / standing / ...
    q_colour  "What colour is the {object}?"    -> expect purple / orange / ...

and we count, inside the relevant ground-truth box, patches whose logit-lens read-out
contains an answer word under ITS question and not under the other. A scene scores
only if BOTH directions work, so the figure shows steering twice over rather than one
lucky patch.

Brightness is scored too, since the figure should be legible in print.

    CUDA_VISIBLE_DEVICES=1 python fig3_bright.py --num 120
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_probe import topk_patch_tokens

IMGDIR = "datasets/COCO/val2014"
ANN = "datasets/COCO/annotations_trainval2014/annotations/instances_val2014.json"
LONG_SIDE = 896
LAYER = 28

ACTORS = ["person", "cat", "dog", "horse", "bird", "elephant", "bear", "zebra",
          "giraffe", "sheep", "cow"]
ACTION = set("""sitting sit seated standing stands stood walking walks running runs
lying laying sleeping asleep eating eats drinking playing plays jumping riding rides
holding looking looks staring stare stares gazing gaze watching resting rests perched
grazing surfing skiing skating climbing reaching leaning bending crouching kneeling
posing waving smiling""".split())
COLOUR = set("""red orange yellow green blue purple violet pink brown black white grey
gray golden gold silver teal turquoise beige cream tan maroon navy lavender crimson
scarlet olive""".split())
COLOURFUL = ["couch", "chair", "umbrella", "car", "bus", "truck", "boat", "bicycle",
             "motorcycle", "kite", "surfboard", "backpack", "handbag", "suitcase",
             "vase", "bench", "train", "tie", "frisbee"]


def load():
    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    by = defaultdict(dict)
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        n = cats[a["category_id"]]
        cur = by[a["image_id"]].get(n)
        if cur is None or a["area"] > cur["area"]:
            by[a["image_id"]][n] = a
    return imgs, by


def answers(toks_box, vocab):
    return sum(1 for t in toks_box if {x.lower() for x in t} & vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=120)
    ap.add_argument("--min-brightness", type=float, default=118.0, dest="min_bright")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="fig3_bright.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    imgs, by = load()
    pool = []
    for iid, objs in by.items():
        im = imgs[iid]
        A = im["width"] * im["height"]
        if im["width"] < 560 or im["height"] < 380:
            continue
        actor = next((a for a in ACTORS if a in objs and objs[a]["area"] / A > 0.06), None)
        col = next((c for c in COLOURFUL if c in objs and objs[c]["area"] / A > 0.06), None)
        if not actor or not col or actor == col:
            continue
        pool.append((iid, im, objs, actor, col))
    print(f"[bright] {len(pool)} scenes with an actor and a colourful object", flush=True)

    # brightest first, cheaply, before touching the GPU
    scored = []
    for iid, im, objs, actor, col in pool:
        try:
            p = Image.open(os.path.join(IMGDIR, im["file_name"])).convert("L")
            b = float(np.asarray(p.resize((64, 64))).mean())
        except Exception:
            continue
        if b >= args.min_bright:
            scored.append((b, iid, im, objs, actor, col))
    scored.sort(key=lambda t: -t[0])
    scored = scored[: args.num]
    print(f"[bright] {len(scored)} pass the brightness bar; probing", flush=True)

    mm = ModelManager("qwen3-vl-8b")
    rows = []
    for k, (b, iid, im, objs, actor, col) in enumerate(scored):
        q1 = f"What is the {actor} doing?"
        q2 = f"What colour is the {col}?"
        pil = Image.open(os.path.join(IMGDIR, im["file_name"])).convert("RGB")
        W0, H0 = pil.size
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            base, _, gh, gw = topk_patch_tokens(mm, pil, LAYER, q1, "SIT", SYSTEM_MESSAGE, english=True)
            t1 = topk_patch_tokens(mm, pil, LAYER, q1, "STI", SYSTEM_MESSAGE, english=True)[0]
            t2 = topk_patch_tokens(mm, pil, LAYER, q2, "STI", SYSTEM_MESSAGE, english=True)[0]
        except Exception as e:
            print(f"  [skip] {im['file_name']}: {e}", flush=True)
            continue

        def cells(box, toks):
            x, y, w, h = box
            c0, c1 = int(x / W0 * gw), int(np.ceil((x + w) / W0 * gw))
            r0, r1 = int(y / H0 * gh), int(np.ceil((y + h) / H0 * gh))
            return [toks[r * gw + c] for r in range(max(0, r0), min(gh, r1))
                    for c in range(max(0, c0), min(gw, c1))]

        ab, cb = objs[actor]["bbox"], objs[col]["bbox"]
        # answer words must appear under their own question and not the other
        act_hit = answers(cells(ab, t1), ACTION) - answers(cells(ab, t2), ACTION)
        col_hit = answers(cells(cb, t2), COLOUR) - answers(cells(cb, t1), COLOUR)
        base_act = answers(cells(ab, base), ACTION)
        base_col = answers(cells(cb, base), COLOUR)
        score = min(act_hit, col_hit) * 3 + act_hit + col_hit
        rows.append(dict(file=im["file_name"], brightness=round(b, 1), actor=actor,
                         colour_obj=col, q1=q1, q2=q2, act_hit=act_hit, col_hit=col_hit,
                         base_act=base_act, base_col=base_col, score=score))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(scored)}] best={max(r['score'] for r in rows)}", flush=True)

    rows.sort(key=lambda r: -r["score"])
    json.dump(rows, open(args.out, "w"), indent=2)
    print("\n===== brightest scenes where BOTH questions are answered in the patches =====")
    for r in rows[: args.top]:
        print(f"  {r['file']}  bright={r['brightness']:.0f}  score={r['score']}")
        print(f"      action  +{r['act_hit']:2d} patches (base {r['base_act']})  | {r['q1']}")
        print(f"      colour  +{r['col_hit']:2d} patches (base {r['base_col']})  | {r['q2']}")


if __name__ == "__main__":
    main()
