#!/usr/bin/env python3
"""
Pick candidate scenes for the new Fig. 3 (steering + localization) from COCO val2014.

A good Fig. 3 scene needs, in order of importance:
  1. several LARGE objects (each callout must cover many vision patches, or the
     decoded word does not correspond to a readable region),
  2. from DISTINCT categories that are spatially SEPARATED (callouts must not
     overlap, and "each region decodes to its own object" must be checkable),
  3. at least two categories that afford two natural, contrasting questions
     (the steering half of the figure needs q1 and q2 to point at different objects),
  4. landscape aspect and decent resolution (the figure spans the text width).

Writes the ranked shortlist to fig3_candidates.json and a contact sheet to
fig3_candidates.png.

    python fig3_pick_scene.py --top 12
"""
import argparse, json, os
from collections import defaultdict

from PIL import Image, ImageDraw

ANN = "datasets/COCO/annotations_trainval2014/annotations/instances_val2014.json"
IMGDIR = "datasets/COCO/val2014"

# Categories that read well as a logit-lens callout: a human-nameable thing whose
# identity a patch can plausibly decode to a single vocabulary word.
GOOD = {
    "cat", "dog", "horse", "bird", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "person", "tv", "laptop", "pizza", "cake", "clock", "bicycle",
    "motorcycle", "bus", "train", "truck", "boat", "airplane", "car",
    "potted plant", "couch", "bed", "dining table", "refrigerator", "oven",
    "toilet", "sink", "vase", "teddy bear", "surfboard", "skateboard", "kite",
    "umbrella", "traffic light", "stop sign", "fire hydrant", "bench",
    "banana", "apple", "orange", "broccoli", "donut", "sandwich", "hot dog",
    "wine glass", "bottle", "keyboard", "book", "chair", "guitar",
}
MIN_FRAC = 0.035          # each callout object covers >= 3.5% of the image
MAX_FRAC = 0.55           # ...and does not swallow the whole scene
MIN_SEP = 0.22            # normalized center-to-center distance between objects


def iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="fig3_candidates")
    args = ap.parse_args()

    print("[fig3] loading COCO instances (this takes ~20 s) ...", flush=True)
    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}

    by_img = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        by_img[a["image_id"]].append(a)

    scored = []
    for iid, anns in by_img.items():
        im = imgs[iid]
        W, H = im["width"], im["height"]
        area = float(W * H)
        if W < 500 or H < 350 or W / H < 1.15 or W / H > 1.75:
            continue                                   # landscape, print-worthy

        # one representative (the largest instance) per category
        best = {}
        for a in anns:
            name = cats[a["category_id"]]
            if name not in GOOD:
                continue
            f = a["area"] / area
            if not (MIN_FRAC <= f <= MAX_FRAC):
                continue
            if name not in best or a["area"] > best[name]["area"]:
                best[name] = a
        objs = sorted(best.values(), key=lambda a: -a["area"])[:5]
        if len(objs) < 3:
            continue

        # spatial separation: every pair of callout anchors must be far apart
        ok, min_sep, max_iou = True, 9.9, 0.0
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                bi, bj = objs[i]["bbox"], objs[j]["bbox"]
                ci = ((bi[0] + bi[2] / 2) / W, (bi[1] + bi[3] / 2) / H)
                cj = ((bj[0] + bj[2] / 2) / W, (bj[1] + bj[3] / 2) / H)
                sep = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
                min_sep = min(min_sep, sep)
                max_iou = max(max_iou, iou(bi, bj))
        if min_sep < MIN_SEP or max_iou > 0.12:
            continue

        fracs = [o["area"] / area for o in objs]
        names = [cats[o["category_id"]] for o in objs]
        # prefer: many objects, big objects, well separated, non-overlapping,
        # and at least one animal/appliance pair (affords two contrasting questions)
        score = (2.2 * len(objs)
                 + 7.0 * sum(fracs)
                 + 3.5 * min_sep
                 - 6.0 * max_iou
                 + 1.5 * (min(fracs) / max(fracs)))
        scored.append({
            "image_id": iid, "file": im["file_name"], "w": W, "h": H,
            "objects": [{"name": cats[o["category_id"]], "bbox": o["bbox"],
                         "frac": round(o["area"] / area, 4)} for o in objs],
            "names": names, "min_sep": round(min_sep, 3),
            "max_iou": round(max_iou, 3), "score": round(score, 3),
        })

    scored.sort(key=lambda r: -r["score"])
    top = scored[:args.top]
    json.dump(top, open(args.out + ".json", "w"), indent=2)
    print(f"[fig3] {len(scored)} qualifying scenes; top {len(top)}:")
    for k, r in enumerate(top):
        print(f"  {k:2d}. {r['file']}  {r['w']}x{r['h']}  score={r['score']:.2f}  "
              f"sep={r['min_sep']}  {', '.join(r['names'])}")

    # contact sheet with the callout anchors boxed
    cols, cell = 4, 420
    rows = (len(top) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 26)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, r in enumerate(top):
        im = Image.open(os.path.join(IMGDIR, r["file"])).convert("RGB")
        s = cell / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        d2 = ImageDraw.Draw(im)
        for o in r["objects"]:
            x, y, w, h = [v * s for v in o["bbox"]]
            d2.rectangle((x, y, x + w, y + h), outline=(220, 20, 60), width=3)
        cx, cy = (k % cols) * cell, (k // cols) * (cell + 26)
        sheet.paste(im, (cx, cy))
        dr.text((cx + 4, cy + im.height + 6), f"{k}: {', '.join(r['names'])}", fill="black")
    sheet.save(args.out + ".png")
    print(f"[fig3] contact sheet -> {args.out}.png")


if __name__ == "__main__":
    main()
