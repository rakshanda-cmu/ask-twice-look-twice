#!/usr/bin/env python3
"""
Build a CONSTRUCTED stimulus for Fig. 3: four unmistakable objects, well separated.

Natural photos make the figure's two claims hard to read: objects overlap, some are
small, and a patch can sit on two things at once. Here each object is cut out of a
COCO photo along its ground-truth segmentation mask and composited onto a plain
background at a known position, so every callout sits on exactly one object and each
object covers many patches.

The image is synthetic in LAYOUT only. The pixels are real photographs and the model
is run on the result, so the decodings in the figure are genuine model behaviour, not
an illustration of it. The caption must say the stimulus is constructed.

    python fig3_compose.py --objects cat,pizza,laptop,"potted plant"
"""
import argparse, json, os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_fill_holes, binary_closing

IMGDIR = "/data2/datasets/COCO/val2014"
ANN = "/data2/datasets/COCO/annotations_trainval2014/annotations/instances_val2014.json"


def load_index():
    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    return d, cats, imgs


def best_instances(d, cats, imgs, name, k=6):
    """Large, polygon-segmented, non-crowd instances of one category."""
    out = []
    for a in d["annotations"]:
        if a.get("iscrowd") or cats[a["category_id"]] != name:
            continue
        seg = a.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue                                  # RLE masks are skipped
        im = imgs[a["image_id"]]
        frac = a["area"] / (im["width"] * im["height"])
        x, y, w, h = a["bbox"]
        if w < 150 or h < 150:
            continue
        if 0.45 > w / h or w / h > 2.2:               # avoid extreme slivers
            continue
        # A whole-image annotation yields a rectangular crop, not a cutout, and a
        # close-up is often unrecognisable once isolated. Want a mid-sized instance
        # whose mask genuinely differs from its bounding box.
        if not (0.06 <= frac <= 0.45):
            continue
        fill = a["area"] / float(w * h)
        boxy = name in {"tv", "laptop", "couch", "bed", "refrigerator", "oven", "book"}
        if not (0.25 <= fill <= (0.97 if boxy else 0.82)):
            continue
        pts = sum(len(pp) for pp in seg) // 2
        if pts < 24:                                   # a real outline, not a quad
            continue
        out.append((fill, a, im))
    out.sort(key=lambda t: (-t[1]["area"]))
    return out[:k]


def cutout(a, im, pad=6):
    """RGBA cutout of one instance, cropped to its bbox, feathered at the edge."""
    src = Image.open(os.path.join(IMGDIR, im["file_name"])).convert("RGB")
    mask = Image.new("L", src.size, 0)
    dr = ImageDraw.Draw(mask)
    for poly in a["segmentation"]:
        if len(poly) >= 6:
            dr.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=255)
    x, y, w, h = [int(v) for v in a["bbox"]]
    box = (max(0, x - pad), max(0, y - pad),
           min(src.width, x + w + pad), min(src.height, y + h + pad))
    # An occluder (a laptop on a couch, a cat in front of a TV) is excluded from the
    # object's own polygon and leaves a hole. Fill interior holes so the cut-out is a
    # solid object; the occluding thing simply stays part of the scene.
    m = mask.crop(box)
    arr = np.asarray(m) > 127
    # a notch that opens to the object's edge is not an interior hole, so close it
    # morphologically first; radius scales with the object so small parts survive
    rad = max(3, int(0.045 * min(arr.shape)))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk = (xx ** 2 + yy ** 2) <= rad ** 2
    filled = binary_fill_holes(binary_closing(arr, structure=disk))
    m = Image.fromarray((filled * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.2))
    rgb = src.crop(box)
    out = rgb.convert("RGBA")
    out.putalpha(m)
    return out


def fit(img, target_h):
    s = target_h / img.height
    return img.resize((max(1, int(img.width * s)), target_h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default='cat,pizza,laptop,potted plant')
    ap.add_argument("--pick", default="0,0,0,0",
                    help="which candidate to use per object (comma-separated ranks)")
    ap.add_argument("--size", default="960,720")
    ap.add_argument("--out", default="paper/figs/fig3_stimulus.png")
    ap.add_argument("--contact", default="fig3_stimulus_candidates.png")
    args = ap.parse_args()

    names = [n.strip() for n in args.objects.split(",")]
    picks = [int(p) for p in args.pick.split(",")]
    W, H = [int(v) for v in args.size.split(",")]

    print("[compose] indexing COCO ...", flush=True)
    d, cats, imgs = load_index()

    cands = {n: best_instances(d, cats, imgs, n) for n in names}
    for n in names:
        print(f"  {n:14s} {len(cands[n])} candidates: "
              + ", ".join(f"{i}:{c[0]:.2f}" for i, c in enumerate(cands[n])))

    # contact sheet of the candidates, so a bad cutout can be swapped by rank
    cs = 190
    sheet = Image.new("RGB", (cs * 6, cs * len(names)), "white")
    for r, n in enumerate(names):
        for c, (frac, a, im) in enumerate(cands[n]):
            t = fit(cutout(a, im), cs - 20)
            bg = Image.new("RGB", (cs, cs), "white")
            bg.paste(t, ((cs - t.width) // 2, 10), t)
            sheet.paste(bg, (c * cs, r * cs))
    sheet.save(args.contact)
    print(f"[compose] candidate sheet -> {args.contact}")

    # ── compose ONE scene: wall + floor, objects standing on the ground plane at
    #    different depths, each with a soft contact shadow. Farther objects sit
    #    higher and smaller, which is what makes it read as a single room rather
    #    than four pasted cut-outs.
    canvas = Image.new("RGB", (W, H), (226, 222, 214))
    horizon = int(H * 0.52)
    floor = Image.new("RGB", (W, H - horizon), (196, 182, 166))
    canvas.paste(floor, (0, horizon))
    # gentle vertical shading on the wall and floor so the scene is not flat
    sh = Image.new("L", (1, H), 0)
    for yy in range(H):
        t = yy / H
        sh.putpixel((0, yy), int(18 + 26 * (1 - abs(t - 0.5) * 2)))
    canvas = Image.composite(canvas, Image.new("RGB", (W, H), (255, 255, 255)),
                             sh.resize((W, H)).point(lambda v: 255 - v))
    d = ImageDraw.Draw(canvas)
    d.line([(0, horizon), (W, horizon)], fill=(178, 166, 152), width=3)

    # (centre x, ground y, height) as fractions of the canvas
    SCENE = {0: (0.22, 0.80, 0.34), 1: (0.70, 0.72, 0.26),
             2: (0.46, 0.97, 0.30), 3: (0.92, 0.86, 0.36)}
    placed = {}
    for idx, ((n, p)) in enumerate(zip(names, picks)):
        cx_f, gy_f, h_f = SCENE[idx]
        frac, a, im = cands[n][p]
        t = fit(cutout(a, im), int(H * h_f))
        if t.width > W * 0.42:
            k = (W * 0.42) / t.width
            t = t.resize((int(t.width * k), int(t.height * k)), Image.LANCZOS)
        x = int(W * cx_f) - t.width // 2
        y = int(H * gy_f) - t.height
        x = max(4, min(W - t.width - 4, x)); y = max(4, min(H - t.height - 4, y))
        # contact shadow: a blurred ellipse just under the object
        sw, shh = int(t.width * 0.86), max(8, int(t.height * 0.10))
        shadow = Image.new("L", (sw + 40, shh + 40), 0)
        ImageDraw.Draw(shadow).ellipse((20, 20, 20 + sw, 20 + shh), fill=96)
        shadow = shadow.filter(ImageFilter.GaussianBlur(11))
        canvas.paste(Image.new("RGB", shadow.size, (120, 110, 100)),
                     (x + (t.width - sw) // 2 - 20, y + t.height - shh // 2 - 20), shadow)
        canvas.paste(t, (x, y), t)
        placed[n] = {"bbox": [x, y, t.width, t.height],
                     "source": im["file_name"], "rank": p}
        print(f"  placed {n:14s} at {(x, y)} size {t.size} from {im['file_name']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    json.dump({"size": [W, H], "objects": placed},
              open(os.path.splitext(args.out)[0] + "_layout.json", "w"), indent=2)
    print(f"[compose] stimulus -> {args.out}")


if __name__ == "__main__":
    main()
