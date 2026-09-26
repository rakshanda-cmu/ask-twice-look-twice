#!/usr/bin/env python3
"""
Generate the paper's "steering" figure crops (paper/figs/steer_{ist,sti,stit}.png).

For one curated NaturalBench example we take the precomputed logit-lens renders
(naturalbench/midlayer/<code>/normal/<ORDER>/final.png), which show every image
patch overlaid with the vocabulary token its final-layer hidden state decodes to,
and crop out *just the image-patch grid* for each ordering (IST / STI / STIT).
Placed side by side, the crops show that question-first (STI) and bracketing
(STIT) steer the patches toward the question, while STIT (unlike STI) still answers
correctly -- "best of both worlds".

The renders stack several labelled sections (SYSTEM / TASK / "Logit Lens - Layer N"
/ GENERATED). The image-patch grid is the block directly under the black
"Logit Lens" title bar. We locate that bar from strong horizontal banner colors
(near-black title/colorbar bars, teal SYSTEM header, blue TASK header) and crop a
fixed grid height -- the patch grid has the same pixel height across orderings
because it is the same source image.

Usage:
    python make_steering_fig.py                 # default example g118_i1_q1
    python make_steering_fig.py --code g30_i0_q0
    python make_steering_fig.py --code g118_i1_q1 --grid-height 654
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

MIDLAYER = "naturalbench/midlayer"
OUT_DIR = "paper/figs"
ORDERS = ("IST", "STI", "STIT")
BOX_RGB = (220, 20, 60)   # crimson zoom-region box


def _row_class(rgb):
    """Classify a row's mean RGB into a strong banner type or 'grid'."""
    r, g, b = rgb
    if r < 22 and g < 22 and b < 22:
        return "BLACK"                      # title bar / colorbar frame
    if b > 120 and b > r + 40 and b > g + 30:
        return "BLUE"                       # TASK header
    if g > 90 and g > r + 25 and abs(g - b) < 55 and b > 70:
        return "TEAL"                       # SYSTEM header
    return "grid"


def _segments(img):
    """List of (y_start, class) where the row class changes (banners only)."""
    rows = np.asarray(img.convert("RGB")).astype(int).mean(axis=1)  # (H,3)
    segs, prev = [], None
    for y, rgb in enumerate(rows):
        c = _row_class(rgb)
        if c != prev:
            if c != "grid":
                segs.append((y, c))
            prev = c
    return segs, rows.shape[0]


def _grid_top(img):
    """Bottom y of the black 'Logit Lens - Layer N' title bar (grid starts here).

    The title bar is the black banner directly followed by the tall patch grid:
    the black segment whose gap to the next strong banner is the largest.
    """
    segs, H = _segments(img)
    blacks = [y for y, c in segs if c == "BLACK"]
    banner_ys = sorted(y for y, _ in segs) + [H]
    best_end, best_gap = None, -1
    for y in blacks:
        nxt = min((by for by in banner_ys if by > y + 4), default=H)
        # skip the 1-2px black underline of the title text; the title bar is a
        # thick black band -> take the black run's end as grid top.
        end = y
        for yy, c in segs:
            if c == "BLACK" and abs(yy - y) < 45:
                end = max(end, yy)
        gap = nxt - end
        if gap > best_gap:
            best_gap, best_end = gap, end
    return best_end


def _grid_height(imgs):
    """Shared patch-grid pixel height: min top->next-banner distance across
    orderings (the patch grid is identical across orderings for one image)."""
    heights = []
    for img in imgs.values():
        segs, H = _segments(img)
        top = _grid_top(img)
        nxt = min((y for y, _ in segs if y > top + 20), default=H)
        d = nxt - top
        if 400 < d < 720:            # a plausible single grid block
            heights.append(d)
    return min(heights) if heights else 650


def _zoom_region(arg):
    vals = [float(v) for v in arg.split(",")]
    if len(vals) != 4:
        raise SystemExit("--zoom needs x0,y0,x1,y1 (normalized 0..1)")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="g118_i1_q1", help="example code under %s" % MIDLAYER)
    ap.add_argument("--grid-height", type=int, default=None,
                    help="override auto grid height (px)")
    ap.add_argument("--margin", type=int, default=2, help="trim px off each grid edge")
    ap.add_argument("--zoom", type=_zoom_region, default="0.06,0.28,0.62,0.66",
                    help="normalized x0,y0,x1,y1 region to zoom (of the patch grid)")
    ap.add_argument("--upscale", type=int, default=3,
                    help="nearest-neighbour upscale factor for zoom crops (keeps cell edges crisp)")
    args = ap.parse_args()

    base = os.path.join(MIDLAYER, args.code, "normal")
    imgs = {o: Image.open(os.path.join(base, o, "final.png")).convert("RGB")
            for o in ORDERS if os.path.isdir(os.path.join(base, o))}
    if not imgs:
        raise SystemExit("no renders found under %s" % base)

    gh = args.grid_height or _grid_height(imgs)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) full patch-grid crops (steer_*.png) + zoomed region crops (zoom_*.png)
    rx0, ry0, rx1, ry1 = args.zoom
    grids = {}
    for o, img in imgs.items():
        top = _grid_top(img) + args.margin
        box = (0, top, img.width, top + gh - 2 * args.margin)
        crop = img.crop(box)
        grids[o] = crop
        crop.save(os.path.join(OUT_DIR, "steer_%s.png" % o.lower()))
        W, H = crop.size
        zbox = (int(rx0 * W), int(ry0 * H), int(rx1 * W), int(ry1 * H))
        zoom = crop.crop(zbox)
        if args.upscale > 1:
            zoom = zoom.resize((zoom.width * args.upscale, zoom.height * args.upscale),
                               Image.NEAREST)
        # crimson frame to tie the zoom back to the box on the original
        dz = ImageDraw.Draw(zoom)
        fw = max(4, int(0.012 * zoom.width))
        dz.rectangle((0, 0, zoom.width - 1, zoom.height - 1), outline=BOX_RGB, width=fw)
        zoom.save(os.path.join(OUT_DIR, "zoom_%s.png" % o.lower()))
        print("%-5s grid=%s zoom_px=%s" % (o, box, zbox))

    # 2) original photo with the zoom region boxed (normalized coords match the
    #    grid: the vision encoder resizes the photo, so patch (u,v) <-> photo (u,v))
    man = json.load(open(os.path.join(MIDLAYER, "manifest.json")))
    ex = next((e for e in man["examples"] if e["code"] == args.code), None)
    if ex:
        photo = Image.open(os.path.join("naturalbench", ex["image_rel"])).convert("RGB")
        d = ImageDraw.Draw(photo)
        pw, ph = photo.size
        lw = max(3, int(0.012 * max(pw, ph)))
        d.rectangle((rx0 * pw, ry0 * ph, rx1 * pw, ry1 * ph), outline=BOX_RGB, width=lw)
        photo.save(os.path.join(OUT_DIR, "steer_original.png"))
        print("photo %s boxed -> steer_original.png" % (photo.size,))
        print("question:", ex["question"])
        print("expected:", ex["expected"],
              "| answers:", ex.get("answers_generated", {}).get("normal", {}))


if __name__ == "__main__":
    main()
