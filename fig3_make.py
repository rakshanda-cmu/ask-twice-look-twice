#!/usr/bin/env python3
"""
Render the new Fig. 3 from the probe dump (no GPU needed).

One scene, one layer, the same four patches throughout; a 2x2 grid crosses the
two questions with the two orderings:

  top row     image-first (SIT): the question sits after the image, so the causal
              mask hides it from the patches. The two panels are *identical* --
              the visual read-out cannot depend on the question -- and each
              region still names its own object (the localization claim).
  bottom row  question-first (STI): the question now precedes the image and
              rewrites what the patches decode to, differently for each question.
              Callouts whose tokens moved are drawn in the highlight colour.

    python fig3_make.py --probe fig3_probe.json --layer 28
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

IMGDIR = "/data2/datasets/COCO/val2014"
LONG_SIDE = 896

HL = "#d81b60"        # this callout moved with the question
COLD = "#1565c0"      # frozen read-out
INK = "#111111"
ANCH = "#ffd400"

# Callout box position per region, in axes fraction (y measured from the BOTTOM),
# chosen so each box sits beside its own anchor and no leader line crosses the scene.
# Keyed by scene so one renderer serves whichever scene was probed.
PLACE_BY_SCENE = {
    # constructed 4-object stimulus (fig3_compose.py): one callout per quadrant
    "stimulus": {
        "cat":          (0.012, 0.988, "left", "top"),
        "dog":          (0.988, 0.988, "right", "top"),
        "pizza":        (0.012, 0.015, "left", "bottom"),
        "potted plant": (0.988, 0.015, "right", "bottom"),
    },
    # callouts on chosen patches (fig3_patches.py) rather than object regions
    "patches": {
        "patch A": (0.988, 0.988, "right", "top"),
        "patch B": (0.988, 0.360, "right", "bottom"),
        "patch C": (0.012, 0.015, "left", "bottom"),
    },
    "livingroom": {
        "potted plant": (0.012, 0.988, "left", "top"),
        "chair":        (0.988, 0.988, "right", "top"),
        "couch":        (0.012, 0.015, "left", "bottom"),
    },
    "fruit": {
        "banana": (0.012, 0.988, "left", "top"),
        "apple":  (0.988, 0.988, "right", "top"),
        "orange": (0.988, 0.015, "right", "bottom"),
    },
    "cat_tv": {
        "tv":    (0.012, 0.988, "left", "top"),
        "chair": (0.988, 0.988, "right", "top"),
        "cat":   (0.012, 0.015, "left", "bottom"),
    },
    "shopfront": {
        "potted plant": (0.012, 0.655, "left", "bottom"),
        "bicycle":      (0.988, 0.470, "right", "bottom"),
        "bench":        (0.012, 0.015, "left", "bottom"),
        "chair":        (0.988, 0.015, "right", "bottom"),
    },
}
PRETTY = {"potted plant": "flower", "bicycle": "bicycles", "bench": "bench",
          "tv": "monitor", "cat": "cat", "chair": "chair"}


def pick_places(anchors):
    """The placement table whose keys cover this scene's regions."""
    have = {a["name"] for a in anchors}
    for tbl in PLACE_BY_SCENE.values():
        if have <= set(tbl):
            return tbl
    raise SystemExit(f"no callout placement for regions {sorted(have)}")


def panel(ax, img, anchors, toks, gh, gw, ref=None, place=None):
    ax.imshow(img)
    ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0); ax.axis("off")

    for a in anchors:
        name, (rr, cc) = a["name"], a["rc"]
        px = (cc + 0.5) / gw * img.width
        py = (rr + 0.5) / gh * img.height
        i = rr * gw + cc
        w, h = img.width / gw, img.height / gh
        ax.add_patch(Rectangle((px - w / 2, py - h / 2), w, h,
                               fill=False, ec=ANCH, lw=1.5, zorder=6))

        words = [t[:11] for t in toks[i]]
        moved = ref is not None and words != [t[:11] for t in ref[i]]
        col = HL if moved else INK

        tx, ty, ha, va = place[name]
        label = f"{PRETTY.get(name, name)}\n" + " · ".join(words)
        ax.annotate(
            label, xy=(px, py), xycoords="data",
            xytext=(tx, ty), textcoords="axes fraction",
            ha=ha, va=va, zorder=7, fontsize=6.6, color=col, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=col,
                      lw=1.0 if moved else 0.7, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=ANCH, lw=1.1,
                            shrinkA=1, shrinkB=1, alpha=0.95,
                            connectionstyle="arc3,rad=0.06"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="fig3_probe.json")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default="paper/figs/fig3_steering")
    args = ap.parse_args()

    d = json.load(open(args.probe))
    res = d[str(args.layer)]
    anchors, q1, q2 = d["anchors"], d["q1"], d["q2"]
    gh, gw = res["no question"]["gh"], res["no question"]["gw"]
    chg = res["frac_changed"]

    src = d.get("image_path") or os.path.join(IMGDIR, d["file"])
    pil = Image.open(src).convert("RGB")
    s = LONG_SIDE / max(pil.size)
    pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
    place = pick_places(anchors)

    sit1, sit2 = res["SIT + q1"]["toks"], res["SIT + q2"]["toks"]
    sti1, sti2 = res["STI + q1"]["toks"], res["STI + q2"]["toks"]
    assert sit1 == sit2, "image-first read-out should be identical across questions"

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.75), dpi=400)
    fig.subplots_adjust(left=0.058, right=0.995, top=0.880, bottom=0.090,
                        wspace=0.030, hspace=0.235)

    panel(axes[0][0], pil, anchors, sit1, gh, gw, ref=None, place=place)
    panel(axes[0][1], pil, anchors, sit2, gh, gw, ref=sit1, place=place)
    panel(axes[1][0], pil, anchors, sti1, gh, gw, ref=sit1, place=place)
    panel(axes[1][1], pil, anchors, sti2, gh, gw, ref=sit1, place=place)

    # column headers (the two questions) and row headers (the two orderings)
    for ax, q, tag in ((axes[0][0], q1, "q1"), (axes[0][1], q2, "q2")):
        ax.set_title(f"{tag}: “{q}”", fontsize=7.2, pad=3.5, color=INK)

    def rowlab(ax, txt, col):
        ax.text(-0.028, 0.5, txt, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=8.0, fontweight="bold", color=col)

    rowlab(axes[0][0], "image-first (SIT)", COLD)
    rowlab(axes[1][0], "question-first (STI)", HL)

    fig.text(0.524, 0.990,
             "Visual steering: the question changes what the image patches decode to,\n"
             "but only when it precedes the image",
             ha="center", va="top", fontsize=8.8, fontweight="bold", linespacing=1.45)

    fig.text(0.524, 0.482,
             f"read-out FROZEN: the two panels above are identical, patch for patch "
             f"({chg['SIT']*100:.0f}% differ)",
             ha="center", va="center", fontsize=7.5, fontweight="bold", color=COLD)
    fig.text(0.524, 0.048,
             f"read-out STEERED: {chg['STI']*100:.0f}% of patches decode to a different "
             f"word; the coloured callouts are the ones that moved",
             ha="center", va="center", fontsize=7.5, fontweight="bold", color=HL)
    fig.text(0.524, 0.008,
             f"Qwen3-VL-8B, layer {args.layer}; each callout lists the three highest-probability "
             "English word tokens at that one patch.",
             ha="center", va="center", fontsize=6.2, color="#555555")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png/.pdf")


if __name__ == "__main__":
    main()
