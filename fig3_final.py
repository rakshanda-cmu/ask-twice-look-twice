#!/usr/bin/env python3
"""
Fig. 3, stated plainly: perception is LOCALIZED, and the question STEERS the read-out.

One scene, one model, one layer. Four panels, read left to right, top to bottom:

  (1) input        the image the model is given, nothing overlaid
  (2) LOCALIZATION image-first (question AFTER the image). Each object region's
                   patch decodes, through the logit lens, to that object. The
                   causal mask hides the question from the patches, so this panel
                   is identical whichever question is asked (0% of patches differ).
  (3),(4) STEERING question-first (question BEFORE the image), the same scene under
                   two questions. A large share of patches now decode differently,
                   and the three marked patches follow whichever object was asked
                   about: couch words for one question, plant words for the other.

Panels 2 and 3/4 mark DIFFERENT patches on purpose. Object regions carry the
localization claim; the steering is spatially diffuse (it is not a spotlight on the
queried object), so it is shown where it is strongest, on background patches. The
caption must say so.

    python fig3_final.py --loc fig3_probe_room.json --steer fig3_probe_perfect.json
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

LONG_SIDE = 896
COLD = "#1565c0"     # localization / frozen read-out
HOT = "#d81b60"      # steering
INK = "#111111"
ANCH = "#ffd400"

PRETTY = {"potted plant": "plant", "couch": "couch", "chair": "chair",
          "tv": "television", "cat": "cat", "laptop": "laptop"}
LOC_PLACE = {                      # object regions -> callout corners
    "umbrella": (0.012, 0.988, "left", "top"),
    "cat":      (0.988, 0.015, "right", "bottom"),
}
STEER_PLACE = {                    # patches that answer the question asked
    "on the umbrella":    (0.988, 0.988, "right", "top"),
    "under the umbrella": (0.012, 0.988, "left", "top"),
    "on the cat":         (0.012, 0.015, "left", "bottom"),
}


def callouts(ax, img, anchors, toks, gh, gw, place, colour, ref=None):
    for a in anchors:
        name, (rr, cc) = a["name"], a["rc"]
        px, py = (cc + .5) / gw * img.width, (rr + .5) / gh * img.height
        w, h = img.width / gw, img.height / gh
        ax.add_patch(Rectangle((px - w / 2, py - h / 2), w, h,
                               fill=False, ec=ANCH, lw=1.6, zorder=6))
        words = [t[:11] for t in toks[rr * gw + cc]]
        moved = ref is not None and words != [t[:11] for t in ref[rr * gw + cc]]
        col = colour if (ref is None or moved) else INK
        tx, ty, ha, va = place[name]
        ax.annotate(f"{PRETTY.get(name, name)}\n" + " · ".join(words),
                    xy=(px, py), xycoords="data", xytext=(tx, ty),
                    textcoords="axes fraction", ha=ha, va=va, zorder=7,
                    fontsize=6.5, color=col, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=col,
                              lw=1.1, alpha=0.96),
                    arrowprops=dict(arrowstyle="-", color=ANCH, lw=1.2,
                                    shrinkA=1, shrinkB=1, alpha=0.95,
                                    connectionstyle="arc3,rad=0.06"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="fig3_probe_room.json")
    ap.add_argument("--steer", default="fig3_probe_perfect.json")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default="paper/figs/fig3_final")
    args = ap.parse_args()

    L = json.load(open(args.loc)); S = json.load(open(args.steer))
    lr, sr = L[str(args.layer)], S[str(args.layer)]
    gh, gw = lr["no question"]["gh"], lr["no question"]["gw"]

    pil = Image.open(L["image_path"]).convert("RGB")
    s = LONG_SIDE / max(pil.size)
    pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)

    frozen = lr["frac_changed"]["SIT"] * 100
    steered = sr["frac_changed"]["STI"] * 100
    q1, q2 = S["q1"], S["q2"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=400)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.845, bottom=0.075,
                        wspace=0.035, hspace=0.300)
    for ax in axes.ravel():
        ax.imshow(pil); ax.set_xlim(0, pil.width); ax.set_ylim(pil.height, 0); ax.axis("off")

    # (1) the raw input
    axes[0][0].set_title("1 · input image", fontsize=8.6, fontweight="bold",
                         color=INK, pad=4)

    # (2) localization, read off the image-first pass
    callouts(axes[0][1], pil, L["anchors"], lr["SIT + q1"]["toks"], gh, gw,
             LOC_PLACE, COLD)
    axes[0][1].set_title("2 · image-first: each region names its own object",
                         fontsize=8.6, fontweight="bold", color=COLD, pad=4)

    # (3),(4) steering, on the patches where the rewrite is strongest
    for ax, key, q, tag in ((axes[1][0], "STI + q1", q1, "3"),
                            (axes[1][1], "STI + q2", q2, "4")):
        callouts(ax, pil, S["steer_anchors"], sr[key]["toks"], gh, gw, STEER_PLACE, HOT,
                 ref=sr["SIT + q1"]["toks"])
        ax.set_title(f"{tag} · question-first  +  “{q}”", fontsize=8.0,
                     fontweight="bold", color=HOT, pad=4)

    fig.text(0.5, 0.988,
             "Perception is localized; the question steers the read-out",
             ha="center", va="top", fontsize=11.0, fontweight="bold")
    fig.text(0.5, 0.945,
             f"Qwen3-VL-8B, layer {args.layer}. Every panel is the same image and the "
             f"same layer; only where the question sits changes.\n"
             f"Each callout lists the three highest-probability English word tokens at "
             f"that one patch.",
             ha="center", va="top", fontsize=6.6, color="#444444", linespacing=1.5)

    fig.text(0.5, 0.462,
             f"LOCALIZATION: the question sits after the image, so the causal mask hides "
             f"it.\nPanel 2 is identical for both questions ({frozen:.0f}% of patches differ).",
             ha="center", va="center", fontsize=7.2, fontweight="bold", color=COLD,
             linespacing=1.5)
    fig.text(0.5, 0.030,
             f"STEERING: the question now precedes the image. {steered:.0f}% of patches "
             f"decode differently, and the patches\nanswer the question that was "
             f"asked: the cat reads \u201csitting\u201d in panel 3, and the umbrella reads "
             f"\u201cpurple\u201d in panel 4. Neither appears under the other question.",
             ha="center", va="center", fontsize=7.4, fontweight="bold", color=HOT,
             linespacing=1.5)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png/.pdf")
    print(f"  frozen {frozen:.1f}%   steered {steered:.1f}%")


if __name__ == "__main__":
    main()
