#!/usr/bin/env python3
"""
Three-panel version of Fig. 3, side by side:

    image-first (SIT)  |  question-first + q1  |  question-first + q2

Panel 1 carries the localization claim: with the question after the image, the mask
hides it from the patches, each region names its own object, and the panel is the
same whichever question is asked. Panels 2 and 3 carry the steering claim: with the
question before the image, the marked patches answer the question that was asked.

Same data and same callouts as fig3_final.py, laid out in one row instead of a 2x2.

    python fig3_row.py --probe fig3_probe_umbrella.json
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from fig3_final import callouts, LOC_PLACE, STEER_PLACE, COLD, HOT, INK, LONG_SIDE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="fig3_probe_umbrella.json")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default="paper/figs/fig3_row")
    args = ap.parse_args()

    d = json.load(open(args.probe))
    r = d[str(args.layer)]
    gh, gw = r["no question"]["gh"], r["no question"]["gw"]
    q1, q2 = d["q1"], d["q2"]
    frozen = r["frac_changed"]["SIT"] * 100
    steered = r["frac_changed"]["STI"] * 100

    pil = Image.open(d["image_path"]).convert("RGB")
    s = LONG_SIDE / max(pil.size)
    pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), dpi=400)
    fig.subplots_adjust(left=0.006, right=0.994, top=0.800, bottom=0.150, wspace=0.030)
    for ax in axes:
        ax.imshow(pil); ax.set_xlim(0, pil.width); ax.set_ylim(pil.height, 0); ax.axis("off")

    callouts(axes[0], pil, d["anchors"], r["SIT + q1"]["toks"], gh, gw, LOC_PLACE, COLD)
    axes[0].set_title("image-first (SIT): question after the image",
                      fontsize=8.0, fontweight="bold", color=COLD, pad=4)
    axes[0].text(0.5, -0.045, f"each region names its own object;\nidentical for q1 and q2 "
                 f"({frozen:.0f}% of patches differ)", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=6.8, color=COLD, fontweight="bold",
                 linespacing=1.45)

    for ax, key, q in ((axes[1], "STI + q1", q1), (axes[2], "STI + q2", q2)):
        callouts(ax, pil, d["steer_anchors"], r[key]["toks"], gh, gw, STEER_PLACE, HOT,
                 ref=r["SIT + q1"]["toks"])
        ax.set_title(f"question-first (STI)  +  “{q}”", fontsize=8.0,
                     fontweight="bold", color=HOT, pad=4)

    axes[1].text(1.02, -0.045,
                 f"the marked patches answer the question that was asked "
                 f"({steered:.0f}% of patches decode differently)",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=6.8, color=HOT, fontweight="bold")

    fig.text(0.5, 0.985, "Perception is localized; the question steers the read-out",
             ha="center", va="top", fontsize=10.6, fontweight="bold")
    fig.text(0.5, 0.930,
             f"Qwen3-VL-8B, layer {args.layer}. All three panels are the same image and the "
             f"same layer; only where the question sits changes. "
             f"Each callout lists the three highest-probability English word tokens at that patch.",
             ha="center", va="top", fontsize=6.2, color="#444444")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png/.pdf")
    print(f"  frozen {frozen:.1f}%   steered {steered:.1f}%")


if __name__ == "__main__":
    main()
