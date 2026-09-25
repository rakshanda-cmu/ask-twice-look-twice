#!/usr/bin/env python3
"""
Blue-heatmap variant of Fig. 2, on the SAME example as fig2_steering.

Instead of printing a word in every patch cell, this renders the website's
evidence heat field: patches whose logit-lens read-out decodes one of the
QUESTION's content words are marked, smoothed into a continuous field, and
painted with the Blues colormap over a desaturated copy of the photo, exactly as
website/build_heatmap.py does (light = no evidence, dark blue = evidence).

The point is the same as the word grid but readable at a glance: question-first
(STI) lights up with the question's own vocabulary and still answers wrong;
question-last (SIT) stays pale and answers right; echoing (STIT) keeps the lit-up
field and recovers the answer.

    CUDA_VISIBLE_DEVICES=1 python fig2_heat.py --group 1007
"""
import argparse, json, os

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Rectangle
from PIL import Image

from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from naturalbench_eval import answer_suffix
from fig2_search import run, matches, ok, VAGUE, STOP, NB_ROOT, LOW_MAX, ORDERS

CMAP = cm.get_cmap("Blues")          # light = 0, dark blue = 1 (matches the site)
OK_C, BAD_C = "#1b7a35", "#c62828"
TITLE = {"SIT": "SIT (question-last)", "STI": "STI (question-first)",
         "STIT": "STIT (ours, echoing)"}


def heat_field(words, targets, gh, gw, W, H):
    """Patch evidence mask -> smooth field in [0,1] at image resolution."""
    mask = np.array([[1.0 if any(matches(words[r * gw + c], t) for t in targets) else 0.0
                      for c in range(gw)] for r in range(gh)])
    zy, zx = H / gh, W / gw
    heat = zoom(mask, (zy, zx), order=1)
    out = np.zeros((H, W))
    h2 = heat[:H, :W]
    out[:h2.shape[0], :h2.shape[1]] = h2
    out = gaussian_filter(out, sigma=max(zy, zx) * 0.5)
    return out, mask.sum()


def overlay(raw, heat, vmax):
    """Blues field over a faint grayscale photo, as website/build_heatmap.py."""
    gray = np.asarray(raw.convert("L").convert("RGB")).astype(float)
    color = CMAP(np.clip(heat / vmax, 0, 1) if vmax > 0 else heat)[:, :, :3] * 255.0
    return Image.fromarray((gray * 0.40 + color * 0.60).clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=1007)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hits", default="fig2_hits.json")
    ap.add_argument("--out", default="paper/figs/fig2_steering_heat")
    args = ap.parse_args()

    rec = next(r for r in json.load(open(args.hits)) if r["key"][0] == args.group)
    layer = args.layer if args.layer is not None else rec["layer"]
    DROP = VAGUE | STOP | {"visible", "present", "depicted", "shown"}
    targets = [t for t in rec["targets"] if t not in DROP]

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pil = Image.open(os.path.join(NB_ROOT, rec["image_file"])).convert("RGB")
    s = LOW_MAX / max(pil.size)
    small = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))),
                       Image.LANCZOS) if s < 1 else pil
    query = rec["question"] + answer_suffix("yes_no")

    mm = ModelManager("qwen3-vl-8b")
    res = {o: run(mm, small, query, o, [layer]) for o in ORDERS}
    gh, gw = res["SIT"][3], res["SIT"][4]
    W = {o: res[o][1][layer] for o in ORDERS}
    A = {o: res[o][0] for o in ORDERS}

    # render the heat on the FULL-resolution photo, from the low-res patch grid
    disp = pil if max(pil.size) <= 900 else pil.resize(
        (int(pil.width * 900 / max(pil.size)), int(pil.height * 900 / max(pil.size))),
        Image.LANCZOS)
    PW, PH = disp.size
    fields, counts = {}, {}
    for o in ORDERS:
        fields[o], counts[o] = heat_field(W[o], targets, gh, gw, PW, PH)
    vmax = max(f.max() for f in fields.values()) or 1.0

    print(f"[heat] g{args.group} layer {layer} grid {gh}x{gw} targets={targets}")
    for o in ORDERS:
        print(f"   {o:5s} answer={A[o]!r:8s} question-word patches={int(counts[o])}")

    fig = plt.figure(figsize=(11.2, 3.5), dpi=400)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.04],
                          left=0.006, right=0.962, top=0.760, bottom=0.055, wspace=0.045)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(disp); ax0.axis("off")
    ax0.set_title("Input", fontsize=9.4, fontweight="bold", pad=5)

    for k, o in enumerate(ORDERS):
        ax = fig.add_subplot(gs[0, k + 1])
        ax.imshow(overlay(disp, fields[o], vmax)); ax.axis("off")
        right = ok(A[o], rec["gt"])
        col = OK_C if right else BAD_C
        ax.add_patch(Rectangle((0, 0), PW - 1, PH - 1, fill=False, ec=col, lw=3.2))
        ax.set_title(f"{TITLE[o]}\n→ “{A[o]}” {'✓' if right else '✗'}",
                     fontsize=8.6, pad=5, color=col, fontweight="bold", linespacing=1.35)
        n = int(counts[o])
        ax.text(0.5, -0.035,
                f"{n} patch{'' if n == 1 else 'es'} decode{'s' if n == 1 else ''} "
                f"a question word",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color="#333333")

    cax = fig.add_subplot(gs[0, 4])
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap="Blues")
    fig.colorbar(sm, cax=cax)
    cax.tick_params(labelsize=5.5)
    cax.set_ylabel("question-word evidence", fontsize=6.4)

    fig.text(0.5, 0.985, f"Q: “{rec['question']}”   (ground truth: {rec['gt']})",
             ha="center", va="top", fontsize=9.6, fontweight="bold")
    fig.text(0.5, 0.918,
             f"dark blue = patches whose logit lens decodes a word from the question "
             f"({', '.join(targets)});  Qwen3-VL-8B, layer {layer}",
             ha="center", va="top", fontsize=6.8, color="#444444")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png/.pdf")


if __name__ == "__main__":
    main()
