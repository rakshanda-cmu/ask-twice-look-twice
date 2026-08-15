#!/usr/bin/env python3
"""
Render the new Fig. 2: "steering happens, yet is not read out".

Four panels across one NaturalBench pair:

    Input (crimson box = magnified region) | SIT | STI | STIT

Each of the three ordering panels shows the magnified region's image patches, one
cell per patch, labelled with the vocabulary token that patch's hidden state
decodes to under the logit lens and coloured by that token's probability
(viridis). Above each panel is the answer the model actually generated in that
ordering. Cells whose token is one of the QUESTION's own content words are ringed,
so the steering is countable rather than a matter of impression:

    SIT  (question-last)  generic tokens, answers RIGHT
    STI  (question-first) question's words appear on the patches, answers WRONG
    STIT (echoing)        same steered patches, answers RIGHT again

    CUDA_VISIBLE_DEVICES=1 python fig2_make.py --group 140 --layer 32
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm, colors as mcolors
from PIL import Image

from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from naturalbench_eval import answer_suffix
from fig2_search import run, matches, ok, VAGUE, STOP, NB_ROOT, LOW_MAX, ORDERS

CRIMSON = "#dc143c"
RING = "#ff2d55"
HITC = "#d81b26"      # red boxes mark the patches decoding a question word
GRIDC = "#ffffff"     # thin cell separators in the translucent "wash" style
WASH_LO, WASH_HI = 0.14, 0.82   # slice of Blues used by the wash (stays pale)
TITLE = {"SIT": "SIT (question-last)", "STI": "STI (question-first)",
         "STIT": "STIT (ours, echoing)"}


def zoom_box(words, targets, gh, gw, pad=2, min_side=5):
    """Tight box around the patches that decode a question word, padded."""
    hit = [(i // gw, i % gw) for i, w in enumerate(words)
           if any(matches(w, t) for t in targets)]
    if not hit:
        return 0, 0, gh, gw
    rs = [r for r, _ in hit]; cs = [c for _, c in hit]
    r0, r1 = max(0, min(rs) - pad), min(gh, max(rs) + pad + 1)
    c0, c1 = max(0, min(cs) - pad), min(gw, max(cs) + pad + 1)
    while r1 - r0 < min_side and (r0 > 0 or r1 < gh):
        if r0 > 0: r0 -= 1
        if r1 < gh: r1 += 1
    while c1 - c0 < min_side and (c0 > 0 or c1 < gw):
        if c0 > 0: c0 -= 1
        if c1 < gw: c1 += 1
    return r0, c0, r1, c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=None)
    ap.add_argument("--groups", default=None,
                    help="comma-separated groups to render in one model load")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hits", default="fig2_hits.json")
    ap.add_argument("--out", default="paper/figs/fig2_steering")
    ap.add_argument("--cmap", default="viridis",
                    help="cell shading colormap; 'Blues' matches the website")
    ap.add_argument("--wash", action="store_true",
                    help="translucent pale shading over the COLOUR photo, as on the site")
    ap.add_argument("--alpha", type=float, default=None,
                    help="cell fill opacity (default 0.44, or 0.45 with --wash)")
    ap.add_argument("--mm", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.alpha is None:
        args.alpha = 0.55 if args.wash else 0.44
    hits = json.load(open(args.hits))
    groups = ([int(g) for g in args.groups.split(",")] if args.groups
              else [args.group])
    if groups == [None]:
        raise SystemExit("pass --group or --groups")

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    mm = ModelManager("qwen3-vl-8b")
    for g in groups:
        render(mm, hits, g, args)


def render(mm, hits, group, args):
    rec = next(r for r in hits if r["key"][0] == group)
    layer = args.layer if args.layer is not None else rec["layer"]
    # drop words that name no content ("image", "visible"): a patch decoding
    # them is not evidence that the question steered perception
    DROP = VAGUE | STOP | {"visible", "present", "depicted", "shown"}
    targets = [t for t in rec["targets"] if t not in DROP]

    pil = Image.open(os.path.join(NB_ROOT, rec["image_file"])).convert("RGB")
    s = LOW_MAX / max(pil.size)
    small = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))),
                       Image.LANCZOS) if s < 1 else pil
    query = rec["question"] + answer_suffix("yes_no")

    res = {o: run(mm, small, query, o, [layer]) for o in ORDERS}
    gh, gw = res["SIT"][3], res["SIT"][4]
    W = {o: res[o][1][layer] for o in ORDERS}
    P = {o: np.array(res[o][2][layer]) for o in ORDERS}
    A = {o: res[o][0] for o in ORDERS}

    r0, c0, r1, c1 = 0, 0, gh, gw
    print(f"[fig2] g{group} layer {layer} grid {gh}x{gw} zoom rows {r0}:{r1} cols {c0}:{c1}")
    for o in ORDERS:
        n = sum(any(matches(w, t) for t in targets) for w in W[o])
        print(f"   {o:5s} answer={A[o]!r:8s} question-word patches={n}")

    disp = small.resize((gw * 64, gh * 64), Image.LANCZOS)
    cell = 64

    fig = plt.figure(figsize=(11.2, 3.35), dpi=400)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.06, 1, 1, 1, 0.045],
                          left=0.006, right=0.965, top=0.775, bottom=0.035, wspace=0.055)

    # ── input with the magnified region boxed
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(small); ax0.axis("off")
    bw, bh = small.width / gw, small.height / gh
    ax0.set_title("Input", fontsize=9.2, fontweight="bold", pad=4)
    ax0.text(0.5, -0.035, "every image patch is shown at right", transform=ax0.transAxes,
             ha="center", va="top", fontsize=6.6, color="#444444")

    norm = mcolors.Normalize(0, 1)
    sm = cm.ScalarMappable(norm=norm, cmap=args.cmap)
    # Blues runs light->dark with probability, viridis dark->light, so the
    # legible text colour flips between them
    dark_at_high = args.cmap.lower().startswith("blues")

    for k, o in enumerate(ORDERS):
        ax = fig.add_subplot(gs[0, k + 1])
        crop = disp.crop((c0 * cell, r0 * cell, c1 * cell, r1 * cell))
        if dark_at_high and not args.wash:
            # a Blues field over a colour photo reads teal where the photo is green;
            # desaturating underneath keeps the shading true blue (as the website does)
            crop = crop.convert("L").convert("RGB")
        ax.imshow(crop, extent=[c0, c1, r1, r0])
        right = ok(A[o], rec["gt"])
        for r in range(r0, r1):
            for c in range(c0, c1):
                i = r * gw + c
                w, p = W[o][i], float(P[o][i])
                hit = any(matches(w, t) for t in targets)
                if args.wash:
                    # translucent light-blue wash: compress the ramp into the pale
                    # half of Blues and stay see-through, so the colour photo reads
                    # underneath and the shading still orders the probabilities
                    face = sm.to_rgba(WASH_LO + (WASH_HI - WASH_LO) * p)
                    ax.add_patch(Rectangle((c, r), 1, 1, fc=face, alpha=args.alpha,
                                           ec=HITC if hit else GRIDC,
                                           lw=2.4 if hit else 0.35,
                                           zorder=3 if hit else 2))
                    txt = "black" if p < 0.80 else "white"
                else:
                    ax.add_patch(Rectangle((c, r), 1, 1, fc=sm.to_rgba(p),
                                           alpha=args.alpha,
                                           ec=RING if hit else "none",
                                           lw=1.8 if hit else 0))
                    txt = ("white" if p > 0.55 else "black") if dark_at_high \
                        else ("white" if p < 0.55 else "black")
                ax.text(c + 0.5, r + 0.5, w[:9], ha="center", va="center",
                        fontsize=4.5, color=txt, zorder=4,
                        fontweight="bold" if hit else "normal")
        ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.axis("off")
        mark = "✓" if right else "✗"
        col = "#1b7a35" if right else "#c62828"
        ax.set_title(f"{TITLE[o]}\n→ “{A[o]}” {mark}", fontsize=8.4, pad=4,
                     color=col, fontweight="bold", linespacing=1.35)

    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(sm, cax=cax)
    cax.tick_params(labelsize=5.5)
    cax.set_ylabel("token probability", fontsize=6.2)

    fig.text(0.5, 0.985,
             f"Q: “{rec['question']}”   (ground truth: {rec['gt']})",
             ha="center", va="top", fontsize=9.4, fontweight="bold")
    fig.text(0.5, 0.925,
             f"ringed cells decode to a word from the question "
             f"({', '.join(targets)});  Qwen3-VL-8B, layer {layer}",
             ha="center", va="top", fontsize=6.6, color="#444444")

    out = args.out if not args.groups else f"{args.out}_g{group}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".png/.pdf")


if __name__ == "__main__":
    main()
