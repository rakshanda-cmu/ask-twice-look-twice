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
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib import cm, colors as mcolors
from PIL import Image

from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from naturalbench_eval import answer_suffix
from fig2_search import run, matches, ok, VAGUE, STOP, NB_ROOT, LOW_MAX, ORDERS

CRIMSON = "#dc143c"
LINKC = "#f5a800"   # amber: the individually traced cells
INKC = "#1a1a1a"
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
    ap.add_argument("--extra", default="",
                    help="extra words to ring alongside the question's own words")
    ap.add_argument("--cell-fontsize", type=float, default=4.0, dest="cell_fontsize",
                    help="font size inside each patch cell")
    ap.add_argument("--cell-chars", type=int, default=8, dest="cell_chars",
                    help="max characters shown per cell before truncation")
    ap.add_argument("--link", default="",
                    help='cells to trace from SIT to STI, e.g. "3,1;2,3;1,5"; each is '
                         'boxed in the SIT panel with a dotted arrow to the SAME patch '
                         'in the STI panel, showing what that one patch decoded to before '
                         'and after the question moved in front')
    ap.add_argument("--figheight", type=float, default=None, dest="figheight",
                    help="figure height in inches; default keeps the 11.2x3.35 aspect")
    ap.add_argument("--figwidth", type=float, default=11.2, dest="figwidth",
                    help="figure width in inches; set to the target \\textwidth "
                         "so cell text renders at its true size instead of being "
                         "scaled down by includegraphics")
    ap.add_argument("--orders", default="",
                    help="comma list of panels to draw, e.g. 'SIT,STI'; default all")
    ap.add_argument("--fit-text", action="store_true", dest="fit_text",
                    help="shrink a cell's font until the whole token fits instead "
                         "of truncating it at --cell-chars")
    ap.add_argument("--diff", action="store_true",
                    help="one combined grid instead of one panel per ordering: "
                         "a changed cell carries the question-last token above "
                         "and the question-first token below")
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
    # words worth ringing that the question does not contain (e.g. "ahead", which
    # reports the spatial relation the question is about without naming it)
    extra = [w.strip() for w in args.extra.split(",") if w.strip()]
    ring_on = targets + extra

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

    if args.diff:
        import matplotlib.patheffects as pe
        sit_w, sti_w = W["SIT"], W["STI"]
        changed = [i for i in range(gh * gw)
                   if sit_w[i].strip() and sti_w[i].strip()
                   and sit_w[i].strip() != sti_w[i].strip()]
        linked = set()
        for spec in (args.link.split(";") if args.link else []):
            if spec.strip():
                rr, cc = [int(v) for v in spec.split(",")]
                linked.add(rr * gw + cc)

        fh = args.figheight if args.figheight else args.figwidth * 0.58
        fig = plt.figure(figsize=(args.figwidth, fh), dpi=400)
        # the grid carries the content, so the input thumbnail stays small
        gs = fig.add_gridspec(1, 2, width_ratios=[0.60, 3.05],
                              left=0.004, right=0.996, top=0.845, bottom=0.01,
                              wspace=0.02)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(small); ax0.axis("off")
        ax0.set_title("Input", fontsize=7.6, fontweight="bold", pad=3)

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(disp, extent=[0, gw, gh, 0])
        rend = fig.canvas.get_renderer()
        cell_px = ax.get_window_extent(rend).width / max(1, gw)
        for r_ in range(1, gh):
            ax.axhline(r_, color="white", lw=0.35, alpha=0.35)
        for c_ in range(1, gw):
            ax.axvline(c_, color="white", lw=0.35, alpha=0.35)

        def _fit(t, frac):
            if not args.fit_text:
                return
            lim = cell_px * frac
            f_ = t.get_fontsize()
            for _ in range(16):
                if t.get_window_extent(rend).width <= lim:
                    return
                f_ *= 0.88
                t.set_fontsize(f_)

        for i in range(gh * gw):
            r_, c_ = divmod(i, gw)
            if i in changed:
                t0 = ax.text(c_ + 0.5, r_ + 0.30, sit_w[i].strip(),
                             ha="center", va="center", fontsize=args.cell_fontsize,
                             color="#f4f4f4", zorder=4)
                t1 = ax.text(c_ + 0.5, r_ + 0.72, sti_w[i].strip(),
                             ha="center", va="center",
                             fontsize=args.cell_fontsize * 1.05,
                             color="#ff5252", fontweight="bold", zorder=5)
                for t_, lw_ in ((t0, 1.0), (t1, 1.2)):
                    t_.set_path_effects([pe.withStroke(linewidth=lw_,
                                                       foreground="#000000a0")])
                    _fit(t_, 0.82)
                ec = LINKC if i in linked else HITC
                ax.add_patch(Rectangle((c_, r_), 1, 1, fill=False, ec=ec,
                                       lw=2.2 if i in linked else 1.0, zorder=6))
            else:
                t_ = ax.text(c_ + 0.5, r_ + 0.5, sit_w[i].strip(),
                             ha="center", va="center", fontsize=args.cell_fontsize,
                             color="white", alpha=0.6, zorder=4)
                t_.set_path_effects([pe.withStroke(linewidth=0.9,
                                                   foreground="#00000080")])
                _fit(t_, 0.88)
        ax.set_xlim(0, gw); ax.set_ylim(gh, 0); ax.axis("off")

        fig.text(0.5, 0.985, f"Q: \u201c{rec['question']}\u201d   "
                 f"(ground truth: {rec['gt']})", ha="center", va="top",
                 fontsize=8.8, fontweight="bold")
        fig.text(0.5, 0.905,
                 f"{len(changed)} of {gh * gw} patches change what they decode to",
                 ha="center", va="top", fontsize=7.4, color=INKC)
        fig.text(0.5, 0.863,
                 f"white above: question-last \u2192 \u201c{A['SIT']}\u201d      "
                 f"red below: question-first \u2192 \u201c{A['STI']}\u201d",
                 ha="center", va="top", fontsize=7.0, color=INKC)
        fig.savefig(args.out + ".pdf"); fig.savefig(args.out + ".png", dpi=300)
        plt.close(fig)
        print(f"wrote {args.out}.png/.pdf  ({len(changed)} changed cells)")
        return


    panels = [o for o in (args.orders.split(",") if args.orders else ORDERS)
              if o in ORDERS]
    ratios = [1.06] + [1.0] * len(panels) + [0.045]
    fh = args.figheight if args.figheight else args.figwidth * 0.299
    fig = plt.figure(figsize=(args.figwidth, fh), dpi=400)
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios,
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

    panel_ax = {}
    for k, o in enumerate(panels):
        ax = fig.add_subplot(gs[0, k + 1])
        panel_ax[o] = ax
        crop = disp.crop((c0 * cell, r0 * cell, c1 * cell, r1 * cell))
        if dark_at_high and not args.wash:
            # a Blues field over a colour photo reads teal where the photo is green;
            # desaturating underneath keeps the shading true blue (as the website does)
            crop = crop.convert("L").convert("RGB")
        ax.imshow(crop, extent=[c0, c1, r1, r0])
        right = ok(A[o], rec["gt"])
        # width of one cell in points, so a token can be sized to actually fit
        # inside its box rather than spilling over the neighbouring cells
        rend = fig.canvas.get_renderer()
        _bb = ax.get_window_extent(rend)
        cell_px = _bb.width / max(1, (c1 - c0))
        for r in range(r0, r1):
            for c in range(c0, c1):
                i = r * gw + c
                w, p = W[o][i], float(P[o][i])
                hit = any(matches(w, t) for t in ring_on)
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
                label = w.strip() if args.fit_text else w[:args.cell_chars]
                fs = args.cell_fontsize
                t = ax.text(c + 0.5, r + 0.5, label,
                            ha="center", va="center",
                            fontsize=fs, color=txt, zorder=4,
                            fontweight="bold" if hit else "normal")
                if args.fit_text:
                    # measure what was actually drawn and shrink until it fits
                    # inside its own cell; estimating glyph widths left tokens
                    # clipped at the box edge
                    limit = cell_px * (0.80 if hit else 0.90)
                    for _ in range(14):
                        if t.get_window_extent(rend).width <= limit:
                            break
                        fs *= 0.88
                        if fs < args.cell_fontsize * 0.30:
                            break
                        t.set_fontsize(fs)
        ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.axis("off")
        mark = "✓" if right else "✗"
        col = "#1b7a35" if right else "#c62828"
        ax.set_title(f"{TITLE[o]}\n→ “{A[o]}” {mark}", fontsize=8.4, pad=4,
                     color=col, fontweight="bold", linespacing=1.35)

    # ── trace individual patches from question-last to question-first
    if args.link:
        LINK = "#f5a800"      # amber: source box and arrow
        cells_ = []
        for spec in args.link.split(";"):
            spec = spec.strip()
            if spec:
                r, c = [int(v) for v in spec.split(",")]
                cells_.append((r, c))
        # deeper rows bow further, so the arcs nest instead of crossing each other
        cells_.sort(key=lambda rc: rc[0])
        if "SIT" in panel_ax and "STI" in panel_ax:
            a, b = panel_ax["SIT"], panel_ax["STI"]
        else:
            cells_ = []
        for k_, (r, c) in enumerate(cells_):
            # source is amber, destination red, so the eye reads "this cell
            # became that cell" rather than seeing two identical marks
            a.add_patch(Rectangle((c, r), 1, 1, fill=False, ec=LINK,
                                  lw=2.6, zorder=6))
            b.add_patch(Rectangle((c, r), 1, 1, fill=False, ec=HITC,
                                  lw=2.6, zorder=6))
            fig.add_artist(ConnectionPatch(
                # ride just under the top edge of the row: at the cell centre
                # the dashes struck through the tokens they passed over
                xyA=(c + 1.0, r + 0.16), coordsA=a.transData,
                xyB=(c + 0.0, r + 0.16), coordsB=b.transData,
                arrowstyle="-|>", mutation_scale=7, linewidth=1.0,
                linestyle=(0, (3.0, 2.0)), color=LINK, alpha=0.95, zorder=11,
                connectionstyle="arc3,rad=0"))

    cax = fig.add_subplot(gs[0, -1])
    fig.colorbar(sm, cax=cax)
    cax.tick_params(labelsize=5.5)
    cax.set_ylabel("token probability", fontsize=6.2)

    fig.text(0.5, 0.985,
             f"Q: “{rec['question']}”   (ground truth: {rec['gt']})",
             ha="center", va="top", fontsize=9.4, fontweight="bold")
    fig.text(0.5, 0.925,
             (f"ringed cells decode to a word from the question "
              f"({', '.join(targets)})"
              + (f" or to “{', '.join(extra)}”" if extra else "")
              + f";  Qwen3-VL-8B, layer {layer}"),
             ha="center", va="top", fontsize=6.6, color="#444444")

    out = args.out if not args.groups else f"{args.out}_g{group}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".png/.pdf")


if __name__ == "__main__":
    main()
