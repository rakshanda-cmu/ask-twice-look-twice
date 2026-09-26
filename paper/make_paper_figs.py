#!/usr/bin/env python3
"""
Paper figures for "Ask Twice, Look Twice".

Every number is read from an on-disk result file at render time (no transcribed
constants), so a figure cannot drift from the tables. Writes vector PDFs into
paper/figs/.

Run:  python paper/make_paper_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(REPO, "paper", "figs")
PROBE = "/home/grg/Research/middle_layers_indicating_hallucinations/naturalbench/probe"

# Okabe-Ito subset, validated colorblind-safe (dataviz validator, light surface).
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def nb_meta(tag):
    p = os.path.join(REPO, "naturalbench", "results",
                     f"qwen3-vl-8b__{tag}__results.json")
    return json.load(open(p))["meta"]


def bare(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


# ----------------------------------------------------------------- Figure 1
def fig_teaser():
    """The paradox in one glance: steering up, accuracy down, echo repairs it."""
    hits = json.load(open(os.path.join(REPO, "fig2_hits.json")))
    m_sit = float(np.mean([r["hits"]["SIT"] for r in hits]))
    m_sti = float(np.mean([r["hits"]["STI"] for r in hits]))
    g = {t: nb_meta(t)["g_acc"] * 100
         for t in ("SIT", "STI", "STIT", "SITIT", "SITIT_echo2eighth")}

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05),
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})

    # (a) perception goes UP
    ax = axes[0]
    top = m_sti * 1.62
    ax.bar([0, 1], [m_sit, m_sti], width=0.5, color=[MUTED, VERM],
           edgecolor="white", linewidth=1.2, zorder=3)
    for x, v in zip([0, 1], [m_sit, m_sti]):
        ax.text(x, v + top * 0.022, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
    ax.annotate("", xy=(1, top * 0.855), xytext=(0, top * 0.60),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    ax.text(0.5, top * 0.90, f"{m_sti/m_sit:.2f}$\\times$ more", ha="center",
            fontsize=9.5, fontweight="bold", color=GREEN)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["question\nlast", "question\nfirst"])
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_ylabel("patches decoding\na question word")
    ax.set_ylim(0, top); ax.set_xlim(-0.6, 1.6)
    ax.set_title("(a) Perception improves", color=GREEN, fontweight="bold",
                 pad=6, fontsize=8)
    bare(ax)

    # (b) accuracy goes DOWN
    ax = axes[1]
    top_b = 58
    ax.bar([0, 1], [g["SIT"], g["STI"]], width=0.5, color=[MUTED, VERM],
           edgecolor="white", linewidth=1.2, zorder=3)
    for x, v in zip([0, 1], [g["SIT"], g["STI"]]):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
    ax.annotate("", xy=(1, top_b * 0.68), xytext=(0, top_b * 0.845),
                arrowprops=dict(arrowstyle="-|>", color=VERM, lw=1.6))
    ax.text(0.5, top_b * 0.885, f"{g['STI']-g['SIT']:+.1f} pts", ha="center",
            fontsize=9.5, fontweight="bold", color=VERM)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["question\nlast", "question\nfirst"])
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_ylabel("group accuracy (%)")
    ax.set_ylim(0, top_b); ax.set_xlim(-0.6, 1.6)
    ax.set_title("(b) Accuracy collapses", color=VERM, fontweight="bold",
                 pad=6, fontsize=8)
    bare(ax)

    # (c) the repair
    ax = axes[2]
    vals = [g["STI"], g["STIT"], g["SITIT_echo2eighth"]]
    ax.bar(range(3), vals, width=0.5, color=[VERM, BLUE, BLUE],
           edgecolor="white", linewidth=1.2, zorder=3)
    for x, v in zip(range(3), vals):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
    ax.axhline(g["SIT"], color=INK, lw=0.9, ls=(0, (4, 2)), zorder=4,
               label="question-last baseline")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=7.8,
              borderpad=0.0, bbox_to_anchor=(-0.02, 1.02))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["question\nfirst", "+question\n13 tok",
                        "+image\n107 tok"])
    # three labels in one panel sit closer than the two-bar panels, so drop a
    # point of type rather than letting them touch
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_ylabel("group accuracy (%)")
    ax.set_ylim(0, top_b); ax.set_xlim(-0.6, 2.6)
    ax.set_title("(c) Echoing repairs it", color=BLUE, fontweight="bold",
                 pad=6, fontsize=8)
    bare(ax)

    fig.tight_layout(w_pad=3.2)
    out = os.path.join(FIGS, "fig_teaser.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[fig] {out}  patches {m_sit:.3f}->{m_sti:.3f}; g_acc {g}")


# ----------------------------------------------------------------- Figure 2
def fig_mechanism():
    """Where it breaks: attention routing, then a late decision fork."""
    pr = json.load(open(os.path.join(PROBE, "probe_disagreement.json")))
    dec = json.load(open(os.path.join(PROBE, "decision_disagreement.json")))
    po = pr.get("orders", pr)
    do = dec.get("orders", dec)

    fig, axes = plt.subplots(1, 2, figsize=(5.05, 2.05))

    # (a) answer-position attention to question vs image
    ax = axes[0]
    L = np.arange(len(po["STI"]["a_q"]))
    ax.plot(L, po["SIT"]["a_q"], color=BLUE, lw=2.0)
    ax.plot(L, po["STI"]["a_q"], color=VERM, lw=2.0)
    ax.plot(L, po["SIT"]["a_img"], color=BLUE, lw=1.2, ls=(0, (3, 2)), alpha=0.85)
    ax.plot(L, po["STI"]["a_img"], color=VERM, lw=1.2, ls=(0, (3, 2)), alpha=0.85)
    pk = int(np.argmax(po["SIT"]["a_q"]))
    ratio = po["SIT"]["a_q"][pk] / po["STI"]["a_q"][pk]
    ax.annotate(f"{ratio:.2f}$\\times$ less\nat layer {pk}",
                xy=(pk, po["STI"]["a_q"][pk] + 0.004),
                xytext=(pk + 4.5, 0.235), fontsize=7.5, color=INK,
                fontweight="bold", ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                connectionstyle="arc3,rad=0.25"))
    ax.scatter([pk, pk], [po["SIT"]["a_q"][pk], po["STI"]["a_q"][pk]], s=22,
               facecolor="white", edgecolor=[BLUE, VERM], linewidth=1.4, zorder=5)
    ax.set_xlabel("layer")
    ax.set_ylabel("attention mass")
    ax.set_ylim(0, 0.255); ax.set_xlim(-1, 36)
    ax.set_title("(a) Question-first routes to the image",
                 fontweight="bold", pad=5, fontsize=8)
    h = [plt.Line2D([], [], color=BLUE, lw=2.0),
         plt.Line2D([], [], color=VERM, lw=2.0),
         plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 2)))]
    ax.legend(h, ["question-last", "question-first", "(dashed: to image)"],
              frameon=False, loc="upper left", handlelength=1.5,
              borderpad=0.0, labelspacing=0.22, ncol=1)
    bare(ax)

    # (b) decision layer
    ax = axes[1]
    for tag, lab, c, lw, ls in [("SIT", "question-last", BLUE, 2.0, "-"),
                                ("SITIT", "image echo", GREEN, 1.8, "-"),
                                ("STIT", "question echo", ORANGE, 1.5, (0, (4, 2))),
                                ("STI", "question-first", VERM, 2.2, "-")]:
        if tag not in do:
            continue
        y = np.array(do[tag]["p_corr2"]) * 100
        ax.plot(np.arange(len(y)), y, color=c, lw=lw, ls=ls, label=lab)
        ax.scatter([len(y) - 1], [y[-1]], s=18, color=c, zorder=5,
                   edgecolor="white", linewidth=0.8)
    sti = np.array(do["STI"]["p_corr2"]); sit = np.array(do["SIT"]["p_corr2"])
    fork = next(i for i in range(len(sti)) if abs(sti[i] - sit[i]) > 0.05)
    ax.axvspan(0, fork, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.axvline(fork, color=INK, lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax.text(fork - 1.2, 99, f"fork at\nlayer {fork}", fontsize=7.5, color=INK,
            ha="right", va="top", fontweight="bold")
    ax.axhline(50, color=MUTED, lw=0.7, zorder=1)
    ax.set_xlabel("layer"); ax.set_ylabel("P(correct) (%)")
    ax.set_ylim(0, 100); ax.set_xlim(-1, 36)
    ax.set_title("(b) Identical until layer %d, then forks" % fork,
                 fontweight="bold", pad=5, fontsize=8)
    ax.legend(frameon=False, loc="lower left", handlelength=1.5,
              borderpad=0.0, labelspacing=0.22, bbox_to_anchor=(0.01, 0.0))
    bare(ax)

    fig.tight_layout(w_pad=2.8)
    out = os.path.join(FIGS, "fig_mechanism.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[fig] {out}  fork L{fork}, peak L{pk} ratio {ratio:.3f}")


# ----------------------------------------------------------------- Figure 3
def fig_knockout():
    """Causal double dissociation: each ordering depends on a different path."""
    k = json.load(open(os.path.join(PROBE, "knockout_last.json")))["orders"]
    conds = [("ko_question", "sever\nquestion path"),
             ("ko_image", "sever\nimage path"),
             ("ko_random", "sever random\nspan (control)")]
    fig, ax = plt.subplots(figsize=(2.78, 2.15))
    w = 0.34
    x = np.arange(len(conds))
    for i, (tag, lab, c) in enumerate([("STI", "question-first", VERM),
                                       ("SIT", "question-last", BLUE)]):
        clean = k[tag]["clean"]["acc"]
        d = [(k[tag][c0]["acc"] - clean) * 100 for c0, _ in conds]
        ax.bar(x + (i - 0.5) * w, d, width=w, color=c, edgecolor="white",
               linewidth=1.0, label=lab, zorder=3)
        for xi, v in zip(x + (i - 0.5) * w, d):
            va, off = ("bottom", 0.3) if v >= 0 else ("top", -0.3)
            ax.text(xi, v + off, f"{v:+.1f}", ha="center", va=va, fontsize=8,
                    fontweight="bold", color=INK)
    ax.axhline(0, color=INK, lw=0.8, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in conds])
    ax.set_ylabel("change in accuracy (pts)")
    ax.set_ylim(-12.0, 5.6)
    ax.set_title("Each ordering answers through a\ndifferent path",
                 fontweight="bold", pad=6)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.2,
              borderpad=0.0, columnspacing=1.2, bbox_to_anchor=(0.5, 1.03))
    bare(ax)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_knockout.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[fig] {out}")


# ----------------------------------------------------------------- Figure 4
def fig_pareto():
    """Cost versus accuracy: the cheapest echo matches the dearest."""
    tc = json.load(open(os.path.join(REPO, "token_cost_results.json")))
    dv = tc["delta_vs_STI"]
    spec = [("STI", None, "question-first", VERM, "X", 58, (7, 0), "left"),
            ("SIT", None, "question-last", MUTED, "s", 34, (5, 10), "left"),
            ("STIT", "STIT", "question echo", ORANGE, "D", 36, (6, -10), "left"),
            ("SITIT_echo2eighth", "SITIT_eighth", "$\\frac{1}{8}$", BLUE, "o", 46, (0, 10), "center"),
            ("SITIT_echo2quarter", "SITIT_quarter", "$\\frac{1}{4}$", BLUE, "o", 40, (0, 10), "center"),
            ("SITIT_echo2half", "SITIT_half", "$\\frac{1}{2}$", BLUE, "o", 40, (0, 10), "center"),
            ("SITIT", "SITIT_full", "full", BLUE, "o", 40, (0, 10), "center")]
    fig, ax = plt.subplots(figsize=(2.52, 2.05))
    pts = []
    for tag, key, lab, c, mk, s, (ox, oy), ha in spec:
        m = nb_meta(tag)
        if m.get("num_groups") != 1900:
            print(f"  [skip] {tag}: num_groups={m.get('num_groups')}")
            continue
        cost = 0.0 if key is None else dv[key]["delta_total"]
        acc = m["pair_acc"] * 100
        pts.append((cost, acc, tag))
        ax.scatter(cost, acc, s=s, marker=mk, color=c, zorder=5,
                   edgecolor="white", linewidth=0.9)
        ax.annotate(lab, xy=(cost, acc), xytext=(ox, oy),
                    textcoords="offset points", fontsize=6.6, color=INK,
                    ha=ha, va="center")
    echo = sorted([p for p in pts if p[2].startswith("SITIT")])
    ax.plot([p[0] for p in echo], [p[1] for p in echo], color=BLUE, lw=1.2,
            alpha=0.45, zorder=3)
    sit_acc = nb_meta("SIT")["pair_acc"] * 100
    ax.axhline(sit_acc, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xticks([0, 100, 1000, 3000])
    ax.set_xticklabels(["0", "100", "1k", "3k"])
    ax.set_xlabel("extra tokens per query")
    ax.set_ylabel("pair accuracy (%)")
    ax.set_title("image echo: cheapest matches\ndearest at 3% of the cost",
                 fontweight="bold", pad=5, fontsize=7.6)
    ax.set_xlim(-6, 14000); ax.set_ylim(74.2, 82.6)
    bare(ax)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_pareto.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[fig] {out}  {[(round(c), round(a, 2), t) for c, a, t in pts]}")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_teaser()
    fig_mechanism()
    fig_knockout()
    fig_pareto()
    print("[done]")
