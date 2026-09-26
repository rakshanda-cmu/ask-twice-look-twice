"""
Token Change Explorer: which patch tokens the question rewrites, animated.

A separate page (per the "new results go in a new tab" rule), built on the
per-patch logit-lens decodes already stored in website/assets/manifest.json:
each example holds the top-1 vocabulary token for every image patch under
question-last (SIT) and question-first (STI) at that example's peak layer.

The static figure in the paper can only call out a handful of patches. Here
every changed patch is listed, and the animation steps through them one at a
time: the cell is boxed amber showing the token it decoded to with the question
last, then flips to red showing what it became with the question first, with an
arrow drawn between the two panels. That is the "old token -> new token" motion
the paper figure freezes into a single frame.

No GPU: the decodes are precomputed. Reading is from manifest.json only, so
nothing here can disagree with the rendered website assets.
"""
import base64
import io
import json
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle, ConnectionPatch
from PIL import Image

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "website", "assets")
MANIFEST = os.path.join(ASSETS, "manifest.json")

AMBER, RED, INK = "#f5a800", "#d81b26", "#1a1a1a"
# many decoded tokens are CJK; without a fallback family they render as boxes
FONTS = ["DejaVu Sans", "Noto Sans CJK JP", "Droid Sans Fallback"]


@st.cache_data(show_spinner=False)
def _load_examples():
    if not os.path.exists(MANIFEST):
        return []
    man = json.load(open(MANIFEST))
    out = []
    for e in man.get("examples", []):
        if e.get("sti_words") and e.get("sit_words") and e.get("image"):
            out.append(e)
    return out


def _norm(w):
    return (w or "").strip()


def _changes(ex):
    """Every patch whose decoded token changed, as (flat_index, old, new)."""
    gw = ex["grid_w"]
    out = []
    for i, (a, b) in enumerate(zip(ex["sit_words"], ex["sti_words"])):
        a, b = _norm(a), _norm(b)
        if a and b and a != b:
            out.append((i, a, b, divmod(i, gw)))
    return out


def _question_words(ex):
    """Words the question is about, used to rank the interesting changes first."""
    ws = {str(ex.get("key", "")).lower()}
    ws |= {str(w).lower() for w in (ex.get("evidence") or [])}
    ws |= {w.strip(".,?").lower() for w in ex["question"].split() if len(w) > 3}
    return {w for w in ws if w}


def _rank(ex, changes):
    """Changes that land on the question's own vocabulary come first."""
    qw = _question_words(ex)
    def key(c):
        new = c[2].lower()
        hit = any(new.startswith(q[:5]) or q.startswith(new[:5]) for q in qw if len(q) > 3)
        return (0 if hit else 1, c[0])
    return sorted(changes, key=key)


def _grid(ax, img, ex, changed, focus, reveal):
    """One combined grid. A cell whose token changed carries both words, the
    question-last one above and the question-first one below; everything else
    carries its single unchanged word. `reveal` gates whether the new words are
    shown yet, which is what the animation steps through."""
    gh, gw = ex["grid_h"], ex["grid_w"]
    ax.imshow(img, extent=[0, gw, gh, 0])
    for r in range(1, gh):
        ax.axhline(r, color="white", lw=0.4, alpha=0.35)
    for c in range(1, gw):
        ax.axvline(c, color="white", lw=0.4, alpha=0.35)

    chg = {i: (old, new) for i, old, new, _ in changed}
    for i, (a, b) in enumerate(zip(ex["sit_words"], ex["sti_words"])):
        r, c = divmod(i, gw)
        if i in chg:
            old, new = chg[i]
            shown = reveal is True or (focus is not None and i == focus[0])
            # old token sits above, dimmed once its replacement is visible
            t0 = ax.text(c + 0.5, r + 0.32, old, ha="center", va="center",
                         fontsize=5.2, color="#f2f2f2", fontfamily=FONTS,
                         zorder=4, alpha=0.8 if shown else 1.0)
            t0.set_path_effects([pe.withStroke(linewidth=1.5, foreground="#00000090")])
            if shown:
                t1 = ax.text(c + 0.5, r + 0.70, new, ha="center", va="center",
                             fontsize=5.8, color="#ff5a5a", fontweight="bold",
                             fontfamily=FONTS, zorder=5)
                t1.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#3a0000")])
            ec = AMBER if (focus is not None and i == focus[0]) else RED
            lw = 2.8 if (focus is not None and i == focus[0]) else 1.4
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False, ec=ec, lw=lw,
                                   zorder=6))
        else:
            tu = ax.text(c + 0.5, r + 0.5, _norm(a), ha="center", va="center",
                         fontsize=5.2, color="white", fontfamily=FONTS,
                         zorder=4, alpha=0.7)
            tu.set_path_effects([pe.withStroke(linewidth=1.2, foreground="#00000070")])
    ax.set_xlim(0, gw); ax.set_ylim(gh, 0); ax.axis("off")


def _frame(ex, img, changed, focus, reveal):
    """One animation frame over the combined grid."""
    gh, gw = ex["grid_h"], ex["grid_w"]
    fig, ax = plt.subplots(figsize=(6.6, 6.6 * gh / gw * 1.02 + 0.9), dpi=110)
    _grid(ax, img, ex, changed, focus, reveal)
    fig.suptitle(f"\u201c{ex['question']}\u201d   (truth: {ex['gt']})",
                 fontsize=10.5, fontweight="bold", y=0.975)
    sub = (f"question-last \u2192 \u201c{ex['SIT']['answer']}\u201d      "
           f"question-first \u2192 \u201c{ex['STI']['answer']}\u201d")
    fig.text(0.5, 0.935, sub, ha="center", fontsize=9, color=INK)
    if focus is not None:
        fig.text(0.5, 0.028, f"{focus[1]}  \u2192  {focus[2]}", ha="center",
                 fontsize=14, fontweight="bold", color=RED, fontfamily=FONTS)
    else:
        fig.text(0.5, 0.022, "white above: question-last     red below: question-first",
                 ha="center", fontsize=9, color=INK)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


@st.cache_data(show_spinner=False)
def _build_gif(idx, picked, hold_ms):
    ex = next(e for e in _load_examples() if e["idx"] == idx)
    img = Image.open(os.path.join(ASSETS, ex["image"])).convert("RGB")
    changed = _changes(ex)
    chosen = [c for c in changed if c[0] in set(picked)]
    frames = [_frame(ex, img, changed, None, False)]
    for c in chosen:
        frames.append(_frame(ex, img, changed, c, False))
        frames.append(_frame(ex, img, changed, c, False))
    frames.append(_frame(ex, img, changed, None, True))
    frames.append(_frame(ex, img, changed, None, True))
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=hold_ms / 1000.0, loop=0)
    return buf.getvalue()


def render_token_change_page():
    st.title("🔤 Token Change Explorer")
    st.caption(
        "Which image patches change what they decode to when the question moves "
        "in front of the image, shown one at a time. Amber is the token under "
        "question-last, red is what the same patch became under question-first. "
        "Per-patch decodes are the precomputed logit-lens grids behind the "
        "website examples, so no GPU is used here."
    )

    examples = _load_examples()
    if not examples:
        st.info("No manifest with per-patch word grids found at website/assets/manifest.json.")
        return

    def _label(e):
        flips = "paradox" if e.get("demonstrates") else "no flip"
        return f"[{e['idx']}] {e['question'][:58]}  ({flips})"

    ex = st.selectbox("Example", examples, format_func=_label, key="tc_ex")
    img = Image.open(os.path.join(ASSETS, ex["image"])).convert("RGB")
    changes = _rank(ex, _changes(ex))
    gh, gw = ex["grid_h"], ex["grid_w"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("question-last", ex["SIT"]["answer"], "correct" if ex["SIT"]["correct"] else "wrong")
    c2.metric("question-first", ex["STI"]["answer"], "correct" if ex["STI"]["correct"] else "wrong")
    c3.metric("image echo", ex["SITIT"]["answer"], "correct" if ex["SITIT"]["correct"] else "wrong")
    c4.metric("patches changed", f"{len(changes)} / {gh * gw}", f"layer {ex['peak_layer']}")

    st.markdown("#### Changed patches")
    st.caption(
        "Ordered so the changes that land on the question's own vocabulary come "
        "first. Tick the ones to animate."
    )
    default = [c[0] for c in changes[:3]]
    picked = []
    for c in changes[:24]:
        i, old, new, (r, cc) = c
        on = st.checkbox(f"`({r},{cc})`  **{old}**  →  **{new}**",
                         value=i in default, key=f"tc_{ex['idx']}_{i}")
        if on:
            picked.append(i)

    st.markdown("#### Animation")
    hold = st.slider("frame hold (ms)", 250, 1500, 650, 50, key="tc_hold")
    if not picked:
        st.info("Tick at least one changed patch above.")
        return
    gif = _build_gif(ex["idx"], tuple(picked), hold)
    if gif:
        st.image(gif, caption=f"{len(picked)} patch change(s), example {ex['idx']}")
        st.download_button("Download GIF", gif,
                           file_name=f"token_change_ex{ex['idx']}.gif", mime="image/gif")
