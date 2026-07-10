"""
Streamlit page: Mechanism Probe — attention + answer-emergence by layer for
STI / STIT / IST, produced by mechanism_probe.py + make_probe_figs.py.
"""

import json
import os

import numpy as np
import streamlit as st

PROBE_DIR = "./naturalbench/probe"
ORDERS = ("STI", "STIT", "IST")


def _load(pset):
    p = os.path.join(PROBE_DIR, f"probe_{pset}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _summary_table(data):
    rows = {}
    for o in ORDERS:
        d = data["orders"][o]
        aq, ai, pg = np.array(d["a_q"]), np.array(d["a_img"]), np.array(d["p_gt"])
        mid = slice(16, 26)  # middle-layer band
        rows[o] = {
            "acc": f"{d['acc']*100:.0f}%",
            "ans→Q attn (mid-layer peak)": f"{aq[mid].max():.3f}",
            "ans→Q attn (mean)": f"{aq.mean():.3f}",
            "ans→image attn (mid-layer peak)": f"{ai[mid].max():.3f}",
            "final P(correct)": f"{pg[-1]:.3f}",
        }
    return rows


def render_probe_page():
    st.subheader("🧠 Mechanism Probe — why STI fails and STIT fixes it")
    st.caption(
        "For each (image, question) pair under STI / STIT / IST we run one eager "
        "forward pass and measure, **per layer**, how much the answer position "
        "attends to the QUESTION vs the IMAGE, plus logit-lens answer-emergence. "
        "Pairs the behavioral result with the internal mechanism."
    )

    with st.expander("ℹ️ What this page measures — method & how to read it",
                     expanded=False):
        st.markdown(
            "**Goal.** Go *inside* the model to explain why the **STI** ordering "
            "(System·Task·Image) loses accuracy, and why repeating the question "
            "after the image (**STIT**) — or putting the image first (**IST**) — "
            "recovers it. Behavioral accuracy tells us *that* STI is worse; this page "
            "asks *why*.\n\n"
            "**Method.** For every (image, question) pair we run **one eager forward "
            "pass** of Qwen3-VL-8B under each ordering and record, **at every layer**, "
            "three things about the **answer position** (the token that is about to "
            "emit *Yes/No*):\n"
            "- **ans→Q attention** — how much that position attends to the **question** "
            "tokens (averaged over heads). *Does the answer actually 'look at' the "
            "question?*\n"
            "- **ans→image attention** — how much it attends to the **image** tokens "
            "instead.\n"
            "- **answer emergence** — a **logit-lens** read of **P(correct token)** from "
            "the hidden state at each layer: *when*, in the depth of the network, the "
            "right answer becomes decodable.\n\n"
            "**Two pair sets.**\n"
            "- **Neutral** — a random, outcome-independent sample. The honest baseline, "
            "so the effect isn't cherry-picked.\n"
            "- **Diagnostic (disagreement)** — only the pairs where **STI is wrong but "
            "IST is right**, to view the mechanism on the exact failure cases.\n\n"
            "**How to read it.** Compare the three orderings in the **middle layers "
            "(~16–26)** — that band is where the answer is decided. The pattern: under "
            "**STI** the answer barely attends to the now-distant question, leans on the "
            "image, and the correct token emerges **late and weak**. Under **STIT/IST** a "
            "question representation sits right next to the answer, **~doubling** ans→Q "
            "attention and making the answer emerge **earlier and stronger** — which "
            "tracks the accuracy recovery.\n\n"
            "**Why it matters (the control).** Adding or removing image tokens did **not** "
            "change accuracy, so this is **not** memory decay/recency over the image. The "
            "real cause is that STI leaves the answer position with **no adjacent question "
            "representation to attend to**. STIT supplies one → **question-conditioned, "
            "top-down reprocessing** is what fixes it."
        )

    neu = _load("neutral")
    dis = _load("disagreement")
    if not neu and not dis:
        st.info("No probe results yet. Run "
                "`CUDA_VISIBLE_DEVICES=0 python mechanism_probe.py --set neutral "
                "--num-pairs 250` then `python make_probe_figs.py`.")
        return

    if neu:
        st.markdown(f"#### Neutral sample (n={neu['n_pairs']}, outcome-independent)")
        st.table(_summary_table(neu))
        st.caption(
            "**Key:** in the **middle layers**, STI's answer attends ~half as much "
            "to the question and far more to the image than IST/STIT. STIT restores "
            "answer→question attention to IST levels — the internal signature of the "
            "behavioral recovery."
        )
        fig = os.path.join(PROBE_DIR, "probe_neutral.png")
        if os.path.exists(fig):
            st.image(fig, width="stretch")

    if dis:
        with st.expander(f"Diagnostic set — STI-wrong / IST-right "
                         f"(n={dis['n_pairs']})"):
            st.caption(
                "On the pairs where STI fails but IST succeeds: STIT recovers "
                f"{dis['orders']['STIT']['acc']*100:.0f}% of them, and its "
                "answer→question attention matches IST while STI's stays low."
            )
            st.table(_summary_table(dis))
            fig = os.path.join(PROBE_DIR, "probe_disagreement.png")
            if os.path.exists(fig):
                st.image(fig, width="stretch")

    st.markdown("---")
    st.markdown(
        "**Interpretation.** The deficit is **not** memory decay over the image "
        "tokens (adding/removing image tokens left accuracy unchanged). It is that "
        "in STI the answer position has **no adjacent question representation to "
        "attend to** — its attention goes to the image instead. STIT (and IST) "
        "place a question representation next to the answer, roughly **doubling** "
        "answer→question attention in the middle layers, which tracks the accuracy "
        "recovery. → question-conditioned (top-down) reprocessing, not recency."
    )

    # ── Winoground probe (whole dataset) ───────────────────────────────────────
    _wp = os.path.join(PROBE_DIR, "winoground_stit_probe.json")
    wino = json.load(open(_wp)) if os.path.exists(_wp) else None
    st.markdown("---")
    st.markdown("### 🟪 Winoground — whole-dataset probe (STIT vs STI / IST)")
    if not wino:
        st.info("No Winoground probe yet. Run "
                "`CUDA_VISIBLE_DEVICES=0 python winoground_stit_probe.py` "
                "(all 1600 pairs; produces the a_q / a_img / P(correct) curves).")
    else:
        st.caption(
            f"Every Winoground image–caption pair (n={wino['n_pairs']}) under STI / "
            "STIT / IST — per-layer answer→question attention, answer→image attention, "
            "and logit-lens P(correct Yes/No). STIT is the strong ordering here "
            "(Group-Acc 37.5%)."
        )
        rows = {}
        for o in ("STI", "STIT", "IST"):
            d = wino["orders"].get(o)
            if not d:
                continue
            aq, pg = np.array(d["a_q"]), np.array(d["p_gt"])
            rows[o] = {
                "acc": f"{d['acc']*100:.1f}%",
                "ans→Q attn (mid peak)": f"{aq[16:26].max():.3f}" if len(aq) > 26 else "—",
                "final P(correct)": f"{pg[-1]:.3f}" if len(pg) else "—",
            }
        if rows:
            st.table(rows)
        fig = os.path.join(PROBE_DIR, "winoground_stit_probe.png")
        if os.path.exists(fig):
            st.image(fig, width="stretch")
