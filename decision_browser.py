"""
Streamlit page: Decision Layer — when, across depth, the model commits to its
yes/no answer for STI / STIT / IST, produced by decision_layer.py.
"""

import json
import os

import streamlit as st

PROBE_DIR = "./naturalbench/probe"
ORDERS = ("STI", "STIT", "IST")


def _load(pset):
    p = os.path.join(PROBE_DIR, f"decision_{pset}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _table(data):
    rows = {}
    for o in ORDERS:
        d = data["orders"][o]
        cf = d["cls_frac"]
        rows[o] = {
            "acc": f"{d['acc']*100:.0f}%",
            "commit layer (median)": f"L{d['commit_median']:.0f}"
                if d["commit_median"] is not None else "—",
            "correct-onset (median)": f"L{d['onset_median']:.0f}"
                if d.get("onset_median") is not None else "—",
            "final P(correct|2)": f"{d['p_corr2'][-1]:.2f}" if d["p_corr2"] else "—",
            "committed-wrong": f"{cf['committed_wrong']*100:.0f}%",
            "flipped-to-wrong": f"{cf['flipped_to_wrong']*100:.0f}%",
        }
    return rows


def render_decision_page():
    st.subheader("🎯 Decision Layer — when the yes/no answer locks in")
    st.caption(
        "For each yes/no pair under STI / STIT / IST we read the logit-lens "
        "distribution **restricted to {Yes, No}** at the answer position, every "
        "layer. P(correct | {Yes,No}) by layer shows *when* the correct answer "
        "wins; the commitment layer is where the argmax stops changing. Tests "
        "whether STI commits **early to an image-anchored (often wrong) answer** "
        "before the question is integrated."
    )

    neu = _load("neutral")
    dis = _load("disagreement")
    if not neu and not dis:
        st.info("No decision-layer results yet. Run "
                "`CUDA_VISIBLE_DEVICES=0 python decision_layer.py --set neutral "
                "--num-pairs 250` (and `--set disagreement`).")
        return

    if neu:
        st.markdown(f"#### Neutral sample (n={neu['n_pairs']}, outcome-independent)")
        st.table(_table(neu))
        fig = os.path.join(PROBE_DIR, "decision_neutral.png")
        if os.path.exists(fig):
            st.image(fig, width="stretch")
        st.caption(
            "Left: P(correct among the two) by layer — higher/earlier crossing of "
            "0.5 means the correct answer wins sooner. Right: histogram of the "
            "layer at which each pair's decision locks in."
        )

    if dis:
        st.markdown("---")
        st.markdown(f"#### Diagnostic set — STI-wrong / IST-right (n={dis['n_pairs']})")
        st.caption(
            "By construction STI ends wrong and IST ends right on these pairs. "
            "The decision-layer view shows *how*: if STI's P(correct|2) stays "
            "below 0.5 with an early commit while IST/STIT climb above, the "
            "correct answer only wins once a question representation sits next to "
            "the answer."
        )
        st.table(_table(dis))
        fig = os.path.join(PROBE_DIR, "decision_disagreement.png")
        if os.path.exists(fig):
            st.image(fig, width="stretch")

    st.markdown("---")
    st.markdown(
        "**Flip classes.** *committed-wrong* = answer was wrong throughout the "
        "stack; *flipped-to-wrong* = correct in the middle layers but wrong at the "
        "output (a late readout failure). A high STI *committed-wrong* fraction "
        "points to a representation problem; a high *flipped-to-wrong* fraction "
        "points to a readout/attention problem that steering could recover."
    )
