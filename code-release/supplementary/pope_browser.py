"""
Streamlit page: POPE (object hallucination) — IST / STI / STIT results and a
cross-model comparison. Purely additive and fully separate from the NaturalBench
page: reads pope/results/<model>__<order>__results.json written by pope_eval.py.

POPE asks "Is there a <object> in the image?" (yes/no) under three negative regimes
(random / popular / adversarial). The headline metric is F1; yes-ratio exposes a
model's bias toward answering "yes" (over-claiming = hallucination).
"""

import json
import os

import streamlit as st

RESULTS_DIR = "./pope/results"
# models shown if results exist (Gemma first, Qwen reference) — mirrors the NB tab
MODELS = [
    ("gemma-3-27b", "Gemma 3 (27B)"),
    ("qwen3-vl-8b", "Qwen3-VL (8B)"),
    ("qwen2.5-vl-7b", "Qwen2.5-VL (7B)"),
]
ORDERS = [
    ("IST", "IST — Image · System · Task"),
    ("STI", "STI — System · Task · Image"),
    ("STIT", "STIT — System · Task · Image · Task"),
]
CATEGORIES = ["random", "popular", "adversarial"]
RUN_CMD = ("CUDA_VISIBLE_DEVICES=1 python pope_eval.py "
           "--model gemma-3-27b --order IST,STI,STIT")


def _meta(model, tag):
    p = os.path.join(RESULTS_DIR, f"{model}__{tag}__results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["meta"]
    except Exception:
        return None


def _overall_row(meta):
    o = meta.get("overall", {})
    return {
        "Accuracy": f"{o.get('acc', 0)*100:.1f}%",
        "F1": f"{o.get('f1', 0)*100:.1f}%",
        "Precision": f"{o.get('precision', 0)*100:.1f}%",
        "Recall": f"{o.get('recall', 0)*100:.1f}%",
        "Yes-ratio": f"{o.get('yes_ratio', 0)*100:.1f}%",
        "# Q": str(o.get("n", 0)),
    }


def _cat_f1_row(meta):
    bc = meta.get("by_category", {})
    return {c.capitalize() + " F1": f"{bc.get(c, {}).get('f1', 0)*100:.1f}%"
            for c in CATEGORIES}


def _gap_caption(metas, label):
    """IST vs STI gap + STIT recovery on F1 — same narrative as the NB/Gemma tab."""
    f1 = lambda t: metas[t]["overall"]["f1"] * 100
    if metas.get("IST") and metas.get("STI"):
        d = f1("IST") - f1("STI")
        better = "IST" if d > 0 else ("STI" if d < 0 else "tie")
        msg = (f"**{label}** F1 gap (IST − STI): **{d:+.1f} pts** "
               f"→ {'**' + better + '** higher' if better != 'tie' else 'tie'}")
        if metas.get("STIT"):
            g = f1("STIT")
            parts = []
            if metas.get("STI"):
                parts.append(f"vs STI **{g - f1('STI'):+.1f}**")
            if metas.get("IST"):
                parts.append(f"vs IST **{g - f1('IST'):+.1f}**")
            msg += f". STIT F1 **{g:.1f}%** (" + " · ".join(parts) + ")"
        st.caption(msg)


def render_pope_page():
    st.subheader("🟣 POPE — Object Hallucination (IST / STI / STIT)")
    st.caption(
        "Does the prompt-ordering effect also govern **object hallucination**? "
        "POPE asks yes/no presence questions under random / popular / adversarial "
        "negatives. Same S/I/T content layout as the NaturalBench runs, so the "
        "IST-vs-STI gap and the STIT recovery transfer head-to-head to a second, "
        "hallucination-specific benchmark. Higher F1 = less hallucination; "
        "**yes-ratio** above 50% flags a bias toward over-claiming objects."
    )

    present = [(m, lab) for m, lab in MODELS if any(_meta(m, t) for t, _ in ORDERS)]
    if not present:
        st.info(
            "**No POPE results yet.** Run (resumable; downloads `lmms-lab/POPE` "
            f"on first use — 9000 yes/no Qs over COCO):\n\n`{RUN_CMD}`"
        )
        return

    # ── per-model tables ───────────────────────────────────────────────────────
    for model, label in present:
        metas = {t: _meta(model, t) for t, _ in ORDERS}
        st.markdown(f"#### {label}")
        overall = {lab: _overall_row(metas[t]) for t, lab in ORDERS if metas.get(t)}
        st.table(overall)
        cat = {lab: _cat_f1_row(metas[t]) for t, lab in ORDERS if metas.get(t)}
        st.caption("F1 by negative-sampling regime:")
        st.table(cat)
        _gap_caption(metas, label)
        st.markdown("")

    # ── cross-model comparison (F1 by ordering) ────────────────────────────────
    if len(present) >= 2:
        st.markdown("#### Cross-model — F1 by ordering")
        comp = {}
        for t, _ in ORDERS:
            comp[t] = {}
            for model, label in present:
                m = _meta(model, t)
                comp[t][label] = f"{m['overall']['f1']*100:.1f}%" if m else "—"
        st.table(comp)
        # does the IST-vs-STI direction agree across models?
        gem, ref = present[0][0], present[1][0]
        gm = {t: _meta(gem, t) for t, _ in ORDERS}
        rm = {t: _meta(ref, t) for t, _ in ORDERS}
        if all(gm.get(o) for o in ("IST", "STI")) and all(rm.get(o) for o in ("IST", "STI")):
            dg = (gm["IST"]["overall"]["f1"] - gm["STI"]["overall"]["f1"]) * 100
            dr = (rm["IST"]["overall"]["f1"] - rm["STI"]["overall"]["f1"]) * 100
            agree = (dg > 0) == (dr > 0)
            st.caption(
                f"IST − STI F1 gap: {present[0][1]} **{dg:+.1f} pts** · "
                f"{present[1][1]} **{dr:+.1f} pts** — "
                + ("**same direction** → the ordering effect replicates on POPE too."
                   if agree else
                   "**opposite direction** → the effect is model-specific here.")
            )

    st.markdown("---")
    st.caption(f"Re-run / extend with:  `{RUN_CMD}`")
