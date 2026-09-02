"""
Streamlit page: "New Experiments" — a dedicated, additive tab for the newest
round of ablations (RF20 half-vs-quarter resolution, token-cost accounting,
GEPA prompt-optimization baseline, logit-lens STI/IST word-diff), kept
SEPARATE from the existing per-dataset tabs so nothing there is touched.
Purely additive and read-only; reads whatever result JSONs each sub-section
needs and renders "not run yet" placeholders for anything still pending.
"""
import json
import os

import pandas as pd
import streamlit as st

RF20_RESULTS_DIR = "./rf20/results"
RF20_MODEL = "qwen3-vl-8b"
RF20_ECHO_COLS = [
    ("SITIT", "SITIT baseline (both full-res)"),
    ("SITIT_echo2half", "SITIT (2nd/echoed occ. at 0.5x)"),
    ("SITIT_echo2quarter", "SITIT (2nd/echoed occ. at 0.25x)"),
]


def _rf20_meta(tag):
    p = os.path.join(RF20_RESULTS_DIR, f"{RF20_MODEL}__{tag}__results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["meta"]
    except Exception:
        return None


def _rf20_resolution_section():
    st.subheader("🔍 RF20 — echoed-occurrence resolution: half vs quarter")
    st.caption(
        "SITIT shows the image twice; here the SECOND (echoed) occurrence is "
        "downscaled to 0.5x or 0.25x while the first stays full-res, isolating "
        "how much detail the echo actually needs to carry. All rows: "
        f"**{RF20_MODEL}**, full 20-dataset RF20 (24,449 samples)."
    )
    metas = {tag: _rf20_meta(tag) for tag, _ in RF20_ECHO_COLS}
    if not any(metas.values()):
        st.info("No RF20 resolution-sweep results yet.")
        return

    rows = []
    for tag, label in RF20_ECHO_COLS:
        m = metas.get(tag)
        if not m:
            rows.append({"Ordering": label, "N": "—", "F1": "—", "Accuracy": "—"})
            continue
        o = m.get("overall", {})
        rows.append({
            "Ordering": label, "N": str(o.get("n", "—")),
            "F1": f"{o.get('f1', 0)*100:.2f}%",
            "Accuracy": f"{o.get('acc', 0)*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    base, half, quarter = metas.get("SITIT"), metas.get("SITIT_echo2half"), metas.get("SITIT_echo2quarter")
    if base and half and quarter:
        f1_base = base["overall"]["f1"] * 100
        f1_half = half["overall"]["f1"] * 100
        f1_quarter = quarter["overall"]["f1"] * 100
        d_half = f1_half - f1_base
        d_quarter = f1_quarter - f1_base
        st.caption(
            f"vs baseline F1 — half: **{d_half:+.2f} pts**, quarter: "
            f"**{d_quarter:+.2f} pts**. "
            + (f"**Half beats quarter** ({f1_half:.2f}% > {f1_quarter:.2f}%) — "
               "the expected/safer outcome; per your instruction, this would "
               "extend the half-resolution echo ablation to the other benchmarks."
               if f1_half > f1_quarter else
               f"**Quarter beats half** ({f1_quarter:.2f}% > {f1_half:.2f}%) — "
               "the more surprising outcome; per your instruction, this should "
               "be discussed before extending it anywhere.")
        )
    elif base and half:
        st.caption("Quarter-resolution run not finished yet — showing baseline vs half so far.")
    elif base:
        st.caption("Half/quarter runs in progress — showing baseline only so far.")


def _token_cost_section():
    st.subheader("🔢 Token-cost Delta — text repetition vs image repetition")
    st.caption(
        "How many extra tokens does each intervention cost, relative to STI "
        "baseline: repeating the TASK text (STIT) vs repeating the IMAGE "
        "(SITIT), at full/½/¼ resolution for the echoed occurrence."
    )
    st.info("Not computed yet — planned next (no GPU generation needed, just "
            "processor/tokenizer introspection on a representative image sample).")


def _gepa_section():
    st.subheader("🧬 GEPA baseline — prompt optimization")
    st.caption(
        "GEPA (Reflective Prompt Evolution, arxiv.org/abs/2507.19457) as an "
        "automated prompt-optimization baseline, run on datasets other than "
        "RF20. Training-time cost (wall-clock, # calls, # tokens) and "
        "inference-time token overhead for the optimized prompt, tracked "
        "alongside accuracy."
    )
    st.info("Integration in progress — GEPA supports wrapping this repo's local "
            "vLLM/HF models directly via custom Python callables (no API keys "
            "needed), confirmed via the official repo source. Scope (which "
            "dataset(s), train/val split) still being finalized.")


def _logit_lens_diff_section():
    st.subheader("🧠 Logit-lens word-diff — STI vs IST")
    st.caption(
        "Per layer/position, diff the top-predicted word between STI and IST "
        "orderings on the same (image, question) pairs, keeping only genuine "
        "word changes (synonym-only differences filtered out)."
    )
    st.info("Not computed yet — will reuse logit_lens_runner.run_logit_lens() "
            "and logit_lens_overlay.py's per-layer word extraction, which "
            "already exist in this repo.")


def render_new_experiments_page():
    st.title("🆕 New Experiments")
    st.caption(
        "A separate tab for the newest round of ablations, kept apart from "
        "the existing per-dataset tabs so nothing there is modified."
    )
    _rf20_resolution_section()
    st.markdown("---")
    _token_cost_section()
    st.markdown("---")
    _gepa_section()
    st.markdown("---")
    _logit_lens_diff_section()
