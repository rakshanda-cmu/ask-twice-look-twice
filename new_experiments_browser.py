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


TOKEN_COST_PATH = "./token_cost_results.json"
TOKEN_COST_VARIANTS = [
    ("STI", "STI (baseline: 1 image, 1 task)"),
    ("STIT", "STIT (text repetition: task x2)"),
    ("SITIT_full", "SITIT (image repetition, full-res echo)"),
    ("SITIT_half", "SITIT (image repetition, ½-res echo)"),
    ("SITIT_quarter", "SITIT (image repetition, ¼-res echo)"),
]


def _token_cost_section():
    st.subheader("🔢 Token-cost Delta — text repetition vs image repetition")
    st.caption(
        "How many extra tokens does each intervention cost, relative to STI "
        "baseline: repeating the TASK text (STIT) vs repeating the IMAGE "
        "(SITIT), at full/½/¼ resolution for the echoed occurrence."
    )
    if not os.path.exists(TOKEN_COST_PATH):
        st.info("Not computed yet — run token_cost_analysis.py (CPU-only, no "
                "GPU generation needed).")
        return
    d = json.load(open(TOKEN_COST_PATH))
    st.caption(
        f"Mean over **{d['n_images']}** representative RF20 images (one per "
        f"dataset), **{d['model']}** tokenizer/processor. Task text: "
        f"\"{d['task_text']}\""
    )
    rows = []
    for key, label in TOKEN_COST_VARIANTS:
        s, delta = d["summary"][key], d["delta_vs_STI"][key]
        rows.append({
            "Variant": label,
            "Image tokens": f"{s['image']:.0f}",
            "Text tokens": f"{s['text']:.0f}",
            "Total tokens": f"{s['total']:.0f}",
            "Δ total vs STI": f"{delta['delta_total']:+.0f}",
            "Δ image vs STI": f"{delta['delta_image']:+.0f}",
            "Δ text vs STI": f"{delta['delta_text']:+.0f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    stit_d = d["delta_vs_STI"]["STIT"]["delta_total"]
    full_d = d["delta_vs_STI"]["SITIT_full"]["delta_total"]
    half_d = d["delta_vs_STI"]["SITIT_half"]["delta_total"]
    quarter_d = d["delta_vs_STI"]["SITIT_quarter"]["delta_total"]
    st.caption(
        f"Text repetition (STIT) costs only **+{stit_d:.0f} tokens** vs STI "
        f"(the repeated task text is short); image repetition (SITIT) costs "
        f"far more, and scales with the echoed image's resolution: "
        f"**+{full_d:.0f}** at full res, **+{half_d:.0f}** at half "
        f"(~{100*half_d/full_d:.0f}% of full), **+{quarter_d:.0f}** at quarter "
        f"(~{100*quarter_d/full_d:.0f}% of full) — consistent with vision "
        f"tokens scaling with pixel count (quadratically in linear scale)."
    )


GEPA_RESULTS_DIR = "./gepa_results"
GEPA_MODEL = "qwen3-vl-8b"
GEPA_DATASETS = ["pope", "vqa"]


def _gepa_result(dataset):
    p = os.path.join(GEPA_RESULTS_DIR, f"{dataset}__{GEPA_MODEL}__gepa.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _gepa_section():
    st.subheader("🧬 GEPA baseline — prompt optimization")
    st.caption(
        "GEPA (Reflective Prompt Evolution, arxiv.org/abs/2507.19457) as an "
        "automated prompt-optimization baseline, run on datasets other than "
        "RF20. One in-process vLLM engine (qwen3-vl-8b) serves as both the "
        "task model (answering under a fixed STI order) and the reflection "
        "model (proposing improved prompt text from failure feedback). "
        "Train/val subsets are carved out of each benchmark's existing full "
        "pool (seeded, disjoint from each other); the held-out eval subset "
        "is scored for both the baseline SYSTEM_MESSAGE and the GEPA-"
        "optimized prompt on identical examples."
    )
    results = {d: _gepa_result(d) for d in GEPA_DATASETS}
    if not any(results.values()):
        st.info("Not run yet.")
        return

    rows = []
    for d in GEPA_DATASETS:
        r = results.get(d)
        if not r:
            rows.append({"Dataset": d, "N (train/val/eval)": "—",
                        "Baseline acc": "—", "GEPA acc": "—", "Δ acc": "—",
                        "Train wall-clock": "—", "Metric calls": "—",
                        "Train tokens": "—", "Δ inference tokens": "—"})
            continue
        sp, tr, ic, ac = r["split"], r["training"], r["inference_cost"], r["accuracy"]
        rows.append({
            "Dataset": d,
            "N (train/val/eval)": f"{sp['train']}/{sp['val']}/{sp['held_out_eval']}",
            "Baseline acc": f"{ac['baseline_system_message']['acc']*100:.2f}%",
            "GEPA acc": f"{ac['gepa_optimized']['acc']*100:.2f}%",
            "Δ acc": f"{ac['delta']*100:+.2f} pts",
            "Train wall-clock": f"{tr['wall_clock_s']:.0f}s",
            "Metric calls": str(tr["total_metric_calls"]),
            "Train tokens": f"{tr['total_tokens']:,}",
            "Δ inference tokens": f"{ic['delta_tokens']:+d}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    for d in GEPA_DATASETS:
        r = results.get(d)
        if not r:
            continue
        with st.expander(f"{d}: optimized prompt + cost breakdown"):
            tr = r["training"]
            st.caption(
                f"Training calls: **{tr['task_calls']}** task_lm "
                f"({tr['task_tokens_in']:,} in / {tr['task_tokens_out']:,} out tok) + "
                f"**{tr['reflect_calls']}** reflection ({tr['reflect_tokens_in']:,} in / "
                f"{tr['reflect_tokens_out']:,} out tok) = **{tr['total_tokens']:,}** total "
                f"tokens over **{tr['wall_clock_s']:.0f}s**."
            )
            st.markdown("**Baseline SYSTEM_MESSAGE:**")
            st.code(r["baseline_prompt"], language=None)
            st.markdown("**GEPA-optimized prompt:**")
            st.code(r["gepa_optimized_prompt"], language=None)


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
