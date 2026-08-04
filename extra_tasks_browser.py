"""
Streamlit page: new CV areas for the STI/SIT/STIT/SITIT ordering ablation --
open-ended VQA (VQAv2), counting (TallyQA), perception probes (MMVP, BLINK),
and multi-frame video QA (NExT-QA). Read-only; reads JSON written by the
extra_tasks/*_eval_vllm.py scripts.

Engine: all five run on the vLLM library (STI/SIT/STIT/SITIT only). SITIT_rev
needs the local HF patch-reversal hooks (see detpo_map/PROMPTS.md) and is not
run for these tasks yet -- a later extension, same as RF20's initial pass.
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "extra_tasks", "results")
ORDER_LIST = ["STI", "SIT", "STIT", "SITIT"]
ORDER_DESC = {
    "STI": "S·T·I — question-first", "SIT": "S·I·T — question-last",
    "STIT": "S·T·I·T — question echo", "SITIT": "S·I·T·I·T — image echo",
}
RUN_PREFIX = ("HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 "
              "/home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/")


def _load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def _fmt(x, mult=1.0, nd=1):
    return "—" if x is None else f"{x * mult:.{nd}f}"


def _order_key(tag):
    return ORDER_LIST.index(tag) if tag in ORDER_LIST else len(ORDER_LIST)


# Short display label per model tag as it appears in meta["model"]. Falls
# back to the raw tag for anything unrecognized so nothing is ever silently
# dropped if a new model gets added later.
MODEL_LABEL = {"qwen3-vl-8b": "Qwen3-VL (8B)", "gemma-3-27b": "Gemma-3 (27B)",
              "gemma-4-31b": "Gemma-4 (31B)"}
MODELS = ["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"]


def _model_label(tag):
    return MODEL_LABEL.get(tag, tag)


def _model_of(meta):
    return meta.get("model", "qwen3-vl-8b")


def _delta_line(by, unit):
    if by.get("STI") is None:
        return None
    parts = [f"{t} {by[t]-by['STI']:+.1f}" for t in ("SIT", "STIT", "SITIT")
             if by.get(t) is not None]
    if not parts:
        return None
    return f"Δ vs STI (question-first), {unit}: " + " · ".join(parts) + \
        f"  (STI = {by['STI']:.1f})."


def _simple_section(title, caption, pattern, script, acc_key="accuracy",
                    extra_cols=None, by_key=None, by_label=None):
    """Generic single-number-per-ordering section (MMVP / TallyQA / VQA-pooled),
    one sub-table per model present -- results for different models share the
    same (ordering, ...) keys, so they must never be merged into one row."""
    st.subheader(title)
    st.caption(caption)
    runs = _load(pattern)
    if not runs:
        st.info(f"**No results yet.** Run:\n\n```\n{RUN_PREFIX}{script}\n```")
        return
    runs.sort(key=lambda r: (_order_key(r["meta"]["ordering"]), _model_of(r["meta"])))
    models_present = [m for m in MODELS if any(_model_of(r["meta"]) == m for r in runs)]
    by_model = {}
    for model in models_present:
        mruns = [r for r in runs if _model_of(r["meta"]) == model]
        if len(models_present) > 1:
            st.markdown(f"**{_model_label(model)}**")
        rows = []
        for r in mruns:
            m = r["meta"]
            row = {"Ordering": m["ordering"], "What": ORDER_DESC.get(m["ordering"], ""),
                   "N": str(m.get("n", "")),
                   "Accuracy (%)": _fmt(m.get(acc_key), 100.0)}
            if extra_cols:
                for label, key, mult in extra_cols:
                    row[label] = _fmt(m.get(key), mult)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {r["meta"]["ordering"]: (r["meta"].get(acc_key) or 0) * 100 for r in mruns}
        by_model[model] = by
        line = _delta_line(by, "accuracy %")
        if line:
            st.caption(line)
        if by_key:
            st.markdown(f"**By {by_label}**" + (f" — {_model_label(model)}"
                                                 if len(models_present) > 1 else ""))
            sub_names = sorted({k for r in mruns for k in r["meta"].get(by_key, {})})
            srows = []
            for r in mruns:
                m = r["meta"]
                row = {"Ordering": m["ordering"]}
                for s in sub_names:
                    d = m.get(by_key, {}).get(s)
                    row[s] = _fmt(d["accuracy"] * 100) if d else "—"
                srows.append(row)
            st.dataframe(pd.DataFrame(srows).astype(str), hide_index=True,
                         use_container_width=True)

    if len(models_present) > 1:
        st.markdown("**Cross-model — Accuracy (%) by ordering**")
        comp_rows = []
        for o in ORDER_LIST:
            if not any(by_model[m].get(o) is not None for m in models_present):
                continue
            row = {"Ordering": o}
            for model in models_present:
                row[_model_label(model)] = _fmt(by_model[model].get(o))
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows).astype(str), hide_index=True,
                     use_container_width=True)

    st.caption(f"Re-run:  `{RUN_PREFIX}{script}`")


def _vqa_section():
    st.subheader("💬 VQAv2 (val) — Open-ended VQA by prompt ordering")
    st.caption(
        "Free-text answers (not multiple-choice), scored with the **official VQA "
        "accuracy** (min(#matching human answers / 3, 1)), broken out by answer "
        "type: **yes/no**, **number** (implicit counting), **other**. Tests "
        "whether the ordering paradox holds outside constrained yes/no framing."
    )
    # Excludes *_paperconfig.json on purpose -- that's a separate diagnostic
    # re-run (no answer suffix, max_tokens=16, N=2000) matching the paper's
    # exact VQAv2 config, kept out of the main "our protocol" comparison below
    # (see _vqa_paperconfig_section).
    runs = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "vqa_order-*.json"))):
        if "_paperconfig" in os.path.basename(p):
            continue
        try:
            runs.append(json.load(open(p)))
        except Exception:
            pass
    if not runs:
        st.info(f"**No results yet.** Run:\n\n```\n{RUN_PREFIX}vqa_eval_vllm.py\n```")
        return
    runs.sort(key=lambda r: (_order_key(r["meta"]["ordering"]), _model_of(r["meta"])))
    models_present = [m for m in MODELS if any(_model_of(r["meta"]) == m for r in runs)]
    by_model = {}
    for model in models_present:
        mruns = [r for r in runs if _model_of(r["meta"]) == model]
        if len(models_present) > 1:
            st.markdown(f"**{_model_label(model)}**")
        rows = []
        for r in mruns:
            m = r["meta"]
            row = {"Ordering": m["ordering"], "N": str(m["n"]),
                   "VQA-score (%)": _fmt(m["vqa_score"], 100.0),
                   "Accuracy (%)": _fmt(m["accuracy"], 100.0)}
            for t, d in m.get("by_answer_type", {}).items():
                row[f"{t} (%)"] = _fmt(d["accuracy"], 100.0)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {r["meta"]["ordering"]: r["meta"]["vqa_score"] * 100 for r in mruns}
        by_model[model] = by
        line = _delta_line(by, "VQA-score %")
        if line:
            st.caption(line)

    if len(models_present) > 1:
        st.markdown("**Cross-model — VQA-score (%) by ordering**")
        comp_rows = []
        for o in ORDER_LIST:
            if not any(by_model[m].get(o) is not None for m in models_present):
                continue
            row = {"Ordering": o}
            for model in models_present:
                row[_model_label(model)] = _fmt(by_model[model].get(o))
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows).astype(str), hide_index=True,
                     use_container_width=True)
    st.caption(f"Re-run:  `{RUN_PREFIX}vqa_eval_vllm.py`")
    _vqa_paperconfig_section()


def _vqa_paperconfig_section():
    """Diagnostic: our normal vqa_eval_vllm.py protocol (SHORT_ANSWER_SUFFIX,
    max_tokens=32/96, N=3000) doesn't match the paper's own VQAv2 numbers
    (Table 4). This re-run drops the suffix and uses max_tokens=16, N=2000 --
    the paper's exact Appendix A config -- to confirm the discrepancy is
    config, not a regression."""
    runs = _load("vqa_order-*_paperconfig.json")
    if not runs:
        return
    st.markdown("#### 📄 Paper-config confirm run (no suffix, max_tokens=16, N=2000)")
    st.caption(
        "Our production VQAv2 numbers above use SHORT_ANSWER_SUFFIX + a larger "
        "token budget + N=3000, which don't match the paper's Table 4 (Appendix A "
        "config: no suffix, max_tokens=16, N=2000). This re-run matches the "
        "paper's exact config on Qwen3-VL-8B to confirm that's the actual source "
        "of the mismatch."
    )
    runs.sort(key=lambda r: _order_key(r["meta"]["ordering"]))
    rows = []
    for r in runs:
        m = r["meta"]
        rows.append({"Ordering": m["ordering"], "N": str(m["n"]),
                     "VQA-score (%)": _fmt(m["vqa_score"], 100.0),
                     "Accuracy (%)": _fmt(m["accuracy"], 100.0)})
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)
    st.caption("Paper (Table 4, Qwen3-VL-8B, N=2000): STI 81.1% · SIT 83.4% · "
              "STIT 82.1% · SITIT 83.8% (VQA accuracy). Compare against the row "
              "above to see how much of the gap the config difference explains.")


def _nextqa_section():
    st.subheader("🎬 NExT-QA — Multi-frame video QA by prompt ordering")
    st.caption(
        "The cleanest generalization of the paper's mechanism: **I** expands to "
        "K uniformly-sampled frames, so SITIT re-shows the *whole clip* a second "
        "time. Multiple-choice (5-way), broken out by question type: **C**ausal, "
        "**T**emporal, **D**escriptive."
    )
    runs = _load("nextqa_order-*.json")
    if not runs:
        st.info(f"**No results yet.** Run:\n\n```\n{RUN_PREFIX}nextqa_eval_vllm.py\n```")
        return
    runs.sort(key=lambda r: (_order_key(r["meta"]["ordering"]), _model_of(r["meta"])))
    models_present = [m for m in MODELS if any(_model_of(r["meta"]) == m for r in runs)]
    for model in models_present:
        mruns = [r for r in runs if _model_of(r["meta"]) == model]
        if len(models_present) > 1:
            st.markdown(f"**{_model_label(model)}**")
        rows = []
        for r in mruns:
            m = r["meta"]
            row = {"Ordering": m["ordering"], "N": str(m["n"]),
                   "Frames": str(m.get("k_frames", "")),
                   "Accuracy (%)": _fmt(m["accuracy"], 100.0)}
            for t, d in m.get("by_type", {}).items():
                row[f"type={t} (%)"] = _fmt(d["accuracy"], 100.0)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {r["meta"]["ordering"]: r["meta"]["accuracy"] * 100 for r in mruns}
        line = _delta_line(by, "accuracy %")
        if line:
            st.caption(line)
    st.caption(f"Re-run:  `{RUN_PREFIX}nextqa_eval_vllm.py`")


def _mvbench_section():
    st.subheader("🎥 MVBench — Multi-frame video QA by prompt ordering")
    st.caption(
        "13 of MVBench's 20 tasks (Li et al. 2024) with confirmed video coverage "
        "in our download (7 excluded: their video content wasn't available in the "
        "subset we pulled, see mvbench_eval_vllm.py's module docstring) -- action "
        "recognition/prediction/localization, object existence/interaction, "
        "counting, direction, scene transitions, egocentric navigation, "
        "counterfactual inference. Multiple-choice, same K-frame SITIT-echo "
        "generalization as NExT-QA."
    )
    runs = _load("mvbench_order-*.json")
    if not runs:
        st.info(f"**No results yet.** Run:\n\n```\n{RUN_PREFIX}mvbench_eval_vllm.py\n```")
        return
    runs.sort(key=lambda r: (_order_key(r["meta"]["ordering"]), _model_of(r["meta"])))
    models_present = [m for m in MODELS if any(_model_of(r["meta"]) == m for r in runs)]
    for model in models_present:
        mruns = [r for r in runs if _model_of(r["meta"]) == model]
        if len(models_present) > 1:
            st.markdown(f"**{_model_label(model)}**")
        rows = []
        for r in mruns:
            m = r["meta"]
            row = {"Ordering": m["ordering"], "N": str(m["n"]),
                   "Frames": str(m.get("k_frames", "")),
                   "Accuracy (%)": _fmt(m.get("accuracy"), 100.0)}
            for t, d in m.get("by_task", {}).items():
                row[f"task={t} (%)"] = _fmt(d.get("accuracy"), 100.0)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {r["meta"]["ordering"]: (r["meta"].get("accuracy") or 0) * 100 for r in mruns}
        line = _delta_line(by, "accuracy %")
        if line:
            st.caption(line)
    st.caption(f"Re-run:  `{RUN_PREFIX}mvbench_eval_vllm.py`")


def _video_qa_section(dataset, title, caption):
    st.subheader(title)
    st.caption(caption)
    runs = _load(f"{dataset}qa_order-*.json")
    script = f"video_qa_eval_vllm.py --dataset {dataset}"
    if not runs:
        st.info(f"**No results yet.** Run:\n\n```\n{RUN_PREFIX}{script}\n```")
        return
    runs.sort(key=lambda r: (_order_key(r["meta"]["ordering"]), _model_of(r["meta"])))
    models_present = [m for m in MODELS if any(_model_of(r["meta"]) == m for r in runs)]
    for model in models_present:
        mruns = [r for r in runs if _model_of(r["meta"]) == model]
        if len(models_present) > 1:
            st.markdown(f"**{_model_label(model)}**")
        rows = [{"Ordering": r["meta"]["ordering"], "N": str(r["meta"]["n"]),
                "Frames": str(r["meta"].get("k_frames", "")),
                "Accuracy (%)": _fmt(r["meta"].get("accuracy"), 100.0)} for r in mruns]
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {r["meta"]["ordering"]: (r["meta"].get("accuracy") or 0) * 100 for r in mruns}
        line = _delta_line(by, "accuracy %")
        if line:
            st.caption(line)
    st.caption(f"Re-run:  `{RUN_PREFIX}{script}`")


def render_extra_tasks_page():
    st.title("🧪 New CV Areas — extending the ordering ablation beyond yes/no + boxes")
    st.caption(
        "STI / SIT / STIT / SITIT applied to task types not covered elsewhere in "
        "this repo: open-ended text answers (VQA), numeric/counting answers "
        "(TallyQA), curated perception-blind-spot probes (MMVP, BLINK), and "
        "multi-frame video QA (NExT-QA, MVBench, MSVD-QA, TGIF-QA). Run on "
        "**Qwen3-VL-8B** and, where noted, **Gemma-3-27B** (the paper's second "
        "model) via the vLLM library; SITIT_rev (needs local HF + "
        "patch-reversal hooks) is not yet run for these tasks."
    )
    _vqa_section()
    st.markdown("---")
    _simple_section(
        "🔢 TallyQA — Counting by prompt ordering",
        "Free-form counting questions (\"how many X\"), exact-match accuracy + "
        "mean absolute error (MAE). Counting forces *exhaustive* scanning of the "
        "image rather than one-object verification — a different failure mode "
        "than presence/detection.",
        "tallyqa_order-*.json", "tallyqa_eval_vllm.py",
        extra_cols=[("MAE", "mae", 1.0)])
    st.markdown("---")
    _simple_section(
        "👁️ MMVP — Perception blind-spot probes by prompt ordering",
        "300 CLIP-blind image pairs with multiple-choice questions (Tong et al. "
        "2024), purpose-built to find VLM perceptual failures (orientation, "
        "counting, state, viewpoint, text, camera, …). Closest existing "
        "benchmark to this repo's own steering/logit-lens analysis.",
        "mmvp_order-*.json", "mmvp_eval_vllm.py")
    st.markdown("---")
    _simple_section(
        "🧩 BLINK — Classic perception tasks (multiple choice) by prompt ordering",
        "Single-image subtasks: Counting, Relative Depth, Relative Reflectance, "
        "Object Localization, IQ Test, Spatial Relation (Fu et al. 2024). "
        "Multi-image subtasks (Jigsaw, correspondence, …) are not run yet.",
        "blink_order-*.json", "blink_eval_vllm.py",
        by_key="by_subtask", by_label="subtask")
    st.markdown("---")
    _nextqa_section()
    st.markdown("---")
    _mvbench_section()
    st.markdown("---")
    _video_qa_section(
        "msvd", "🐕 MSVD-QA — Open-ended video QA by prompt ordering",
        "Classic short-answer video QA (Xu et al. 2017, Chen & Dolan 2011 videos), "
        "whole-word-containment scoring against a single canonical answer per "
        "question. Same K-frame SITIT-echo generalization as NExT-QA/MVBench.")
    st.markdown("---")
    _video_qa_section(
        "tgif", "🎞️ TGIF-QA — Open-ended video QA by prompt ordering",
        "Short-answer video QA over animated GIFs (Jang et al. 2017; Repeating "
        "Action / State Transition / FrameQA / Count subtasks pooled), same "
        "scoring and frame-echo generalization as MSVD-QA.")
