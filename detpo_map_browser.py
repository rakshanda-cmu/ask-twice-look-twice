"""
Streamlit page: DetPO Detection — mAP on the RF20-VL *Aerial* domain and referring
[email protected] on RefCOCOg (val), both from the served Qwen3-VL model via the DetPO
setup. Purely additive and read-only; reads JSON written by the eval scripts in
detpo_map/:
    detpo_map/results/rf20_aerial_<config>_<model>.json
    detpo_map/results/refcocog_val_refacc_<model>.json

This is distinct from the "🟩 RF20" page (that one is the prompt-ordering yes/no
object-presence study); this page is the DetPO object-detection benchmark (COCO mAP).
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "detpo_map", "results")

RF20_RUN = ("HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \\\n"
            "  /home/grg/anaconda3/envs/qwen-vllm-env/bin/python "
            "detpo_map/rf20_aerial_eval.py --model Qwen3-VL-30B-A3B-Instruct")
REF_RUN = ("/home/grg/anaconda3/envs/qwen-vllm-env/bin/python "
           "detpo_map/refcocog_eval.py --model Qwen/Qwen3-VL-30B-A3B-Instruct \\\n"
           "  --n 2573 --out detpo_map/results/refcocog_val_refacc_qwen3vl-30b-a3b.json")


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


def _rf20_section():
    st.subheader("🛩️ RF20-VL — Aerial (DetPO, COCO mAP)")
    st.caption(
        "Few-shot object **detection** on the Aerial domain of the Roboflow20-VL "
        "benchmark (`wildfire-smoke`, `aerial-airport`), evaluated through the DetPO "
        "vLLM setup. The model is prompted per class for boxes; scoring is standard "
        "COCO **mAP@[.50:.95]** and **[email protected]** on the test split. Configuration below "
        "is the zero-shot **baseline** (default class descriptions parsed from each "
        "dataset README — no DetPO prompt optimization)."
    )
    runs = _load("rf20_aerial_*.json")
    if not runs:
        st.info("**No RF20 Aerial results yet.** Run:\n\n```\n" + RF20_RUN + "\n```")
        return
    for r in runs:
        m = r["meta"]
        st.markdown(f"### {m['model']}  ·  _{m['config']}_")
        rows = []
        pd_ = m["per_dataset"]
        for ds in sorted(pd_):
            d = pd_[ds]
            rows.append({
                "Dataset": ds,
                "Classes": ", ".join(d.get("classes", [])),
                "# test img": str(d.get("n_images", "—")),
                "mAP@[.5:.95]": _fmt(d.get("mAP")),
                "[email protected]": _fmt(d.get("mAP50")),
                "[email protected]": _fmt(d.get("mAP75")),
            })
        mean = m.get("mean", {})
        rows.append({
            "Dataset": "Aerial (mean)", "Classes": "", "# test img": "",
            "mAP@[.5:.95]": _fmt(mean.get("mAP")),
            "[email protected]": _fmt(mean.get("mAP50")),
            "[email protected]": _fmt(mean.get("mAP75")),
        })
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        st.caption(
            f"Aerial-mean **mAP {_fmt(mean.get('mAP'))}** · "
            f"**[email protected] {_fmt(mean.get('mAP50'))}** (measured here). "
            "Config: raw per-class detection scored with COCO mAP, no NMS / VQA "
            "rescoring (this DetPO `run_evaluation` baseline path). The DetPO paper "
            "lists the Qwen3-VL-30B Aerial baseline at ~9.0 mAP under its fuller "
            "pipeline; absolute values shift with decoding, resolution and "
            "post-processing, so treat this as our reproduction under the simple "
            "baseline, not a paper-identical number."
        )
    st.caption("Re-run:  `" + RF20_RUN.replace("\\\n  ", " ") + "`")


def _refcoco_section():
    st.subheader("🎯 RefCOCOg (val) — Referring [email protected]")
    st.caption(
        "Referring-expression grounding: for each RefCOCOg (umd split) validation "
        "expression the served model predicts one box; it is correct if IoU with the "
        "ground-truth box ≥ 0.5. Headline metric is referring **[email protected]** — the "
        "standard RefCOCO metric (RefCOCO is not a multi-class detection set, so COCO "
        "mAP does not apply here)."
    )
    runs = _load("refcocog_*.json")
    if not runs:
        st.info("**No RefCOCOg results yet.** Run:\n\n```\n" + REF_RUN + "\n```")
        return
    rows = []
    for r in runs:
        m = r["meta"]
        rows.append({
            "Model": m["model"], "Variant": m.get("variant", "umd"),
            "Split": m.get("split", "val"),
            "N": m.get("n"), "Parsed": m.get("parsed"),
            "Correct": m.get("correct"),
            "[email protected] (%)": _fmt(m.get("acc"), 100.0),
        })
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)
    best = max(runs, key=lambda r: r["meta"].get("acc", 0))["meta"]
    st.caption(f"**[email protected] = {_fmt(best.get('acc'), 100.0)}%** "
               f"on {best.get('n')} val expressions ({best['model']}).")
    st.caption("Re-run:  `" + REF_RUN.replace("\\\n  ", " ") + "`")


def render_detpo_map_page():
    st.title("🛩️ DetPO Detection — RF20 Aerial mAP + RefCOCO")
    st.caption(
        "Object-detection evaluation of the served **Qwen3-VL-30B-A3B** model via the "
        "DetPO setup (vLLM OpenAI-compatible server). Two benchmarks: RF20-VL Aerial "
        "(COCO mAP) and RefCOCOg val (referring [email protected])."
    )
    _rf20_section()
    st.markdown("---")
    _refcoco_section()
