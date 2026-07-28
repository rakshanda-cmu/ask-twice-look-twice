"""
Streamlit page: DetPO Detection — mAP on the RF20-VL *Aerial* domain and referring
[email protected] on RefCOCOg (val), for the served Qwen3-VL model, and across the prompt
orderings STI / SIT / STIT / SITIT / SITIT_rev (question-first paradox applied to
localization). Read-only; reads JSON written by the eval scripts in detpo_map/:
    detpo_map/results/rf20_aerial_baseline_<model>.json          (30B baseline)
    detpo_map/results/rf20_aerial_order-<ORD>_<model>.json        (8B orderings)
    detpo_map/results/refcocog_val_refacc_<model>.json            (30B baseline)
    detpo_map/results/refcocog_val_order-<ORD>_<model>.json       (8B orderings)

Distinct from the "🟩 RF20" page (that one is the prompt-ordering yes/no
object-presence study); this page is the DetPO localization benchmark (mAP / [email protected]).
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "detpo_map", "results")

# Canonical row order: 30B baseline first, then the orderings.
ORDER_ROWS = ["baseline", "STI", "SIT", "STIT", "SITIT", "SITIT_rev"]
ORDER_DESC = {
    "baseline": "DetPO default (S·T·I, question-first)",
    "STI": "S·T·I — question-first",
    "SIT": "S·I·T — question-last",
    "STIT": "S·T·I·T — question echo",
    "SITIT": "S·I·T·I·T — image echo",
    "SITIT_rev": "S·I·T·Ī·T — image echo, 2nd image reversed",
}

RF20_RUN = ("CUDA_VISIBLE_DEVICES=0 HF_HOME=/data2/hf_cache "
            "/home/grg/anaconda3/envs/logitlens/bin/python "
            "detpo_map/ordering_eval.py --benchmark rf20 "
            "--orders STI,SIT,STIT,SITIT,SITIT_rev")
REF_RUN = ("CUDA_VISIBLE_DEVICES=1 HF_HOME=/data2/hf_cache "
           "/home/grg/anaconda3/envs/logitlens/bin/python "
           "detpo_map/ordering_eval.py --benchmark refcoco "
           "--orders STI,SIT,STIT,SITIT,SITIT_rev --n 0")


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


def _tag(meta):
    return meta.get("ordering", "baseline")


def _order_key(tag):
    return ORDER_ROWS.index(tag) if tag in ORDER_ROWS else len(ORDER_ROWS)


def _rf20_section():
    st.subheader("🛩️ RF20-VL Aerial — DetPO detection mAP by prompt ordering")
    st.caption(
        "Few-shot object **detection** on the Aerial domain of Roboflow20-VL "
        "(`wildfire-smoke`, `aerial-airport`) through the DetPO setup, scored with "
        "COCO **mAP@[.50:.95]** / **[email protected]** on the test split. Each row re-orders the "
        "same system (**S**), task/question (**T**) and image (**I**) tokens: "
        "question-first (STI) vs question-last (SIT), question echo (STIT), image "
        "echo (SITIT) and image echo with the 2nd image block reversed (SITIT_rev). "
        "Orderings run on local **Qwen3-VL-8B** (so SITIT_rev's patch reversal can be "
        "applied); the **baseline** row is the served **Qwen3-VL-30B-A3B** for "
        "reference."
    )
    runs = _load("rf20_aerial_*.json")
    if not runs:
        st.info("**No RF20 results yet.** Run:\n\n```\n" + RF20_RUN + "\n```")
        return
    runs.sort(key=lambda r: (_order_key(_tag(r["meta"])), r["meta"]["model"]))
    rows = []
    for r in runs:
        m = r["meta"]
        tag = _tag(m)
        pd_ = m["per_dataset"]; mean = m.get("mean", {})
        row = {"Ordering": tag, "Model": m["model"].replace("Instruct", "").rstrip("-"),
               "What": ORDER_DESC.get(tag, "")}
        for ds in ("wildfire-smoke", "aerial-airport"):
            row[f"{ds} mAP"] = _fmt(pd_.get(ds, {}).get("mAP"))
        row["mean mAP"] = _fmt(mean.get("mAP"))
        row["mean [email protected]"] = _fmt(mean.get("mAP50"))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)
    by = {_tag(r["meta"]): r["meta"].get("mean", {}).get("mAP")
          for r in runs if "8b" in r["meta"].get("model", "")}
    _delta_caption(by, "mAP")
    st.caption("Re-run orderings:  `" + RF20_RUN + "`")


def _delta_caption(by, unit):
    """by: {ordering_tag: display_value}. State the ordering effect vs STI."""
    if by.get("STI") is not None:
        parts = [f"{t} {by[t]-by['STI']:+.1f}" for t in ("SIT", "STIT", "SITIT",
                 "SITIT_rev") if by.get(t) is not None]
        if parts:
            st.caption(f"8B ordering Δ vs STI (question-first), {unit}: "
                       + " · ".join(parts) + f"  (STI = {by['STI']:.1f}).")


def _refcoco_section():
    st.subheader("🎯 RefCOCOg (val) — Referring [email protected] by prompt ordering")
    st.caption(
        "Referring-expression grounding on RefCOCOg (umd val): one predicted box per "
        "expression, correct if IoU with the GT box ≥ 0.5 (**[email protected]**). Same "
        "orderings as above; orderings on local **Qwen3-VL-8B**, baseline on served "
        "**Qwen3-VL-30B-A3B**."
    )
    runs = _load("refcocog_*.json")
    if not runs:
        st.info("**No RefCOCOg results yet.** Run:\n\n```\n" + REF_RUN + "\n```")
        return
    runs.sort(key=lambda r: (_order_key(_tag(r["meta"])), r["meta"]["model"]))
    rows = []
    for r in runs:
        m = r["meta"]
        tag = _tag(m)
        rows.append({
            "Ordering": tag, "Model": m["model"].replace("Instruct", "").rstrip("-"),
            "What": ORDER_DESC.get(tag, ""),
            "N": str(m.get("n", "")), "Parsed": str(m.get("parsed", "")),
            "[email protected] (%)": _fmt(m.get("acc"), 100.0),
        })
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)
    by = {_tag(r["meta"]): (r["meta"].get("acc") or 0) * 100
          for r in runs if "8b" in r["meta"].get("model", "")}
    _delta_caption(by, "[email protected] %")
    st.caption("Re-run orderings:  `" + REF_RUN + "`")


def render_detpo_map_page():
    st.title("🛩️ DetPO Detection — RF20 Aerial mAP + RefCOCO (prompt orderings)")
    st.caption(
        "The question-first paradox applied to **localization**: does moving the "
        "question before/after the image, echoing it, or echoing the image change "
        "detection mAP and referring accuracy? Two benchmarks (RF20-VL Aerial COCO "
        "mAP; RefCOCOg val referring [email protected]) across STI / SIT / STIT / SITIT / "
        "SITIT_rev, plus the served Qwen3-VL-30B-A3B baseline."
    )
    _rf20_section()
    st.markdown("---")
    _refcoco_section()
