"""
Streamlit page: DetPO Detection — mAP on the RF20-VL *Aerial* domain and referring
accuracy (IoU>=0.5) on RefCOCOg (val), for the served Qwen3-VL model, and the prompt
orderings STI / SIT / STIT / SITIT / SITIT_rev (question-first paradox applied to
localization). Read-only; reads JSON written by the eval scripts in detpo_map/:
    detpo_map/results/rf20_aerial_baseline_<model>.json          (30B baseline)
    detpo_map/results/rf20_aerial_order-<ORD>_<model>.json        (8B orderings)
    detpo_map/results/refcocog_val_refacc_<model>.json            (30B baseline)
    detpo_map/results/refcocog_val_order-<ORD>_<model>.json       (8B orderings)

Distinct from the "🟩 RF20" page (that one is the prompt-ordering yes/no
object-presence study); this page is the DetPO localization benchmark (mAP / AP50).
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

# RF20-VL 20-dataset subset grouped into the paper's super-categories.
RF20_CATS = {
    "wildfire-smoke": "Aerial", "aerial-airport": "Aerial",
    "paper-parts": "Document", "all-elements": "Document",
    "trail-camera": "Flora/Fauna", "gwhd2021": "Flora/Fauna",
    "wb-prova": "Flora/Fauna", "aquarium-combined": "Flora/Fauna",
    "recode-waste": "Industrial", "defect-detection": "Industrial",
    "water-meter": "Industrial",
    "dentalai": "Lab Imaging", "x-ray-id": "Lab Imaging",
    "orionproducts": "Misc", "the-dreidel-project": "Misc", "soda-bottles": "Misc",
    "flir-camera-objects": "Misc", "new-defects-in-wood": "Misc",
    "lacrosse-object-detection": "Sport", "actions": "Sport",
}
CAT_ORDER = ["Aerial", "Document", "Flora/Fauna", "Industrial", "Lab Imaging",
             "Misc", "Sport"]
ORD_ORDER = ["STI", "SIT", "STIT", "SITIT", "SITIT_rev"]

# DetPO paper reference numbers (RF20-VL Aerial mAP), Tables 1–2 of the DetPO paper.
# Baseline is the default "class names + instructions" (C+I) prompt.
PAPER_REF_RF20 = [
    {"Model": "Qwen3-VL-8B (C+I)", "Aerial baseline": "7.1",
     "Aerial +DetPO": "8.3", "Aerial +DetPO+VQA": "12.3", "All (baseline)": "11.4"},
    {"Model": "Qwen3-VL-30B-A3B (C+I)", "Aerial baseline": "9.0",
     "Aerial +DetPO": "13.8", "Aerial +DetPO+VQA": "16.1", "All (baseline)": "11.9"},
]

RF20_RUN = ("CUDA_VISIBLE_DEVICES=0 HF_HOME=/data2/hf_cache "
            "/home/grg/anaconda3/envs/logitlens/bin/python "
            "detpo_map/ordering_eval.py --benchmark rf20 "
            "--orders STI,SIT,STIT,SITIT,SITIT_rev --datasets <ds1,ds2,...>")
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


def _rf20_by_ordering():
    """Unified {ordering: {dataset: mAP}} from both the aerial ordering files and
    the per-dataset ordering files."""
    mp, mp50 = {}, {}
    for r in _load("rf20_aerial_order-*.json"):
        m = r["meta"]; o = m["ordering"]
        for ds, d in m.get("per_dataset", {}).items():
            mp.setdefault(o, {})[ds] = d.get("mAP")
            mp50.setdefault(o, {})[ds] = d.get("mAP50")
    for r in _load("rf20ds_*.json"):
        m = r["meta"]; o = m["ordering"]; ds = m["dataset"]
        mp.setdefault(o, {})[ds] = m.get("mAP")
        mp50.setdefault(o, {})[ds] = m.get("mAP50")
    return mp, mp50


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rf20_section():
    st.subheader("🛩️ RF20-VL — DetPO detection mAP by prompt ordering (all 20 domains)")
    st.caption(
        "Few-shot object **detection** across the full **Roboflow20-VL** benchmark "
        "(20 datasets in 7 super-categories) through the DetPO setup, **multi-class "
        "prompting** (all classes at once, as in the DetPO paper), scored with COCO "
        "**mAP (IoU .50 to .95)** on each test split. Each row re-orders the same "
        "system (**S**), task/question (**T**) and image (**I**) tokens: question-first "
        "(STI), question-last (SIT), question echo (STIT), image echo (SITIT) and image "
        "echo with the 2nd image block reversed (SITIT_rev). Run on local "
        "**Qwen3-VL-8B** (so SITIT_rev's patch reversal applies)."
    )
    mp, _ = _rf20_by_ordering()
    if not mp:
        st.info("**No RF20 results yet.** Run:\n\n```\n" + RF20_RUN + "\n```")
        return
    present_ord = [o for o in ORD_ORDER if o in mp]

    # coverage note
    ndone = len({ds for o in mp for ds in mp[o]})
    st.caption(f"Coverage: **{ndone}/20** datasets have results so far "
               f"(table fills in as the run completes).")

    # --- Super-category summary (paper Table-1 layout) ---
    st.markdown("**By super-category — mean mAP** (paper layout)")
    cat_ds = {c: [d for d in RF20_CATS if RF20_CATS[d] == c] for c in CAT_ORDER}
    rows = []
    for o in present_ord:
        row = {"Ordering": o, "What": ORDER_DESC.get(o, "")}
        allvals = []
        for c in CAT_ORDER:
            vals = [mp[o].get(d) for d in cat_ds[c]]
            row[c] = _fmt(_mean(vals))
            allvals += [v for v in vals if v is not None]
        row["All"] = _fmt(_mean(allvals))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)
    by = {o: _mean([mp[o].get(d) for d in RF20_CATS]) for o in present_ord}
    _delta_caption(by, "mAP (All-mean)")

    # --- Per-dataset breakdown (expandable) ---
    with st.expander("Per-dataset mAP breakdown (20 datasets × orderings)"):
        drows = []
        for c in CAT_ORDER:
            for ds in cat_ds[c]:
                row = {"Category": c, "Dataset": ds}
                for o in present_ord:
                    row[o] = _fmt(mp[o].get(ds))
                drows.append(row)
        st.dataframe(pd.DataFrame(drows).astype(str), hide_index=True,
                     use_container_width=True)

    # --- DetPO paper reference (Aerial column) ---
    st.markdown("**DetPO paper reference — RF20-VL Aerial mAP** (Tables 1–2)")
    st.dataframe(pd.DataFrame(PAPER_REF_RF20), hide_index=True,
                 use_container_width=True)
    st.caption(
        "Published DetPO numbers for context. Baseline = default class-names + "
        "instructions (C+I) prompt; **+DetPO** = optimized prompt; **+VQA** = with "
        "VQA-Score confidence re-ranking. \"All\" is the 20-dataset mean; the paper's "
        "8B baseline is 11.4 All-mean / 7.1 Aerial. Compare against the STI row's "
        "All-mean above once coverage reaches 20/20."
    )
    st.caption("Re-run orderings (dataset-split across GPUs, resumable):  `"
               + RF20_RUN + "`")


def _delta_caption(by, unit):
    """by: {ordering_tag: display_value}. State the ordering effect vs STI."""
    if by.get("STI") is not None:
        parts = [f"{t} {by[t]-by['STI']:+.1f}" for t in ("SIT", "STIT", "SITIT",
                 "SITIT_rev") if by.get(t) is not None]
        if parts:
            st.caption(f"8B ordering Δ vs STI (question-first), {unit}: "
                       + " · ".join(parts) + f"  (STI = {by['STI']:.1f}).")


def _refcoco_section():
    st.subheader("🎯 RefCOCOg (val) — Referring accuracy (IoU ≥ 0.5) by prompt ordering")
    st.caption(
        "Referring-expression grounding on RefCOCOg (umd val): one predicted box per "
        "expression, correct if IoU with the GT box ≥ 0.5 (**referring accuracy**). Same "
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
            "Ref acc IoU>=0.5 (%)": _fmt(m.get("acc"), 100.0),
        })
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)
    by = {_tag(r["meta"]): (r["meta"].get("acc") or 0) * 100
          for r in runs if "8b" in r["meta"].get("model", "")}
    _delta_caption(by, "referring acc (IoU>=0.5) %")

    # Focused comparison: 30B baseline vs the 8B orderings (STI/SIT baselines +
    # the three echo orderings) requested.
    acc = {_tag(r["meta"]): (r["meta"].get("acc") or 0) * 100 for r in runs}
    base = acc.get("baseline")
    st.markdown("**Baseline (30B) vs 8B orderings** — STI / SIT baselines and "
                "STIT / SITIT / SITIT_rev echoes")
    comp = []
    comp.append({"Row": "baseline (Qwen3-VL-30B, S·T·I)",
                 "Ref acc IoU>=0.5 (%)": _fmt(base), "Δ vs baseline": "—"})
    for t in ("STI", "SIT", "STIT", "SITIT", "SITIT_rev"):
        v = acc.get(t)
        comp.append({"Row": f"{t}  ·  {ORDER_DESC[t]}  (8B)",
                     "Ref acc IoU>=0.5 (%)": _fmt(v),
                     "Δ vs baseline": "—" if (v is None or base is None)
                     else f"{v - base:+.1f}"})
    st.dataframe(pd.DataFrame(comp).astype(str), hide_index=True,
                 use_container_width=True)
    st.caption(
        "All three echo orderings beat the DetPO-default question-first prompt; "
        "**STIT** (question echo) is strongest. Note the DetPO paper reports **no "
        "RefCOCO metric** (RefCOCO appears only qualitatively in its intro; its "
        "quantitative results are RF20-VL and LVIS), so there is no published "
        "baseline to overlay here — the baseline shown is our own served 30B run."
    )
    st.caption("Re-run orderings:  `" + REF_RUN + "`")


def render_detpo_map_page():
    st.title("🛩️ DetPO Detection — RF20 Aerial mAP + RefCOCO (prompt orderings)")
    st.caption(
        "The question-first paradox applied to **localization**: does moving the "
        "question before/after the image, echoing it, or echoing the image change "
        "detection mAP and referring accuracy? Two benchmarks (RF20-VL Aerial COCO "
        "mAP; RefCOCOg val referring accuracy at IoU ≥ 0.5) across STI / SIT / STIT / SITIT / "
        "SITIT_rev, plus the served Qwen3-VL-30B-A3B baseline."
    )
    _rf20_section()
    st.markdown("---")
    _refcoco_section()
