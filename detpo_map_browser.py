"""
Streamlit page: DetPO Detection — mAP on RF20-VL (all 20 datasets) and referring
accuracy (IoU>=0.5) on RefCOCO / RefCOCO+ / RefCOCOg, for the served Qwen3-VL
model and the prompt orderings STI / SIT / STIT / SITIT / STITI / SITIT_rev
(question-first paradox applied to localization). Read-only; reads JSON written
by the eval scripts in detpo_map/:
    detpo_map/results/rf20_aerial_baseline_<model>.json          (30B baseline)
    detpo_map/results/rf20ds_<dataset>_order-<ORD>_<model>.json  (8B orderings, RF20)
    detpo_map/results/refcocog_val_refacc_<model>.json           (30B baseline)
    detpo_map/results/<dataset>_<split>_order-<ORD>_<model>.json (8B orderings, grounding)

Distinct from the "🟩 RF20" page (that one is the prompt-ordering yes/no
object-presence study); this page is the DetPO localization benchmark (mAP / AP50
/ referring accuracy).
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "detpo_map", "results")

# Canonical row order: 30B baseline first, then the orderings.
ORDER_ROWS = ["baseline", "STI", "SIT", "STIT", "SITIT", "STITI", "SITIT_rev"]
ORDER_DESC = {
    "baseline": "DetPO default (S·T·I, question-first)",
    "STI": "S·T·I — question-first",
    "SIT": "S·I·T — question-last",
    "STIT": "S·T·I·T — question echo",
    "SITIT": "S·I·T·I·T — image echo",
    "STITI": "S·T·I·T·I — exploratory (ends on image, not text)",
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
ORD_ORDER = ["STI", "SIT", "STIT", "SITIT", "STITI", "SITIT_rev"]

# DetPO paper reference numbers (RF20-VL Aerial mAP), Tables 1–2 of the DetPO paper.
# Baseline is the default "class names + instructions" (C+I) prompt.
PAPER_REF_RF20 = [
    {"Model": "Qwen3-VL-8B (C+I)", "Aerial baseline": "7.1",
     "Aerial +DetPO": "8.3", "Aerial +DetPO+VQA": "12.3", "All (baseline)": "11.4"},
    {"Model": "Qwen3-VL-30B-A3B (C+I)", "Aerial baseline": "9.0",
     "Aerial +DetPO": "13.8", "Aerial +DetPO+VQA": "16.1", "All (baseline)": "11.9"},
]

RF20_RUN = ("HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 "
            "/home/grg/anaconda3/envs/qwen-vllm-env/bin/python "
            "detpo_map/ordering_eval_vllm.py --orders STI,SIT,STIT,SITIT,STITI   "
            "# SITIT_rev: detpo_map/ordering_eval.py (HF+hooks)")
GROUNDING_RUN = ("HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 "
                 "/home/grg/anaconda3/envs/qwen-vllm-env/bin/python "
                 "detpo_map/grounding_eval_vllm.py --datasets refcoco,refcoco+ "
                 "--splits testA,testB --orders STI,SIT,STIT,SITIT,STITI   "
                 "# SITIT_rev: detpo_map/ordering_eval.py --benchmark refcoco "
                 "--ref-dataset <ds> --ref-split <split>")


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


# Short display label per model tag as it appears in meta["model"]. Falls
# back to the raw tag for anything unrecognized (e.g. a future model added
# without updating this map) so nothing is ever silently dropped.
MODEL_LABEL = {
    "qwen3-vl-8b": "Qwen3-VL (8B)",
    "gemma-3-27b": "Gemma-3 (27B)",
    "gemma-4-31b": "Gemma-4 (31B)",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B (served baseline)",
}
# Ordered list of the "local ordering sweep" models -- iterate this, not a
# hardcoded tuple, so a newly-added model (e.g. gemma-4-31b) automatically
# gets its own section everywhere instead of needing every call site updated.
MODELS = ["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"]


def _model_label(tag):
    return MODEL_LABEL.get(tag, tag)


def _model_of(meta):
    return meta.get("model", "qwen3-vl-8b")


def _rf20_by_ordering():
    """{model: {ordering: {dataset: mAP}}} from both the aerial ordering files
    and the per-dataset ordering files. Keyed by model as well as ordering --
    Qwen and Gemma both write rf20ds_<dataset>_order-<tag>_<model>.json for the
    same (dataset, ordering), and a plain {ordering: {dataset: ...}} dict would
    silently let one model's value clobber the other's for shared keys."""
    mp, mp50 = {}, {}
    for r in _load("rf20_aerial_order-*.json"):
        m = r["meta"]; o = m["ordering"]; model = _model_of(m)
        for ds, d in m.get("per_dataset", {}).items():
            mp.setdefault(model, {}).setdefault(o, {})[ds] = d.get("mAP")
            mp50.setdefault(model, {}).setdefault(o, {})[ds] = d.get("mAP50")
    for r in _load("rf20ds_*.json"):
        m = r["meta"]; o = m["ordering"]; ds = m["dataset"]; model = _model_of(m)
        mp.setdefault(model, {}).setdefault(o, {})[ds] = m.get("mAP")
        mp50.setdefault(model, {}).setdefault(o, {})[ds] = m.get("mAP50")
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
        "(STI), question-last (SIT), question echo (STIT), image echo (SITIT), the "
        "exploratory STITI (ends on an image, not text), and image echo with the "
        "2nd image block reversed (SITIT_rev). Run on local **Qwen3-VL-8B** (so "
        "SITIT_rev's patch reversal applies)."
    )
    mp, _ = _rf20_by_ordering()
    if not mp:
        st.info("**No RF20 results yet.** Run:\n\n```\n" + RF20_RUN + "\n```")
        return
    active = ["STI", "SIT", "STIT", "SITIT", "STITI", "SITIT_rev"]
    models_present = [m for m in MODELS if m in mp]
    cat_ds = {c: [d for d in RF20_CATS if RF20_CATS[d] == c] for c in CAT_ORDER}
    all_mean_by_model = {}
    coverage_by_model = {}

    for model in models_present:
        mp_m = mp[model]
        present_ord = [o for o in active if o in mp_m]
        st.markdown(f"#### {_model_label(model)}")

        def _cov(o):
            return len(mp_m.get(o, {}))
        cov = " · ".join(f"{o} {_cov(o)}/20" for o in present_ord)
        st.caption("Coverage: " + cov + ".")

        st.markdown("**By super-category — mean mAP** (paper layout)")
        st.caption(
            "Cells show **n/N** whenever a category or the All-mean is averaged "
            "over fewer than the full dataset count for that ordering (e.g. a "
            "still-running sweep) -- an incomplete-coverage mean is NOT "
            "comparable to a complete one, so it's never shown as a bare number."
        )
        rows = []
        for o in present_ord:
            row = {"Ordering": o, "What": ORDER_DESC.get(o, "")}
            allvals = []
            for c in CAT_ORDER:
                vals = [mp_m[o].get(d) for d in cat_ds[c]]
                present = [v for v in vals if v is not None]
                cell = _fmt(_mean(present))
                if present and len(present) < len(vals):
                    cell += f" ({len(present)}/{len(vals)})"
                row[c] = cell
                allvals += present
            n_total = len(RF20_CATS)
            all_cell = _fmt(_mean(allvals))
            if allvals and len(allvals) < n_total:
                all_cell += f" ({len(allvals)}/{n_total})"
            row["All"] = all_cell
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {o: _mean([mp_m[o].get(d) for d in RF20_CATS]) for o in present_ord}
        cov_o = {o: (sum(1 for d in RF20_CATS if mp_m[o].get(d) is not None),
                    len(RF20_CATS)) for o in present_ord}
        all_mean_by_model[model] = by
        coverage_by_model[model] = cov_o
        _delta_caption(by, "mAP (All-mean)", coverage=cov_o)

        with st.expander(f"Per-dataset mAP breakdown — {_model_label(model)} "
                         "(20 datasets × orderings)"):
            drows = []
            for c in CAT_ORDER:
                for ds in cat_ds[c]:
                    row = {"Category": c, "Dataset": ds}
                    for o in present_ord:
                        row[o] = _fmt(mp_m[o].get(ds))
                    drows.append(row)
            st.dataframe(pd.DataFrame(drows).astype(str), hide_index=True,
                         use_container_width=True)
        st.markdown("")

    # --- Cross-model comparison (All-mean mAP by ordering) ---
    if len(models_present) > 1:
        st.markdown("**Cross-model — All-mean mAP by ordering**")
        comp_rows = []
        for o in active:
            row = {"Ordering": o, "What": ORDER_DESC.get(o, "")}
            for model in models_present:
                cell = _fmt(all_mean_by_model[model].get(o))
                cov = coverage_by_model[model].get(o)
                if cov and cov[0] < cov[1]:
                    cell += f" ({cov[0]}/{cov[1]})"
                row[_model_label(model)] = cell
            if any(all_mean_by_model[m].get(o) is not None for m in models_present):
                comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows).astype(str), hide_index=True,
                     use_container_width=True)
        st.caption(
            "Gemma-3-27B has no dedicated grounding/localization training the way "
            "Qwen2/2.5/3-VL does (it is a general chat-tuned multimodal model), so "
            "its raw detection boxes are markedly less precise even after "
            "correcting for its coordinate convention (fixed 896×896 canvas per "
            "its own image processor, vs. Qwen's 0-1000 normalized convention) — "
            "expect much lower absolute mAP, not a bug."
        )

    # --- DetPO paper reference (Aerial column) ---
    st.markdown("**DetPO paper reference — RF20-VL Aerial mAP** (Tables 1–2)")
    st.dataframe(pd.DataFrame(PAPER_REF_RF20), hide_index=True,
                 use_container_width=True)
    st.caption(
        "Published DetPO numbers for context. Baseline = default class-names + "
        "instructions (C+I) prompt; **+DetPO** = optimized prompt; **+VQA** = with "
        "VQA-Score confidence re-ranking. \"All\" is the 20-dataset mean; the paper's "
        "8B baseline is 11.4 All-mean / 7.1 Aerial. Compare against the STI row's "
        "All-mean above."
    )
    st.caption("Re-run orderings (dataset-split across GPUs, resumable):  `"
               + RF20_RUN + "`")


def _delta_caption(by, unit, coverage=None):
    """by: {ordering_tag: display_value}. coverage: optional {tag: (n, total)} --
    flags a delta term with "(n/total)" when that ordering's mean was averaged
    over fewer than the full dataset/item count, so a still-partial mean is never
    silently presented as equivalent to a complete one. State the ordering
    effect vs STI."""
    if by.get("STI") is not None:
        def _term(t):
            d = f"{by[t]-by['STI']:+.1f}"
            if coverage and coverage.get(t) and coverage[t][0] < coverage[t][1]:
                n, total = coverage[t]
                d += f" ({n}/{total})"
            return f"{t} {d}"
        parts = [_term(t) for t in ("SIT", "STIT", "SITIT", "STITI", "SITIT_rev")
                if by.get(t) is not None]
        if parts:
            st.caption(f"8B ordering Δ vs STI (question-first), {unit}: "
                       + " · ".join(parts) + f"  (STI = {by['STI']:.1f}).")


def _refcocog_section():
    st.subheader("🎯 RefCOCOg (val) — Referring accuracy (IoU ≥ 0.5) by prompt ordering")
    st.caption(
        "Referring-expression grounding on RefCOCOg (umd val, 2573 expressions): "
        "one predicted box per expression, correct if IoU with the GT box ≥ 0.5 "
        "(**referring accuracy**). Orderings on local **Qwen3-VL-8B**, baseline on "
        "served **Qwen3-VL-30B-A3B**."
    )
    runs = _load("refcocog_val_*.json")
    if not runs:
        st.info("**No RefCOCOg results yet.**")
        return
    runs.sort(key=lambda r: (_order_key(_tag(r["meta"])), _model_of(r["meta"])))
    # Separate table PER MODEL -- results for different models share the same
    # ordering keys, so merging them into one dataframe would make it easy to
    # misread which row belongs to which model. Never combine.
    models_present = [m for m in MODELS if any(_model_of(r["meta"]) == m for r in runs)]
    for model in models_present:
        mruns = [r for r in runs if _model_of(r["meta"]) == model]
        if len(models_present) > 1:
            st.markdown(f"**{_model_label(model)}**")
        rows = []
        for r in mruns:
            m = r["meta"]
            tag = _tag(m)
            rows.append({
                "Ordering": tag, "What": ORDER_DESC.get(tag, ""),
                "N": str(m.get("n", "")), "Parsed": str(m.get("parsed", "")),
                "Ref acc IoU>=0.5 (%)": _fmt(m.get("acc"), 100.0),
            })
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                     use_container_width=True)
        by = {_tag(r["meta"]): (r["meta"].get("acc") or 0) * 100 for r in mruns}
        if by:
            _delta_caption(by, "referring acc (IoU>=0.5) %")

    # Focused comparison: 30B baseline vs each 8B-scale model's orderings.
    # Deliberately cross-model (every row is labeled with its model name) --
    # this is a comparison table by design, not a merge.
    st.markdown("**Baseline (30B) vs orderings, by model**")
    base = None
    for r in runs:
        if _tag(r["meta"]) == "baseline":
            base = (r["meta"].get("acc") or 0) * 100
    comp = [{"Row": "baseline (Qwen3-VL-30B, S·T·I)",
             "Ref acc IoU>=0.5 (%)": _fmt(base), "Δ vs baseline": "—"}]
    for model in models_present:
        acc = {_tag(r["meta"]): (r["meta"].get("acc") or 0) * 100
              for r in runs if _model_of(r["meta"]) == model}
        for t in ("STI", "SIT", "STIT", "SITIT", "STITI", "SITIT_rev"):
            v = acc.get(t)
            if v is None:
                continue
            comp.append({"Row": f"{t}  ·  {ORDER_DESC[t]}  ({_model_label(model)})",
                         "Ref acc IoU>=0.5 (%)": _fmt(v),
                         "Δ vs baseline": "—" if base is None else f"{v - base:+.1f}"})
    st.dataframe(pd.DataFrame(comp).astype(str), hide_index=True,
                 use_container_width=True)
    st.caption(
        "Every 8B ordering beats the DetPO-default question-first prompt. Note "
        "the DetPO paper reports **no RefCOCO metric** (RefCOCO appears only "
        "qualitatively in its intro; its quantitative results are RF20-VL and "
        "LVIS), so there is no published baseline to overlay here — the baseline "
        "shown is our own served 30B run."
    )


def _refcoco_variant_section(dataset, title, caption_extra=""):
    """Generic renderer for refcoco / refcoco+ (testA + testB splits, no 30B
    baseline was run for these -- only the 8B ordering sweep)."""
    st.subheader(title)
    st.caption(
        f"Referring-expression grounding on **{dataset}** (testA + testB, the "
        "standard reporting splits), same referring-accuracy (IoU ≥ 0.5) metric "
        "and orderings as RefCOCOg, on local **Qwen3-VL-8B**. " + caption_extra
    )
    runs = _load(f"{dataset}_test*_order-*.json")
    if not runs:
        st.info(f"**No {dataset} results yet.** Run:\n\n```\n" + GROUNDING_RUN + "\n```")
        return
    by_split = {}
    for r in runs:
        m = r["meta"]
        by_split.setdefault(m["split"], []).append(m)
    for split in sorted(by_split):
        st.markdown(f"**{dataset} / {split}**")
        ms = sorted(by_split[split], key=lambda m: (_order_key(_tag(m)), _model_of(m)))
        # Separate table per model -- never merge different models' rows.
        models_present = [mo for mo in MODELS if any(_model_of(m) == mo for m in ms)]
        for model in models_present:
            mms = [m for m in ms if _model_of(m) == model]
            if len(models_present) > 1:
                st.caption(f"**{_model_label(model)}**")
            rows = [{"Ordering": _tag(m), "What": ORDER_DESC.get(_tag(m), ""),
                    "N": str(m.get("n", "")), "Parsed": str(m.get("parsed", "")),
                    "Ref acc IoU>=0.5 (%)": _fmt(m.get("acc"), 100.0)} for m in mms]
            st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                         use_container_width=True)
            by = {_tag(m): (m.get("acc") or 0) * 100 for m in mms}
            if by:
                _delta_caption(by, f"{split} referring acc %")
    st.caption("Re-run:  `" + GROUNDING_RUN + "`")


def render_detpo_map_page():
    st.title("🛩️ DetPO Detection — RF20 mAP + RefCOCO/RefCOCO+/RefCOCOg (prompt orderings)")
    st.caption(
        "The question-first paradox applied to **localization**: does moving the "
        "question before/after the image, echoing it, or echoing the image change "
        "detection mAP and referring accuracy? RF20-VL (20 datasets, COCO mAP) and "
        "three referring-grounding benchmarks (RefCOCO, RefCOCO+, RefCOCOg — "
        "referring accuracy at IoU ≥ 0.5) across STI / SIT / STIT / SITIT / STITI / "
        "SITIT_rev, plus the served Qwen3-VL-30B-A3B baseline where available."
    )
    _prompt_doc()
    _rf20_section()
    st.markdown("---")
    _refcocog_section()
    st.markdown("---")
    _refcoco_variant_section("refcoco", "🎯 RefCOCO — Referring accuracy by prompt ordering")
    st.markdown("---")
    _refcoco_variant_section(
        "refcoco+", "🎯 RefCOCO+ — Referring accuracy by prompt ordering",
        "RefCOCO+ forbids location words in expressions (appearance-only), so "
        "it's generally harder than RefCOCO.")


def _prompt_doc():
    """Render the exact prompt templates + ordering construction (detpo_map/PROMPTS.md)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "detpo_map", "PROMPTS.md")
    with st.expander("📝 Prompt templates & ordering construction (exact text)"):
        if os.path.exists(path):
            st.markdown(open(path).read())
        else:
            st.info("Prompt documentation file not found: detpo_map/PROMPTS.md")
