"""
Streamlit page: RF20 — object-presence yes/no across 20 RF100-VL detection domains,
under IST / STI / STIT. Purely additive and read-only; reads
rf20/results/<model>__<order>__results.json written by rf20_eval.py, including the
per-dataset and per-category breakdowns stored in meta.
"""

import json
import os

import pandas as pd
import streamlit as st

from rf20_eval import RF20, CATEGORIES

RESULTS_DIR = "./rf20/results"
MODELS = [("qwen3-vl-8b", "Qwen3-VL (8B)"), ("gemma-3-27b", "Gemma 3 (27B)")]
ORDERS = [("IST", "IST — Image · System · Task"),
          ("STI", "STI — System · Task · Image"),
          ("STIT", "STIT — System · Task · Image · Task"),
          ("SITIT", "SITIT — System · Image · Task · Image · Task"),
          ("SITIT_echo2quarter", "SITIT (2nd¼) — echoed occurrence at 0.25x res")]
OKEYS = [o for o, _ in ORDERS]
RUN_CMD = ("CUDA_VISIBLE_DEVICES=0 python rf20_eval.py "
           "--model qwen3-vl-8b --order IST,STI,STIT")


def _meta(model, order):
    p = os.path.join(RESULTS_DIR, f"{model}__{order}__results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["meta"]
    except Exception:
        return None


def _pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def _overall_row(meta):
    o = meta.get("overall", {})
    return {
        "Accuracy": _pct(o.get("acc")), "F1": _pct(o.get("f1")),
        "Precision": _pct(o.get("precision")), "Recall": _pct(o.get("recall")),
        "Yes-ratio": _pct(o.get("yes_ratio")), "# Q": str(o.get("n", 0)),
    }


def _category_df(metas):
    rows = []
    for cat in CATEGORIES:
        row = {"Category": cat}
        for o in OKEYS:
            m = metas.get(o)
            d = (m or {}).get("by_category", {}).get(cat) if m else None
            row[o + " F1"] = f"{d['f1']*100:.1f}" if d else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_df(metas):
    """One row per RF20 dataset: F1 under each ordering + the STIT recovery."""
    rows = []
    for ds in sorted(RF20, key=lambda d: (RF20[d], d)):
        f1 = {}
        n = None
        for o in OKEYS:
            m = metas.get(o)
            d = (m or {}).get("by_dataset", {}).get(ds) if m else None
            f1[o] = d["f1"] * 100 if d else None
            if d and n is None:
                n = d.get("n")
        rec = (f1["STIT"] - f1["STI"]) if (f1.get("STIT") is not None
                                           and f1.get("STI") is not None) else None
        rows.append({
            "Dataset": ds, "Category": RF20[ds],
            "IST F1": None if f1["IST"] is None else round(f1["IST"], 1),
            "STI F1": None if f1["STI"] is None else round(f1["STI"], 1),
            "STIT F1": None if f1["STIT"] is None else round(f1["STIT"], 1),
            "Δ STIT−STI": None if rec is None else round(rec, 1),
            "# Q": n or 0,
        })
    return pd.DataFrame(rows)


def render_rf20_page():
    st.subheader("🟩 RF20 — Object Presence across 20 Roboflow domains")
    st.caption(
        "Object-hallucination generalization test: for each image we ask "
        "\"Is there a <class> in the image?\" (yes/no) across **20 out-of-distribution "
        "RF100-VL domains** (industrial, medical x-ray/dental, aerial, aquatic, "
        "documents, sport, …), under the same IST / STI / STIT / SITIT ordering. "
        "Headline metric is **F1**; **Δ STIT−STI** is the STIT recovery."
    )

    present = [(m, lab) for m, lab in MODELS if any(_meta(m, o) for o in OKEYS)]
    if not present:
        st.info(f"**No RF20 results yet.** Run (resumable):\n\n`{RUN_CMD}`")
        return

    for model, label in present:
        metas = {o: _meta(model, o) for o in OKEYS}
        st.markdown(f"### {label}")

        st.markdown("**Overall (all 20 datasets pooled)**")
        st.table({lab: _overall_row(metas[o]) for o, lab in ORDERS if metas.get(o)})
        if metas.get("IST") and metas.get("STI"):
            f1 = lambda o: metas[o]["overall"]["f1"] * 100
            d = f1("IST") - f1("STI")
            msg = f"F1 gap (IST − STI): **{d:+.1f} pts**"
            if metas.get("STIT"):
                msg += (f". STIT **{f1('STIT'):.1f}%** "
                        f"(vs STI **{f1('STIT')-f1('STI'):+.1f}** · "
                        f"vs IST **{f1('STIT')-f1('IST'):+.1f}**)")
            st.caption(msg)
        if metas.get("SITIT"):
            f1 = lambda o: metas[o]["overall"]["f1"] * 100
            msg = f"SITIT (image echo) F1: **{f1('SITIT'):.1f}%**"
            if metas.get("SITIT_echo2quarter"):
                dd = f1("SITIT_echo2quarter") - f1("SITIT")
                msg += (f"  ·  SITIT with the **echoed occurrence at 0.25x res**: "
                        f"**{f1('SITIT_echo2quarter'):.1f}%** ({dd:+.1f} vs full-res SITIT)")
            st.caption(msg)

        st.markdown("**By super-category (F1)**")
        st.dataframe(_category_df(metas), hide_index=True, use_container_width=True)

        st.markdown("**By dataset (F1) — sorted by category**")
        st.caption("A large positive **Δ STIT−STI** = domains where re-stating the "
                   "question after the image most recovers the STI drop.")
        st.dataframe(_dataset_df(metas), hide_index=True, use_container_width=True)
        st.markdown("---")

    st.caption(f"Re-run / extend with:  `{RUN_CMD}`  ·  data: local RF100-VL (20-domain subset)")
