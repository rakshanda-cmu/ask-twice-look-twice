"""
Streamlit page: Cross-Dataset Summary — every model × dataset × ordering in one
place, plus an interactive 3D drill-down. Purely additive and read-only; it just
reads the result JSONs already written by the three separate runners:

    naturalbench/results/<model>__<order>__results.json   (g_acc / pair_acc)
    pope/results/<model>__<order>__results.json           (overall.f1 / acc)
    winoground/results/<model>__<order>__results.json      (overall.acc / group_acc)

Nothing here writes or mutates results.
"""

import json
import os

import pandas as pd
import streamlit as st

ORDERS = ["IST", "SIT", "STI", "STIT", "SITIT", "SITIT_rev"]
MODELS = [("qwen3-vl-8b", "Qwen3-VL (8B)"), ("gemma-3-27b", "Gemma 3 (27B)")]
# (key, label, results_dir, headline_metric_key, headline_label)
DATASETS = [
    ("naturalbench", "NaturalBench", "./naturalbench/results", "g_acc", "Group-Acc"),
    ("pope", "POPE", "./pope/results", "f1", "F1"),
    ("winoground", "Winoground", "./winoground/results", "group_acc", "Group-Acc"),
    ("rf20", "RF20", "./rf20/results", "f1", "F1"),
]


def _read(dataset_key, results_dir, hkey, model, order):
    """Return {'headline': float, 'acc': float} for one (dataset,model,order) or None."""
    p = os.path.join(results_dir, f"{model}__{order}__results.json")
    if not os.path.exists(p):
        return None
    try:
        meta = json.load(open(p))["meta"]
    except Exception:
        return None
    if dataset_key == "naturalbench":               # metrics live at top level
        return {"headline": meta.get(hkey), "acc": meta.get("pair_acc")}
    o = meta.get("overall", {})                      # pope / winoground
    return {"headline": o.get(hkey), "acc": o.get("acc")}


def _collect():
    """data[(dkey,dlabel,hlabel,model,mlabel)][order] = {'headline','acc'}."""
    data = {}
    for dkey, dlabel, ddir, hkey, hlabel in DATASETS:
        for model, mlabel in MODELS:
            series = {}
            for o in ORDERS:
                r = _read(dkey, ddir, hkey, model, o)
                if r and r.get("headline") is not None:
                    series[o] = r
            if series:
                data[(dkey, dlabel, hlabel, model, mlabel)] = series
    return data


def _fmt(x):
    return "—" if x is None else f"{x*100:.1f}%"


def _summary_table(data):
    rows = []
    for (dkey, dlabel, hlabel, model, mlabel), series in data.items():
        get = lambda o: (series.get(o) or {}).get("headline")
        ist, sit, sti, stit, sitit, sititr = (get("IST"), get("SIT"), get("STI"),
                                              get("STIT"), get("SITIT"), get("SITIT_rev"))
        d_pen = (ist - sti) * 100 if (ist is not None and sti is not None) else None
        d_rec = (stit - sti) * 100 if (stit is not None and sti is not None) else None
        rows.append({
            "Dataset": dlabel, "Model": mlabel, "Metric": hlabel,
            "IST": _fmt(ist), "SIT": _fmt(sit), "STI": _fmt(sti),
            "STIT": _fmt(stit), "SITIT": _fmt(sitit), "SITIT-rev": _fmt(sititr),
            "Δ IST−STI (penalty)": "—" if d_pen is None else f"{d_pen:+.1f}",
            "Δ STIT−STI (recovery)": "—" if d_rec is None else f"{d_rec:+.1f}",
        })
    return pd.DataFrame(rows)


def _plot_bars(data, use_acc):
    """Grouped bar chart: one cluster per (dataset · model), three bars (IST/STI/
    STIT) per cluster, so the ordering effect is the within-cluster bar pattern."""
    import plotly.graph_objects as go
    ocolor = {"IST": "#4c78a8", "SIT": "#b279a2", "STI": "#f58518",
              "STIT": "#54a24b", "SITIT": "#e45756", "SITIT_rev": "#72b7b2"}

    series_keys = list(data.keys())
    xlabels = [f"{d} · {m}" for (_, d, _, _, m) in series_keys]
    fig = go.Figure()
    for o in ORDERS:
        ys, txt, lab = [], [], []
        for k in series_keys:
            _, dlabel, hlabel, _, mlabel = k
            v = (data[k].get(o) or {}).get("acc" if use_acc else "headline")
            ys.append(v * 100 if v is not None else None)
            lab.append(f"{v*100:.1f}" if v is not None else "")
            mname = "Acc" if use_acc else hlabel
            txt.append(f"{dlabel} · {mlabel}<br>{o}: {v*100:.1f}% ({mname})"
                       if v is not None else f"{dlabel} · {mlabel}<br>{o}: —")
        fig.add_trace(go.Bar(
            name=o, x=xlabels, y=ys, marker_color=ocolor[o],
            text=lab, textposition="outside", cliponaxis=False,
            hovertext=txt, hoverinfo="text",
        ))
    fig.update_layout(
        barmode="group", height=520, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Accuracy (%)" if use_acc else "Headline metric (%)",
        xaxis_title="", legend_title="Ordering", bargap=0.25, bargroupgap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_summary_page():
    st.subheader("📊 Cross-Dataset Summary — Models × Datasets × Ordering")
    st.caption(
        "Every model, dataset and prompt ordering (IST / STI / STIT) in one place. "
        "Each dataset reports its headline metric — **NaturalBench & Winoground: "
        "Group-Acc**, **POPE & RF20: F1** — over the same IST/STI/STIT manipulation. "
        "**Δ IST−STI** is the ordering penalty; **Δ STIT−STI** is the STIT recovery. "
        "Higher = better in every column."
    )

    data = _collect()
    if not data:
        st.info("No results found yet under naturalbench/ , pope/ or winoground/ results dirs.")
        return

    df = _summary_table(data)
    st.markdown("#### Summary table")
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.caption(
        "Reading it: a **positive Δ IST−STI** means STI (task-before-image) hurts — "
        "the ordering effect. A **positive Δ STIT−STI** means re-stating the task after "
        "the image recovers it — the STIT effect."
    )

    # ── grouped bar comparison ─────────────────────────────────────────────────
    st.markdown("#### Bar comparison")
    use_acc = st.radio(
        "Y-axis metric",
        ["Accuracy (comparable across datasets)", "Headline metric (per dataset)"],
        horizontal=True, key="summary_bar_metric",
    ).startswith("Accuracy")
    st.caption(
        "One cluster per (dataset · model); the three bars are IST / STI / STIT. "
        "The ordering effect is the within-cluster pattern — STI (orange) dipping "
        "below IST (blue), and STIT (green) recovering. Hover for exact values."
    )
    st.plotly_chart(_plot_bars(data, use_acc), use_container_width=True)

    with st.expander("Which result files feed this page"):
        for (dkey, dlabel, hlabel, model, mlabel), series in data.items():
            st.caption(f"`{dlabel}` · `{model}` → orders present: "
                       f"{', '.join(series.keys())}")
