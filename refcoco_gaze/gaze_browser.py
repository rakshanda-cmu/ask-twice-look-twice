"""
Streamlit page: does the model's Grad-CAM visual-attention saliency track human
gaze scanpaths recorded on the SAME (image, referring-expression) pairs, and
does that alignment change across the STI/SIT/STIT/SITIT prompt orderings?
Read-only; reads JSON written by refcoco_gaze/gaze_eval_gradcam.py.
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER_LIST = ["STI", "SIT", "STIT", "SITIT"]
ORDER_DESC = {
    "STI": "S·T·I — question-first", "SIT": "S·I·T — question-last",
    "STIT": "S·T·I·T — question echo", "SITIT": "S·I·T·I·T — image echo",
}
RUN_CMD = ("CUDA_VISIBLE_DEVICES=0 HF_HOME=/data2/hf_cache "
          "/home/grg/anaconda3/envs/logitlens/bin/python "
          "refcoco_gaze/gaze_eval_gradcam.py --orders STI,SIT,STIT,SITIT")


def _load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def render_gaze_page():
    st.title("👁️ RefCOCO-Gaze — does prompt ordering change how *human-like* the model's attention is?")
    st.caption(
        "A different kind of test than accuracy: instead of asking whether the "
        "model gets the answer right, this asks whether the model **looks at the "
        "same places a human would** while resolving the same referring "
        "expression. Uses RefCOCO-Gaze (Mondal et al., ECCV 2024) — 869 human "
        "eye-tracking scanpaths recorded while 220 participants heard a RefCOCO "
        "referring expression and looked for the target object."
    )
    with st.expander("📐 Method (coordinate transform, saliency signal, metric)"):
        st.markdown(
            "**Coordinate transform** (empirically validated to <3px total error "
            "on 29/29 spot checks): fixations and the target box are recorded in "
            "the 1680×1050 eye-tracker display canvas, which is the original COCO "
            "image letterboxed (\"contain\" fit, centered):\n\n"
            "```\nscale = min(1680/ow, 1050/oh)\n"
            "ox, oy = (1680-ow*scale)/2, (1050-oh*scale)/2\n"
            "orig_xy = (display_xy - (ox,oy)) / scale\n```\n\n"
            "**Saliency signal:** reuses this repo's own validated Grad-CAM "
            "recipe (`attn_heatmap_gen.py`) — raw answer→image attention is "
            "attention-sink-dominated and does **not** localize the queried "
            "object, so we backprop the model's own predicted token instead. "
            "Adapted for referring/grounding: the model generates a bbox JSON "
            "greedily; we locate the **first token that introduces a coordinate "
            "digit** after the `\"bbox_2d\": [` marker — the moment the model "
            "commits to a location, the direct analog of a human fixation "
            "landing on the target — then backprop that token's logit through a "
            "second grad-enabled forward pass. Grad-CAM = "
            "ReLU(Σ grad·activation) over image-token hidden states, averaged "
            "over layers.\n\n"
            "**Metric:** Normalized Scanpath Saliency (NSS) — z-score the "
            "Grad-CAM grid, sample its value at each human fixation's mapped "
            "grid cell, average over fixations. **Higher = the model's attention "
            "matches where humans looked.** Secondary: does the grid's argmax "
            "cell fall inside the target's box (direct analog of humans' "
            "final-fixation-in-bbox)."
        )

    runs = _load("gaze_order-*.json")
    if not runs:
        st.info("**No results yet.** Run:\n\n```\n" + RUN_CMD + "\n```")
        return
    runs.sort(key=lambda r: ORDER_LIST.index(r["meta"]["ordering"])
              if r["meta"]["ordering"] in ORDER_LIST else 99)

    rows = []
    for r in runs:
        m = r["meta"]
        rows.append({
            "Ordering": m["ordering"], "What": ORDER_DESC.get(m["ordering"], ""),
            "N": str(m.get("n", "")),
            "Mean NSS": _fmt(m.get("mean_nss")),
            "Argmax-in-bbox (%)": _fmt((m.get("argmax_in_bbox_rate") or 0) * 100, 1),
        })
    st.subheader("Grad-CAM saliency vs. human gaze, by prompt ordering")
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True,
                 use_container_width=True)

    by = {r["meta"]["ordering"]: r["meta"].get("mean_nss") for r in runs}
    if by.get("STI") is not None:
        parts = [f"{t} {by[t]-by['STI']:+.3f}" for t in ("SIT", "STIT", "SITIT")
                 if by.get(t) is not None]
        if parts:
            st.caption("Δ mean NSS vs STI (question-first): " + " · ".join(parts)
                       + f"  (STI = {by['STI']:.3f}).")

    best = max(rows, key=lambda r: float(r["Mean NSS"]) if r["Mean NSS"] != "—" else -9)
    worst = min(rows, key=lambda r: float(r["Mean NSS"]) if r["Mean NSS"] != "—" else 9)
    st.caption(
        f"**{best['Ordering']}** has the strongest alignment with human gaze "
        f"(NSS={best['Mean NSS']}); **{worst['Ordering']}** the weakest "
        f"(NSS={worst['Mean NSS']}). This directly tests the paper's mechanism: "
        "question-first prompting doesn't just hurt accuracy, it should produce "
        "measurably less human-like visual attention allocation, and echoing "
        "(especially the image) should recover it."
    )
    st.caption("Re-run:  `" + RUN_CMD + "`")
