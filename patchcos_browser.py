"""
Streamlit page: Visual-Patch Perturbation — per-patch cosine similarity of the
image-token LLM hidden states (layer ~18) under IST / STI / STIT vs a base run whose
input is the image alone. Read-only; renders patch_cosine_full.json produced by
scratch_patch_cosine.py (Qwen3-VL-8B over NaturalBench).
"""

import json
import os

import numpy as np
import streamlit as st

RESULT = "patch_cosine_full.json"
ORDERS = ["IST", "STI", "STIT"]
RUN_CMD = "CUDA_VISIBLE_DEVICES=0 python scratch_patch_cosine.py --num-pairs 0 --validate --out patch_cosine_full"


def render_patchcos_page():
    st.subheader("🖼️ Visual-Patch Perturbation — IST vs STI vs STIT")
    st.caption(
        "How much does prompt **ordering** change the model's *image* representation? "
        "For each NaturalBench image we compare the image-token hidden states (layer "
        "~18) under IST / STI / STIT against a **base** run whose input is the image "
        "alone (no system, no question), per patch, by cosine similarity. "
        "**1.0 = untouched; lower = the surrounding text has perturbed that patch.**"
    )

    if not os.path.exists(RESULT):
        st.info(f"**No result yet.** Run (Qwen3-VL-8B, whole NaturalBench):\n\n`{RUN_CMD}`")
        return

    d = json.load(open(RESULT))
    gh, gw = d["grid"]
    maps = d.get("mean_cos_map", {})
    n = max(d.get("n", {}).values()) if d.get("n") else 0
    have = [o for o in ORDERS if o in maps]

    st.markdown(f"#### Mean cosine to image-only base — layer {d['layer']}, "
                f"{n} (image·question) pairs, {gh}×{gw} patch grid")
    cols = st.columns(len(have))
    for c, o in zip(cols, have):
        c.metric(o, f"{np.mean(maps[o]):.3f}")

    ident = d.get("identities", {})
    if ident:
        bi = ident.get("cos(base,IST)")
        ss = ident.get("cos(STI,STIT)")
        st.caption(
            f"**Causal-attention identities (measured):** cos(base, IST) = "
            f"**{bi:.4f}**, cos(STI, STIT) = **{ss:.4f}**. Because attention is causal, "
            "IST's image patches (image is first) equal the base, and STIT's equal STI "
            "(its 2nd question comes *after* the image). So the visual perturbation is "
            "the same for STI and STIT — the STIT fix is **not** in the image "
            "representation."
        )

    # ── heatmaps (shared color scale) ──────────────────────────────────────────
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    allv = [v for o in have for v in maps[o]]
    zmin = min(allv) if allv else 0.0
    fig = make_subplots(rows=1, cols=len(have), subplot_titles=have,
                        horizontal_spacing=0.06)
    for ci, o in enumerate(have, start=1):
        z = np.array(maps[o]).reshape(gh, gw)
        fig.add_trace(
            go.Heatmap(z=z, coloraxis="coloraxis",
                       hovertemplate=f"{o}<br>patch (%{{y}},%{{x}})<br>cos %{{z:.3f}}<extra></extra>"),
            row=1, col=ci)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{ci if ci>1 else ''}",
                         showticklabels=False, row=1, col=ci)
        fig.update_xaxes(showticklabels=False, row=1, col=ci)
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=30, b=0),
        coloraxis=dict(colorscale="Viridis", cmin=zmin, cmax=1.0,
                       colorbar=dict(title="cosine")),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Reading it: **IST is uniformly 1.0** (image first → its patches never see the "
        "text). **STI and STIT are identical** and drop to ~"
        f"{np.mean(maps.get('STI', [1])):.2f} — placing system+question *before* the "
        "image rewrites every visual patch. Since STI and STIT perturb the image the "
        "same way yet differ in accuracy, the STIT benefit must come from the "
        "**answer** position (see the Mechanism Probe), not the image encoding."
    )
    st.markdown("---")
    st.caption(f"Regenerate with:  `{RUN_CMD}`")

    # ── Cosine-similarity heatmaps: raw image vs IST vs STI, 2 questions ─────────
    # (added below; existing content above is unchanged)
    st.markdown("---")
    st.markdown("### 🔥 Cosine-similarity heatmaps — raw image vs IST vs STI")
    st.caption(
        "**Per-patch cosine of the image representation vs the RAW image-only run**, "
        "Qwen3-VL-8B, overlaid on the image. For each ordering we take the image-token "
        "hidden states and, per patch and per layer, measure "
        "cos( h(raw image only) , h(ordering) ); we plot **(1 − cos)** — how much the "
        "surrounding text perturbs that patch (**red = perturbed, blue = untouched**), "
        "on one **fixed** color scale across every cell. Columns: **Image** (raw "
        "reference) · **IST** · **STI**. For each question, three rows: **GIF** (across "
        "all layers) · **Mean** (over layers) · **Final layer**.\n\n"
        "Because attention is **causal** over image tokens: **IST** (image first) is "
        "**flat / untouched** (cos ≈ 1.0) for both questions — the image never sees the "
        "question; **STI** (question first) is perturbed and the map **differs between "
        "the two questions**, since each question rewrites the visual patches "
        "differently."
    )
    _render_cos_overlays("patchcos_demo", "full resolution")

    # ── same analysis at HALF resolution (added below; full-res above unchanged) ──
    st.markdown("---")
    st.markdown("### 🔥 Cosine-similarity heatmaps — HALF resolution")
    st.caption(
        "The **same** raw-vs-IST-vs-STI cosine analysis, but the input image is first "
        "**down-scaled to half resolution** (fewer patch tokens). The causal-attention "
        "result is unchanged: **IST stays flat / untouched** (cos ≈ 1.0), **STI is "
        "perturbed** and question-dependent — showing the effect is not an artifact of "
        "the number of image tokens."
    )
    _render_cos_overlays("patchcos_demo_half", "half resolution")


def _render_cos_overlays(pdir, tag):
    """Render the [Image | IST | STI] × {q1,q2} × {gif,mean,final} cosine overlay grid
    from <pdir>/manifest.json (produced by patchcos_overlay_gen.py)."""
    pmp = os.path.join(pdir, "manifest.json")
    if not os.path.exists(pmp):
        st.info(f"**No {tag} cosine heatmaps yet.** Run:\n\n"
                f"`CUDA_VISIBLE_DEVICES=0 python patchcos_overlay_gen.py "
                f"--image examples/<img>.png"
                + (" --scale 0.5 --out-dir patchcos_demo_half" if "half" in pdir else "")
                + "`")
        return
    pm = json.load(open(pmp))
    psrc = os.path.join(pdir, pm.get("source", ""))
    cols_order = pm.get("cols", ["Image", "IST", "STI"])
    COLS = [(c, "Image (raw)" if c == "Image" else c) for c in cols_order]
    VARIANTS = [("gif", "GIF (all layers)"), ("mean", "Mean over layers"),
                ("final", "Final layer")]
    ident = pm.get("identity", {})
    if ident:
        # average the final-layer relative-L2 over the two questions, per ordering
        def _mid(o):
            vs = [ident[q].get(o) for q in ident if o in ident[q]]
            return sum(vs) / len(vs) if vs else None
        parts = []
        for o in cols_order:
            if o == "Image":
                continue
            v = _mid(o)
            if v is not None:
                parts.append(f"**{o}** ‖Δ‖/‖base‖ = {v:.4f}")
        if parts:
            st.caption("**Identity check (final-layer relative-L2 of the image "
                       "representation vs raw):** " + " · ".join(parts) +
                       ".  IST is exactly **0** — the image is bit-for-bit identical to "
                       "the raw run (not just same-direction); **STI = STIT** > 0 — both "
                       "perturb the image identically, so the STIT accuracy gain is "
                       "**not** in the image encoding.")
    for q in pm["questions"]:
        st.markdown(f"#### “{q['text']}”")
        for vkey, vlab in VARIANTS:
            st.markdown(f"*{vlab}*")
            cc = st.columns(len(COLS))
            for col, (ckey, clab) in zip(cc, COLS):
                with col:
                    st.caption(clab)
                    if ckey == "Image":               # raw reference (cos=1, untouched)
                        if os.path.exists(psrc):
                            st.image(psrc, use_container_width=True)
                        else:
                            st.caption("(missing)")
                        continue
                    p = os.path.join(pdir, pm["rows"].get(q["id"], {}).get(ckey, {}).get(vkey, ""))
                    if p and os.path.exists(p):
                        st.image(p, use_container_width=True)
                    else:
                        st.caption("(missing)")
    sz = pm.get("img_size")
    szs = f" · input {sz[0]}×{sz[1]}px" if sz else ""
    st.caption(f"Fixed scale: (1−cos) ∈ [0, {pm.get('vmax', 0):.3f}] · {pm['grid'][0]}×"
               f"{pm['grid'][1]} patch grid · {pm.get('n_layers','?')} layers{szs}.")
