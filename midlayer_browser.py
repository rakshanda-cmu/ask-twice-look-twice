"""
Streamlit page: Middle-Layer Analysis (IST vs STI logit lens).

For each curated / user-added NaturalBench (image, question) example we show the
logit lens under both orderings side by side — IST on the left, STI on the right.

Summary view : final-layer (model output) only, all examples, grouped by scenario.
Drill-down   : click an example to expand all layers — IST GIF | STI GIF, plus an
               expander with the per-layer frame grid for each ordering.
Add examples : pick any IST✓/STI✗ or STI✓/IST✗ pair from NaturalBench and run it;
               buffered 5 per scenario (FIFO — oldest is evicted).

Separate from the Logit Lens page; that page and its code are untouched.
"""

import os

import streamlit as st

import midlayer_core as mc

MODEL_OPTIONS = {
    "LLaVA-1.5 (7B)":  "llava-1.5",
    "Qwen2.5-VL (7B)": "qwen2.5-vl-7b",
    "Qwen3-VL (8B)":   "qwen3-vl-8b",
}


@st.cache_resource(show_spinner="Loading model … (one-time)")
def _load_model(model_name):
    from utils import setup_seeds, disable_torch_init
    from model_manager import ModelManager
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()
    return ModelManager(model_name)


def _badge(correct):
    return "✅" if correct else "❌"


def _safe_image(path, caption_missing="(render unavailable)", **kw):
    """st.image that never crashes on a missing/partial/corrupt file."""
    if not path or not os.path.exists(path):
        st.caption(f"⚠️ {caption_missing}")
        return False
    try:
        st.image(path, **kw)
        return True
    except Exception:
        st.caption(f"⚠️ corrupt render — re-run precompute "
                   f"({os.path.basename(path)})")
        return False


def _order_dir(e, variant, order):
    """dirs[variant][order] if present (else None); tolerate flat dirs[order]."""
    d = e.get("dirs", {})
    if variant in d and isinstance(d[variant], dict):
        return d[variant].get(order)
    return d.get(order)  # legacy fallback


def _order_answer(e, variant, order):
    """(answer, correct) for one ordering under a variant (judged vs gt)."""
    from naturalbench_eval import judge_pair
    gen = e.get("answers_generated", {})
    gv = gen.get(variant, {}) if isinstance(gen.get(variant), dict) else {}
    ans = gv.get(order)
    if ans is None and order in ("IST", "STI"):
        ans = e.get(f"{order.lower()}_answer", "")
    ans = ans or ""
    try:
        correct, _ = judge_pair(ans, e["expected"],
                                e.get("question_type", "yes_no"), e["question"])
    except Exception:
        correct = (bool(e.get(f"{order.lower()}_correct"))
                   if order in ("IST", "STI") else False)
    return ans, correct


def _example_header(e, variant, orders=("IST", "STI")):
    suffix = "  *(+20 space tokens)*" if variant == "space" else ""
    bits = []
    for o in orders:
        a, c = _order_answer(e, variant, o)
        bits.append(f"{o}: `{a}` {_badge(c)}")
    return (
        f"**{e['code']}** · `{e['question_type']}` · src `{e['source']}`{suffix}  \n"
        f"**Q:** {e['question']}  \n"
        f"**Expected:** `{e['expected']}`  ·  " + "  ·  ".join(bits)
    )


def _render_original(e):
    """Small original image + expander to full size (requirement 3)."""
    p = os.path.join(mc.NB_DIR, e["image_rel"])
    if not os.path.exists(p):
        return
    with st.expander("🖼️ Original image"):
        _safe_image(p, width="stretch", caption=os.path.basename(p))


def _render_passed_generated(e, variant, orders=("IST", "STI")):
    """Show exactly what was passed to the model (prompt incl. spaces) and what
    it generated, for the given orderings."""
    pinfo = e.get("prompt_info", {}).get(variant, {})
    with st.expander("📋 What was passed to the model & what it generated"):
        for order in orders:
            info = pinfo.get(order, {})
            ans, correct = _order_answer(e, variant, order)
            st.markdown(
                f"**{order}** — order `{info.get('order', order)}` · "
                f"seq_len `{info.get('seq_len','?')}` · "
                f"vision `{info.get('n_vision_tokens','?')}` tokens "
                f"(grid {info.get('grid','?')}) · "
                f"spaces inserted `{info.get('n_spaces', 0)}`"
            )
            decoded = info.get("decoded_prompt")
            if decoded:
                st.caption("Exact prompt passed (image patches collapsed; "
                           "inserted spaces shown as ␣):")
                st.code(decoded, language="text")
            st.markdown(f"↳ **Generated answer:** `{ans}` {_badge(correct)}  "
                        f"(expected `{e['expected']}`)")
            st.divider()


def _render_example_block(e, variant, orders=("IST", "STI")):
    """Original image + header + final-layer side-by-side + per-layer drill-down."""
    fl = e.get("final_layer", "?")
    left, right = orders

    top = st.columns([1, 4])
    with top[0]:
        _safe_image(os.path.join(mc.NB_DIR, e["image_rel"]),
                    caption_missing="image missing", width="stretch")
    with top[1]:
        st.markdown(_example_header(e, variant, orders))
        _render_original(e)

    _render_passed_generated(e, variant, orders)

    cL, cR = st.columns(2)
    for col, order in ((cL, left), (cR, right)):
        with col:
            odir = _order_dir(e, variant, order)
            _, oc = _order_answer(e, variant, order)
            st.markdown(f"**{order}** {_badge(oc)} — final layer ({fl})")
            fp = os.path.join(odir, "final.png") if odir else None
            _safe_image(fp, caption_missing="not rendered yet", width="stretch")

    with st.expander(f"🔬 Drill down — all {e.get('num_layers','?')} layers "
                     f"({left} vs {right} · system·image·task·generated token grid)"):
        dL, dR = st.columns(2)
        for col, order in ((dL, left), (dR, right)):
            with col:
                odir = _order_dir(e, variant, order)
                st.markdown(f"**{order}** — animated over layers")
                gif = os.path.join(odir, "anim.gif") if odir else None
                _safe_image(gif, caption_missing="not rendered yet",
                            width="stretch")
                frames = sorted(
                    f for f in os.listdir(odir)
                    if f.startswith("layer_") and f.endswith(".png")
                ) if odir and os.path.isdir(odir) else []
                with st.expander(f"{order}: per-layer frames ({len(frames)})"):
                    for fr in frames:
                        layer_n = fr.replace("layer_", "").replace(".png", "")
                        _safe_image(os.path.join(odir, fr),
                                    caption_missing=f"layer {layer_n} missing",
                                    caption=f"{order} · layer {layer_n}",
                                    width="stretch")
    st.divider()


def _render_scenario(man, scen, variant, orders=("IST", "STI")):
    exs = [e for e in man["examples"] if e["scenario"] == scen]
    exs = sorted(exs, key=lambda e: e.get("ts", 0))
    st.markdown(f"### {mc.SCENARIO_LABEL[scen]} — {len(exs)} example(s)")
    if not exs:
        st.caption("No examples yet. Add one below.")
        return
    for e in exs:
        _render_example_block(e, variant, orders)


# ──────────────────────────────────────────────────────────────────────────────
#  Add-new-example section
# ──────────────────────────────────────────────────────────────────────────────

def _candidate_pairs(model, scen):
    """Return list of (group, image, question, label) for a scenario."""
    try:
        ist = {g["index"]: g for g in mc._load_results(model, "IST")}
        sti = {g["index"]: g for g in mc._load_results(model, "STI")}
    except FileNotFoundError:
        return []
    out = []
    for gidx, gi in ist.items():
        gs = sti.get(gidx)
        if gs is None:
            continue
        bys = {(p["image_index"], p["question_index"]): p for p in gs["pairs"]}
        for pi in gi["pairs"]:
            ps = bys.get((pi["image_index"], pi["question_index"]))
            if ps is None:
                continue
            if mc.scenario_of(pi["correct"], ps["correct"]) != scen:
                continue
            q = pi["question"]
            label = (f"g{gidx} i{pi['image_index']} q{pi['question_index']} · "
                     f"{q[:55]}{'…' if len(q) > 55 else ''} "
                     f"(exp {pi['gt_answer']})")
            out.append((gidx, pi["image_index"], pi["question_index"], label))
    return out


def _add_section(model):
    st.markdown("### ➕ Add an example from NaturalBench")
    st.caption(
        "Pick a pair from either asymmetric scenario and run the logit lens — "
        "this renders BOTH the normal and the +20-space variants (IST + STI). "
        f"Buffered {mc.BUFFER_PER_SCENARIO} per scenario — the oldest is replaced."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        scen_label = st.radio("Scenario", [mc.SCENARIO_LABEL[mc.SCEN_A],
                                           mc.SCENARIO_LABEL[mc.SCEN_B]])
        scen = mc.SCEN_A if scen_label == mc.SCENARIO_LABEL[mc.SCEN_A] else mc.SCEN_B
    with c2:
        cands = _candidate_pairs(model, scen)
        if not cands:
            st.info("No candidate pairs (need IST & STI results).")
            return
        labels = [c[3] for c in cands]
        pick = st.selectbox(f"Candidate pair ({len(cands)} available)", labels)
        chosen = cands[labels.index(pick)]

    if st.button("▶ Run logit lens for this example", type="primary"):
        g, i, q, _ = chosen
        rec = mc.build_example_record(model, g, i, q)
        mm = _load_model(model)
        bar = st.progress(0.0, text="starting …")
        msgs = {"n": 0}

        n_runs = len(mc.VARIANTS) * len(mc.ORDERS)

        def cb(msg, _bar=bar, _m=msgs, _n=n_runs):
            _m["n"] += 1
            _bar.progress(min(0.95, _m["n"] / float(_n)), text=msg)

        import time as _t
        rec = mc.process_example(mm, rec, resolution=640, progress_cb=cb,
                                 ts=_t.time())
        man = mc.load_manifest()
        man = mc.add_example(man, rec)
        mc.save_manifest(man)
        bar.empty()
        ag = rec["answers_generated"]
        st.success(
            f"Added {rec['code']} ({mc.SCENARIO_LABEL[rec['scenario']]}). "
            f"normal IST={ag['normal']['IST']!r}/STI={ag['normal']['STI']!r} · "
            f"space IST={ag['space']['IST']!r}/STI={ag['space']['STI']!r}"
        )
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  Page entry point
# ──────────────────────────────────────────────────────────────────────────────

def _render_page(variant, title, caption, key, orders=("IST", "STI"),
                 show_add=True):
    st.subheader(title)
    st.caption(caption)

    model = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=2,
                         key=f"mid_model_label_{key}")
    model_name = MODEL_OPTIONS[model]

    man = mc.load_manifest()
    have = [e for e in man["examples"] if e.get("model", model_name) == model_name]
    man = {"examples": have}

    if not man["examples"]:
        st.info(
            "No precomputed examples found. Run "
            "`CUDA_VISIBLE_DEVICES=1 python precompute_midlayer.py` to populate "
            "the curated examples, or add one below."
        )
    else:
        n_a = sum(1 for e in man["examples"] if e["scenario"] == mc.SCEN_A)
        n_b = sum(1 for e in man["examples"] if e["scenario"] == mc.SCEN_B)
        st.caption(f"{n_a} IST✓/STI✗ · {n_b} STI✓/IST✗ examples loaded "
                   f"(model: {model_name}).")
        tab_a, tab_b = st.tabs([f"{mc.SCENARIO_LABEL[mc.SCEN_A]} ({n_a})",
                                f"{mc.SCENARIO_LABEL[mc.SCEN_B]} ({n_b})"])
        with tab_a:
            _render_scenario(man, mc.SCEN_A, variant, orders)
        with tab_b:
            _render_scenario(man, mc.SCEN_B, variant, orders)

    if show_add:
        st.divider()
        _add_section(model_name)


def render_midlayer_page():
    """Tab 3 — normal prompts."""
    _render_page(
        variant="normal",
        title="🔬 Middle-Layer Analysis — IST vs STI Logit Lens",
        caption=(
            "Where do IST and STI diverge inside the model? For each NaturalBench "
            "pair we project every layer's hidden states through the LM head "
            "(logit lens) under both orderings, showing the full "
            "system·image·task·generated token grid. Summary = final layer; "
            "expand for all layers, IST (left) vs STI (right)."
        ),
        key="normal",
    )


def render_midlayer_space_page():
    """Tab 4 — same examples with 20 space tokens added."""
    _render_page(
        variant="space",
        title="␣ Middle-Layer Analysis — +20 Space Tokens (IST vs STI)",
        caption=(
            "Same examples and side-by-side logit lens as the Middle-Layer tab, but "
            "with **20 space tokens** inserted after the Task (IST) / after the "
            "image (STI). Answers are re-judged with the space tokens present — "
            "compare against the no-spaces tab to see the effect."
        ),
        key="space",
    )


def render_stit_compare_page():
    """STIT vs STI and STIT vs IST per-layer comparisons (same 80 examples)."""
    st.subheader("🔁 STIT vs STI / IST — per-layer logit-lens comparison")
    st.caption(
        "STIT = System·Task·Image·**Task** (question repeated *after* the image — "
        "the recency-fixed ordering). Each example shows the per-layer logit lens "
        "for STIT (left) vs STI or IST (right) over the same 80 curated pairs."
    )
    model = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=2,
                         key="mid_model_label_stit")
    model_name = MODEL_OPTIONS[model]

    man = mc.load_manifest()
    have = [e for e in man["examples"] if e.get("model", model_name) == model_name]
    man = {"examples": have}
    if not man["examples"]:
        st.info("No examples yet. Run `python precompute_midlayer.py` then "
                "`python precompute_stit.py`.")
        return

    n_stit = sum(1 for e in man["examples"]
                 if _order_dir(e, "normal", "STIT")
                 and os.path.exists(os.path.join(_order_dir(e, "normal", "STIT"),
                                                 "final.png")))
    st.caption(f"{n_stit}/{len(man['examples'])} examples have STIT rendered "
               "(re-run `precompute_stit.py` to finish any missing).")

    cmp_tabs = st.tabs(["STIT vs STI", "STIT vs IST"])
    for tab, right in zip(cmp_tabs, ("STI", "IST")):
        with tab:
            sa, sb = st.tabs([f"{mc.SCENARIO_LABEL[mc.SCEN_A]}",
                              f"{mc.SCENARIO_LABEL[mc.SCEN_B]}"])
            with sa:
                _render_scenario(man, mc.SCEN_A, "normal", ("STIT", right))
            with sb:
                _render_scenario(man, mc.SCEN_B, "normal", ("STIT", right))
