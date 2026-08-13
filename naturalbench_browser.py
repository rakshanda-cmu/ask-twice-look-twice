"""
Streamlit page for the NaturalBench prompt-ordering experiments.

Two experiments are supported:
    Experiment 1 — IST : Image · System · Task
    Experiment 2 — STI : System · Task · Image

This page lets you:
  • pick a model (dropdown), system prompt, and #groups, then RUN either/both
    experiments (writing results/correct/wrong JSONs), and
  • BROWSE the results — compare IST vs STI metrics and inspect the
    correct/wrong groups (2 images × 2 questions, gold vs model answers).

Self-contained; used as a separate page from logit_lens_app.py:
    from naturalbench_browser import render_naturalbench_page
    render_naturalbench_page()
"""

import json
import os

import streamlit as st

from constants import SYSTEM_MESSAGE

NB_DIR = "./naturalbench"
RESULTS_DIR = os.path.join(NB_DIR, "results")

MODEL_OPTIONS = {
    "LLaVA-1.5 (7B)":  "llava-1.5",
    "Qwen2.5-VL (7B)": "qwen2.5-vl-7b",
    "Qwen3-VL (8B)":   "qwen3-vl-8b",
}

EXPERIMENTS = {
    "IST — Image · System · Task": "IST",
    "STI — System · Task · Image": "STI",
}


# ── model cache (separate from the logit-lens page's loader) ───────────────────
@st.cache_resource(show_spinner="Loading model … (one-time)")
def _load_model(model_name: str):
    from utils import setup_seeds, disable_torch_init
    from model_manager import ModelManager
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()
    return ModelManager(model_name)


@st.cache_data(show_spinner=False)
def _load_results(path, _mtime):
    with open(path) as f:
        return json.load(f)


def _results_path(model, order, kind="results"):
    return os.path.join(RESULTS_DIR, f"{model}__{order}__{kind}.json")


# ══════════════════════════════════════════════════════════════════════════════
#  Run section
# ══════════════════════════════════════════════════════════════════════════════

def _run_section():
    st.markdown("### ⚙️ Run experiments")

    if not os.path.exists(os.path.join(NB_DIR, "groups.json")):
        st.warning(
            "NaturalBench data not found. Run "
            "`python download_naturalbench.py` first."
        )
        return

    from naturalbench_eval import load_groups
    all_groups = load_groups(NB_DIR)
    n_available = len(all_groups)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=2,
                                   key="nb_model_label")
        model_name = MODEL_OPTIONS[model_label]
    with c2:
        num_groups = st.number_input(
            "Number of groups", min_value=1, max_value=n_available,
            value=min(100, n_available), step=10,
            help=f"{n_available} groups available locally.",
        )
    with c3:
        max_tokens = st.slider("Max new tokens", 4, 64, 16, step=4)

    exp_labels = st.multiselect(
        "Experiments to run", list(EXPERIMENTS.keys()),
        default=list(EXPERIMENTS.keys()),
    )
    orders = [EXPERIMENTS[l] for l in exp_labels]

    system_prompt = st.text_area(
        "System prompt (the 'S' in IST / STI)", value=SYSTEM_MESSAGE, height=80,
        help="Must be non-empty for the two orderings to differ by system-prompt "
             "position. Task = each NaturalBench question (added automatically).",
    )

    est = int(num_groups) * 4 * len(orders)
    st.caption(
        f"Will run **{est}** generations "
        f"({num_groups} groups × 4 pairs × {len(orders)} experiment(s)). "
        "Results are written to `naturalbench/results/`."
    )

    run = st.button("▶ Run experiment(s)", type="primary",
                    disabled=(not orders), width="stretch")

    if run and orders:
        from naturalbench_eval import run_experiment, write_experiment_outputs
        mm = _load_model(model_name)
        groups = all_groups[: int(num_groups)]

        for order in orders:
            st.write(f"**Running {order} …**")
            bar = st.progress(0.0, text=f"{order}: starting …")
            meta, records = run_experiment(
                mm, groups, order=order, system_prompt=system_prompt,
                nb_dir=NB_DIR, max_tokens=int(max_tokens),
                progress_cb=lambda fr, msg, _b=bar: _b.progress(fr, text=msg),
            )
            write_experiment_outputs(meta, records, RESULTS_DIR)
            bar.empty()
            st.success(
                f"{order} done — G-Acc {meta['g_acc']*100:.1f}% · "
                f"Q-Acc {meta['q_acc']*100:.1f}% · I-Acc {meta['i_acc']*100:.1f}% · "
                f"pair {meta['pair_acc']*100:.1f}%"
            )
        st.session_state["nb_view_model"] = model_name
        st.cache_data.clear()  # so the results viewer reloads fresh files
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  Results / browse section
# ══════════════════════════════════════════════════════════════════════════════

def _available_models():
    if not os.path.isdir(RESULTS_DIR):
        return []
    models = set()
    for f in os.listdir(RESULTS_DIR):
        if f.endswith("__results.json"):
            models.add(f.split("__")[0])
    return sorted(models)


def _metric_row(meta):
    return {
        "G-Acc (all 4)": f"{meta.get('g_acc', 0)*100:.1f}%",
        "Q-Acc (per question)": f"{meta.get('q_acc', 0)*100:.1f}%",
        "I-Acc (per image)": f"{meta.get('i_acc', 0)*100:.1f}%",
        "Pair-Acc": f"{meta.get('pair_acc', 0)*100:.1f}%",
        "# groups": str(meta.get("num_groups", 0)),
    }


def _pair_badge(correct):
    return "✅" if correct else "❌"


def _render_group_card(g, key_prefix):
    icon = "🟢" if g["group_correct"] else "🔴"
    st.markdown(
        f"**{icon} Group {g['index']}** · `{g['question_type']}` · "
        f"src `{g['source']}` · **{g['num_pairs_correct']}/4** pairs correct"
    )

    ci0, ci1 = st.columns(2)
    for col, img_idx in ((ci0, 0), (ci1, 1)):
        with col:
            ip = os.path.join(NB_DIR, g[f"image_{img_idx}"])
            if os.path.exists(ip):
                st.image(ip, caption=f"Image {img_idx}", width="stretch")
            else:
                st.caption(f"(missing {ip})")

    # index pairs by (image, question) for tabular display
    by = {(p["image_index"], p["question_index"]): p for p in g["pairs"]}
    for q_idx in (0, 1):
        st.markdown(f"**Q{q_idx}:** {g[f'question_{q_idx}']}")
        rows = []
        for img_idx in (0, 1):
            p = by[(img_idx, q_idx)]
            rows.append({
                "image": f"Image {img_idx}",
                "expected": p["gt_answer"],
                "model answer": p["model_answer_raw"],
                "✓": _pair_badge(p["correct"]),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
    st.divider()


def _browse_experiment(model, order):
    rpath = _results_path(model, order)
    if not os.path.exists(rpath):
        st.info(f"No results for {order}. Run it above.")
        return
    data = _load_results(rpath, os.path.getmtime(rpath))
    records = data["results"]

    verdict = st.radio(
        f"{order}: show", ["Wrong only", "Correct only", "All"],
        index=0, horizontal=True, key=f"verdict_{model}_{order}",
    )
    qtypes = sorted({r["question_type"] for r in records})
    qfilter = st.selectbox(f"{order}: question type", ["all"] + qtypes,
                           key=f"qtype_{model}_{order}")

    def keep(r):
        if verdict == "Wrong only" and r["group_correct"]:
            return False
        if verdict == "Correct only" and not r["group_correct"]:
            return False
        if qfilter != "all" and r["question_type"] != qfilter:
            return False
        return True

    filtered = [r for r in records if keep(r)]
    st.caption(f"**{len(filtered)}** group(s) match.")
    if not filtered:
        return

    per_page = 5
    n_pages = (len(filtered) + per_page - 1) // per_page
    page = st.number_input(f"{order}: page (1–{n_pages})", min_value=1,
                           max_value=n_pages, value=1, step=1,
                           key=f"page_{model}_{order}")
    start = (page - 1) * per_page
    for g in filtered[start: start + per_page]:
        _render_group_card(g, key_prefix=f"{order}_{g['index']}")


def _render_comparison_card(gi, gs):
    """
    Render one group with BOTH experiments' answers side by side.
    gi = IST group record, gs = STI group record (same group index).
    """
    ist_icon = "🟢" if gi["group_correct"] else "🔴"
    sti_icon = "🟢" if gs["group_correct"] else "🔴"
    st.markdown(
        f"**Group {gi['index']}** · `{gi['question_type']}` · src `{gi['source']}`  "
        f"— IST {ist_icon} **{gi['num_pairs_correct']}/4** · "
        f"STI {sti_icon} **{gs['num_pairs_correct']}/4**"
    )

    ci0, ci1 = st.columns(2)
    for col, img_idx in ((ci0, 0), (ci1, 1)):
        with col:
            ip = os.path.join(NB_DIR, gi[f"image_{img_idx}"])
            if os.path.exists(ip):
                st.image(ip, caption=f"Image {img_idx}", width="stretch")
            else:
                st.caption(f"(missing {ip})")

    byi = {(p["image_index"], p["question_index"]): p for p in gi["pairs"]}
    bys = {(p["image_index"], p["question_index"]): p for p in gs["pairs"]}
    for q_idx in (0, 1):
        st.markdown(f"**Q{q_idx}:** {gi[f'question_{q_idx}']}")
        rows = []
        for img_idx in (0, 1):
            pi = byi[(img_idx, q_idx)]
            ps = bys[(img_idx, q_idx)]
            rows.append({
                "image": f"Image {img_idx}",
                "expected": pi["gt_answer"],
                "IST answer": f"{pi['model_answer_raw']} {_pair_badge(pi['correct'])}",
                "STI answer": f"{ps['model_answer_raw']} {_pair_badge(ps['correct'])}",
            })
        st.dataframe(rows, hide_index=True, width="stretch")
    st.divider()


def _render_pair_comparison_card(gi, img_idx, q_idx, pi, ps):
    """
    Render ONE (image, question) pair with both experiments' answers.
    gi = group record (for image path / question / meta), pi/ps = IST/STI pair.
    """
    st.markdown(
        f"**Group {gi['index']}** · `{gi['question_type']}` · src `{gi['source']}`  "
        f"— **Image {img_idx} · Q{q_idx}**  "
        f"(IST {_pair_badge(pi['correct'])} · STI {_pair_badge(ps['correct'])})"
    )
    cimg, cinfo = st.columns([1, 2], gap="medium")
    with cimg:
        ip = os.path.join(NB_DIR, gi[f"image_{img_idx}"])
        if os.path.exists(ip):
            st.image(ip, caption=f"Image {img_idx}", width="stretch")
        else:
            st.caption(f"(missing {ip})")
    with cinfo:
        st.markdown(f"**Q{q_idx}:** {gi[f'question_{q_idx}']}")
        st.dataframe([{
            "expected": pi["gt_answer"],
            "IST answer": f"{pi['model_answer_raw']} {_pair_badge(pi['correct'])}",
            "STI answer": f"{ps['model_answer_raw']} {_pair_badge(ps['correct'])}",
        }], hide_index=True, width="stretch")
    st.divider()


# mode -> predicate over (IST correct, STI correct) — works for group OR pair flags
_COMPARE_MODES = {
    "ist_right_sti_wrong": lambda ic, sc: ic and not sc,
    "sti_right_ist_wrong": lambda ic, sc: sc and not ic,
    "both_wrong":          lambda ic, sc: (not ic) and (not sc),
    "both_correct":        lambda ic, sc: ic and sc,
}


def _browse_comparison(model, mode, label, granularity="group"):
    """
    Cross-experiment browser: align IST & STI, filter by `mode`.

    granularity="group": unit = a whole group (group_correct = all 4 pairs).
    granularity="pair" : unit = a single (image, question) pair.
    """
    ist_p = _results_path(model, "IST")
    sti_p = _results_path(model, "STI")
    if not (os.path.exists(ist_p) and os.path.exists(sti_p)):
        st.info("This view needs both IST and STI results.")
        return

    ist = _load_results(ist_p, os.path.getmtime(ist_p))["results"]
    sti = _load_results(sti_p, os.path.getmtime(sti_p))["results"]
    sti_by = {g["index"]: g for g in sti}
    pred = _COMPARE_MODES[mode]
    unit = "pair(s)" if granularity == "pair" else "group(s)"

    # ── collect matched units ──────────────────────────────────────────────────
    matched = []  # group mode: (gi, gs); pair mode: (gi, img, q, pi, ps)
    for gi in ist:
        gs = sti_by.get(gi["index"])
        if gs is None:
            continue
        if granularity == "group":
            if pred(gi["group_correct"], gs["group_correct"]):
                matched.append((gi, gs))
        else:
            byi = {(p["image_index"], p["question_index"]): p for p in gi["pairs"]}
            bys = {(p["image_index"], p["question_index"]): p for p in gs["pairs"]}
            for key in sorted(byi):
                pi, ps = byi[key], bys.get(key)
                if ps is None:
                    continue
                if pred(pi["correct"], ps["correct"]):
                    matched.append((gi, key[0], key[1], pi, ps))

    st.markdown(f"#### {label} — **{len(matched)}** {unit} total")
    if not matched:
        st.caption(f"No {unit} in this category.")
        return

    qtypes = sorted({m[0]["question_type"] for m in matched})
    qfilter = st.selectbox(f"{label}: question type", ["all"] + qtypes,
                           key=f"qtype_cmp_{model}_{mode}_{granularity}")
    if qfilter != "all":
        matched = [m for m in matched if m[0]["question_type"] == qfilter]
        st.caption(f"**{len(matched)}** {unit} after type filter.")
    if not matched:
        return

    per_page = 5 if granularity == "group" else 8
    n_pages = (len(matched) + per_page - 1) // per_page
    page = st.number_input(f"{label}: page (1–{n_pages})", min_value=1,
                           max_value=n_pages, value=1, step=1,
                           key=f"page_cmp_{model}_{mode}_{granularity}")
    start = (page - 1) * per_page
    for m in matched[start: start + per_page]:
        if granularity == "group":
            _render_comparison_card(m[0], m[1])
        else:
            gi, img_idx, q_idx, pi, ps = m
            _render_pair_comparison_card(gi, img_idx, q_idx, pi, ps)


def _render_resolution_table(model):
    """Separate table: IST/STI at the dataset mean resolution vs full-res baseline."""
    COLS = [("IST", "IST (full-res)"), ("IST_meanres", "IST mean-res"),
            ("STI", "STI (full-res)"), ("STI_meanres", "STI mean-res")]
    comp, metas = {}, {}
    for tag, label in COLS:
        p = _results_path(model, tag)
        if os.path.exists(p):
            meta = _load_results(p, os.path.getmtime(p))["meta"]
            metas[tag] = meta
            comp[label] = _metric_row(meta)

    if "IST_meanres" not in metas and "STI_meanres" not in metas:
        return  # nothing to show yet

    mk = next((k for k in ("STI_meanres", "IST_meanres")
               if k in metas and metas[k].get("resize")), None)
    rz = metas[mk].get("resize") if mk else None
    mode = metas[mk].get("resize_mode") if mk else None
    how = (f"large images downscaled to ≤ {rz[0]}×{rz[1]} px (aspect preserved)"
           if mode == "cap" else
           (f"images resized to {rz[0]}×{rz[1]} px" if rz else ""))
    st.markdown("---")
    st.markdown("#### 🖼️ Resolution control — mean-resolution vs full-res"
                + (f" ({how})" if how else ""))
    st.caption("Tests whether *fewer image tokens* between question and answer "
               "helps STI (your forgetting hypothesis). Full-res baselines are the "
               "same runs shown in the main table — unchanged.")
    st.table(comp)

    mr = []
    for base, res, name in (("IST", "IST_meanres", "IST"),
                            ("STI", "STI_meanres", "STI")):
        if base in metas and res in metas:
            dd = (metas[res]["g_acc"] - metas[base]["g_acc"]) * 100
            arrow = "▲" if dd > 0 else ("▼" if dd < 0 else "▬")
            mr.append(f"{name} {arrow} **{dd:+.1f} pts**")
    if mr:
        st.caption("Mean-res vs full-res, Group-Acc — " + " · ".join(mr))


def _render_echo_resolution_table(model):
    """Separate table: SITIT echo-resolution ablation — which image occurrence
    (1st vs 2nd/echoed) matters more when it's shown at reduced resolution,
    at two scales (half, quarter)."""
    COLS = [("SITIT", "SITIT (both full-res)"),
            ("SITIT_echo1half", "SITIT (1st half-res, 2nd full-res)"),
            ("SITIT_echo2half", "SITIT (1st full-res, 2nd half-res)"),
            ("SITIT_echo1quarter", "SITIT (1st 0.25x, 2nd full-res)"),
            ("SITIT_echo2quarter", "SITIT (1st full-res, 2nd 0.25x)")]
    comp, metas = {}, {}
    for tag, label in COLS:
        p = _results_path(model, tag)
        if os.path.exists(p):
            meta = _load_results(p, os.path.getmtime(p))["meta"]
            metas[tag] = meta
            comp[label] = _metric_row(meta)

    if not any(k in metas for k in ("SITIT_echo1half", "SITIT_echo2half",
                                    "SITIT_echo1quarter", "SITIT_echo2quarter")):
        return  # nothing to show yet

    st.markdown("---")
    st.markdown("#### 🖼️½ SITIT echo-resolution ablation")
    st.caption(
        "SITIT shows the image twice (S·I·T·I·T). Here ONE of the two "
        "occurrences is downscaled (to half, or a more aggressive 0.25x, "
        "width/height) while the other stays full-res, isolating which "
        "occurrence — the first viewing or the post-task echo — carries more "
        "of the group-accuracy benefit, and whether that holds as the "
        "downscaled occurrence loses even more detail."
    )
    st.table(comp)

    mr = []
    for tag, name in (("SITIT_echo1half", "1st-half"), ("SITIT_echo2half", "2nd-half"),
                      ("SITIT_echo1quarter", "1st-0.25x"), ("SITIT_echo2quarter", "2nd-0.25x")):
        if "SITIT" in metas and tag in metas:
            dd = (metas[tag]["g_acc"] - metas["SITIT"]["g_acc"]) * 100
            arrow = "▲" if dd > 0 else ("▼" if dd < 0 else "▬")
            mr.append(f"{name} {arrow} **{dd:+.1f} pts**")
    if mr:
        st.caption("vs full-res SITIT, Group-Acc — " + " · ".join(mr))


def _render_imagecopies_table(model):
    """Separate table: image-copy distance sweep (N=1,2,3) for IST and STI."""
    COLS = [("IST", "IST ×1"), ("IST_copies2", "IST ×2"), ("IST_copies3", "IST ×3"),
            ("STI", "STI ×1"), ("STI_copies2", "STI ×2"), ("STI_copies3", "STI ×3")]
    comp, metas = {}, {}
    for tag, label in COLS:
        p = _results_path(model, tag)
        if os.path.exists(p):
            meta = _load_results(p, os.path.getmtime(p))["meta"]
            metas[tag] = meta
            comp[label] = _metric_row(meta)

    if "IST_copies2" not in metas and "STI_copies2" not in metas \
            and "IST_copies3" not in metas and "STI_copies3" not in metas:
        return

    st.markdown("---")
    st.markdown("#### 🖼️×N Image-copy distance sweep")
    st.caption(
        "The image is repeated N× at the image position → more image tokens "
        "between the question and the answer, **same content**. If the model "
        "*forgets the question across the image tokens*, STI should drop as N "
        "grows while IST (question adjacent to the answer) stays flat."
    )
    st.table(comp)

    def trend(prefix):
        gs = [metas[t]["g_acc"] * 100 for t in (prefix, f"{prefix}_copies2",
              f"{prefix}_copies3") if t in metas]
        if len(gs) >= 2:
            return f"{prefix}: " + " → ".join(f"{g:.1f}%" for g in gs) + \
                   f"  (Δ {gs[-1]-gs[0]:+.1f} pts over the sweep)"
        return None
    lines = [t for t in (trend("IST"), trend("STI")) if t]
    if lines:
        st.caption("Group-Acc vs #image-copies — " + "  ·  ".join(lines))


def _results_section():
    st.markdown("### 📈 Results")
    models = _available_models()
    if not models:
        st.info("No results yet. Configure and run an experiment above.")
        return

    default_model = st.session_state.get("nb_view_model", models[-1])
    idx = models.index(default_model) if default_model in models else len(models) - 1
    model = st.selectbox("Results for model", models, index=idx,
                         key="nb_results_model")

    # ── metric comparison: baseline vs +5 vs +20 spaces, IST & STI ──────────────
    # (columns appear only if their result file exists; grouped per ordering)
    COLS = [
        ("IST", "IST"), ("IST_space5", "IST +5sp"),
        ("IST_space", "IST +20sp"), ("IST_space40", "IST +40sp"),
        ("SIT", "SIT"),
        ("STI", "STI"), ("STI_space5", "STI +5sp"),
        ("STI_space", "STI +20sp"), ("STI_space40", "STI +40sp"),
        ("STI_space120", "STI +120sp"),
        ("STIT", "STIT (Q-rep)"), ("STIT_cue", "STIT cue-only"),
        ("StIT", "StIT (generic t)"),
        ("STITI", "STITI"), ("STITI_rev", "STITI-rev (bidir)"), ("STITIT", "STITIT"),
        ("STSIT", "STSIT"), ("STIST", "STIST"), ("SITI", "SITI"), ("SITIT", "SITIT"),
        ("SITIT_rev", "SITIT-rev (S·I·T·Ī·T)"),
        ("SITT", "SITT"),
        ("SITIT_echo1half", "SITIT (1st½)"), ("SITIT_echo2half", "SITIT (2nd½)"),
        ("SITIT_echo1quarter", "SITIT (1st¼)"), ("SITIT_echo2quarter", "SITIT (2nd¼)"),
        ("IST_think", "IST +think"), ("STIT_think", "STIT +think"),
        ("IST_copies2", "IST ×2img"), ("IST_copies3", "IST ×3img"),
        ("STI_copies2", "STI ×2img"), ("STI_copies3", "STI ×3img"),
    ]
    comp, metas = {}, {}
    for tag, col_label in COLS:
        p = _results_path(model, tag)
        if os.path.exists(p):
            meta = _load_results(p, os.path.getmtime(p))["meta"]
            metas[tag] = meta
            comp[col_label] = _metric_row(meta)

    if not comp:
        st.info("No IST/STI result files for this model yet.")
        return

    has_space = any(k.endswith("_space") or "_space" in k for k in metas
                    if k not in ("IST", "STI"))
    st.markdown("#### Experiment comparison"
                + (" — without vs **with extra space tokens**" if has_space else ""))
    if st.checkbox("↔ Expand table — orderings as rows, scrollable / full-screen",
                   key="nb_expand_cmp", value=False):
        import pandas as pd
        _dft = pd.DataFrame(comp).T            # orderings as rows, metrics as columns
        _dft.index.name = "Ordering"
        st.dataframe(_dft, use_container_width=True,
                     height=min(700, 44 + 35 * len(_dft)))
    else:
        st.table(comp)

    # IST vs STI gap (baseline)
    if "IST" in metas and "STI" in metas:
        d = (metas["IST"]["g_acc"] - metas["STI"]["g_acc"]) * 100
        better = "IST" if d > 0 else ("STI" if d < 0 else "tie")
        st.caption(
            f"Group-accuracy gap (IST − STI), no spaces: **{d:+.1f} pts** "
            f"→ {'**' + better + '** higher' if better != 'tie' else 'tie'}."
        )

    # SIT (System·Image·Task) — the conventional system→image→question ordering
    if "SIT" in metas:
        g = metas["SIT"]["g_acc"] * 100
        parts = []
        if "IST" in metas:
            parts.append(f"vs IST **{g - metas['IST']['g_acc']*100:+.1f} pts**")
        if "STI" in metas:
            parts.append(f"vs STI **{g - metas['STI']['g_acc']*100:+.1f} pts**")
        st.caption(f"**SIT** (System·Image·Task): Group-Acc **{g:.1f}%**"
                   + (" — " + " · ".join(parts) if parts else ""))

    # Effect of extra spaces (variant − baseline) on Group-Acc, per ordering
    eff = []
    for base, sp, name in (("IST", "IST_space5", "IST +5sp"),
                           ("IST", "IST_space", "IST +20sp"),
                           ("IST", "IST_space40", "IST +40sp"),
                           ("STI", "STI_space5", "STI +5sp"),
                           ("STI", "STI_space", "STI +20sp"),
                           ("STI", "STI_space40", "STI +40sp"),
                           ("STI", "STI_space120", "STI +120sp")):
        if base in metas and sp in metas:
            dd = (metas[sp]["g_acc"] - metas[base]["g_acc"]) * 100
            arrow = "▲" if dd > 0 else ("▼" if dd < 0 else "▬")
            eff.append(f"{name} {arrow} **{dd:+.1f} pts**")
    if eff:
        st.caption("Effect of extra space tokens on Group-Acc — " + " · ".join(eff))
    elif not has_space:
        st.caption(
            "ℹ️ Space-token columns appear once you run, e.g.:  \n"
            "`CUDA_VISIBLE_DEVICES=0 python naturalbench_eval.py "
            "--order IST,STI --n-spaces 5 --num-groups 1900` (resumable)."
        )

    # STIT: System·Task·Image·Task (question repeated after image — recency fix)
    if "STIT" in metas:
        g = metas["STIT"]["g_acc"] * 100
        parts = []
        if "STI" in metas:
            parts.append(f"vs STI **{g - metas['STI']['g_acc']*100:+.1f} pts**")
        if "IST" in metas:
            parts.append(f"vs IST **{g - metas['IST']['g_acc']*100:+.1f} pts**")
        st.caption(
            f"**STIT** (question repeated *after* the image — recency-fixed STI): "
            f"Group-Acc **{g:.1f}%** — " + " · ".join(parts)
        )
        if "STIT_cue" in metas:
            gc = metas["STIT_cue"]["g_acc"] * 100
            gs = metas["STI"]["g_acc"] * 100 if "STI" in metas else None
            msg = (f"**STIT cue-only** (post-image repeat = answer instruction, "
                   f"NO question content): Group-Acc **{gc:.1f}%**. ")
            if gs is not None:
                msg += (f"If STIT ({g:.1f}%) ≫ cue-only ({gc:.1f}%) ≈ STI ({gs:.1f}%) "
                        "→ it's the **question content** reprocessed with the image, "
                        "not a generic post-image prompt.")
            st.caption(msg)

    # StIT: System · t · Image · Task — generic pre-image instruction, real question after
    if "StIT" in metas:
        g = metas["StIT"]["g_acc"] * 100
        parts = []
        for ref in ("STIT", "IST", "STI"):
            if ref in metas:
                parts.append(f"vs {ref} **{g - metas[ref]['g_acc']*100:+.1f}**")
        st.caption(
            f"**StIT** (System·**t**·Image·Task — pre-image slot is a *fixed generic* "
            f"image-analysis instruction, the real question follows the image): "
            f"Group-Acc **{g:.1f}%**" + (" — " + " · ".join(parts) if parts else "")
            + ". If StIT ≈ STIT, the pre-image text need not be the question — a generic "
            "instruction suffices as long as the question comes after the image.")

    # SITT: System · Image · Task · Task — image shown once, task stated twice
    # (no image echo — isolates whether repeating the TASK alone, without a
    # 2nd image occurrence, recovers STI, vs STIT which repeats task+image).
    if "SITT" in metas:
        g = metas["SITT"]["g_acc"] * 100
        parts = []
        for ref in ("STIT", "SIT", "STI", "IST"):
            if ref in metas:
                parts.append(f"vs {ref} **{g - metas[ref]['g_acc']*100:+.1f}**")
        st.caption(
            f"**SITT** (System·Image·Task·Task — image shown once, question "
            f"repeated twice with *no* 2nd image occurrence): Group-Acc "
            f"**{g:.1f}%**" + (" — " + " · ".join(parts) if parts else "")
            + ". If SITT ≈ STIT, repeating the task alone (no image echo) is "
            "enough for the recovery.")

    # ── separate tables: resolution control + image-copy distance sweep ─────────
    _render_resolution_table(model)
    _render_echo_resolution_table(model)
    _render_imagecopies_table(model)

    # ── per-experiment + cross-experiment browsers ─────────────────────────────
    st.markdown("#### Browse groups")
    available_orders = [o for o in ("IST", "STI") if o in comp]

    # cross-experiment comparison tabs only make sense when both exist
    have_both = "IST" in comp and "STI" in comp
    cmp_specs = []
    if have_both:
        cmp_specs = [
            ("IST ✓ / STI ✗", "ist_right_sti_wrong"),
            ("STI ✓ / IST ✗", "sti_right_ist_wrong"),
            ("Both ✗ (IST & STI)", "both_wrong"),
            ("Both ✓ (IST & STI)", "both_correct"),
        ]

    gran = "group"
    if have_both:
        gran_label = st.radio(
            "Cross-experiment tabs — comparison unit",
            ["Group level (all 4 pairs)", "Pair level (single image-question)"],
            index=0, horizontal=True, key=f"cmp_gran_{model}",
            help="Group = the whole group counts right only if all 4 pairs are "
                 "right. Pair = compare each (image, question) pair independently.",
        )
        gran = "pair" if gran_label.startswith("Pair") else "group"

    labels = [f"Experiment {o}" for o in available_orders] + [s[0] for s in cmp_specs]
    tabs = st.tabs(labels)

    for i, order in enumerate(available_orders):
        with tabs[i]:
            _browse_experiment(model, order)

    offset = len(available_orders)
    for j, (lab, mode) in enumerate(cmp_specs):
        with tabs[offset + j]:
            _browse_comparison(model, mode, lab, granularity=gran)


# ══════════════════════════════════════════════════════════════════════════════
#  Page entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_naturalbench_page():
    st.subheader("🧪 NaturalBench — Prompt-Ordering Experiments (IST vs STI)")
    st.caption(
        "NaturalBench pairs **2 images × 2 questions** (4 pairs/group) so a "
        "vision-blind model can't win. We compare two prompt orderings — "
        "**IST** (Image·System·Task) vs **STI** (System·Task·Image) — and report "
        "Group / Question / Image / Pair accuracy. Correct (all-4-right) and "
        "wrong groups are saved to separate JSON files."
    )
    _run_section()
    st.divider()
    _results_section()
