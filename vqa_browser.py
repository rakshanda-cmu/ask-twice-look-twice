"""
Streamlit VQA evaluation browser.

Reads the analysis JSON produced by `vqa_eval.py` and lets you browse the
examples the model got right vs. wrong — showing, for each, the image, the
question, the dataset's expected answer(s), and the model's predicted answer.

Used as a tab inside logit_lens_app.py:
    from vqa_browser import render_vqa_browser
    render_vqa_browser()
"""

import glob
import json
import os

import streamlit as st
from PIL import Image


def _find_analysis_files(vqa_dir="./vqa"):
    return sorted(glob.glob(os.path.join(vqa_dir, "vqa_analysis_*.json")))


@st.cache_data(show_spinner="Loading VQA analysis …")
def _load_analysis(path, _mtime):
    with open(path) as f:
        return json.load(f)


def _answer_badge(correct):
    if correct:
        return ":green-background[✓ CORRECT]"
    return ":red-background[✗ WRONG]"


def _render_example(ex, data_dir, idx):
    """Render a single example card."""
    img_path = os.path.join(data_dir, ex["image_file"])
    col_img, col_info = st.columns([1, 2], gap="medium")

    with col_img:
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")
        else:
            st.warning(f"Image not found:\n{img_path}")
        st.caption(f"`{ex['image_file']}`  ·  qid `{ex['question_id']}`")

    with col_info:
        st.markdown(f"**Q:** {ex['question']}  {_answer_badge(ex['correct'])}")

        gt_common = ex.get("gt_most_common", "")
        st.markdown(f"**Expected answer (dataset):** `{gt_common}`")

        # full distribution of the 10 human answers
        from collections import Counter
        dist = Counter(ex.get("gt_answers", []))
        dist_str = " · ".join(f"`{a}`×{c}" for a, c in dist.most_common())
        st.caption(f"All human answers: {dist_str}")

        if ex["correct"]:
            st.markdown(f"**Model answer:** :green[{ex['model_answer_raw']}]")
        else:
            st.markdown(f"**Model answer:** :red[{ex['model_answer_raw']}]")

        meta_bits = [
            f"VQA score: **{ex.get('vqa_score', 0):.2f}**",
            f"type: `{ex.get('answer_type', '?')}`",
        ]
        if ex.get("matched_gt"):
            meta_bits.append(f"matched: {', '.join('`'+m+'`' for m in ex['matched_gt'])}")
        st.caption("  ·  ".join(meta_bits))

        # hand off to the Logit Lens tab
        if st.button("🔍 Inspect in Logit Lens",
                     key=f"send_{ex['question_id']}_{idx}"):
            st.session_state["lens_coco_id"] = str(ex["image_id"])
            st.session_state["lens_query"] = ex["question"]
            st.session_state["jump_to_lens"] = True
            st.rerun()

    st.divider()


def render_vqa_browser(default_data_dir="./COCO/val2014", vqa_dir="./vqa"):
    st.subheader("📊 VQA Evaluation Browser")
    st.caption(
        "Browse VQA v2 (val) examples the model answered right vs. wrong. "
        "Generate the data with `python download_vqa.py` then "
        "`python vqa_eval.py --num-samples 5000 --model qwen3-vl-8b`."
    )

    files = _find_analysis_files(vqa_dir)
    if not files:
        st.info(
            f"No analysis file found in `{vqa_dir}/`. "
            "Run `python vqa_eval.py` to generate `vqa_analysis_<model>.json`."
        )
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        path = st.selectbox(
            "Analysis file",
            files,
            format_func=lambda p: os.path.basename(p),
        )
    with c2:
        data_dir = st.text_input("COCO val2014 dir", value=default_data_dir)

    data = _load_analysis(path, os.path.getmtime(path))
    meta = data.get("meta", {})
    results = data.get("results", [])

    # ── summary metrics ────────────────────────────────────────────────────────
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_wrong = n - n_correct
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Examples", n)
    m2.metric("Correct", n_correct, f"{(n_correct/max(1,n))*100:.1f}%")
    m3.metric("Wrong", n_wrong, f"{(n_wrong/max(1,n))*100:.1f}%",
              delta_color="inverse")
    m4.metric("VQA accuracy", f"{meta.get('vqa_accuracy', 0)*100:.1f}%")

    with st.expander("Run metadata"):
        st.json(meta)

    # ── filters ─────────────────────────────────────────────────────────────────
    st.markdown("##### Filters")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        verdict = st.radio(
            "Show", ["Wrong only", "Correct only", "All"],
            index=0, horizontal=False,
        )
    with f2:
        answer_types = sorted({r.get("answer_type", "?") for r in results})
        atype = st.selectbox("Answer type", ["all"] + answer_types)
    with f3:
        search = st.text_input(
            "Search question / answer text", value="",
            placeholder="e.g. color, dog, how many …",
        )

    # apply filters
    def keep(r):
        if verdict == "Wrong only" and r["correct"]:
            return False
        if verdict == "Correct only" and not r["correct"]:
            return False
        if atype != "all" and r.get("answer_type", "?") != atype:
            return False
        if search:
            s = search.lower()
            hay = (r["question"] + " " + r["model_answer_raw"] + " "
                   + " ".join(r.get("gt_answers", []))).lower()
            if s not in hay:
                return False
        return True

    filtered = [r for r in results if keep(r)]
    st.caption(f"**{len(filtered)}** example(s) match the current filters.")

    if not filtered:
        return

    # ── pagination ──────────────────────────────────────────────────────────────
    per_page = st.select_slider("Examples per page", [5, 10, 20, 50], value=10)
    n_pages = (len(filtered) + per_page - 1) // per_page
    page = st.number_input(
        f"Page (1–{n_pages})", min_value=1, max_value=n_pages, value=1, step=1,
    )
    start = (page - 1) * per_page
    page_items = filtered[start: start + per_page]

    st.divider()
    for i, ex in enumerate(page_items):
        _render_example(ex, data_dir, start + i)
