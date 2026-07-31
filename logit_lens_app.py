"""
Streamlit UI for the Logit Lens overlay visualizer.

Run:
    CUDA_VISIBLE_DEVICES=0 conda run -n logitlens streamlit run logit_lens_app.py
"""

import io
import os
import sys

import imageio
import matplotlib
matplotlib.use("Agg")
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

st.set_page_config(
    page_title="VLM Logit Lens",
    page_icon="🔍",
    layout="wide",
)

sys.path.insert(0, os.path.dirname(__file__))

from logit_lens_overlay import (
    logit_lens_all_vision_tokens,
    compute_text_logit_lens,
    render_combined_frame,
    _make_colorbar_frame,
    GRID_SIZE,
)
from model_manager import ModelManager, QWEN_MODELS
from constants import SYSTEM_MESSAGE
from naturalbench_browser import render_naturalbench_page
from midlayer_browser import (
    render_midlayer_page, render_midlayer_space_page, render_stit_compare_page,
)
from probe_browser import render_probe_page
from decision_browser import render_decision_page
from gemma_browser import render_gemma_page
from pope_browser import render_pope_page
from rf20_browser import render_rf20_page
from detpo_map_browser import render_detpo_map_page
from extra_tasks_browser import render_extra_tasks_page
from refcoco_gaze.gaze_browser import render_gaze_page
from patchcos_browser import render_patchcos_page
from sitit_compare_browser import render_sitit_compare_page
from logitlens_demo_browser import render_logitlens_demo_page
from summary_browser import render_summary_page


# ── model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model … (one-time)")
def load_model(model_name: str):
    from utils import setup_seeds, disable_torch_init
    setup_seeds()
    disable_torch_init()
    return ModelManager(model_name)


# ── pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(
        model_manager, img: Image.Image, query: str,
        system_prompt: str, order: str,
        layer_range: list, top_k: int, max_tokens: int,
        resolution: int, alpha: float, fps: float,
        show_text_lens: bool,
        progress_cb=None,
):
    """Returns (answer, frames, gif_bytes)."""
    if progress_cb:
        progress_cb(0.05, "Preparing inputs …")

    _, input_ids, kwargs = model_manager.prepare_inputs_from_pil(
        [query], img, system_prompt=system_prompt, order=order,
    )
    grid_h       = model_manager.grid_h
    grid_w       = model_manager.grid_w
    section_info = model_manager.section_info

    if progress_cb:
        progress_cb(0.10, "Running model inference …")

    with torch.inference_mode():
        outputs = model_manager.llm_model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )

    answer = model_manager.tokenizer.batch_decode(
        outputs["sequences"], skip_special_tokens=True
    )[0].strip()

    logits_warper    = TopKLogitsWarper(top_k=top_k, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])

    if progress_cb:
        progress_cb(0.30, "Computing vision logit lens …")

    all_probs, all_words = logit_lens_all_vision_tokens(
        model_manager.llm_model, model_manager.tokenizer,
        input_ids, outputs,
        model_manager.img_start_idx,
        layer_range, logits_warper, logits_processor,
        grid_h=grid_h, grid_w=grid_w,
    )

    text_data = None
    if show_text_lens:
        if progress_cb:
            progress_cb(0.45, "Computing text logit lens …")
        text_data = compute_text_logit_lens(
            model_manager.llm_model, model_manager.tokenizer,
            input_ids, outputs,
            model_manager.img_start_idx, model_manager.img_end_idx,
            layer_range, logits_warper, logits_processor,
        )

    disp_image = img.resize((resolution, resolution), Image.LANCZOS)
    colorbar   = _make_colorbar_frame(resolution, resolution)
    frames     = []
    n_layers   = len(layer_range)

    for fi, layer_idx in enumerate(layer_range):
        if progress_cb:
            pct = 0.50 + 0.48 * (fi + 1) / n_layers
            progress_cb(pct, f"Rendering layer {layer_idx} ({fi+1}/{n_layers}) …")

        frame = render_combined_frame(
            disp_image, all_probs[fi], all_words[fi],
            layer_idx, fi, text_data, alpha,
            section_info=section_info,
            grid_h=grid_h, grid_w=grid_w,
        )
        full = Image.new("RGB", (frame.width, frame.height + colorbar.height))
        full.paste(frame,    (0, 0))
        full.paste(colorbar, (0, frame.height))
        frames.append(full)

    duration_ms = int(1000 / fps)
    buf = io.BytesIO()
    imageio.mimsave(buf, [np.array(f) for f in frames],
                    format="GIF", duration=duration_ms, loop=0)
    gif_bytes = buf.getvalue()

    if progress_cb:
        progress_cb(1.0, "Done!")

    return answer, frames, gif_bytes


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🔍 VLM Logit Lens — Vision & Text Token Grid")
st.caption(
    "Projects each token's hidden state through the LM head at every selected layer. "
    "**Vision panel**: spatial heatmap of what text concept each image patch represents. "
    "**Token grid** (below): all tokens in prompt-sequence order — "
    "system → image → task → generated — each cell the same size as one image patch."
)

# ── page switch ───────────────────────────────────────────────────────────────
# Adds a separate NaturalBench experiments page without altering the existing
# logit-lens / heatmap UI below.
_page = st.sidebar.radio(
    "Page",
    ["🔍 Logit Lens", "🧪 NaturalBench Experiments", "🔬 Middle-Layer Analysis",
     "␣ Middle-Layer + Spaces", "🔁 STIT vs STI / IST", "🧠 Mechanism Probe",
     "🎯 Decision Layer", "🔷 Gemma 3", "🟣 POPE", "🟩 RF20",
     "🛩️ DetPO mAP (RF20 Aerial + RefCOCO)",
     "🧪 New CV Areas (VQA/Counting/MMVP/BLINK/NExT-QA)",
     "👁️ RefCOCO-Gaze (Grad-CAM vs human attention)",
     "🖼️ Patch Perturbation", "🔬 Logit Lens (this image)", "🎞️ SITIT vs STIT",
     "📊 Cross-Dataset Summary"],
    key="app_page",
)
if _page == "🧪 NaturalBench Experiments":
    render_naturalbench_page()
    st.stop()
if _page == "🔬 Middle-Layer Analysis":
    render_midlayer_page()
    st.stop()
if _page == "␣ Middle-Layer + Spaces":
    render_midlayer_space_page()
    st.stop()
if _page == "🔁 STIT vs STI / IST":
    render_stit_compare_page()
    st.stop()
if _page == "🧠 Mechanism Probe":
    render_probe_page()
    st.stop()
if _page == "🎯 Decision Layer":
    render_decision_page()
    st.stop()
if _page == "🔷 Gemma 3":
    render_gemma_page()
    st.stop()
if _page == "🟣 POPE":
    render_pope_page()
    st.stop()
if _page == "🟩 RF20":
    render_rf20_page()
    st.stop()
if _page == "🛩️ DetPO mAP (RF20 Aerial + RefCOCO)":
    render_detpo_map_page()
    st.stop()
if _page == "🧪 New CV Areas (VQA/Counting/MMVP/BLINK/NExT-QA)":
    render_extra_tasks_page()
    st.stop()
if _page == "👁️ RefCOCO-Gaze (Grad-CAM vs human attention)":
    render_gaze_page()
    st.stop()
if _page == "🖼️ Patch Perturbation":
    render_patchcos_page()
    st.stop()
if _page == "🔬 Logit Lens (this image)":
    render_logitlens_demo_page()
    st.stop()
if _page == "🎞️ SITIT vs STIT":
    render_sitit_compare_page()
    st.stop()
if _page == "📊 Cross-Dataset Summary":
    render_summary_page()
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # ── model ─────────────────────────────────────────────────────────────────
    st.subheader("Model")
    MODEL_OPTIONS = {
        "LLaVA-1.5 (7B)":   "llava-1.5",
        "Qwen2.5-VL (7B)":  "qwen2.5-vl-7b",
        "Qwen3-VL (8B)":    "qwen3-vl-8b",
    }
    model_label    = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=2)
    selected_model = MODEL_OPTIONS[model_label]

    # ── prompt format ─────────────────────────────────────────────────────────
    st.subheader("Prompt format")

    ORDER_OPTIONS = {
        "SIT — System · Image · Task":  "SIT",
        "IST — Image · System · Task":  "IST",
        "ITS — Image · Task · System":  "ITS",
        "STI — System · Task · Image":  "STI",
        "TIS — Task · Image · System":  "TIS",
        "TSI — Task · System · Image":  "TSI",
        "IT  — Image · Task (no sys)":  "IT",
        "TI  — Task · Image (no sys)":  "TI",
    }
    # Sensible defaults per model family
    default_order_label = (
        "IT  — Image · Task (no sys)" if selected_model in QWEN_MODELS
        else "SIT — System · Image · Task"
    )
    order_label = st.selectbox(
        "Section order", list(ORDER_OPTIONS.keys()),
        index=list(ORDER_OPTIONS.keys()).index(default_order_label),
    )
    order = ORDER_OPTIONS[order_label]

    default_sys = "" if selected_model in QWEN_MODELS else SYSTEM_MESSAGE
    system_prompt = st.text_area(
        "System prompt",
        value=default_sys,
        height=100,
        help="Ignored if 'S' is not in the order string above.",
    )

    # ── layers ────────────────────────────────────────────────────────────────
    st.subheader("Layers")
    NUM_LAYERS_BY_MODEL = {"llava-1.5": 32, "qwen2.5-vl-7b": 28, "qwen3-vl-8b": 36}
    if selected_model in st.session_state.get("loaded_models", {}):
        n_layers_total = st.session_state["loaded_models"][selected_model]
    else:
        n_layers_total = NUM_LAYERS_BY_MODEL.get(selected_model, 32)
    ALL_LAYERS = list(range(n_layers_total))
    step = max(1, n_layers_total // 18)
    default_layers = list(range(0, n_layers_total, step))
    layer_range = st.multiselect(
        "Layers to visualize", options=ALL_LAYERS, default=default_layers,
    )
    if not layer_range:
        st.warning("Select at least one layer.")
    layer_range = sorted(layer_range)

    # ── vision overlay ────────────────────────────────────────────────────────
    st.subheader("Vision overlay")
    resolution = st.select_slider(
        "Output resolution (px)",
        options=[576, 768, 960, 1152, 1440], value=1152,
    )
    alpha = st.slider("Heatmap blend α", 0.1, 0.9, 0.55, step=0.05,
                      help="0 = image only · 1 = heatmap only")

    # ── text token grid ───────────────────────────────────────────────────────
    st.subheader("Text token grid")
    show_text_lens = st.toggle("Show token grid", value=True)

    # ── animation ─────────────────────────────────────────────────────────────
    st.subheader("Animation")
    fps = st.slider("GIF speed (fps)", 0.5, 5.0, 1.5, step=0.5)

    # ── inference ─────────────────────────────────────────────────────────────
    st.subheader("Inference")
    max_tokens = st.slider("Max new tokens", 64, 1024, 256, step=64)
    top_k      = st.slider("Logit lens top-k", 10, 200, 50, step=10)

# ── main panel ────────────────────────────────────────────────────────────────
col_img, col_prompt = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("Image")
    upload = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    coco_id   = st.text_input("…or COCO val2014 image ID", placeholder="e.g. 499775", value="000000499775")
    data_path = st.text_input("COCO val2014 directory", value="./COCO/val2014/")

with col_prompt:
    st.subheader("Task prompt")
    query = st.text_area(
        "Prompt",
        value="Please help me describe the image in detail.",
        height=120,
        label_visibility="collapsed",
    )

# resolve image
img_pil = None
if upload is not None:
    img_pil = Image.open(upload).convert("RGB")
    col_img.image(img_pil, caption="Uploaded image", width='stretch')
elif coco_id.strip():
    try:
        path    = os.path.join(data_path, f"COCO_val2014_{str(int(coco_id)).zfill(12)}.jpg")
        img_pil = Image.open(path).convert("RGB")
        col_img.image(img_pil, caption=f"COCO id={coco_id}", width='stretch')
    except Exception as e:
        col_img.error(f"Could not load image: {e}")

run_btn = st.button(
    "▶ Run Logit Lens",
    type="primary",
    disabled=(img_pil is None or not layer_range),
    width='stretch',
)

if img_pil is None:
    st.info("Upload an image or enter a COCO image ID to get started.")

if run_btn and img_pil is not None and layer_range:
    model_manager = load_model(selected_model)
    if "loaded_models" not in st.session_state:
        st.session_state["loaded_models"] = {}
    st.session_state["loaded_models"][selected_model] = model_manager.num_layers

    progress_bar = st.progress(0.0, text="Starting …")
    def _progress(pct, msg):
        progress_bar.progress(pct, text=msg)

    with st.spinner(""):
        answer, frames, gif_bytes = run_pipeline(
            model_manager, img_pil, query,
            system_prompt=system_prompt,
            order=order,
            layer_range=layer_range,
            top_k=top_k, max_tokens=max_tokens,
            resolution=resolution, alpha=alpha, fps=fps,
            show_text_lens=show_text_lens,
            progress_cb=_progress,
        )

    progress_bar.empty()

    st.subheader("Generated answer")
    st.info(answer)

    st.subheader("Logit Lens animation")
    st.image(gif_bytes, width='stretch')
    st.download_button("⬇ Download GIF", data=gif_bytes,
                       file_name="logit_lens_overlay.gif", mime="image/gif")

    st.subheader("Per-layer frames")
    n_cols = min(3, len(frames))
    cols   = st.columns(n_cols)
    for fi, (layer_idx, frame) in enumerate(zip(layer_range, frames)):
        with cols[fi % n_cols]:
            st.image(frame, caption=f"Layer {layer_idx}", width='stretch')
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            st.download_button(
                f"⬇ Layer {layer_idx}",
                data=buf.getvalue(),
                file_name=f"layer_{layer_idx:03d}.png",
                mime="image/png",
                key=f"dl_{layer_idx}",
            )
