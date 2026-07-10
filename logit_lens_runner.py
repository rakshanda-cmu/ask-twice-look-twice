"""
Reusable, non-Streamlit logit-lens computation.

This mirrors `run_pipeline()` in logit_lens_app.py but is import-safe (it does NOT
run any Streamlit code) and returns the per-layer frames directly so other pages
(e.g. the Middle-Layer Analysis tab) can render them. The existing Logit Lens page
and its run_pipeline are left untouched.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

sys.path.insert(0, os.path.dirname(__file__))

from logit_lens_overlay import (
    logit_lens_all_vision_tokens,
    compute_text_logit_lens,
    render_combined_frame,
    _make_colorbar_frame,
)
from model_manager import QWEN_MODELS


def run_logit_lens(model_manager, img, query, system_prompt, order, layer_range,
                   top_k=50, max_tokens=16, resolution=768, alpha=0.55,
                   show_text_lens=False, n_spaces=0):
    """
    Run the logit lens for one (image, query) under a given prompt `order`.

    n_spaces > 0 inserts that many space tokens after the Task (IST) / image (STI)
    via model_manager.prepare_inputs_with_spaces.

    Returns (answer:str, frames:list[PIL.Image]) where frames[i] corresponds to
    layer_range[i] (vision heatmap panel + optional token grid + colorbar).
    """
    if n_spaces and n_spaces > 0:
        _, input_ids, kwargs = model_manager.prepare_inputs_with_spaces(
            [query], img, system_prompt=system_prompt, order=order, n_spaces=n_spaces,
        )
    else:
        _, input_ids, kwargs = model_manager.prepare_inputs_from_pil(
            [query], img, system_prompt=system_prompt, order=order,
        )
    grid_h = model_manager.grid_h
    grid_w = model_manager.grid_w
    section_info = model_manager.section_info

    with torch.inference_mode():
        outputs = model_manager.llm_model.generate(
            input_ids,
            do_sample=False, num_beams=1,
            max_new_tokens=max_tokens, use_cache=True,
            output_hidden_states=True, return_dict_in_generate=True,
            **kwargs,
        )

    # answer (slice off prompt for Qwen; split ASSISTANT turn for LLaVA)
    if model_manager.model_name in QWEN_MODELS:
        gen = outputs["sequences"][:, input_ids.shape[1]:]
        answer = model_manager.tokenizer.batch_decode(
            gen, skip_special_tokens=True)[0].strip()
    else:
        answer = model_manager.tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=True)[0].strip()
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()

    logits_warper = TopKLogitsWarper(top_k=top_k, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])

    all_probs, all_words = logit_lens_all_vision_tokens(
        model_manager.llm_model, model_manager.tokenizer,
        input_ids, outputs, model_manager.img_start_idx,
        layer_range, logits_warper, logits_processor,
        grid_h=grid_h, grid_w=grid_w,
    )

    text_data = None
    if show_text_lens:
        text_data = compute_text_logit_lens(
            model_manager.llm_model, model_manager.tokenizer,
            input_ids, outputs,
            model_manager.img_start_idx, model_manager.img_end_idx,
            layer_range, logits_warper, logits_processor,
        )

    disp_image = img.resize((resolution, resolution), Image.LANCZOS)
    colorbar = _make_colorbar_frame(resolution, resolution)
    frames = []
    for fi, layer_idx in enumerate(layer_range):
        frame = render_combined_frame(
            disp_image, all_probs[fi], all_words[fi],
            layer_idx, fi, text_data, alpha,
            section_info=section_info, grid_h=grid_h, grid_w=grid_w,
        )
        full = Image.new("RGB", (frame.width, frame.height + colorbar.height))
        full.paste(frame, (0, 0))
        full.paste(colorbar, (0, frame.height))
        frames.append(full)

    return answer, frames
