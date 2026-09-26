#!/usr/bin/env python3
"""Smoke test: one image-only logit-lens pass, print the decoded patch-word grid."""
import sys
import numpy as np, torch
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from logit_lens_overlay import logit_lens_all_vision_tokens

LAYER = 28

def main():
    path = sys.argv[1]
    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    pil = Image.open(path).convert("RGB")
    long = 896
    s = long / max(pil.size)
    if s < 1:
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
    print("image", pil.size)
    mm = ModelManager("qwen3-vl-8b")
    print("layers", mm.num_layers)
    _, input_ids, kwargs = mm.prepare_inputs_from_pil([""], pil, system_prompt="", order="I")
    print("grid", mm.grid_h, mm.grid_w, "span", mm.img_start_idx, mm.img_end_idx)
    with torch.inference_mode():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
    outputs = {"hidden_states": (out.hidden_states,)}
    warper = TopKLogitsWarper(top_k=50, filter_value=float("-inf"))
    probs, words = logit_lens_all_vision_tokens(
        mm.llm_model, mm.tokenizer, input_ids, outputs, mm.img_start_idx,
        [LAYER], warper, LogitsProcessorList([]), grid_h=mm.grid_h, grid_w=mm.grid_w)
    w = np.array(words[0]).reshape(mm.grid_h, mm.grid_w)
    for r in range(0, mm.grid_h, 2):
        print(" ".join(f"{x.strip()[:8]:>8s}" for x in w[r][::2]))

if __name__ == "__main__":
    main()
