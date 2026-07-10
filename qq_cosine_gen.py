"""
Question↔Question image-representation cosine, on ONE image (the shared example),
Qwen3-VL-8B. Two different questions are asked about the SAME image
  q1 = "What sport is on TV?"   q2 = "What is the cat doing?"
and for each prompt ordering (IST / STI / STIT) we take the image-token LLM hidden
states and measure, per patch, cos( h(q1) , h(q2) ) — how much does *which question
you ask* rewrite each visual patch?

Because attention is causal over image tokens:
  IST  (image first)          → the image never sees the question → cos ≈ 1.0
  STI/STIT (question first)    → the question precedes the image  → cos < 1.0 where the
                                 differing questions perturb the patch differently.

Read-only artifact: writes attn_demo/qq_cosine.json (per-patch map at a mid layer +
mean-cos-vs-layer curve + scalar means), rendered by the 🖼️ Patch Perturbation tab.

    CUDA_VISIBLE_DEVICES=0 python qq_cosine_gen.py --image examples/<img>.png
"""
import argparse, json, os
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from constants import SYSTEM_MESSAGE

OUT = "attn_demo"
QUESTIONS = [("q1", "What sport is on TV?"), ("q2", "What is the cat doing?")]
ORDERS = ["SIT", "STI", "STIT"]
LAYER = 18                      # representative mid layer (matches the patch-cosine tab)


def img_hidden(mm, pil, question, order):
    """All-layer image-token hidden states for one (question, ordering) → [n_layers+1, P, hid]."""
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [question], pil, system_prompt=SYSTEM_MESSAGE, order=order)
    gh, gw = mm.grid_h, mm.grid_w
    s, e = mm.img_start_idx, mm.img_end_idx
    with torch.inference_mode():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
    hs = torch.stack([h[0, s:e].float() for h in out.hidden_states])   # [L+1, P, hid]
    return hs.cpu(), gh, gw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="qwen3-vl-8b")
    args = ap.parse_args()
    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    setup_seeds(); disable_torch_init()
    os.makedirs(OUT, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    print(f"[qq-cos] {args.image} {pil.size} · Qwen sdpa", flush=True)
    mm = ModelManager(args.model, attn_implementation="sdpa")

    man = {"image": os.path.basename(args.image), "source": "_source.png",
           "questions": [{"id": q, "text": t} for q, t in QUESTIONS],
           "layer": LAYER, "orders": {}}
    gh = gw = None
    for order in ORDERS:
        h1, gh, gw = img_hidden(mm, pil, QUESTIONS[0][1], order)
        h2, _, _ = img_hidden(mm, pil, QUESTIONS[1][1], order)
        P = min(h1.shape[1], h2.shape[1])
        cos = F.cosine_similarity(h1[:, :P], h2[:, :P], dim=-1)   # [L+1, P]
        by_layer = cos.mean(1).tolist()                          # mean over patches, per layer
        li = min(LAYER, cos.shape[0] - 1)
        man["orders"][order] = {
            "cos_map": cos[li].tolist(),                         # per-patch at LAYER
            "mean": float(cos[li].mean()),
            "cos_by_layer": by_layer,
        }
        print(f"  {order:4s}  layer{li} mean cos(q1,q2) = {cos[li].mean():.4f}  "
              f"(range {cos[li].min():.3f}-{cos[li].max():.3f})", flush=True)
    man["grid"] = [gh, gw]
    man["n_layers"] = len(man["orders"][ORDERS[0]]["cos_by_layer"])
    json.dump(man, open(f"{OUT}/qq_cosine.json", "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
