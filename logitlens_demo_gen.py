"""
Per-layer logit-lens GIFs for ONE image (the shared example), Qwen3-VL-8B, for the
two questions crossed with the two orderings:
    q1="What sport is on TV?"  q2="What is the cat doing?"   ×   IST , STI     (4 GIFs)

Each GIF animates the logit lens through EVERY layer — the image-patch heatmap (what
each patch decodes to) plus the text/GENERATED token grid (including the model's
generated answer decoded at each layer). Reuses sitit_stit_gif_gen.one() (the same
single-image render used elsewhere), so the 4 GIFs are directly comparable.

Read-only artifact → logitlens_demo/{qid}_{order}.gif + manifest.json, rendered side
by side by the "🔬 Logit-lens (this image)" tab.

    CUDA_VISIBLE_DEVICES=0 python logitlens_demo_gen.py --image examples/<img>.png
"""
import argparse, json, os
from PIL import Image
from sitit_stit_gif_gen import one
from model_manager import ModelManager

OUT = "logitlens_demo"
QUESTIONS = [("q1", "What sport is on TV?"), ("q2", "What is the cat doing?")]
ORDERS = ["IST", "STI"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--layer-step", type=int, default=1, dest="layer_step")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize the image by this factor before the model (0.5 = half res)")
    ap.add_argument("--out-dir", default=OUT, dest="out_dir")
    args = ap.parse_args()
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    setup_seeds(); disable_torch_init()
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    if args.scale != 1.0:
        pil = pil.resize((max(1, int(pil.width * args.scale)),
                          max(1, int(pil.height * args.scale))), Image.LANCZOS)
    mm = ModelManager(args.model)
    layer_range = list(range(0, mm.num_layers, args.layer_step))
    print(f"[ll] {args.image} scale={args.scale} -> {pil.size} · "
          f"{len(QUESTIONS)}q × {len(ORDERS)} orders · {len(layer_range)} layers", flush=True)

    man = {"image": os.path.basename(args.image), "layers": layer_range,
           "scale": args.scale, "img_size": list(pil.size),
           "questions": [{"id": q, "text": t} for q, t in QUESTIONS],
           "orders": ORDERS, "cells": {}}
    for qid, qtext in QUESTIONS:
        for order in ORDERS:
            ans, gif = one(mm, pil, qtext, order, layer_range)
            fn = f"{qid}_{order}.gif"
            open(f"{outdir}/{fn}", "wb").write(gif)
            man["cells"].setdefault(qid, {})[order] = {"answer": ans, "gif": fn}
            json.dump(man, open(f"{outdir}/manifest.json", "w"), indent=2)
            print(f"  {qid} {order:4s} -> answer={ans!r}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
