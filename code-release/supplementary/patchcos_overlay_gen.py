"""
Per-patch cosine-similarity HEATMAPS (overlaid on the image) of the image
representation under an ordering vs the RAW image-only run, on ONE image, Qwen3-VL-8B.

Columns:  [Image = raw reference]  [IST]  [STI]
For each ordering we take the image-token LLM hidden states and, per patch and per
layer, measure cos( h(raw image only) , h(ordering) ). We overlay (1 - cos) — how much
the surrounding text has perturbed that patch — on the image (red = perturbed).

Two questions on the SAME image; for each: GIF (all layers) / Mean (over layers) /
Final layer  →  6 rows × 3 columns.

Because attention is causal over image tokens:
  IST (image first)        → image never sees the question → cos ≈ 1.0 (flat, cold) for
                             BOTH questions.
  STI (question first)     → the question precedes the image → cos < 1.0, and the map
                             DIFFERS between the two questions (each question rewrites
                             different patches).

A single FIXED color scale is used across every cell/layer/question so IST reads as
flat and STI's perturbation is directly comparable.

Read-only artifact → patchcos_demo/ (+ manifest.json), rendered by the 🖼️ Patch
Perturbation tab, added below the existing content.

    CUDA_VISIBLE_DEVICES=0 python patchcos_overlay_gen.py --image examples/<img>.png
"""
import argparse, io, json, os
import numpy as np, torch
import torch.nn.functional as F
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from constants import SYSTEM_MESSAGE

OUT = "patchcos_demo"
DISP_W, ALPHA, FPS, CMAP = 620, 0.6, 2.0, "jet"
QUESTIONS = [("q1", "What sport is on TV?"), ("q2", "What is the cat doing?")]
ORDERS = [("IST", "IST"), ("STI", "STI"), ("STIT", "STIT")]   # cols after raw "Image"


def img_hidden(mm, pil, question, order, with_system=True):
    """All-layer image-token hidden states for one (question, ordering) → [L+1, P, hid]."""
    sys_text = SYSTEM_MESSAGE if with_system else ""
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [question], pil, system_prompt=sys_text, order=order)
    gh, gw = mm.grid_h, mm.grid_w
    s, e = mm.img_start_idx, mm.img_end_idx
    with torch.inference_mode():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
    hs = torch.stack([h[0, s:e].float() for h in out.hidden_states])   # [L+1, P, hid]
    return hs.cpu(), gh, gw


def overlay(disp, grid01):
    hm = plt.get_cmap(CMAP)(grid01)[:, :, :3]
    hm = Image.fromarray((hm * 255).astype(np.uint8)).resize(disp.size, Image.BICUBIC)
    return Image.blend(disp, hm, ALPHA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize the image by this factor before the model (0.5 = half res)")
    ap.add_argument("--out-dir", default=OUT, dest="out_dir")
    args = ap.parse_args()
    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    setup_seeds(); disable_torch_init()
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    if args.scale != 1.0:
        pil = pil.resize((max(1, int(pil.width * args.scale)),
                          max(1, int(pil.height * args.scale))), Image.LANCZOS)
    disp = pil.resize((DISP_W, int(pil.height * DISP_W / pil.width)), Image.LANCZOS)
    disp.save(f"{outdir}/_source.png")
    print(f"[patchcos] {args.image} scale={args.scale} -> {pil.size} · Qwen sdpa", flush=True)
    mm = ModelManager(args.model, attn_implementation="sdpa")

    # base = image alone (no system, no question) — the raw image representation
    base, gh, gw = img_hidden(mm, pil, "", "I", with_system=False)
    P0 = base.shape[1]

    # cos-to-base per (question, ordering) → dev[qid][oid] = (1-cos) grid stack [L, gh, gw]
    # identity[qid][oid] = mean relative-L2 ‖h-base‖/‖base‖ at the final layer: cosine
    # only measures direction, so this proves IST is *bit-for-bit* identical (→ 0), not
    # merely same-direction.
    dev, identity = {}, {}
    for qid, qtext in QUESTIONS:
        dev[qid] = {}; identity[qid] = {}
        for oid, order in ORDERS:
            h, _, _ = img_hidden(mm, pil, qtext, order)
            P = min(P0, h.shape[1])
            cos = F.cosine_similarity(base[:, :P], h[:, :P], dim=-1)   # [L+1, P]
            d = (1.0 - cos).clamp(min=0)                              # perturbation
            grids = d[:, :gh * gw].reshape(d.shape[0], gh, gw).numpy()
            dev[qid][oid] = grids
            rel = ((h[-1, :P] - base[-1, :P]).norm(dim=-1) /
                   (base[-1, :P].norm(dim=-1) + 1e-9)).mean().item()
            identity[qid][oid] = rel
            print(f"  {qid} {oid:4s}  mean(1-cos)@final={grids[-1].mean():.4f} "
                  f"max={grids[-1].max():.3f}  rel-L2@final={rel:.4f}", flush=True)

    # one FIXED scale across everything so IST reads flat and STI is comparable
    allv = np.concatenate([dev[q][o].ravel() for q in dev for o in dev[q]])
    vmax = float(np.percentile(allv, 99)) or 1e-6
    print(f"  fixed vmax(1-cos) = {vmax:.4f}", flush=True)

    def norm(g):
        return np.clip(g / vmax, 0, 1)

    man = {"image": os.path.basename(args.image), "source": "_source.png",
           "questions": [{"id": q, "text": t} for q, t in QUESTIONS],
           "cols": ["Image"] + [o for o, _ in ORDERS], "grid": [gh, gw],
           "scale": args.scale, "img_size": list(pil.size), "identity": identity,
           "vmax": vmax, "n_layers": int(base.shape[0]), "rows": {}}
    for qid, _ in QUESTIONS:
        man["rows"][qid] = {}
        for oid, _ in ORDERS:
            grids = dev[qid][oid]
            frames = [np.array(overlay(disp, norm(grids[l]))) for l in range(len(grids))]
            buf = io.BytesIO()
            imageio.mimsave(buf, frames, format="GIF", duration=int(1000 / FPS), loop=0)
            open(f"{outdir}/{qid}_{oid}_gif.gif", "wb").write(buf.getvalue())
            overlay(disp, norm(grids.mean(0))).save(f"{outdir}/{qid}_{oid}_mean.png")
            overlay(disp, norm(grids[-1])).save(f"{outdir}/{qid}_{oid}_final.png")
            man["rows"][qid][oid] = {"gif": f"{qid}_{oid}_gif.gif",
                                     "mean": f"{qid}_{oid}_mean.png",
                                     "final": f"{qid}_{oid}_final.png"}
    json.dump(man, open(f"{outdir}/manifest.json", "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
