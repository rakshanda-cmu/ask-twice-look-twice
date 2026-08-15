"""
Question-SPECIFIC patch-perturbation heatmaps on ONE image, Qwen3-VL-8B.

Motivation / fix. The earlier maps compared either cos(h(q1), h(q2)) (washed out:
at a mid layer cos ~ 0.95, so ~flat) or (1 - cos) of each question vs the image-only
baseline (which made q1 and q2 look IDENTICAL, because ~95% of each question's
perturbation is a shared "a question is present" component -- see scratch_qqcos_diag).

We isolate the question-specific part. With b = image-only hidden states and
e_q = h_STI(q) - b (what question q added to each patch), the shared component is
comm = mean_q e_q and the question-specific residual is

        spec_q = e_q - comm .

We plot ||spec_q|| per patch: how differently THIS question rewrote each patch than
the average question. Under image-first (IST) the image never sees the question, so
e_q = 0 and the map is flat by construction -- the built-in control. The specific
signal is sharpest around layer 18 (highest spatial CV); we default there and also
render a mean over a mid-layer band.

Outputs overlays + a single side-by-side comparison PNG (rows = questions,
cols = [image, IST-specific, STI-specific]) into patchcos_contrast/.

    CUDA_VISIBLE_DEVICES=0 python patchcos_contrast_gen.py --image examples/<img>.png
"""
import argparse, json, os
import numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from core.constants import SYSTEM_MESSAGE

OUT = "patchcos_contrast"
DISP_W, ALPHA, CMAP = 620, 0.55, "turbo"
QUESTIONS = [("q1", "What sport is on TV?"), ("q2", "What is the cat doing?")]
ORDERS = ["IST", "STI"]
LAYER = 18                       # sharpest question-specific layer (see diag)
BAND = (12, 25)                  # mid-layer band for the averaged map


def img_hidden(mm, pil, q, order, sys=True):
    _, ids, kw = mm.prepare_inputs_from_pil(
        [q], pil, system_prompt=SYSTEM_MESSAGE if sys else "", order=order)
    gh, gw, s, e = mm.grid_h, mm.grid_w, mm.img_start_idx, mm.img_end_idx
    with torch.inference_mode():
        out = mm.llm_model(ids, output_hidden_states=True, use_cache=False, **kw)
    return torch.stack([h[0, s:e].float() for h in out.hidden_states]).cpu(), gh, gw


def overlay(disp, grid01):
    hm = plt.get_cmap(CMAP)(grid01)[:, :, :3]
    hm = Image.fromarray((hm * 255).astype(np.uint8)).resize(disp.size, Image.BICUBIC)
    return Image.blend(disp, hm, ALPHA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out-dir", default=OUT, dest="out_dir")
    args = ap.parse_args()
    from core.model_manager import ModelManager
    from core.utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    setup_seeds(); disable_torch_init()
    os.makedirs(args.out_dir, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    disp = pil.resize((DISP_W, int(pil.height * DISP_W / pil.width)), Image.LANCZOS)
    disp.save(f"{args.out_dir}/_source.png")
    print(f"[contrast] {args.image} {pil.size} · Qwen sdpa · layer {args.layer}", flush=True)
    mm = ModelManager(args.model, attn_implementation="sdpa")

    base, gh, gw = img_hidden(mm, pil, "", "I", sys=False)      # image-only baseline
    P0 = base.shape[1]

    # e_q per (order, question); then spec_q = e_q - mean_q e_q  (question-specific)
    maps = {}   # maps[order][qid] = {"layer": grid[gh,gw], "band": grid[gh,gw]}
    for order in ORDERS:
        E = {}
        for qid, qt in QUESTIONS:
            h, _, _ = img_hidden(mm, pil, qt, order)
            P = min(P0, h.shape[1])
            E[qid] = (h[:, :P] - base[:, :P])                  # [L+1,P,H]
        comm = torch.stack([E[q] for q, _ in QUESTIONS]).mean(0)
        maps[order] = {}
        for qid, _ in QUESTIONS:
            spec = E[qid] - comm                               # [L+1,P,H]
            mag = spec.norm(dim=-1)                            # [L+1,P]
            Pn = gh * gw
            lay = mag[args.layer, :Pn].reshape(gh, gw).numpy()
            band = mag[BAND[0]:BAND[1], :Pn].mean(0).reshape(gh, gw).numpy()
            maps[order][qid] = {"layer": lay, "band": band}
            print(f"  {order} {qid}: layer{args.layer} spec |.| mean={lay.mean():.2f} "
                  f"max={lay.max():.2f} CV={lay.std()/(lay.mean()+1e-9):.2f}", flush=True)

    # one fixed scale from the STI maps (IST is ~0 by construction => reads flat)
    sti_vals = np.concatenate([maps["STI"][q][k].ravel()
                               for q, _ in QUESTIONS for k in ("layer", "band")])
    vmax = float(np.percentile(sti_vals, 99)) or 1e-6
    nrm = lambda g: np.clip(g / vmax, 0, 1)

    # individual overlays + manifest
    man = {"image": os.path.basename(args.image), "source": "_source.png",
           "questions": [{"id": q, "text": t} for q, t in QUESTIONS],
           "layer": args.layer, "band": list(BAND), "vmax": vmax,
           "grid": [gh, gw], "rows": {}}
    for qid, _ in QUESTIONS:
        man["rows"][qid] = {}
        for order in ORDERS:
            for k in ("layer", "band"):
                fn = f"{qid}_{order}_{k}.png"
                overlay(disp, nrm(maps[order][qid][k])).save(f"{args.out_dir}/{fn}")
                man["rows"][qid][f"{order}_{k}"] = fn
    json.dump(man, open(f"{args.out_dir}/manifest.json", "w"), indent=2)

    # single side-by-side comparison figure (rows = questions, cols = image/IST/STI)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for r, (qid, qt) in enumerate(QUESTIONS):
        axes[r, 0].imshow(disp); axes[r, 0].set_ylabel(f"{qid}: {qt}", fontsize=10)
        axes[r, 0].set_title("image" if r == 0 else "")
        for c, order in enumerate(ORDERS, start=1):
            axes[r, c].imshow(overlay(disp, nrm(maps[order][qid]["layer"])))
            axes[r, c].set_title(f"{order} (question-specific)" if r == 0 else "")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle(f"Question-specific patch perturbation, layer {args.layer} "
                 f"(IST flat by construction; STI differs by question)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/comparison.png", dpi=130, bbox_inches="tight")
    print(f"[done] wrote {args.out_dir}/comparison.png", flush=True)


if __name__ == "__main__":
    main()
