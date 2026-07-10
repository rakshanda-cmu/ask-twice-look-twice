"""
Grad-CAM heatmaps over the image for image-only vs IST vs STI, on Qwen3-VL-8B, for
two questions on the SAME image. For each (question, ordering) we backprop the model's
answer-token logit and form ReLU(Σ grad·activation) over the image-token hidden states
per layer (Grad-CAM), then overlay it on the image. Raw answer→image attention is
attention-sink dominated and does not localise the queried object; Grad-CAM highlights
the patches the model actually uses for its answer.

Layout produced (fed to the 🖼️ Patch Perturbation tab, added below the existing
content — nothing removed):
    columns  : [image-only]  [IST]  [STI]     (answer→image-patch attention)
    per question, 3 rows     : GIF (all layers) · Mean (over layers) · Final layer
    2 questions              : 6 rows total.

Note: the image-only column has no question, so it is identical across the two
questions (base attention).

Run (needs one GPU + the image ON THIS SERVER):
    CUDA_VISIBLE_DEVICES=0 python attn_heatmap_gen.py --image ./attn_demo/tv_cat.png
"""
import argparse, io, json, os
import numpy as np, torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from constants import SYSTEM_MESSAGE

OUT = "attn_demo"
DISP_W, ALPHA, FPS, CMAP = 620, 0.55, 2.0, "jet"
QUESTIONS = [("q1", "What sport is on TV?"), ("q2", "What is the cat doing?")]
CONDS = [("base", "I"), ("IST", "IST"), ("STI", "STI")]   # base = image-only


def attn_maps(mm, pil, question, order):
    """Per-layer **Grad-CAM** of the answer token over image patches → [n_layers, gh, gw].
    Raw answer→image attention (and attention rollout) is dominated by attention sinks
    on the border/background patches and does not localise the queried object. Instead
    we take the model's own answer token (argmax at the answer position), backprop its
    logit, and for each layer form ReLU(Σ_d grad·activation) over the image-token hidden
    states — the gradient×activation Grad-CAM. This highlights the patches the model
    actually *uses* to produce the answer (the TV for a sport question, the cat for a
    cat question), and still gives one map per layer for the GIF / Mean / Final views."""
    sys_text = "" if order == "I" else SYSTEM_MESSAGE
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [question], pil, system_prompt=sys_text, order=order)
    gh, gw = mm.grid_h, mm.grid_w
    s, e = mm.img_start_idx, mm.img_end_idx
    side = gh * gw
    mm.llm_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
        hs = out.hidden_states                    # tuple len n_layers+1, each [1, seq, hid]
        for h in hs:
            h.retain_grad()
        target = out.logits[0, -1].max()          # the answer token's logit
        target.backward()
    maps = []
    for h in hs[1:]:                              # skip the embedding layer
        g = h.grad[0]                             # [seq, hid]
        a = h[0].detach()                         # [seq, hid]
        cam = torch.relu((g.detach() * a).sum(-1))   # [seq]  grad·activation, positive part
        row = cam[s:s + min(e - s, side)].float().cpu().numpy()
        if row.size < side:
            row = np.pad(row, (0, side - row.size))
        maps.append(row[:side].reshape(gh, gw))
    mm.llm_model.zero_grad(set_to_none=True)
    del out, hs
    torch.cuda.empty_cache()
    return np.array(maps), gh, gw


def _norm(m):
    # Grad-CAM map: light spatial smoothing, then robust min-max so the queried
    # object's patches read as the hot region
    from scipy.ndimage import gaussian_filter
    m = gaussian_filter(m, sigma=1.0)
    lo, hi = np.percentile(m, 40), np.percentile(m, 99)
    return np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)


def overlay(disp, grid01):
    hm = plt.get_cmap(CMAP)(grid01)[:, :, :3]
    hm = Image.fromarray((hm * 255).astype(np.uint8)).resize(disp.size, Image.BICUBIC)
    return Image.blend(disp, hm, ALPHA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="qwen3-vl-8b")
    args = ap.parse_args()
    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()
    os.makedirs(OUT, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    disp = pil.resize((DISP_W, int(pil.height * DISP_W / pil.width)), Image.LANCZOS)
    disp.save(f"{OUT}/_source.png")
    print(f"[attn] image {args.image} {pil.size} · Grad-CAM (sdpa, frozen)", flush=True)
    # Grad-CAM needs hidden-state gradients, NOT attention weights, so use SDPA (no
    # materialized attention matrices) and freeze all params (we want activation grads
    # only — this avoids storing ~16 GB of parameter gradients). A forward hook makes
    # the embedding output require grad so the graph still tracks through frozen weights.
    mm = ModelManager(args.model, attn_implementation="sdpa")
    for p in mm.llm_model.parameters():
        p.requires_grad_(False)
    mm.llm_model.get_input_embeddings().register_forward_hook(
        lambda mod, inp, out: out.requires_grad_(True))

    manifest = {"image": os.path.basename(args.image), "source": "_source.png",
                "questions": [{"id": q, "text": t} for q, t in QUESTIONS],
                "conds": [c for c, _ in CONDS], "n_layers": None, "rows": {}}
    base = None
    for qid, qtext in QUESTIONS:
        for cid, order in CONDS:
            if cid == "base" and base is not None:
                maps, gh, gw = base
            else:
                maps, gh, gw = attn_maps(mm, pil, qtext, order)
                if cid == "base":
                    base = (maps, gh, gw)
            manifest["n_layers"] = len(maps)
            # GIF over layers
            frames = [np.array(overlay(disp, _norm(maps[l]))) for l in range(len(maps))]
            buf = io.BytesIO()
            imageio.mimsave(buf, frames, format="GIF", duration=int(1000 / FPS), loop=0)
            open(f"{OUT}/{qid}_{cid}_gif.gif", "wb").write(buf.getvalue())
            # mean + final
            overlay(disp, _norm(maps.mean(0))).save(f"{OUT}/{qid}_{cid}_mean.png")
            overlay(disp, _norm(maps[-1])).save(f"{OUT}/{qid}_{cid}_final.png")
            manifest["rows"].setdefault(qid, {})[cid] = {
                "gif": f"{qid}_{cid}_gif.gif", "mean": f"{qid}_{cid}_mean.png",
                "final": f"{qid}_{cid}_final.png"}
            print(f"  {qid} {cid:5s} ({order}) done · {len(maps)} layers · grid {gh}x{gw}", flush=True)
        json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
