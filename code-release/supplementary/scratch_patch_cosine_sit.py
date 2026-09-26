"""
Per-patch cosine similarity of the IMAGE-token LLM hidden states (layer ~18) under
IST / STI / STIT vs a BASE run whose input is the image alone (no system, no task).

Produces a spatial heatmap per ordering (mean over NaturalBench images at a fixed
resolution). Qwen3-VL-8B.

    base  : order "I"     (image only)
    IST   : order "IST"   (image first -> patches see no preceding text)
    STI   : order "STI"   (system+task precede the image)
    STIT  : order "STIT"  (system+task precede; 2nd task after image)

Causal attention => IST patches ≡ base, and STIT patches ≡ STI (validated below).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from constants import SYSTEM_MESSAGE
from naturalbench_eval import load_groups, _question_for
from model_manager import ModelManager

LAYER = 18
RESIZE = 448


def img_hidden(mm, pil, question, order):
    """Return layer-LAYER hidden states at the image token positions: [P, H]."""
    q = question
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [q], pil, system_prompt=("" if order == "I" else SYSTEM_MESSAGE), order=order)
    s, e = mm.img_start_idx, mm.img_end_idx
    gh, gw = mm.grid_h, mm.grid_w
    with torch.inference_mode():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
    hs = out.hidden_states[LAYER][0, s:e].float().cpu()   # [P, H]
    return hs, (gh, gw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=30)
    ap.add_argument("--validate", action="store_true",
                    help="also compute IST & STIT to check causal identities")
    ap.add_argument("--out", default="patch_cosine")
    args = ap.parse_args()

    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()
    mm = ModelManager("qwen3-vl-8b")

    groups = load_groups("./naturalbench")
    # build (image, question) pairs
    pairs = []
    for g in groups:
        for qi in (0, 1):
            q = _question_for(g, qi)
            for ii in (0, 1):
                pairs.append((g[f"image_{ii}"], q))
    if args.num_pairs > 0:
        pairs = pairs[:args.num_pairs]
    print(f"[data] {len(pairs)} (image,question) pairs  layer={LAYER}  resize={RESIZE}", flush=True)

    base_cache = {}
    orders = ["SIT", "STI", "STIT"]
    acc = {o: None for o in orders}
    cnt = {o: 0 for o in orders}
    ident = {"cos(base,SIT)": [], "cos(STI,STIT)": []}
    grid = None
    t0 = time.time()

    for pi, (imgrel, question) in enumerate(pairs):
        path = os.path.join("./naturalbench", imgrel)
        try:
            pil = Image.open(path).convert("RGB").resize((RESIZE, RESIZE), Image.LANCZOS)
        except Exception:
            continue
        if imgrel not in base_cache:
            hb, g = img_hidden(mm, pil, question, "I")
            base_cache[imgrel] = hb
            grid = grid or g
        hb = base_cache[imgrel]

        per = {}
        for o in orders:
            ho, g = img_hidden(mm, pil, question, o)
            if ho.shape != hb.shape:
                break
            cos = F.cosine_similarity(ho, hb, dim=-1).numpy()  # [P]
            per[o] = (ho, cos)
            acc[o] = cos if acc[o] is None else acc[o] + cos
            cnt[o] += 1

        if "SIT" in per:
            ident["cos(base,SIT)"].append(float(per["SIT"][1].mean()))
            if "STI" in per and "STIT" in per:
                c = F.cosine_similarity(per["STI"][0], per["STIT"][0], dim=-1).mean()
                ident["cos(STI,STIT)"].append(float(c))

        if (pi + 1) % 20 == 0 or pi + 1 == len(pairs):
            r = (pi + 1) / (time.time() - t0)
            msg = " ".join(f"{o}:{(acc[o]/max(1,cnt[o])).mean():.3f}" for o in orders)
            print(f"  [{pi+1}/{len(pairs)}] meanCos {msg} ({r:.1f} pair/s)", flush=True)

    res = {"layer": LAYER, "resize": RESIZE, "grid": grid, "n": cnt,
           "mean_cos_map": {o: (acc[o] / cnt[o]).tolist() for o in orders if cnt[o]}}
    if True:
        for k, v in ident.items():
            if v:
                print(f"[identity] {k} = {np.mean(v):.4f} (min {np.min(v):.4f})", flush=True)
        res["identities"] = {k: (float(np.mean(v)) if v else None) for k, v in ident.items()}
    json.dump(res, open(f"{args.out}.json", "w"))
    print(f"[saved] {args.out}.json  grid={grid}", flush=True)


if __name__ == "__main__":
    main()
