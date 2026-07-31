"""
Mechanistic comparison: does the model's Grad-CAM visual-attention saliency (the
patches it actually uses to decide the object's location) track human gaze
scanpaths recorded on the SAME (image, referring-expression) pairs, and does
that alignment change across the STI/SIT/STIT/SITIT prompt orderings?

Data: RefCOCO-Gaze (Mondal et al., ECCV 2024) val split, 869 scanpaths, joined
back to RefCOCO's own images/boxes via REF_ID (RefCOCO-Gaze reuses RefCOCO's
referring expressions; it adds no images/boxes of its own).

Coordinate transform (empirically validated to <3px total error on 29/29 spot
checks): fixations/target-box are recorded in the 1680x1050 eye-tracker display
canvas, which is the original COCO image letterboxed ("contain" fit, centered):
    scale = min(1680/ow, 1050/oh)
    ox, oy = (1680-ow*scale)/2, (1050-oh*scale)/2
    display_xy = orig_xy*scale + (ox,oy)   /   orig_xy = (display_xy-(ox,oy))/scale

Saliency signal: reuses attn_heatmap_gen.py's validated Grad-CAM recipe (raw
answer->image attention is attention-sink-dominated and does NOT localise the
queried object -- backprop the model's own predicted token instead). Adapted
for referring/grounding: the model is asked to output a bbox JSON; we let it
generate greedily, locate the FIRST token that introduces a coordinate digit
after the `"bbox_2d": [` marker (the moment the model commits to a location --
the direct analog of a human fixation landing on the target), then do a second
grad-enabled forward pass truncated to just before that token and backprop its
logit. Grad-CAM = ReLU(sum_d grad*activation) over image-token hidden states,
per layer, averaged over layers.

Metric: Normalized Scanpath Saliency (NSS) -- z-score the (mean-over-layers)
Grad-CAM grid, sample its value at each human fixation's mapped grid cell,
average over fixations. Higher = model attention matches where humans looked.
Secondary: does the grid's argmax cell fall inside the target's grid-mapped
bbox (direct analog of humans' final-fixation-in-bbox).

Run (HF only -- Grad-CAM needs backprop through hidden states, not expressible
via an inference-only engine like vLLM):
  CUDA_VISIBLE_DEVICES=0 /home/grg/anaconda3/envs/logitlens/bin/python \
    refcoco_gaze/gaze_eval_gradcam.py --orders STI,SIT,STIT,SITIT
"""
import argparse, json, os, pickle, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from constants import SYSTEM_MESSAGE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
GAZE_DIR = "/data2/hf_cache/newtasks/refcoco_gaze"
RC = "/home/grg/Research/rf-20-vl-benchmark/datasets/RefCOCO"
MODEL_TAG = "qwen3-vl-8b"

DISP_W, DISP_H = 1680, 1050
REF_PROMPT = ('Locate "{phrase}" in the image and output its bounding box. '
              'Return valid JSON only, no extra text, in the form '
              '{{"bbox_2d": [x1, y1, x2, y2]}} where (x1,y1) is the top-left and '
              '(x2,y2) the bottom-right corner.')
MAX_NEW_TOKENS = 48


def load_gaze_val():
    return json.load(open(os.path.join(GAZE_DIR, "val_gaze.json")))


def load_refcoco_index():
    """ref_id -> (image_path, W, H, bbox[x,y,w,h]) using RefCOCO's own data
    (RefCOCO-Gaze contributes no images/boxes of its own -- see module docstring)."""
    refs = pickle.load(open(f"{RC}/refcoco/refs(unc).p", "rb"))
    inst = json.load(open(f"{RC}/refcoco/instances.json"))
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {im["id"]: im for im in inst["images"]}
    idx = {}
    for r in refs:
        a, im = ann.get(r["ann_id"]), img.get(r["image_id"])
        if a is None or im is None:
            continue
        path = os.path.join(RC, "train2014", im["file_name"])
        if os.path.isfile(path):
            idx[r["ref_id"]] = {"path": path, "W": im["width"], "H": im["height"],
                                "bbox": a["bbox"]}
    return idx


def display_to_orig(x, y, ow, oh):
    scale = min(DISP_W / ow, DISP_H / oh)
    ox, oy = (DISP_W - ow * scale) / 2, (DISP_H - oh * scale) / 2
    return (x - ox) / scale, (y - oy) / scale


def find_target_index(tokenizer, prompt_len, gen_ids):
    """Return the full-sequence index of the first token that introduces a digit
    occurring after the `"bbox_2d": [` marker in the decoded continuation."""
    marker = '"bbox_2d"'
    text_so_far = ""
    seen_marker = False
    for k in range(1, len(gen_ids) + 1):
        text_so_far = tokenizer.decode(gen_ids[:k], skip_special_tokens=True)
        if not seen_marker and marker in text_so_far:
            seen_marker = True
            after = text_so_far[text_so_far.index(marker) + len(marker):]
            if re.search(r"\d", after):
                return prompt_len + k - 1
            continue
        if seen_marker:
            after = text_so_far[text_so_far.index(marker) + len(marker):]
            if re.search(r"\d", after):
                return prompt_len + k - 1
    return None


def grad_cam_grid(mm, pil, task, order, target_index_in_full, full_ids):
    """Grad-CAM at `target_index_in_full` (backprop logits predicting that
    token), mean over layers -> [gh, gw]. Mirrors attn_heatmap_gen.py exactly,
    generalized to backprop an arbitrary future position instead of always the
    prompt's last position."""
    sys_text = SYSTEM_MESSAGE
    _, prompt_ids, kwargs = mm.prepare_inputs_from_pil(
        [task], pil, system_prompt=sys_text, order=order)
    gh, gw = mm.grid_h, mm.grid_w
    s, e = mm.img_start_idx, mm.img_end_idx
    side = gh * gw

    input_ids = full_ids[:, :target_index_in_full]
    # kwargs' attention_mask / mm_token_type_ids are sized for the PROMPT only;
    # input_ids may be longer (prompt + partial generated continuation up to
    # the target token). Extend both with the value that means "ordinary
    # attended text token" for the extra positions -- attention_mask=1 (no
    # padding), mm_token_type_ids=0 (text; confirmed against the prompt's own
    # trailing text-span values).
    extra = input_ids.shape[1] - kwargs["attention_mask"].shape[1]
    if extra > 0:
        kwargs = dict(kwargs)
        am = kwargs["attention_mask"]
        kwargs["attention_mask"] = torch.cat(
            [am, am.new_ones((am.shape[0], extra))], dim=1)
        if "mm_token_type_ids" in kwargs:
            tt = kwargs["mm_token_type_ids"]
            kwargs["mm_token_type_ids"] = torch.cat(
                [tt, tt.new_zeros((tt.shape[0], extra))], dim=1)
    mm.llm_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
        hs = out.hidden_states
        for h in hs:
            h.retain_grad()
        target = out.logits[0, -1].max()
        target.backward()
    maps = []
    for h in hs[1:]:
        g = h.grad[0]
        a = h[0].detach()
        cam = torch.relu((g.detach() * a).sum(-1))
        row = cam[s:s + min(e - s, side)].float().cpu().numpy()
        if row.size < side:
            row = np.pad(row, (0, side - row.size))
        maps.append(row[:side].reshape(gh, gw))
    mm.llm_model.zero_grad(set_to_none=True)
    del out, hs
    torch.cuda.empty_cache()
    return np.array(maps).mean(0), gh, gw   # mean-over-layers grid


def nss_score(grid, fixations_grid_xy):
    """grid: [gh,gw] raw Grad-CAM. fixations_grid_xy: list of (col,row) float
    grid coords. Returns mean NSS over fixations (None if no valid fixations)."""
    g = gaussian_filter(grid.astype(np.float64), sigma=0.6)
    mu, sd = g.mean(), g.std()
    if sd < 1e-9:
        return None
    z = (g - mu) / sd
    gh, gw = grid.shape
    vals = []
    for cx, cy in fixations_grid_xy:
        r, c = int(np.clip(cy, 0, gh - 1)), int(np.clip(cx, 0, gw - 1))
        vals.append(z[r, c])
    return float(np.mean(vals)) if vals else None


def run_order(mm, tag, letters, gaze_entries, ref_idx, log_every):
    out = os.path.join(OUT_DIR, f"gaze_order-{tag}_{MODEL_TAG}.json")
    results, done = [], set()
    if os.path.exists(out):
        try:
            d = json.load(open(out))
            results = d["results"]; done = {r["ref_gaze_id"] for r in results}
            print(f"  [resume] {len(done)} already done", flush=True)
        except Exception:
            results, done = [], set()

    t0 = time.time()
    for i, entry in enumerate(gaze_entries):
        rgid = entry["REF_GAZE_ID"]
        if rgid in done:
            continue
        info = ref_idx.get(entry["REF_ID"])
        if info is None:
            continue
        pil = Image.open(info["path"]).convert("RGB")
        W, H = info["W"], info["H"]
        task = REF_PROMPT.format(phrase=entry["REF_SENTENCE"])

        try:
            _, prompt_ids, kwargs = mm.prepare_inputs_from_pil(
                [task], pil, system_prompt=SYSTEM_MESSAGE, order=letters)
            with torch.inference_mode():
                gen = mm.llm_model.generate(
                    prompt_ids, do_sample=False, num_beams=1,
                    max_new_tokens=MAX_NEW_TOKENS, use_cache=True, **kwargs)
            gen_ids = gen[0, prompt_ids.shape[1]:].tolist()
            tidx = find_target_index(mm.tokenizer, prompt_ids.shape[1], gen_ids)
            if tidx is None:
                continue
            grid, gh, gw = grad_cam_grid(mm, pil, task, letters, tidx, gen)
        except Exception as ex:
            print(f"    [{tag}] err {rgid}: {ex}", flush=True)
            continue

        fix_grid = []
        for fx, fy in zip(entry["FIX_X"], entry["FIX_Y"]):
            ox, oy = display_to_orig(fx, fy, W, H)
            fix_grid.append((ox / W * gw, oy / H * gh))
        nss = nss_score(grid, fix_grid)

        bx, by, bw, bh = info["bbox"]
        bcx, bcy = (bx + bw / 2) / W * gw, (by + bh / 2) / H * gh
        argmax_r, argmax_c = np.unravel_index(np.argmax(grid), grid.shape)
        in_bbox = (bx / W * gw <= argmax_c <= (bx + bw) / W * gw and
                  by / H * gh <= argmax_r <= (by + bh) / H * gh)
        human_final_in_bbox = bool(entry["FIX_IN_BBOX"][-1]) if entry["FIX_IN_BBOX"] else None

        results.append({"ref_gaze_id": rgid, "ref_id": entry["REF_ID"],
                        "nss": nss, "argmax_in_bbox": bool(in_bbox),
                        "human_final_fix_in_bbox": human_final_in_bbox,
                        "n_fixations": len(entry["FIX_X"])})
        if (len(results)) % log_every == 0:
            vals = [r["nss"] for r in results if r["nss"] is not None]
            print(f"    [{tag}] {len(results)}/{len(gaze_entries)} "
                  f"mean_NSS={np.mean(vals):.3f} "
                  f"{len(results)/(time.time()-t0+1e-9):.2f}/s", flush=True)
        if len(results) % 50 == 0:
            _save(out, tag, results)
    _save(out, tag, results)
    vals = [r["nss"] for r in results if r["nss"] is not None]
    print(f"  [{tag}] n={len(results)} mean_NSS={np.mean(vals):.3f} -> {out}", flush=True)


def _save(out, tag, results):
    vals = [r["nss"] for r in results if r["nss"] is not None]
    argmax_hits = [r["argmax_in_bbox"] for r in results]
    meta = {"benchmark": "RefCOCO-Gaze-vs-GradCAM", "ordering": tag,
            "model": MODEL_TAG, "n": len(results),
            "mean_nss": float(np.mean(vals)) if vals else None,
            "argmax_in_bbox_rate": float(np.mean(argmax_hits)) if argmax_hits else None,
            "complete": False}
    json.dump({"meta": meta, "results": results}, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="STI,SIT,STIT,SITIT")
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--log-every", type=int, default=20, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds(); disable_torch_init()

    gaze_entries = load_gaze_val()
    ref_idx = load_refcoco_index()
    print(f"[data] {len(gaze_entries)} gaze scanpaths, "
          f"{sum(1 for e in gaze_entries if e['REF_ID'] in ref_idx)} resolve to RefCOCO",
          flush=True)

    print(f"[load] ModelManager({args.model}, sdpa, frozen)", flush=True)
    mm = ModelManager(args.model, attn_implementation="sdpa")
    for p in mm.llm_model.parameters():
        p.requires_grad_(False)
    mm.llm_model.get_input_embeddings().register_forward_hook(
        lambda mod, inp, out: out.requires_grad_(True))

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        print(f"=== gaze · ordering {tag} ===", flush=True)
        run_order(mm, tag, tag, gaze_entries, ref_idx, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
