"""
Generate per-layer logit-lens GIFs comparing SITIT vs STIT on NaturalBench
(Qwen3-VL-8B), for N examples. Reuses the existing logit_lens_overlay machinery.

SITIT (S·I·T·I·T) has TWO image blocks. The single-image render only showed the
first, so SITIT GIFs now use a dedicated multi-image renderer that lays out the FULL
flow in prompt order:
    SYSTEM · IMAGE 1 · TASK · IMAGE 2 · TASK · GENERATED
with the per-layer logit lens on every section (both image heatmaps + all text +
generated tokens). STIT (one image) keeps the original single-image render.

Output:
  sitit_compare/ex{idx:02d}_{order}.gif      per-layer animation
  sitit_compare/manifest.json                answers, correctness, layer list

Run (needs one free GPU):
  CUDA_VISIBLE_DEVICES=0 python sitit_stit_gif_gen.py --num 40 --layer-step 1
"""
import argparse, io, json, os
import imageio, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from constants import SYSTEM_MESSAGE
from naturalbench_eval import load_groups, _question_for, _gt_for, answer_suffix, judge_pair
from model_manager import ModelManager
from logit_lens_overlay import (
    logit_lens_all_vision_tokens, compute_text_logit_lens,
    render_combined_frame, render_vision_frame, _render_token_sections,
    _make_colorbar_frame,
)

OUT = "sitit_compare"
TOP_K, MAX_TOK, RESO, ALPHA, FPS = 50, 16, 768, 0.55, 2.0


def _image_runs(ids_row, image_token_id):
    """Contiguous runs of image placeholder tokens → [(start, end), ...]."""
    pos = [i for i, t in enumerate(ids_row) if t == image_token_id]
    if not pos:
        return []
    runs, s = [], pos[0]
    for a, b in zip(pos[:-1], pos[1:]):
        if b != a + 1:
            runs.append((s, a + 1)); s = b
    runs.append((s, pos[-1] + 1))
    return runs


def _label(W, text, color):
    im = Image.new("RGB", (W, 22), color)
    ImageDraw.Draw(im).text((6, 4), text, fill=(255, 255, 255))
    return im


def _fit(img, W):
    if img is not None and img.width != W:
        return img.resize((W, img.height), Image.NEAREST)
    return img


def _seg(text_data, key, fi):
    sec = text_data.get(key, {})
    p = sec.get("probs", np.zeros((1, 0))); w = sec.get("words", [[]]); l = sec.get("labels", [])
    if p.shape[0] == 0 or fi >= p.shape[0]:
        return np.array([]), [], []
    return p[fi], (w[fi] if fi < len(w) else []), l


def render_sitit_frame(disp, vis1, vis2, text_data, fi, layer_idx, grid_h, grid_w,
                       rel_s, rel_e):
    """Full S·I·T·I·T flow: system, image1, task1, image2, task2, generated."""
    W = disp.width; cell = W // grid_w; cmap = plt.get_cmap("plasma")
    bp, bw, bl = _seg(text_data, "before", fi)
    ap, aw, al = _seg(text_data, "after", fi)
    gp, gw, gl = _seg(text_data, "generated", fi)
    rel_s = max(0, min(rel_s, len(al))); rel_e = max(rel_s, min(rel_e, len(al)))

    def txt(name, i, j):
        seg = (name, ap[i:j] if len(ap) else np.array([]),
               list(aw[i:j]) if aw else [], list(al[i:j]))
        return _fit(_render_token_sections([seg], grid_w, cell, cmap), W)

    div = Image.new("RGB", (W, 6), (60, 60, 80))
    panels = []
    if len(bl):
        panels += [_label(W, "SYSTEM PROMPT", (0, 90, 90)),
                   _fit(_render_token_sections([("system", bp, bw, bl)], grid_w, cell, cmap), W), div]
    panels += [_label(W, "IMAGE 1", (90, 40, 90)),
               render_vision_frame(disp, vis1[0][fi], vis1[1][fi], layer_idx, ALPHA,
                                   grid_h=grid_h, grid_w=grid_w), div]
    if rel_s > 0:
        panels += [_label(W, "TASK PROMPT", (0, 40, 120)), txt("task", 0, rel_s), div]
    panels += [_label(W, "IMAGE 2  (2nd I)", (110, 50, 90)),
               render_vision_frame(disp, vis2[0][fi], vis2[1][fi], layer_idx, ALPHA,
                                   grid_h=grid_h, grid_w=grid_w), div]
    if rel_e < len(al):
        panels += [_label(W, "TASK PROMPT", (0, 40, 120)), txt("task", rel_e, len(al)), div]
    if len(gl):
        panels += [_label(W, "GENERATED TOKENS", (150, 80, 0)),
                   _fit(_render_token_sections([("generated", gp, gw, gl)], grid_w, cell, cmap), W)]

    H = sum(p.height for p in panels)
    out = Image.new("RGB", (W, H), (20, 20, 20)); y = 0
    for p in panels:
        out.paste(p, (0, y)); y += p.height
    return out


def one(mm, pil, query, order, layer_range):
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [query], pil, system_prompt=SYSTEM_MESSAGE, order=order)
    grid_h, grid_w, section_info = mm.grid_h, mm.grid_w, mm.section_info
    with torch.inference_mode():
        outputs = mm.llm_model.generate(
            input_ids, do_sample=False, num_beams=1, max_new_tokens=MAX_TOK,
            use_cache=True, output_hidden_states=True,
            return_dict_in_generate=True, **kwargs)
    answer = mm.tokenizer.batch_decode(outputs["sequences"][:, input_ids.shape[1]:],
                                       skip_special_tokens=True)[0].strip()
    warper = TopKLogitsWarper(top_k=TOP_K, filter_value=float("-inf"))
    proc = LogitsProcessorList([])

    vis1 = logit_lens_all_vision_tokens(
        mm.llm_model, mm.tokenizer, input_ids, outputs, mm.img_start_idx,
        layer_range, warper, proc, grid_h=grid_h, grid_w=grid_w)
    text_data = compute_text_logit_lens(
        mm.llm_model, mm.tokenizer, input_ids, outputs,
        mm.img_start_idx, mm.img_end_idx, layer_range, warper, proc)

    # detect a 2nd image block (SITIT) → multi-image render
    itid = getattr(mm.llm_model.config, "image_token_id", None)
    runs = _image_runs(input_ids[0].tolist(), itid) if itid is not None else []
    disp = pil.resize((RESO, RESO), Image.LANCZOS)
    cbar = _make_colorbar_frame(RESO, RESO)

    two = len(runs) >= 2
    if two:
        vis2 = logit_lens_all_vision_tokens(
            mm.llm_model, mm.tokenizer, input_ids, outputs, runs[1][0],
            layer_range, warper, proc, grid_h=grid_h, grid_w=grid_w)
        # map image2's absolute span into the "after"-section coordinates
        hs0 = outputs["hidden_states"][0][layer_range[0] + 1].shape[1]
        ids_len = input_ids.shape[1]
        after_off = mm.img_end_idx + (1 if hs0 == ids_len else 0)
        after_start = ids_len - (hs0 - after_off)
        rel_s, rel_e = runs[1][0] - after_start, runs[1][1] - after_start

    frames = []
    for fi, li in enumerate(layer_range):
        if two:
            fr = render_sitit_frame(disp, vis1, vis2, text_data, fi, li,
                                    grid_h, grid_w, rel_s, rel_e)
        else:
            fr = render_combined_frame(disp, vis1[0][fi], vis1[1][fi], li, fi,
                                       text_data, ALPHA, section_info=section_info,
                                       grid_h=grid_h, grid_w=grid_w)
        full = Image.new("RGB", (fr.width, fr.height + cbar.height))
        full.paste(fr, (0, 0)); full.paste(cbar, (0, fr.height))
        frames.append(full)
    buf = io.BytesIO()
    imageio.mimsave(buf, [np.array(f) for f in frames], format="GIF",
                    duration=int(1000 / FPS), loop=0)
    return answer, buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--layer-step", type=int, default=1, dest="layer_step")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--orders", default="SITIT,STIT")
    args = ap.parse_args()
    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()
    os.makedirs(OUT, exist_ok=True)
    mm = ModelManager("qwen3-vl-8b")
    layer_range = list(range(0, mm.num_layers, args.layer_step))
    orders = [o.strip().upper() for o in args.orders.split(",") if o.strip()]
    print(f"[gen] {args.num} examples · orders {orders} · layers {layer_range}", flush=True)

    groups = load_groups("./naturalbench")[args.start:args.start + args.num]
    mpath = f"{OUT}/manifest.json"
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"layers": layer_range, "examples": []}
    manifest["layers"] = layer_range
    by_idx = {e["idx"]: e for e in manifest["examples"]}
    for k, g in enumerate(groups):
        idx = args.start + k
        pil = Image.open(f"./naturalbench/{g['image_0']}").convert("RGB")
        q = _question_for(g, 0); gt = _gt_for(g, 0, 0)
        query = q + answer_suffix(g["question_type"])
        rec = by_idx.get(idx, {"idx": idx, "group": g["index"], "question": q, "gt": gt,
                               "image": g["image_0"], "orders": {}})
        for order in orders:
            ans, gif = one(mm, pil, query, order, layer_range)
            open(f"{OUT}/ex{idx:02d}_{order}.gif", "wb").write(gif)
            ok, pred = judge_pair(ans, gt, g["question_type"], q)
            rec["orders"][order] = {"answer": ans, "pred": pred, "correct": ok,
                                    "gif": f"ex{idx:02d}_{order}.gif"}
        by_idx[idx] = rec
        manifest["examples"] = [by_idx[i] for i in sorted(by_idx)]
        json.dump(manifest, open(mpath, "w"), indent=2)
        s = rec["orders"].get("SITIT", {}); t = rec["orders"].get("STIT", {})
        print(f"  ex{idx:02d} gt={gt} | SITIT={s.get('answer')!r}({'OK' if s.get('correct') else 'x'}) "
              f"STIT={t.get('answer')!r}({'OK' if t.get('correct') else 'x'})", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
