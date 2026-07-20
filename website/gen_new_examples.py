"""
Generate site assets for the steering-contrast examples found by search_steer.py:
for each chosen (STI-wrong, SIT-right) NaturalBench pair, render the zoomed-in
per-layer STI / SIT / SITIT logit-lens GIFs, the raw image, and -- for the static
comparison -- the STI and SIT per-patch decoded-word grids at the peak layer (so the
green/red boxes can be placed on exactly the patches where STI surfaces the action
word and SIT does not).

Runs from the MAIN repo (imports the model, reads the images):
  cd /home/grg/Research/middle_layers_indicating_hallucinations
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
    /home/grg/anaconda3/envs/logitlens/bin/python \
    /home/.../website/gen_new_examples.py
"""
import io, json, os
import imageio, numpy as np, torch
from PIL import Image, ImageSequence
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from constants import SYSTEM_MESSAGE
from naturalbench_eval import answer_suffix, judge_pair
from model_manager import ModelManager
from sitit_stit_gif_gen import one
from logit_lens_overlay import logit_lens_all_vision_tokens

SUPP = "/home/grg/Research/ask-twice-look-twice-supp"
OUT = os.path.join(SUPP, "website", "assets")
NB_ROOT = "/home/grg/Research/middle_layers_indicating_hallucinations/naturalbench"
SCRATCH = ("/tmp/claude-1000/-home-grg-Research-ask-twice-look-twice-supp/"
           "873eb924-61f3-4169-9a36-e6f2390cabf0/scratchpad")
LOW_MAX, LAYER_STEP, GIF_W, TOP_K = 280, 2, 470, 50


def downscale(pil, longest):
    w, h = pil.size
    s = longest / max(w, h)
    return pil if s >= 1.0 else pil.resize((max(1, int(w * s)), max(1, int(h * s))),
                                            Image.LANCZOS)


def shrink_gif(gif_bytes, width):
    im = Image.open(io.BytesIO(gif_bytes))
    frames = []
    for f in ImageSequence.Iterator(im):
        fr = f.convert("RGB")
        if fr.width > width:
            fr = fr.resize((width, int(fr.height * width / fr.width)), Image.LANCZOS)
        frames.append(np.array(fr))
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=int(1000 / 2.0), loop=0)
    return buf.getvalue()


def words_at(mm, pil, query, order, layer, warper, proc):
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [query], pil, system_prompt=SYSTEM_MESSAGE, order=order)
    with torch.inference_mode():
        outputs = mm.llm_model.generate(
            input_ids, do_sample=False, num_beams=1, max_new_tokens=8, use_cache=True,
            output_hidden_states=True, return_dict_in_generate=True, **kwargs)
    _, words = logit_lens_all_vision_tokens(
        mm.llm_model, mm.tokenizer, input_ids, outputs, mm.img_start_idx,
        [layer], warper, proc, grid_h=mm.grid_h, grid_w=mm.grid_w)
    return words[0], mm.grid_h, mm.grid_w


def main():
    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()
    mm = ModelManager("qwen3-vl-8b", attn_implementation="eager")
    layer_range = list(range(0, mm.num_layers, LAYER_STEP))
    warper = TopKLogitsWarper(top_k=TOP_K, filter_value=float("-inf"))
    proc = LogitsProcessorList([])

    chosen = json.load(open(os.path.join(SCRATCH, os.environ.get("CHOSEN_FILE", "chosen.json"))))
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath))

    for w in chosen:
        idx, q, gt, key = w["idx"], w["question"], w["gt"], w["key"]
        peak = w["peak_layer"]
        pil = Image.open(os.path.join(NB_ROOT, w["image"])).convert("RGB")
        low = downscale(pil, LOW_MAX)
        query = q + answer_suffix("yes_no")
        rec = {"idx": idx, "question": q, "gt": gt, "img_size": list(low.size),
               "key": key, "peak_layer": peak}
        for order in ("STI", "SIT", "SITIT"):
            ans, gif = one(mm, low, query, order, layer_range)
            correct, pred = judge_pair(ans, gt, "yes_no", q)
            fn = f"ex{idx}_{order}.gif"
            open(os.path.join(OUT, fn), "wb").write(shrink_gif(gif, GIF_W))
            im = Image.open(os.path.join(OUT, fn)); im.seek(0)
            poster = f"ex{idx}_{order}_poster.png"
            im.convert("RGB").save(os.path.join(OUT, poster))
            rec[order] = {"answer": ans, "pred": pred, "correct": bool(correct),
                          "gif": fn, "poster": poster}
        disp = low.resize((360, int(low.height * 360 / low.width)), Image.LANCZOS)
        disp.save(os.path.join(OUT, f"ex{idx}_image.png"))
        rec["image"] = f"ex{idx}_image.png"
        # per-patch decoded words at the peak layer, for the green/red boxes
        sti_words, gh, gw = words_at(mm, low, query, "STI", peak, warper, proc)
        sit_words, _, _ = words_at(mm, low, query, "SIT", peak, warper, proc)
        rec["grid_h"], rec["grid_w"] = gh, gw
        rec["sti_words"], rec["sit_words"] = sti_words, sit_words
        rec["demonstrates"] = (not rec["STI"]["correct"]) and rec["SIT"]["correct"]
        manifest["examples"] = [e for e in manifest["examples"] if e["idx"] != idx] + [rec]
        json.dump(manifest, open(mpath, "w"), indent=2)
        print(f"  g{idx} key={key!r} gt={gt} | STI={rec['STI']['pred']}"
              f"({'ok' if rec['STI']['correct'] else 'x'}) "
              f"SIT={rec['SIT']['pred']}({'ok' if rec['SIT']['correct'] else 'x'}) "
              f"SITIT={rec['SITIT']['pred']}({'ok' if rec['SITIT']['correct'] else 'x'}) "
              f"demo={rec['demonstrates']} grid={gh}x{gw}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
