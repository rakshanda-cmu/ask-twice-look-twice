"""
Extra assets for the static site:
  (1) SIT (question-last) per-layer logit-lens GIFs for the four examples, so the site
      can open with the PROBLEM: SIT answers right, STI answers wrong.
  (2) Figure-3-style stills: the per-patch logit-lens **words** overlaid on the (low
      token) image at a late layer, under STI, showing the question-steered patch
      decodings.

Run from the MAIN research repo (imports model code, reads NaturalBench images):
    cd /home/grg/Research/middle_layers_indicating_hallucinations
    python /home/grg/Research/ask-twice-look-twice-supp/website/build_extra.py
"""
import io, json, os
import imageio, numpy as np
import torch
from PIL import Image, ImageSequence
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from sitit_stit_gif_gen import one, TOP_K, MAX_TOK
from logit_lens_overlay import logit_lens_all_vision_tokens, render_vision_frame
from model_manager import ModelManager
from constants import SYSTEM_MESSAGE
from naturalbench_eval import answer_suffix, judge_pair
from utils import setup_seeds, disable_torch_init

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
NB = "./naturalbench"
LOW_MAX = 280
LAYER_STEP = 2
GIF_W = 470
STILL_W = 760
STILL_LAYERS = [24, 33]          # late-mid + near-final; pick the clearer per example

EXS = [  # (idx, image_file, question, gt) — the four the site already uses
    (229,  "images/nb_229_0.jpg",  "Is the dog chasing a person?",     "Yes"),
    (638,  "images/nb_638_1.jpg",  "Is the person walking?",           "No"),
    (978,  "images/nb_978_0.jpg",  "Is the man performing on a ramp?", "Yes"),
    (1231, "images/nb_1231_0.jpg", "Is the girl entering the water?",  "No"),
]


def downscale(pil, longest):
    w, h = pil.size
    s = longest / max(w, h)
    return pil if s >= 1 else pil.resize((int(w * s), int(h * s)), Image.LANCZOS)


def shrink_gif(gif_bytes, width):
    im = Image.open(io.BytesIO(gif_bytes)); frames = []
    for f in ImageSequence.Iterator(im):
        fr = f.convert("RGB")
        if fr.width > width:
            fr = fr.resize((width, int(fr.height * width / fr.width)), Image.LANCZOS)
        frames.append(np.array(fr))
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=int(1000 / 2.0), loop=0)
    return buf.getvalue()


def vision_stills(mm, pil, query, order, layers):
    """Render per-patch logit-lens word overlays at the given absolute layers."""
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [query], pil, system_prompt=SYSTEM_MESSAGE, order=order)
    gh, gw = mm.grid_h, mm.grid_w
    with torch.inference_mode():
        outputs = mm.llm_model.generate(
            input_ids, do_sample=False, num_beams=1, max_new_tokens=MAX_TOK,
            use_cache=True, output_hidden_states=True, return_dict_in_generate=True, **kwargs)
    warper = TopKLogitsWarper(top_k=TOP_K, filter_value=float("-inf"))
    proc = LogitsProcessorList([])
    lr = list(layers)
    probs, words = logit_lens_all_vision_tokens(
        mm.llm_model, mm.tokenizer, input_ids, outputs, mm.img_start_idx, lr,
        warper, proc, grid_h=gh, grid_w=gw)
    disp = pil.resize((STILL_W, int(pil.height * STILL_W / pil.width)), Image.LANCZOS)
    out = {}
    for i, L in enumerate(lr):
        out[L] = render_vision_frame(disp, probs[i], words[i], L, alpha=0.55,
                                     grid_h=gh, grid_w=gw)
    return out


def main():
    setup_seeds(); disable_torch_init()
    mm = ModelManager("qwen3-vl-8b", attn_implementation="eager")
    layer_range = list(range(0, mm.num_layers, LAYER_STEP))
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    by = {e["idx"]: e for e in man["examples"]}

    for idx, imgrel, q, gt in EXS:
        pil = downscale(Image.open(os.path.join(NB, imgrel)).convert("RGB"), LOW_MAX)
        query = q + answer_suffix("yes_no")
        # (1) SIT GIF + correctness
        ans, gif = one(mm, pil, query, "SIT", layer_range)
        correct, pred = judge_pair(ans, gt, "yes_no", q)
        open(os.path.join(OUT, f"ex{idx}_SIT.gif"), "wb").write(shrink_gif(gif, GIF_W))
        by[idx]["SIT"] = {"answer": ans, "pred": pred, "correct": bool(correct),
                          "gif": f"ex{idx}_SIT.gif"}
        # (2) STI word-overlay stills (Figure-3 style)
        stills = vision_stills(mm, pil, query, "STI", STILL_LAYERS)
        by[idx]["word_stills"] = {}
        for L, im in stills.items():
            fn = f"ex{idx}_words_L{L}.png"
            im.save(os.path.join(OUT, fn))
            by[idx]["word_stills"][str(L)] = fn
        print(f"  g{idx}: SIT={pred}({'ok' if correct else 'x'}) gt={gt} "
              f"stills={list(by[idx]['word_stills'].values())}", flush=True)
        man["examples"] = [by[i] for i in sorted(by)]
        json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
