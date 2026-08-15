"""
Generate the static-site assets for the review supplement: for a shortlist of clear,
low-token NaturalBench yes/no examples, render the per-layer logit-lens GIF under
STI (question-first, fails) and SITIT (image-echoing, fixes), plus a zoomed low-token
still of the image. Keeps only examples where STI is wrong and SITIT is right at the
chosen low resolution.

Run from the MAIN research repo (so it can import the model code and read the
NaturalBench images); writes assets into this repo's website/assets/ folder:

    cd /home/grg/Research/middle_layers_indicating_hallucinations
    python /home/grg/Research/ask-twice-look-twice-supp/website/build_assets.py
"""
import io, json, os
import imageio, numpy as np
from PIL import Image, ImageSequence

from sitit_stit_gif_gen import one
from model_manager import ModelManager
from naturalbench_eval import answer_suffix, judge_pair
from utils import setup_seeds, disable_torch_init

OUT = "/home/grg/Research/ask-twice-look-twice-supp/website/assets"
NB = "./naturalbench"                 # images live here in the main repo
LOW_MAX = 280                         # downscale longest side -> few image tokens
LAYER_STEP = 2                        # every 2nd layer -> smaller GIFs
GIF_W = 470                           # final GIF width (downscaled for the web)

# shortlist of clear, low-token action/pose yes-no pairs (STI-wrong, SITIT-right at
# full res); we re-verify the contrast at low res and keep the best four.
CANDS = [
    (217,  "images/nb_217_0.jpg",  "Is the dog running?",              "No"),
    (638,  "images/nb_638_1.jpg",  "Is the person walking?",           "No"),
    (36,   "images/nb_36_0.jpg",   "Is someone riding a bicycle?",     "Yes"),
    (229,  "images/nb_229_0.jpg",  "Is the dog chasing a person?",     "Yes"),
    (978,  "images/nb_978_0.jpg",  "Is the man performing on a ramp?", "Yes"),
    (886,  "images/nb_886_0.jpg",  "Is the dog wearing two collars?",  "No"),
    (1231, "images/nb_1231_0.jpg", "Is the girl entering the water?",  "No"),
    (1581, "images/nb_1581_0.jpg", "Is the man wearing a gray suit?",  "Yes"),
]


def downscale(pil, longest):
    w, h = pil.size
    s = longest / max(w, h)
    if s >= 1.0:
        return pil
    return pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def shrink_gif(gif_bytes, width):
    """Downscale every frame of a GIF to `width` and re-encode (smaller for the web)."""
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


def main():
    setup_seeds(); disable_torch_init()
    os.makedirs(OUT, exist_ok=True)
    mm = ModelManager("qwen3-vl-8b", attn_implementation="eager")
    layer_range = list(range(0, mm.num_layers, LAYER_STEP))
    manifest = {"examples": [], "low_max": LOW_MAX, "layer_step": LAYER_STEP,
                "model": "qwen3-vl-8b"}

    for idx, imgrel, q, gt in CANDS:
        path = os.path.join(NB, imgrel)
        if not os.path.exists(path):
            print(f"  g{idx}: image missing, skip", flush=True); continue
        pil = Image.open(path).convert("RGB")
        low = downscale(pil, LOW_MAX)
        query = q + answer_suffix("yes_no")
        rec = {"idx": idx, "question": q, "gt": gt, "img_size": list(low.size)}
        ok = {}
        for order in ("STI", "SITIT"):
            ans, gif = one(mm, low, query, order, layer_range)
            correct, pred = judge_pair(ans, gt, "yes_no", q)
            fn = f"ex{idx}_{order}.gif"
            open(os.path.join(OUT, fn), "wb").write(shrink_gif(gif, GIF_W))
            rec[order] = {"answer": ans, "pred": pred, "correct": bool(correct), "gif": fn}
            ok[order] = bool(correct)
        # zoomed low-token still (upscaled so the coarse patches read as a zoom)
        disp = low.resize((360, int(low.height * 360 / low.width)), Image.LANCZOS)
        disp.save(os.path.join(OUT, f"ex{idx}_image.png"))
        rec["image"] = f"ex{idx}_image.png"
        rec["demonstrates"] = (not ok["STI"]) and ok["SITIT"]   # theory holds here
        manifest["examples"].append(rec)
        print(f"  g{idx}: STI={rec['STI']['pred']}({'ok' if ok['STI'] else 'x'}) "
              f"SITIT={rec['SITIT']['pred']}({'ok' if ok['SITIT'] else 'x'}) "
              f"gt={gt} demo={rec['demonstrates']}", flush=True)
        json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

    keep = [e for e in manifest["examples"] if e["demonstrates"]]
    print(f"[done] {len(keep)}/{len(manifest['examples'])} examples show the contrast "
          f"at low res", flush=True)


if __name__ == "__main__":
    main()
