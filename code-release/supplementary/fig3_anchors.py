#!/usr/bin/env python3
"""
Choose the Fig. 3 scene by the property the figure actually needs: every object
region must own a patch whose logit-lens top-3 tokens NAME that object.

The paper's callout figures quote the top-3 vocabulary tokens at one patch per
region ("cat: asleep - sleeping - breeds"), not the argmax over every patch, so
that is what we score. For each COCO ground-truth object we look at the patches
inside its box and keep the best "anchor": the patch whose top-3 tokens are clean
English AND, ideally, match the object's category name. A scene is a good Fig. 3
scene when several well-separated objects each own such an anchor, which makes
"each region decodes to its own object" literally checkable rather than a claim.

    CUDA_VISIBLE_DEVICES=1 python fig3_anchors.py --pool fig3_coco350.json --top 8
"""
import argparse, json, os, re

import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager

IMGDIR = "datasets/COCO/val2014"
LAYER = 28
LONG_SIDE = 896
TOPK = 3

WORD_RE = re.compile(r"^[a-z][a-z\-]{2,}$")
GENERIC = set("""
background backdrop scene image photo picture view color colour gray grey white
black blue green red yellow brown dark light bright blur blurry blurred texture
pattern surface area region part parts side left right top bottom center centre
middle front back near far above below inside outside something someone thing
things stuff other others more most less least very much many few some any what
which where when who how why the and but for with from into onto over under
looks look looking seems seem appears appear shows show showing being been
having never seventh fifth forth sixth eighth ninth tenth upright vertical
horizontal empty sleek colorful painted wooden rustic first second third
""".split())

# Words that count as naming a COCO category (the model rarely emits the exact
# dataset string, so accept the obvious synonyms/parts).
ALIAS = {
    "tv": "tv television screen monitor telly display broadcast channel",
    "cat": "cat kitten feline tabby paws whisker purr asleep sleeping striped",
    "dog": "dog puppy canine breed retriever paws leash",
    "potted plant": "plant plants foliage leaves leaf greenery pot potted fern",
    "person": "person man woman people guy girl boy face hands wearing shirt",
    "laptop": "laptop notebook macbook keyboard screen computer",
    "couch": "couch sofa cushion cushions upholstery seating",
    "chair": "chair seat stool armchair",
    "dining table": "table tabletop dining desk wooden",
    "pizza": "pizza cheese crust slice toppings pepperoni",
    "cake": "cake frosting icing dessert layers chocolate",
    "bed": "bed pillow pillows mattress blanket duvet sheets",
    "bicycle": "bicycle bike wheels handlebar cycling",
    "motorcycle": "motorcycle motorbike bike engine helmet",
    "car": "car vehicle sedan automobile hood bumper",
    "bus": "bus coach transit double decker",
    "train": "train locomotive railway carriage rail",
    "boat": "boat vessel ship sail hull",
    "horse": "horse mare pony equine saddle hoof",
    "bird": "bird beak feather feathers wings perched",
    "elephant": "elephant trunk tusk tusks",
    "bear": "bear grizzly panda fur",
    "zebra": "zebra stripes striped",
    "giraffe": "giraffe neck spots",
    "sheep": "sheep lamb wool flock",
    "cow": "cow cattle bovine calf",
    "clock": "clock time hands dial",
    "vase": "vase pottery ceramic urn",
    "book": "book books pages novel",
    "bottle": "bottle bottles glass wine label",
    "refrigerator": "refrigerator fridge freezer appliance",
    "oven": "oven stove range appliance",
    "sink": "sink faucet basin tap",
    "toilet": "toilet lavatory bathroom",
    "keyboard": "keyboard keys typing",
    "teddy bear": "teddy bear plush stuffed toy",
    "banana": "banana bananas",
    "apple": "apple apples",
    "orange": "orange oranges citrus",
    "surfboard": "surfboard surf board wave",
    "umbrella": "umbrella parasol canopy",
    "bench": "bench seat park",
    "traffic light": "traffic light signal lights",
}


def clean(w):
    w = w.strip().lower()
    return w if WORD_RE.fullmatch(w) and w not in GENERIC else None


def names(cat, toks):
    """Do this patch's top-k tokens name `cat`?"""
    voc = set(ALIAS.get(cat, cat).split())
    for t in toks:
        c = clean(t)
        if not c:
            continue
        for v in voc:
            if c == v or (len(c) >= 4 and len(v) >= 4 and (c.startswith(v[:4]) or v.startswith(c[:4]))):
                return True
    return False


class Lens:
    """Top-k logit-lens decode of the image patches at one layer."""

    def __init__(self, model, tokenizer):
        self.m, self.t = model, tokenizer

    def run(self, mm, pil, question=None, order="I", system=""):
        q = [question] if question else [""]
        _, input_ids, kwargs = mm.prepare_inputs_from_pil(q, pil, system_prompt=system, order=order)
        with torch.inference_mode():
            out = self.m(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
            h = out.hidden_states[LAYER + 1][0, mm.img_start_idx:
                                             mm.img_start_idx + mm.grid_h * mm.grid_w]
            logits = self.m.lm_head(h).float()
            probs = F.softmax(logits, dim=-1)
            p, idx = probs.topk(TOPK, dim=-1)
        toks = [[self.t.decode(int(i)).strip() for i in row] for row in idx.cpu()]
        return toks, p.cpu().numpy(), mm.grid_h, mm.grid_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="fig3_coco350.json")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="fig3_anchor_ranked.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pool = json.load(open(args.pool))
    print(f"[anchors] {len(pool)} scenes", flush=True)
    mm = ModelManager("qwen3-vl-8b")
    lens = Lens(mm.llm_model, mm.tokenizer)

    rows = []
    for k, r in enumerate(pool):
        path = os.path.join(IMGDIR, r["file"])
        pil = Image.open(path).convert("RGB")
        W0, H0 = pil.size
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            toks, probs, gh, gw = lens.run(mm, pil)
        except Exception as e:
            print(f"  [skip] {r['file']}: {e}", flush=True)
            continue

        found = []
        for o in r["objects"]:
            x, y, w, h = o["bbox"]
            c0, c1 = int(x / W0 * gw), int(np.ceil((x + w) / W0 * gw))
            r0, r1 = int(y / H0 * gh), int(np.ceil((y + h) / H0 * gh))
            best = None
            for rr in range(max(0, r0), min(gh, max(r0 + 1, r1))):
                for cc in range(max(0, c0), min(gw, max(c0 + 1, c1))):
                    t = toks[rr * gw + cc]
                    nice = [clean(x_) for x_ in t]
                    n_clean = sum(1 for x_ in nice if x_)
                    hit = names(o["name"], t)
                    sc = 3.0 * hit + n_clean + float(probs[rr * gw + cc][0])
                    if best is None or sc > best["sc"]:
                        best = {"sc": sc, "rc": [rr, cc], "toks": t,
                                "hit": bool(hit), "n_clean": n_clean}
            if best and best["hit"] and best["n_clean"] >= 2:
                found.append({"name": o["name"], **best})

        if len(found) >= 3:
            ctr = np.array([[f["rc"][0] / gh, f["rc"][1] / gw] for f in found])
            sep = min(np.linalg.norm(ctr[i] - ctr[j])
                      for i in range(len(ctr)) for j in range(i + 1, len(ctr)))
            sc = 3.0 * len(found) + 5.0 * sep + 0.4 * sum(f["n_clean"] for f in found)
            rows.append({"file": r["file"], "score": round(float(sc), 3),
                         "sep": round(float(sep), 3), "grid": [gh, gw],
                         "n_named": len(found),
                         "anchors": [{"name": f["name"], "rc": f["rc"], "toks": f["toks"]}
                                     for f in found]})
        if (k + 1) % 40 == 0:
            b = max((x["score"] for x in rows), default=0)
            print(f"  [{k+1}/{len(pool)}] kept={len(rows)} best={b:.2f}", flush=True)

    rows.sort(key=lambda r: -r["score"])
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\n[anchors] {len(rows)} scenes where >=3 objects name themselves. Top:")
    for i, r in enumerate(rows[: args.top]):
        print(f" {i:2d}. {r['file']}  score={r['score']:.2f} sep={r['sep']:.2f}")
        for a in r["anchors"]:
            print(f"        {a['name']:14s} -> {' / '.join(a['toks'])}")


if __name__ == "__main__":
    main()
