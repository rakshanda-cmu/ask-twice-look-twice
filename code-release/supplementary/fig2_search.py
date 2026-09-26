#!/usr/bin/env python3
"""
Find the Fig. 2 example: a pair where question-first STEERS the patches toward the
question's own words and still answers WRONG, while question-last stays generic and
answers RIGHT, and question echoing repairs it.

For each candidate (from fig2_cands.py) we re-run all three orderings at the
rendering resolution -- the disagreement has to survive the downscale, or the
figure would not match the run it illustrates -- and count, per layer, how many
image patches decode to one of the question's content words. The example we want
maximizes (STI hits - SIT hits) at some mid-to-late layer.

    CUDA_VISIBLE_DEVICES=1 python fig2_search.py --num 120
"""
import argparse, json, os, re

import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from naturalbench_eval import answer_suffix

NB_ROOT = "../middle_layers_indicating_hallucinations/naturalbench"
LOW_MAX = 308            # keeps the patch grid coarse enough to print words in
ORDERS = ("SIT", "STI", "STIT")

STOP = set("""is are am the a an of on in to do does did was were be been being that this
with at it its his her their your our there here they he she you we and or no not any
some by for from as into onto over under image picture photo scene""".split())
VAGUE = set("someone person people individual individuals anyone nobody thing".split())

_MASK = {}


def english_mask(mm):
    key = id(mm.tokenizer)
    if key in _MASK:
        return _MASK[key]
    V = mm.llm_model.lm_head.weight.shape[0]
    keep = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        t = mm.tokenizer.convert_ids_to_tokens(i)
        if t is None:
            continue
        s = t.replace("Ġ", "").replace("▁", "")
        if len(s) >= 3 and s.isascii() and s.isalpha():
            keep[i] = True
    _MASK[key] = keep
    return keep


def content_words(q):
    return [w for w in re.findall(r"[a-zA-Z]+", q.lower())
            if w not in STOP and len(w) >= 3]


def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def matches(pw, target):
    a, b = stem(pw.lower()), stem(target.lower())
    if len(a) < 3 or len(b) < 3:
        return False
    return a.startswith(b) or b.startswith(a)


def run(mm, pil, query, order, layers, english=True):
    """Answer + per-layer (top-1 word, probability) over the image patches."""
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [query], pil, system_prompt=SYSTEM_MESSAGE, order=order)
    n = mm.grid_h * mm.grid_w
    start = mm.img_start_idx
    with torch.inference_mode():
        gen = mm.llm_model.generate(input_ids, do_sample=False, num_beams=1,
                                    max_new_tokens=8, use_cache=True,
                                    output_hidden_states=True,
                                    return_dict_in_generate=True, **kwargs)
        ans = mm.tokenizer.batch_decode(gen["sequences"][:, input_ids.shape[1]:],
                                        skip_special_tokens=True)[0].strip()
        hs = gen["hidden_states"][0]          # the prompt forward pass
        mask = english_mask(mm).to(mm.llm_model.device) if english else None
        words, probs = {}, {}
        for L in layers:
            h = hs[L + 1][0, start:start + n]
            lg = mm.llm_model.lm_head(h).float()
            # Probability is always taken over the FULL vocabulary, so the number
            # shown on the colour scale is the token's real probability. The
            # English mask only decides WHICH token is displayed; masking before
            # the softmax would renormalize over a subset and inflate it.
            p = F.softmax(lg, dim=-1)
            sel = p.masked_fill(~mask.unsqueeze(0), -1.0) if mask is not None else p
            idx = sel.argmax(dim=-1)
            pv = p.gather(1, idx.unsqueeze(1)).squeeze(1)
            words[L] = [mm.tokenizer.decode(int(i)).strip() for i in idx.cpu()]
            probs[L] = pv.cpu().numpy().tolist()
    return ans, words, probs, mm.grid_h, mm.grid_w


def ok(ans, gt):
    a, g = ans.strip().lower(), gt.strip().lower()
    return a.startswith("yes") if g.startswith("y") else a.startswith("no")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="fig2_cands.json")
    ap.add_argument("--num", type=int, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="fig2_hits.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    cands = json.load(open(args.cands))[args.start: args.start + args.num]
    mm = ModelManager("qwen3-vl-8b")
    layers = list(range(16, mm.num_layers, 2))

    out = []
    for n, c in enumerate(cands):
        targets = [w for w in c["content"] if w not in VAGUE]
        if not targets:
            continue
        pil = Image.open(os.path.join(NB_ROOT, c["image_file"])).convert("RGB")
        s = LOW_MAX / max(pil.size)
        if s < 1:
            pil = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))),
                             Image.LANCZOS)
        query = c["question"] + answer_suffix("yes_no")
        try:
            res = {o: run(mm, pil, query, o, layers) for o in ORDERS}
        except Exception as e:
            print(f"  skip g{c['key'][0]}: {e}", flush=True)
            continue

        a = {o: res[o][0] for o in ORDERS}
        holds = (ok(a["SIT"], c["gt"]) and not ok(a["STI"], c["gt"])
                 and ok(a["STIT"], c["gt"]))

        best = None
        for L in layers:
            hs_ = {o: sum(any(matches(w, t) for t in targets) for w in res[o][1][L])
                   for o in ORDERS}
            gap = hs_["STI"] - hs_["SIT"]
            if best is None or gap > best["gap"]:
                best = {"layer": L, "gap": gap, "hits": hs_}
        rec = dict(key=c["key"], question=c["question"], gt=c["gt"],
                   image_file=c["image_file"], targets=targets, answers=a,
                   holds=bool(holds), grid=[res["SIT"][3], res["SIT"][4]], **best)
        out.append(rec)
        flag = "<HOLDS>" if holds else ""
        print(f"[{n+1}/{len(cands)}] g{c['key'][0]} gap={best['gap']:+3d} "
              f"L{best['layer']} STI={best['hits']['STI']} SIT={best['hits']['SIT']} "
              f"| {a['SIT']!r}/{a['STI']!r}/{a['STIT']!r} {flag} | {c['question']}",
              flush=True)

    out.sort(key=lambda r: (r["holds"], r["gap"]), reverse=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n===== top by (disagreement holds, steering gap) =====")
    for r in out[:20]:
        print(f"  {'H' if r['holds'] else ' '} g{r['key'][0]:5d} gap={r['gap']:+3d} "
              f"L{r['layer']:2d} (STI {r['hits']['STI']} vs SIT {r['hits']['SIT']}) "
              f"| {r['question']} [{r['image_file']}]")


if __name__ == "__main__":
    main()
