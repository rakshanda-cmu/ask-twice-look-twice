"""
Search NaturalBench for zoomed-in cases where question-first (STI) patches surface
the question's content words but question-last (SIT) patches do NOT -- i.e. STI
steers perception to the answer object, SIT does not (yet SIT answers correctly).

Runs from the MAIN research repo so it can import the model + read the images.
Reads candidate (STI-wrong, SIT-correct) pairs from cands.json; writes ranked
results to steer_hits.json.

  cd /home/grg/Research/middle_layers_indicating_hallucinations
  CUDA_VISIBLE_DEVICES=0 python /home/.../website/search_steer.py --num 90
"""
import argparse, json, os, re
import numpy as np, torch
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from constants import SYSTEM_MESSAGE
from naturalbench_eval import answer_suffix
from model_manager import ModelManager
from logit_lens_overlay import logit_lens_all_vision_tokens

SUPP = "/home/grg/Research/ask-twice-look-twice-supp"
NB_ROOT = "/home/grg/Research/middle_layers_indicating_hallucinations/naturalbench"
SCRATCH = ("/tmp/claude-1000/-home-grg-Research-ask-twice-look-twice-supp/"
           "873eb924-61f3-4169-9a36-e6f2390cabf0/scratchpad")
LOW_MAX = 280
TOP_K, MAX_TOK = 50, 8
SCORE_LAYERS = None          # set after model load: mid-late band
STOP = set("is are am the a an of on in to do does did was were be been being that "
           "this with at it its his her their your our there here they he she you we "
           "and or no not any some by for from as into onto".split())


def content_words(q):
    ws = [w for w in re.findall(r"[a-zA-Z]+", q.lower()) if w not in STOP and len(w) >= 3]
    return ws


def key_word(words):
    ing = [w for w in words if w.endswith("ing") and len(w) > 4]
    if ing:
        return max(ing, key=len)
    return max(words, key=len) if words else ""


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def matches(patch_word, target):
    """True if an English patch word matches a target content word (stem-wise)."""
    pw = patch_word.strip().lower()
    if not re.fullmatch(r"[a-z]{3,}", pw):
        return False
    a, b = stem(pw), stem(target)
    if len(a) < 3 or len(b) < 3:
        return False
    return a.startswith(b) or b.startswith(a) or a in b or b in a


def patch_hits(words_by_layer, targets):
    """Count (layer,patch) cells whose word matches any target; also per-layer counts."""
    per_layer = []
    for lw in words_by_layer:                       # one layer
        c = sum(any(matches(w, t) for t in targets) for w in lw)
        per_layer.append(c)
    return sum(per_layer), per_layer


def run_order(mm, pil, query, order, warper, proc):
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [query], pil, system_prompt=SYSTEM_MESSAGE, order=order)
    with torch.inference_mode():
        outputs = mm.llm_model.generate(
            input_ids, do_sample=False, num_beams=1, max_new_tokens=MAX_TOK,
            use_cache=True, output_hidden_states=True, return_dict_in_generate=True,
            **kwargs)
    ans = mm.tokenizer.batch_decode(outputs["sequences"][:, input_ids.shape[1]:],
                                    skip_special_tokens=True)[0].strip()
    _, words = logit_lens_all_vision_tokens(
        mm.llm_model, mm.tokenizer, input_ids, outputs, mm.img_start_idx,
        SCORE_LAYERS, warper, proc, grid_h=mm.grid_h, grid_w=mm.grid_w)
    return ans, words


def main():
    global SCORE_LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=90)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cands", default="cands.json")
    ap.add_argument("--out", default="steer_hits.json")
    args = ap.parse_args()
    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()

    mm = ModelManager("qwen3-vl-8b")
    SCORE_LAYERS = list(range(16, mm.num_layers, 2))
    warper = TopKLogitsWarper(top_k=TOP_K, filter_value=float("-inf"))
    proc = LogitsProcessorList([])

    cands = json.load(open(os.path.join(SCRATCH, args.cands)))
    # order by fewest content words (cleaner, more localizable questions)
    cands.sort(key=lambda c: len(content_words(c[1])))
    cands = cands[args.start:args.start + args.num]

    out = []
    for n, c in enumerate(cands):
        (gi, ii, qi), question, gt, image_file, sti_pred, sit_pred = c
        cw = content_words(question)
        kw = key_word(cw)
        if not kw:
            continue
        targets = list(set([kw] + [w for w in cw if w not in
                           ("image", "someone", "person", "people", "individual",
                            "individuals", "anyone", "nobody")]))
        pil = Image.open(os.path.join(NB_ROOT, image_file)).convert("RGB")
        w, h = pil.size
        s = LOW_MAX / max(w, h)
        if s < 1.0:
            pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        query = question + answer_suffix("yes_no")
        try:
            sti_ans, sti_w = run_order(mm, pil, query, "STI", warper, proc)
            sit_ans, sit_w = run_order(mm, pil, query, "SIT", warper, proc)
        except Exception as ex:
            print(f"  skip {gi}: {ex}", flush=True)
            continue
        # key-word hits (the action) and all-content hits
        ksti, ksti_pl = patch_hits(sti_w, [kw])
        ksit, ksit_pl = patch_hits(sit_w, [kw])
        asti, _ = patch_hits(sti_w, targets)
        asit, _ = patch_hits(sit_w, targets)
        peak_layer = SCORE_LAYERS[int(np.argmax(ksti_pl))] if any(ksti_pl) else \
            SCORE_LAYERS[int(np.argmax([sum(any(matches(x, t) for t in targets)
                                            for x in lw) for lw in sti_w]))]
        def ok(ans):
            a = ans.strip().lower()
            g = gt.strip().lower()
            return a.startswith(g[:2]) or (g.startswith("y") and a.startswith("yes")) \
                or (g.startswith("n") and a.startswith("no"))
        sti_ok, sit_ok = ok(sti_ans), ok(sit_ans)
        paradox = (not sti_ok) and sit_ok      # STI wrong, SIT right AT LOW RES
        rec = dict(group=gi, image_index=ii, question_index=qi, question=question,
                   gt=gt, image=image_file, key=kw, targets=targets,
                   sti_ans=sti_ans, sit_ans=sit_ans, sti_ok=sti_ok, sit_ok=sit_ok,
                   paradox=paradox, key_sti=ksti, key_sit=ksit, all_sti=asti,
                   all_sit=asit, key_gap=ksti - ksit, all_gap=asti - asit,
                   peak_layer=peak_layer)
        out.append(rec)
        print(f"[{n+1}/{len(cands)}] g{gi} {question!r} key={kw!r} "
              f"| STI key={ksti} ans={sti_ans!r}{'ok' if sti_ok else 'X'} "
              f"| SIT key={ksit} ans={sit_ans!r}{'ok' if sit_ok else 'X'} "
              f"| gap={ksti-ksit} peakL{peak_layer} {'<PARADOX>' if paradox else ''}",
              flush=True)

    # rank: paradox first, then biggest key-word steering gap
    out.sort(key=lambda r: (r["paradox"], r["key_gap"], r["all_gap"]), reverse=True)
    json.dump(out, open(os.path.join(SCRATCH, args.out), "w"), indent=2)
    npar = sum(r["paradox"] for r in out)
    print(f"\n===== {npar}/{len(out)} hold the paradox at low res. "
          f"TOP by (paradox, key-word steering gap) =====")
    for r in out[:20]:
        print(f"  {'P' if r['paradox'] else ' '} g{r['group']} key={r['key']!r} "
              f"keygap={r['key_gap']} (STI {r['key_sti']} vs SIT {r['key_sit']}) "
              f"peakL{r['peak_layer']} | STI={r['sti_ans']!r} SIT={r['sit_ans']!r} "
              f"| {r['question']!r} [{r['image']}]", flush=True)


if __name__ == "__main__":
    main()
