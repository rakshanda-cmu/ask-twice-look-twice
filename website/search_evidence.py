"""
Evidence-based, per-layer steering search. For each candidate (with a hand-written
answer-evidence token set), measure at EVERY mid-late layer how many image patches
decode an evidence token under STI vs SIT, and report the best layer (largest
STI-minus-SIT evidence gap). Ranks candidates so the strongest "STI identifies the
answer evidence, SIT does not" cases float to the top.

  cd /home/grg/Research/middle_layers_indicating_hallucinations
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
    /home/grg/anaconda3/envs/logitlens/bin/python /home/.../website/search_evidence.py
"""
import json, os, re
import numpy as np, torch
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from constants import SYSTEM_MESSAGE
from naturalbench_eval import answer_suffix
from model_manager import ModelManager
from logit_lens_overlay import logit_lens_all_vision_tokens

NB_ROOT = "/home/grg/Research/middle_layers_indicating_hallucinations/naturalbench"
SCRATCH = ("/tmp/claude-1000/-home-grg-Research-ask-twice-look-twice-supp/"
           "873eb924-61f3-4169-9a36-e6f2390cabf0/scratchpad")
LOW_MAX, TOP_K, MAX_TOK = 280, 50, 8


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def matches(pw, target):
    pw = pw.strip().lower()
    if not re.fullmatch(r"[a-z-]{3,}", pw):
        return False
    a, b = stem(pw), stem(target.lower())
    return len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)
                                            or a in b or b in a)


def hits_by_layer(words_by_layer, evidence):
    """For each layer return (count, {token:count}) of evidence-matching patches."""
    res = []
    for lw in words_by_layer:
        found = {}
        for w in lw:
            for t in evidence:
                if matches(w, t):
                    found[t] = found.get(t, 0) + 1
                    break
        res.append((sum(found.values()), found))
    return res


def run(mm, pil, query, order, layers, warper, proc):
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
        layers, warper, proc, grid_h=mm.grid_h, grid_w=mm.grid_w)
    return ans, words


def main():
    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()
    mm = ModelManager("qwen3-vl-8b")
    layers = list(range(12, mm.num_layers, 2))     # scan layers 12..34
    warper = TopKLogitsWarper(top_k=TOP_K, filter_value=float("-inf"))
    proc = LogitsProcessorList([])

    cands = json.load(open(os.path.join(SCRATCH, os.environ.get("EVID_FILE","evidence_sets.json"))))
    out = []
    for n, (gid, c) in enumerate(cands.items()):
        ev = c["evidence"]; gt = c["gt"]
        pil = Image.open(os.path.join(NB_ROOT, c["image"])).convert("RGB")
        w, h = pil.size
        s = LOW_MAX / max(w, h)
        if s < 1.0:
            pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        query = c["question"] + answer_suffix("yes_no")
        try:
            sti_ans, sti_w = run(mm, pil, query, "STI", layers, warper, proc)
            sit_ans, sit_w = run(mm, pil, query, "SIT", layers, warper, proc)
        except Exception as ex:
            print(f"  skip g{gid}: {ex}", flush=True); continue
        sti_h = hits_by_layer(sti_w, ev)
        sit_h = hits_by_layer(sit_w, ev)
        # best layer = largest STI-minus-SIT evidence gap (tie-break by STI count)
        best_i = max(range(len(layers)), key=lambda i: (sti_h[i][0] - sit_h[i][0],
                                                        sti_h[i][0]))
        bl = layers[best_i]
        ns, found = sti_h[best_i]; nt = sit_h[best_i][0]

        def ok(a):
            a = a.strip().lower()
            return (gt[0].lower() == "y" and a.startswith("y")) or \
                   (gt[0].lower() == "n" and a.startswith("n"))
        paradox = (not ok(sti_ans)) and ok(sit_ans)
        sti_wins = ok(sti_ans) and (not ok(sit_ans))
        rec = dict(group=int(gid), question=c["question"], gt=gt, image=c["image"],
                   evidence=ev, best_layer=bl, sti=ns, sit=nt, gap=ns - nt,
                   sti_tokens=found, sti_ans=sti_ans, sit_ans=sit_ans,
                   paradox=paradox, sti_wins=sti_wins,
                   per_layer=[(layers[i], sti_h[i][0], sit_h[i][0]) for i in range(len(layers))])
        out.append(rec)
        print(f"[{n+1}/{len(cands)}] g{gid} bestL{bl} STI {ns} vs SIT {nt} "
              f"gap {ns-nt} {'STI-WINS' if sti_wins else ('PARADOX' if paradox else '')} tokens={list(found)} "
              f"| {c['question']}", flush=True)

    out.sort(key=lambda r: (r["gap"], r["sti"]), reverse=True)
    json.dump(out, open(os.path.join(SCRATCH, os.environ.get("EVID_OUT","evidence_hits.json")), "w"), indent=2)
    print("\n===== RANKED by best-layer evidence gap (STI - SIT) =====")
    for r in out:
        print(f"  g{r['group']:4d} L{r['best_layer']:2d} STI {r['sti']:2d} vs SIT "
              f"{r['sit']:2d} (gap {r['gap']:2d}) {'P' if r['paradox'] else ' '} "
              f"{list(r['sti_tokens'])} | {r['question']}", flush=True)


if __name__ == "__main__":
    main()
