#!/usr/bin/env python3
"""
Build the candidate list for the new Fig. 2.

Fig. 2 needs a NaturalBench (image, question) pair that carries the whole paper in
one picture: question-last answers it RIGHT, question-first answers it WRONG, and
question echoing repairs it -- while the question-first patch decodings visibly
move toward the question's own words. This script does the cheap half from the
stored per-pair results (which orderings got it right); fig2_search.py then runs
the model to measure the steering.

    python fig2_cands.py --model qwen3-vl-8b
"""
import argparse, json, os, re
from collections import defaultdict

RES = "naturalbench/results"
STOP = set("""is are am the a an of on in to do does did was were be been being that this
with at it its his her their your our there here they he she you we and or no not any
some by for from as into onto over under""".split())


def load(model, order):
    p = os.path.join(RES, f"{model}__{order}__results.json")
    d = json.load(open(p))["results"]
    out = {}
    for g in d:
        for pr in g["pairs"]:
            key = (g["index"], pr["image_index"], pr["question_index"])
            out[key] = pr
    return out


def content_words(q):
    return [w for w in re.findall(r"[a-zA-Z]+", q.lower())
            if w not in STOP and len(w) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--out", default="fig2_cands.json")
    args = ap.parse_args()

    sit = load(args.model, "SIT")
    sti = load(args.model, "STI")
    stit = load(args.model, "STIT")
    try:
        sitit = load(args.model, "SITIT")
    except FileNotFoundError:
        sitit = {}

    cands = []
    for k, s in sit.items():
        t, e = sti.get(k), stit.get(k)
        if not (t and e):
            continue
        if s["correct"] and (not t["correct"]) and e["correct"]:
            if sitit and k in sitit and not sitit[k]["correct"]:
                continue                      # want every "ours" ordering to repair it
            cw = content_words(s["question"])
            cands.append({
                "key": list(k), "question": s["question"], "gt": s["gt_answer"],
                "image_file": s["image_file"],
                "sti_pred": t["model_answer_raw"], "sit_pred": s["model_answer_raw"],
                "stit_pred": e["model_answer_raw"],
                "n_content": len(cw), "content": cw,
            })

    # short, concrete questions localize best in a patch grid
    cands.sort(key=lambda c: c["n_content"])
    json.dump(cands, open(args.out, "w"), indent=2)
    print(f"[fig2] {len(cands)} pairs where SIT right, STI wrong, STIT (and SITIT) right")
    for c in cands[:15]:
        print(f"  g{c['key'][0]:5d} gt={c['gt']:3s} STI={c['sti_pred']!r:6s} "
              f"| {c['question']}")


if __name__ == "__main__":
    main()
