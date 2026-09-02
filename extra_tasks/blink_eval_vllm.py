"""
BLINK perception-probe ordering ablation via the vLLM library.

BLINK (Fu et al. 2024) reformats 14 classic visual-perception tasks as multiple
choice. This first pass covers the **single-image** subtasks only: Counting,
Relative_Depth, Relative_Reflectance, Object_Localization, IQ_Test,
Spatial_Relation. (Forensic_Detection and Art_Style *look* single-image from
their names but their rows carry a populated image_2 -- verified empirically,
not assumed -- so they are multi-image internally and excluded here too.) The
excluded multi-image subtasks (Jigsaw, Multi-view_Reasoning, *_Correspondence,
Visual_Similarity, Forensic_Detection, Art_Style) are a stretch item that would
reuse the same multi-image `common.build_conversation` path used for NExT-QA
video frames.

Source: BLINK-Benchmark/BLINK (HF), val split (has GT; test split does not).
Metric: multiple-choice exact match on the "(A)".."(D)" letter, per-subtask and
pooled.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/blink_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT
"""
import argparse, glob, io, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import pandas as pd
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm, add_echo_args, echo_tag_suffix,
                    echo_occurrence_uris)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 1024
SUBTASKS = ["Counting", "Relative_Depth", "Relative_Reflectance",
            "Object_Localization", "IQ_Test", "Spatial_Relation"]
MC_SUFFIX = "\nAnswer with the option letter in parentheses, e.g. (A)."
LETTER_PAREN = re.compile(r"\(([A-Da-d])\)")


HF_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _find_parquet(subtask):
    hits = glob.glob(os.path.join(
        HF_HUB, "datasets--BLINK-Benchmark--BLINK", "snapshots", "*",
        subtask, "val-00000-of-00001.parquet"))
    if not hits:
        raise FileNotFoundError(f"BLINK/{subtask} not found; run download first.")
    return hits[0]


def load_blink():
    rows = []
    for st in SUBTASKS:
        df = pd.read_parquet(_find_parquet(st))
        for _, r in df.iterrows():
            if r.get("image_2") is not None:
                continue  # multi-image subtask row, skip in v1
            img = r["image_1"]
            if img is None:
                continue
            rows.append({"idx": r["idx"], "subtask": st, "question": r["question"],
                         "prompt": r["prompt"], "answer": str(r["answer"]).strip(),
                         "image_bytes": img["bytes"]})
    return rows


def parse_letter(text):
    # PARENS-ONLY (no standalone fallback) -- see mmvp_eval_vllm.py's
    # parse_letter: a standalone-letter fallback falsely matches the English
    # article "a", confirmed empirically on real Gemma-3-27B output.
    m = LETTER_PAREN.search(text.strip())
    return f"({m.group(1).upper()})" if m else None


def run_order(llm, sp, tag, letters, rows, log_every, echo_scale=None, echo_which=None):
    out = os.path.join(OUT_DIR, f"blink_order-{tag}_{MODEL_TAG}.json")
    if os.path.exists(out):
        try:
            m = json.load(open(out))["meta"]
            if m.get("complete"):
                print(f"  [{tag}] cached (acc={m.get('accuracy'):.3f})", flush=True)
                return
        except Exception:
            pass

    convs = []
    for r in rows:
        pil = downscale(Image.open(io.BytesIO(r["image_bytes"])).convert("RGB"), IMG_CAP)
        task = r["prompt"].strip() + MC_SUFFIX
        if echo_scale is not None:
            occ_uris = echo_occurrence_uris(pil, echo_scale, echo_which)
            convs.append(build_conversation(letters, task, None,
                                            per_occurrence_uris=occ_uris))
        else:
            convs.append(build_conversation(letters, task, data_uri(pil)))

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, by_sub, n_correct = [], {}, 0
    for o, r in zip(outputs, rows):
        text = o.outputs[0].text.strip()
        pred = parse_letter(text)
        correct = pred == r["answer"]
        n_correct += int(correct)
        bs = by_sub.setdefault(r["subtask"], {"n": 0, "correct": 0})
        bs["n"] += 1; bs["correct"] += int(correct)
        results.append({"idx": r["idx"], "subtask": r["subtask"], "gt": r["answer"],
                        "pred": pred, "raw": text, "correct": correct})
        if len(results) % log_every == 0:
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f}", flush=True)

    by_sub_summary = {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                      for k, v in by_sub.items()}
    meta = {"benchmark": "BLINK", "ordering": tag, "model": MODEL_TAG,
            "engine": "vllm", "n": len(results),
            "accuracy": n_correct / max(1, len(results)),
            "by_subtask": by_sub_summary, "runtime_s": dt, "complete": True}
    json.dump({"meta": meta, "results": results}, open(out, "w"), indent=2)
    print(f"  [{tag}] n={len(results)} acc={meta['accuracy']:.3f} "
          f"({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    add_echo_args(ap)
    args = ap.parse_args()
    if args.echo_scale is not None and not args.echo_which:
        ap.error("--echo-scale requires --echo-which {first,second}")
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model  # rebinds the module-global MODEL_TAG other functions read at call time
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_blink()
    print(f"[data] {len(rows)} BLINK rows across {len(SUBTASKS)} single-image subtasks",
          flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model)
    sp = SamplingParams(temperature=0.0, max_tokens=256)  # Gemma-3-27B often
    # emits a verbose reasoning preamble before its final answer -- 48 tokens
    # truncated 16-79% of Gemma's BLINK/MMVP responses before the parseable
    # "(X)" ever appeared (confirmed by inspecting raw truncated text), which
    # also risked confounding the STI-vs-SITIT ordering comparison itself since
    # different orderings truncated at very different rates. Harmless for Qwen,
    # which reliably stops at EOS well before this budget.

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        if args.echo_scale is not None:
            assert tag.count("I") == 2, \
                f"--echo-scale requires a 2-image order, got {tag!r}"
            out_tag_name = echo_tag_suffix(tag, args.echo_scale, args.echo_which)
        else:
            out_tag_name = tag
        print(f"=== BLINK · ordering {out_tag_name} ===", flush=True)
        run_order(llm, sp, out_tag_name, ORDER_LETTERS[tag], rows, args.log_every,
                  echo_scale=args.echo_scale, echo_which=args.echo_which)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
