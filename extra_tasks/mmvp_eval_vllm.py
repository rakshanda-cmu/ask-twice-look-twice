"""
MMVP (Multimodal Visual Patterns) ordering ablation via the vLLM library.

MMVP (Tong et al. 2024) is 300 hand-curated CLIP-blind image pairs with a
multiple-choice question each, purpose-built to find VLM perceptual blind spots
(orientation, counting, presence, state, color, structural, viewpoint, text,
camera). Thematically the closest existing benchmark to this repo's own
steering/logit-lens analysis of *where* perception fails.

Source: MMVP/MMVP (HF) — Questions.csv + "MMVP Images/<Index>.jpg".
Metric: multiple-choice exact match on the "(a)"/"(b)" letter.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/mmvp_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT
"""
import argparse, glob, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import pandas as pd
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 1024
MC_SUFFIX = "\nAnswer with the option letter in parentheses, e.g. (a)."
LETTER_PAREN = re.compile(r"\(([A-Da-d])\)")


HF_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _find_mmvp_root():
    hits = glob.glob(os.path.join(HF_HUB, "datasets--MMVP--MMVP", "snapshots", "*"))
    if not hits:
        raise FileNotFoundError("MMVP not found; run the download script first.")
    return hits[0]


def load_mmvp():
    root = _find_mmvp_root()
    df = pd.read_csv(os.path.join(root, "Questions.csv"))
    rows = []
    for _, r in df.iterrows():
        img_path = os.path.join(root, "MMVP Images", f"{int(r['Index'])}.jpg")
        if os.path.isfile(img_path):
            rows.append({"index": int(r["Index"]), "question": r["Question"],
                         "options": r["Options"], "answer": str(r["Correct Answer"]),
                         "image_path": img_path})
    return rows


def parse_letter(text):
    # PARENS-ONLY, deliberately no standalone-letter fallback: a verbose
    # preamble ("Certainly! Let's take a look...") can contain the English
    # article "a" as an isolated word, which a naive word-boundary fallback
    # (naturalbench_eval.py's _first_letter pattern) would falsely match as
    # the answer -- confirmed empirically on real Gemma-3-27B output. Since
    # the prompt explicitly requests "(a)"-style formatting, requiring the
    # parens is not a loss of recall for compliant answers, and a model that
    # never produces the format within the token budget is honestly unparsed.
    m = LETTER_PAREN.search(text.strip())
    return f"({m.group(1).lower()})" if m else None


def run_order(llm, sp, tag, letters, rows, log_every):
    out = os.path.join(OUT_DIR, f"mmvp_order-{tag}_{MODEL_TAG}.json")
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
        pil = downscale(Image.open(r["image_path"]).convert("RGB"), IMG_CAP)
        task = f"{r['question']}\nOptions: {r['options']}{MC_SUFFIX}"
        convs.append(build_conversation(letters, task, data_uri(pil)))

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, n_correct = [], 0
    for o, r in zip(outputs, rows):
        text = o.outputs[0].text.strip()
        pred = parse_letter(text)
        gt = r["answer"].strip().lower()
        correct = pred == gt
        n_correct += int(correct)
        results.append({"index": r["index"], "question": r["question"], "gt": gt,
                        "pred": pred, "raw": text, "correct": correct})
        if len(results) % log_every == 0:
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f}", flush=True)

    meta = {"benchmark": "MMVP", "ordering": tag, "model": MODEL_TAG,
            "engine": "vllm", "n": len(results),
            "accuracy": n_correct / max(1, len(results)), "runtime_s": dt,
            "complete": True}
    json.dump({"meta": meta, "results": results}, open(out, "w"), indent=2)
    print(f"  [{tag}] n={len(results)} acc={meta['accuracy']:.3f} "
          f"({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model  # rebinds the module-global MODEL_TAG other functions read at call time
    if args.model == "gemma-3-27b" and args.tp != 1:
        print("  [note] forcing --tp 1 for gemma-3-27b (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_mmvp()
    print(f"[data] {len(rows)} MMVP rows", flush=True)

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
        print(f"=== MMVP · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
