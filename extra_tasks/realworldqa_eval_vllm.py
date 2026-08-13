"""
RealWorldQA ordering ablation via the vLLM library.

RealWorldQA (xAI, 2024) is 765 real-world photos (many from vehicles) with a
mix of multiple-choice ("answer with only the letter") and free-form
single-word/number questions ("Yes"/"No", a count, a color) -- each
question's own text already carries its answer-format instruction, so unlike
this repo's other MC harnesses no extra suffix is appended here.

Source: xai-org/RealworldQA (HF), 2 parquet shards. Ground truth ("answer")
is either a bare letter (A-D) or a free-form word/number.
Metric: for letter-GT rows, exact match on the extracted letter (bare, "(X)",
or "X." -- NOT parens-only, since the dataset's own prompt asks for a bare
letter, unlike this repo's other MC harnesses). For free-form GT, VQA-style
normalized token-boundary containment (vqa_eval.normalize_answer), EXCEPT
that normalize_answer strips "a"/"an"/"the" as articles -- which would
silently blank out a genuine gold answer of "A", so letter-GT rows are always
routed through the letter path first, never through normalize_answer.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/realworldqa_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT
"""
import argparse, io, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
from PIL import Image
from vqa_eval import normalize_answer
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 1024
LETTER_GTS = {"A", "B", "C", "D"}


def load_realworldqa():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    for shard in ("test-00000-of-00002", "test-00001-of-00002"):
        p = hf_hub_download("xai-org/RealworldQA", f"data/{shard}.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        for i, r in df.iterrows():
            rows.append({"index": len(rows), "question": r["question"],
                        "answer": str(r["answer"]).strip(),
                        "image_bytes": r["image"]["bytes"]})
    return rows


def extract_letter(text):
    text = text.strip()
    m = re.match(r"^\(?([A-Da-d])\)?[\.\):]?(\s|$)", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", text)
    return m.group(1).upper() if m else None


def grade(raw, gt):
    """Returns (correct, pred_display)."""
    if gt.upper() in LETTER_GTS:
        pred = extract_letter(raw)
        return pred == gt.upper(), pred
    gt_n = normalize_answer(gt)
    pred_n = normalize_answer(raw)
    if not gt_n:
        return gt_n == pred_n, pred_n
    hit = re.search(r"(?<!\w)" + re.escape(gt_n) + r"(?!\w)", pred_n) is not None
    return hit, pred_n


def run_order(llm, sp, tag, letters, rows, log_every):
    out = os.path.join(OUT_DIR, f"realworldqa_order-{tag}_{MODEL_TAG}.json")
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
        convs.append(build_conversation(letters, r["question"], data_uri(pil)))

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, n_correct = [], 0
    for o, r in zip(outputs, rows):
        text = o.outputs[0].text.strip()
        correct, pred = grade(text, r["answer"])
        n_correct += int(correct)
        results.append({"index": r["index"], "question": r["question"],
                        "gt": r["answer"], "pred": pred, "raw": text, "correct": correct})
        if len(results) % log_every == 0:
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f}", flush=True)

    meta = {"benchmark": "RealWorldQA", "ordering": tag, "model": MODEL_TAG,
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
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem",
                    help="vLLM gpu_memory_utilization; lower this if another "
                         "process (e.g. the r1-1.5b server) already holds GPU0.")
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_realworldqa()
    print(f"[data] {len(rows)} RealWorldQA rows", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model, gpu_mem=args.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=64)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        print(f"=== RealWorldQA · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
