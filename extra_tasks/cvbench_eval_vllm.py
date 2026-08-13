"""
CV-Bench ordering ablation via the vLLM library.

CV-Bench (Tong et al., Cambrian-1, 2024) repurposes standard 2D (ADE20K,
COCO) and 3D (Omni3D) vision annotations into 2,638 multiple-choice VQA
questions across 4 tasks: Count, Relation (2D), Depth, Distance (3D) --
testing whether VLM perception of basic spatial/counting relations survives
the same S/T/I ordering manipulation used elsewhere in this repo.

Source: nyu-visionx/CV-Bench (HF), test_2d.parquet + test_3d.parquet. Each
row's "prompt" field already embeds the question + lettered choices; "answer"
is "(X)" for the gold letter (Count questions can have >4 choices, so the
parser is not restricted to A-D).
Metric: multiple-choice exact match on the "(X)" letter, pooled + per-task.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/cvbench_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT
"""
import argparse, io, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 1024
MC_SUFFIX = "\nAnswer with the option letter in parentheses, e.g. (A)."
LETTER_PAREN = re.compile(r"\(([A-Za-z])\)")


def load_cvbench():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    for split in ("test_2d", "test_3d"):
        p = hf_hub_download("nyu-visionx/CV-Bench", f"{split}.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            rows.append({"idx": int(r["idx"]), "type": r["type"], "task": r["task"],
                        "prompt": r["prompt"], "answer": str(r["answer"]).strip(),
                        "image_bytes": r["image"]["bytes"]})
    return rows


def parse_letter(text):
    m = LETTER_PAREN.search(text.strip())
    return f"({m.group(1).upper()})" if m else None


def run_order(llm, sp, tag, letters, rows, log_every):
    out = os.path.join(OUT_DIR, f"cvbench_order-{tag}_{MODEL_TAG}.json")
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
        task = r["prompt"] + MC_SUFFIX
        convs.append(build_conversation(letters, task, data_uri(pil)))

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, n_correct, by_task = [], 0, {}
    for o, r in zip(outputs, rows):
        text = o.outputs[0].text.strip()
        pred = parse_letter(text)
        gt = r["answer"].upper()
        correct = pred == gt
        n_correct += int(correct)
        bt = by_task.setdefault(r["task"], {"n": 0, "correct": 0})
        bt["n"] += 1; bt["correct"] += int(correct)
        results.append({"idx": r["idx"], "type": r["type"], "task": r["task"],
                        "gt": gt, "pred": pred, "raw": text, "correct": correct})
        if len(results) % log_every == 0:
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f}", flush=True)

    by_task_summary = {t: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                       for t, v in by_task.items()}
    meta = {"benchmark": "CV-Bench", "ordering": tag, "model": MODEL_TAG,
            "engine": "vllm", "n": len(results),
            "accuracy": n_correct / max(1, len(results)), "by_task": by_task_summary,
            "runtime_s": dt, "complete": True}
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

    rows = load_cvbench()
    print(f"[data] {len(rows)} CV-Bench rows", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model, gpu_mem=args.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        print(f"=== CV-Bench · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
