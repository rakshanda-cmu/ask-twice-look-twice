"""
TallyQA (counting VQA) ordering ablation via the vLLM library.

Source: moondream/TallyQA-VLMEvalKit (HF) — a clean TSV of base64 image +
question + integer answer, derived from the original TallyQA (Acharya et al.)
test set. Counting forces *exhaustive* scanning of the image rather than
one-object verification -- a different failure mode than presence/detection.

Metric: exact-match accuracy + MAE (mean absolute error) on the parsed integer.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/tallyqa_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT --num-samples 2000
"""
import argparse, base64, io, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
HF_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
TSV_PATH = os.path.join(HF_HUB, "datasets--moondream--TallyQA-VLMEvalKit",
                        "snapshots", "*", "tallyqa_data.tsv")
IMG_CAP = 1024
COUNT_SUFFIX = "\nAnswer with a single integer only."
NUM = re.compile(r"(?<!\d)(-?\d+)(?!\d)")


def _find_tsv():
    import glob
    hits = glob.glob(TSV_PATH)
    if hits:
        return hits[0]
    raise FileNotFoundError("TallyQA tsv not found; run the download script first.")


def load_tallyqa(n, seed=0):
    import csv, random
    csv.field_size_limit(2**28)
    rows = []
    with open(_find_tsv(), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


# TallyQA answers are small object counts (the public dataset's max is well
# under 100); a model occasionally emits a degenerate huge number (observed:
# "1000000000" for a true count of 0), which is a generation artifact, not a
# plausible count. A single such value dominates the MEAN absolute error (one
# outlier of 1e9 over 2000 examples inflates MAE from ~0.3 to ~500,000) without
# affecting exact-match accuracy. Treat implausible values as unparsed rather
# than silently averaging them in.
PLAUSIBLE_MAX_COUNT = 200


def parse_count(text):
    # Isolated digits, LAST match -- see mvbench_eval_vllm.py's parse_choice
    # for why (a verbose preamble that reasons through the scene before
    # stating the final count would otherwise have an early incidental number
    # falsely picked over the actual total).
    matches = NUM.findall(text.strip())
    for v in reversed(matches):
        v = int(v)
        if 0 <= v <= PLAUSIBLE_MAX_COUNT:
            return v
    return None


CHUNK_SIZE = 10000  # see vqa_eval_vllm.py's CHUNK_SIZE comment -- same host-RAM
# OOM risk from holding all conversations (each embedding a base64 image) in
# memory at once before a single llm.chat() call; confirmed via journalctl
# (gemma-3-27b TallyQA full run OOM-killed at ~109GB anon-RSS). Checkpointed
# between chunks so a crash resumes instead of restarting from zero.


def run_order(llm, sp, tag, letters, rows, log_every):
    out = os.path.join(OUT_DIR, f"tallyqa_order-{tag}_{MODEL_TAG}.json")
    results, n_correct, abs_err_sum, n_parsed, done_idx = [], 0, 0.0, 0, set()
    if os.path.exists(out):
        try:
            existing = json.load(open(out))
            m = existing["meta"]
            if m.get("complete"):
                print(f"  [{tag}] cached (acc={m.get('accuracy'):.3f})", flush=True)
                return
            results = existing["results"]
            done_idx = {r["row_idx"] for r in results}
            for r in results:
                n_correct += int(r["correct"])
                if r["pred"] is not None:
                    n_parsed += 1
                    abs_err_sum += abs(r["pred"] - r["gt"])
            print(f"  [resume] {tag}: {len(done_idx)} rows already done", flush=True)
        except Exception:
            results, n_correct, abs_err_sum, n_parsed, done_idx = [], 0, 0.0, 0, set()

    remaining = [(i, r) for i, r in enumerate(rows) if i not in done_idx]

    def _checkpoint(dt_so_far, complete):
        meta_out = {"benchmark": "TallyQA", "ordering": tag, "model": MODEL_TAG,
                    "engine": "vllm", "n": len(results),
                    "accuracy": n_correct / max(1, len(results)),
                    "mae": abs_err_sum / max(1, n_parsed), "parsed": n_parsed,
                    "runtime_s": dt_so_far, "complete": complete}
        json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)
        return meta_out

    t0 = time.time()
    for ci in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[ci:ci + CHUNK_SIZE]
        convs, idxs, gts = [], [], []
        for i, r in chunk:
            try:
                pil = Image.open(io.BytesIO(base64.b64decode(r["image"]))).convert("RGB")
            except Exception:
                continue
            pil = downscale(pil, IMG_CAP)
            task = r["question"].strip() + COUNT_SUFFIX
            convs.append(build_conversation(letters, task, data_uri(pil)))
            idxs.append(i)
            gts.append(int(r["answer"]))

        outputs = llm.chat(convs, sp, use_tqdm=False)
        del convs

        for o, i, gt in zip(outputs, idxs, gts):
            text = o.outputs[0].text.strip()
            pred = parse_count(text)
            correct = pred == gt
            n_correct += int(correct)
            if pred is not None:
                n_parsed += 1
                abs_err_sum += abs(pred - gt)
            results.append({"row_idx": i, "gt": gt, "pred": pred, "raw": text,
                            "correct": correct})
            if len(results) % log_every == 0:
                print(f"    [{tag}] {len(results)}/{len(rows)} "
                      f"acc={n_correct/len(results):.3f}", flush=True)
        _checkpoint(time.time() - t0, complete=False)

    dt = time.time() - t0
    meta = _checkpoint(dt, complete=True)
    print(f"  [{tag}] n={len(results)} acc={meta['accuracy']:.3f} mae={meta['mae']:.2f} "
          f"({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--num-samples", type=int, default=2000, dest="num_samples")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem",
                    help="vLLM gpu_memory_utilization; lower if another process "
                         "already holds GPU memory.")
    ap.add_argument("--log-every", type=int, default=200, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model  # rebinds the module-global MODEL_TAG other functions read at call time
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_tallyqa(args.num_samples)
    print(f"[data] {len(rows)} TallyQA rows", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model, gpu_mem=args.gpu_mem)
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
        print(f"=== TallyQA · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
