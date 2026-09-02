"""
NExT-QA (multi-frame video QA) ordering ablation via the vLLM library.

The cleanest generalization of the paper's core mechanism: instead of one image,
"I" expands to K uniformly-sampled frames, so "SITIT" (image echo) becomes
*re-showing the whole K-frame clip a second time* -- turning "recency loss" into
an explicit, tunable temporal-distance variable (frames between question and
answer), extending what naturalbench_tokensweep.py does with token distance.
Reuses common.build_conversation unchanged: it already accepts a list of image
URIs at each "I" position.

Source: lmms-lab/NExTQA (HF), MC (multiple-choice) test split, 5-way (a0..a4).
Frames extracted from videos.zip with OpenCV (uniform sampling, K frames/video).
Metric: multiple-choice exact match (0-4), overall + by question type
(C=causal, T=temporal, D=descriptive).

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/nextqa_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT --num-samples 1500 --frames 6
"""
import argparse, glob, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import cv2
import pandas as pd
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm, add_echo_args, echo_tag_suffix,
                    echo_occurrence_uris)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
VIDEO_DIR = "/data2/hf_cache/newtasks/nextqa/NExTVideo"
IMG_CAP = 512  # smaller per-frame cap: K frames already multiply token cost
MC_SUFFIX = "\nAnswer with the option number only (0, 1, 2, 3, or 4)."


HF_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _find_mc_parquet():
    hits = glob.glob(os.path.join(
        HF_HUB, "datasets--lmms-lab--NExTQA", "snapshots", "*",
        "MC", "test-00000-of-00001.parquet"))
    if not hits:
        raise FileNotFoundError("NExT-QA MC parquet not found; run download first.")
    return hits[0]


def load_nextqa(n, seed=0):
    import random
    df = pd.read_parquet(_find_mc_parquet())
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        vpath = os.path.join(VIDEO_DIR, f"{r['video']}.mp4")
        if os.path.isfile(vpath):
            rows.append(r.to_dict())
        if len(rows) >= n:
            break
    return rows


def sample_frames(video_path, k):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = sorted(set(int(i * (total - 1) / max(1, k - 1)) for i in range(k)))
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


ISOLATED_NUM_5 = re.compile(r"(?<!\d)([0-4])(?!\d)")


def parse_choice(text):
    # Isolated digits only, LAST match -- see mvbench_eval_vllm.py's
    # parse_choice for why (preamble text can mention unrelated numbers).
    matches = ISOLATED_NUM_5.findall(text.strip())
    return int(matches[-1]) if matches else None


# vLLM's internal multimodal-processor cache has hit an AssertionError
# ("Expected a cached item for mm_hash=...") when a single llm.chat() call
# carries too many distinct images at once (observed: 1500 prompts x 6 frames
# = 9000 images in one batch overwhelmed it). Chunking keeps each batch's
# working set well inside the cache and is also checkpoint-friendly, so a
# crash mid-ordering resumes from the last saved chunk instead of restarting.
CHUNK_SIZE = 100


def run_order(llm, sp, tag, letters, rows, k_frames, log_every,
             echo_scale=None, echo_which=None):
    out = os.path.join(OUT_DIR, f"nextqa_order-{tag}_{MODEL_TAG}.json")
    results, done = [], set()
    if os.path.exists(out):
        try:
            d = json.load(open(out))
            if d["meta"].get("complete"):
                print(f"  [{tag}] cached (acc={d['meta'].get('accuracy'):.3f})", flush=True)
                return
            results = d.get("results", [])
            done = {(r["video"], r["qid"]) for r in results}
            print(f"  [resume] {len(done)} already done", flush=True)
        except Exception:
            results, done = [], set()

    convs, meta = [], []
    for r in rows:
        # NExT-QA's qid is a PER-VIDEO local question index (0..~13), NOT
        # globally unique -- it restarts at 0 for every video, so dedup must
        # key on (video, qid) together or almost every row gets wrongly
        # treated as "already done" once a few small qid values have appeared.
        if (str(r["video"]), int(r["qid"])) in done:
            continue
        vpath = os.path.join(VIDEO_DIR, f"{r['video']}.mp4")
        frames = sample_frames(vpath, k_frames)
        if not frames:
            continue
        dframes = [downscale(f, IMG_CAP) for f in frames]
        opts = "\n".join(f"{i}: {r[f'a{i}']}" for i in range(5))
        task = f"{r['question']}\nOptions:\n{opts}{MC_SUFFIX}"
        if echo_scale is not None:
            occ_uris = echo_occurrence_uris(dframes, echo_scale, echo_which)
            convs.append(build_conversation(letters, task, None,
                                            per_occurrence_uris=occ_uris))
        else:
            uris = [data_uri(f) for f in dframes]
            convs.append(build_conversation(letters, task, uris))
        meta.append({"qid": int(r["qid"]), "video": str(r["video"]),
                     "type": r["type"], "answer": int(r["answer"]),
                     "question": r["question"]})

    t0 = time.time()
    outputs = []
    for i in range(0, len(convs), CHUNK_SIZE):
        chunk_out = llm.chat(convs[i:i + CHUNK_SIZE], sp, use_tqdm=False)
        outputs.extend(chunk_out)
        print(f"    [{tag}] generated {min(i+CHUNK_SIZE, len(convs))}/{len(convs)}",
              flush=True)
        _save(out, tag, results + _score(outputs, meta[:len(outputs)]))
    dt = time.time() - t0

    results = results + _score(outputs, meta)
    n_correct = sum(r["correct"] for r in results)
    _save(out, tag, results, complete=True)
    print(f"  [{tag}] n={len(results)} acc={n_correct/max(1,len(results)):.3f} "
          f"({dt:.0f}s) -> {out}", flush=True)


def _score(outputs, meta):
    scored = []
    for o, m in zip(outputs, meta):
        text = o.outputs[0].text.strip()
        pred = parse_choice(text)
        correct = pred == m["answer"]
        scored.append({**m, "pred": pred, "raw": text, "correct": correct})
    return scored


def _save(out, tag, results, complete=False):
    by_type, n_correct = {}, sum(r["correct"] for r in results)
    for r in results:
        bt = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        bt["n"] += 1; bt["correct"] += int(r["correct"])
    by_type_summary = {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                       for k, v in by_type.items()}
    meta_out = {"benchmark": "NExT-QA", "ordering": tag, "model": MODEL_TAG,
                "engine": "vllm", "n": len(results),
                "accuracy": n_correct / max(1, len(results)),
                "by_type": by_type_summary, "complete": complete}
    json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--num-samples", type=int, default=1500, dest="num_samples")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem",
                    help="vLLM gpu_memory_utilization; lower if another process "
                         "already holds GPU memory.")
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    add_echo_args(ap)
    args = ap.parse_args()
    if args.echo_scale is not None and not args.echo_which:
        ap.error("--echo-scale requires --echo-which {first,second}")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_nextqa(args.num_samples)
    print(f"[data] {len(rows)} NExT-QA MC rows, {args.frames} frames/video", flush=True)

    from vllm import SamplingParams
    # SITIT doubles the frame block -> 2*frames images per prompt; cap accordingly.
    llm = make_llm(tp=args.tp, limit_images=args.frames * 2 + 1, disable_mm_cache=True,
                   gpu_mem=args.gpu_mem)
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
        print(f"=== NExT-QA · ordering {out_tag_name} ===", flush=True)
        run_order(llm, sp, out_tag_name, ORDER_LETTERS[tag], rows, args.frames,
                  args.log_every, echo_scale=args.echo_scale, echo_which=args.echo_which)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
