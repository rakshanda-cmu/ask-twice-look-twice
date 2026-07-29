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
import argparse, glob, json, os, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import cv2
import pandas as pd
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

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


def parse_choice(text):
    t = text.strip()
    for c in t:
        if c in "01234":
            return int(c)
    return None


def run_order(llm, sp, tag, letters, rows, k_frames, log_every):
    out = os.path.join(OUT_DIR, f"nextqa_order-{tag}_{MODEL_TAG}.json")
    if os.path.exists(out):
        try:
            m = json.load(open(out))["meta"]
            if m.get("complete"):
                print(f"  [{tag}] cached (acc={m.get('accuracy'):.3f})", flush=True)
                return
        except Exception:
            pass

    convs, meta = [], []
    for r in rows:
        vpath = os.path.join(VIDEO_DIR, f"{r['video']}.mp4")
        frames = sample_frames(vpath, k_frames)
        if not frames:
            continue
        uris = [data_uri(downscale(f, IMG_CAP)) for f in frames]
        opts = "\n".join(f"{i}: {r[f'a{i}']}" for i in range(5))
        task = f"{r['question']}\nOptions:\n{opts}{MC_SUFFIX}"
        convs.append(build_conversation(letters, task, uris))
        meta.append({"qid": int(r["qid"]), "video": str(r["video"]),
                     "type": r["type"], "answer": int(r["answer"]),
                     "question": r["question"]})

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, by_type, n_correct = [], {}, 0
    for o, m in zip(outputs, meta):
        text = o.outputs[0].text.strip()
        pred = parse_choice(text)
        correct = pred == m["answer"]
        n_correct += int(correct)
        bt = by_type.setdefault(m["type"], {"n": 0, "correct": 0})
        bt["n"] += 1; bt["correct"] += int(correct)
        results.append({**m, "pred": pred, "raw": text, "correct": correct})
        if len(results) % log_every == 0:
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f}", flush=True)

    by_type_summary = {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                       for k, v in by_type.items()}
    meta_out = {"benchmark": "NExT-QA", "ordering": tag, "model": MODEL_TAG,
                "engine": "vllm", "n": len(results), "k_frames": k_frames,
                "accuracy": n_correct / max(1, len(results)),
                "by_type": by_type_summary, "runtime_s": dt, "complete": True}
    json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)
    print(f"  [{tag}] n={len(results)} acc={meta_out['accuracy']:.3f} "
          f"({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--num-samples", type=int, default=1500, dest="num_samples")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_nextqa(args.num_samples)
    print(f"[data] {len(rows)} NExT-QA MC rows, {args.frames} frames/video", flush=True)

    from vllm import SamplingParams
    # SITIT doubles the frame block -> 2*frames images per prompt; cap accordingly.
    llm = make_llm(tp=args.tp, limit_images=args.frames * 2 + 1)
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        print(f"=== NExT-QA · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.frames, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
