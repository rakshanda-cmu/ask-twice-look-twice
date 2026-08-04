"""
MVBench (Li et al. 2024, InternVideo/OpenGVLab) multi-frame video QA ordering
ablation via the vLLM library -- 20 diverse temporal-reasoning tasks (action
sequence/prediction/localization, object existence/interaction/shuffle, moving
count/direction/attribute, scene transition, counterfactual inference, ...),
multiple choice, 200 examples/task.

Scoped to the subset of tasks whose source video archives were downloaded
(action_antonym, action_localization, action_prediction, action_sequence,
counterfactual_inference, egocentric_navigation, fine_grained_action,
moving_attribute, moving_count, moving_direction, object_existence,
object_interaction, scene_transition -- confirmed >=94% video coverage on a
50-example sample per task; excluded: action_count, character_order,
episodic_reasoning, fine_grained_pose, object_shuffle, state_change,
unexpected_action, which need video archives not downloaded here (perception.zip,
NTU external download, etc.) -- documented, not silently dropped).

Source (HF): OpenGVLab/MVBench, json/*.json (all tasks) + selected video/*.zip.
Metric: multiple-choice exact match (candidate index).

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/mvbench_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT --num-samples-per-task 30 --frames 6
"""
import argparse, glob, json, os, random, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import cv2
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
VIDEO_ROOT = "/data2/hf_cache/newtasks/mvbench/videos"
IMG_CAP = 512
MC_SUFFIX = "\nAnswer with the option number only (0-indexed)."

HF_HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

# Tasks confirmed >=94% video coverage against the downloaded archives (see
# module docstring for what was excluded and why).
TASKS = [
    "action_antonym", "action_localization", "action_prediction",
    "action_sequence", "counterfactual_inference", "egocentric_navigation",
    "fine_grained_action", "moving_attribute", "moving_count",
    "moving_direction", "object_existence", "object_interaction",
    "scene_transition",
]


def _find_json_dir():
    hits = glob.glob(os.path.join(HF_HUB, "datasets--OpenGVLab--MVBench",
                                  "snapshots", "*", "json"))
    if not hits:
        raise FileNotFoundError("MVBench json/ not found; run the download script first.")
    return hits[0]


def _find_video(name):
    """name may or may not have an extension; search the extracted video tree."""
    base = os.path.basename(name)
    candidates = [base] if "." in base else [base + e for e in
                 (".mp4", ".avi", ".mov", ".webm", ".gif")]
    for root, _, files in os.walk(VIDEO_ROOT):
        fileset = set(files)
        for c in candidates:
            if c in fileset:
                return os.path.join(root, c)
    return None


def load_rows(n_per_task, seed=0):
    jdir = _find_json_dir()
    rows = []
    skipped_by_task = {}
    for task in TASKS:
        data = json.load(open(os.path.join(jdir, f"{task}.json")))
        random.Random(seed).shuffle(data)
        kept = 0
        for d in data:
            if kept >= n_per_task:
                break
            vpath = _find_video(d["video"])
            if vpath is None:
                skipped_by_task[task] = skipped_by_task.get(task, 0) + 1
                continue
            rows.append({"task": task, "video_path": vpath, "question": d["question"],
                        "candidates": d["candidates"], "answer": d["answer"],
                        "start": d.get("start"), "end": d.get("end")})
            kept += 1
        if skipped_by_task.get(task):
            print(f"  [data] {task}: skipped {skipped_by_task[task]} "
                  f"(video not found)", flush=True)
    return rows


def sample_frames(video_path, k, start=None, end=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    lo = int(start * fps) if start is not None else 0
    hi = int(end * fps) if end is not None else total - 1
    lo, hi = max(0, lo), min(total - 1, max(hi, lo))
    idxs = sorted(set(int(lo + i * (hi - lo) / max(1, k - 1)) for i in range(k)))
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


ISOLATED_NUM = re.compile(r"(?<!\d)(\d+)(?!\d)")


def parse_choice(text, n_choices):
    # Isolated digits only (rejects e.g. the "24" in "24-hour"); take the LAST
    # match, not the first -- a verbose model's reasoning preamble can mention
    # unrelated numbers before converging on its actual final pick.
    matches = ISOLATED_NUM.findall(text.strip())
    for v in reversed(matches):
        v = int(v)
        if 0 <= v < n_choices:
            return v
    return None


def run_order(llm, sp, tag, letters, rows, k_frames, model_tag, log_every):
    out = os.path.join(OUT_DIR, f"mvbench_order-{tag}_{model_tag}.json")
    results, done = [], set()
    if os.path.exists(out):
        try:
            d = json.load(open(out))
            if d["meta"].get("complete"):
                print(f"  [{tag}] cached (acc={d['meta'].get('accuracy'):.3f})", flush=True)
                return
            results = d.get("results", [])
            done = {(r["task"], r["video_path"], r["question"]) for r in results}
            print(f"  [resume] {len(done)} already done", flush=True)
        except Exception:
            results, done = [], set()

    convs, meta = [], []
    for r in rows:
        key = (r["task"], r["video_path"], r["question"])
        if key in done:
            continue
        frames = sample_frames(r["video_path"], k_frames, r["start"], r["end"])
        if not frames:
            continue
        uris = [data_uri(downscale(f, IMG_CAP)) for f in frames]
        opts = "\n".join(f"{i}: {c}" for i, c in enumerate(r["candidates"]))
        task_text = f"{r['question']}\nOptions:\n{opts}{MC_SUFFIX}"
        convs.append(build_conversation(letters, task_text, uris))
        try:
            gt_idx = r["candidates"].index(r["answer"])
        except ValueError:
            gt_idx = None
        meta.append({"task": r["task"], "video_path": r["video_path"],
                     "question": r["question"], "n_choices": len(r["candidates"]),
                     "answer_idx": gt_idx})

    CHUNK = 100
    t0 = time.time()
    outputs = []
    for i in range(0, len(convs), CHUNK):
        chunk_out = llm.chat(convs[i:i + CHUNK], sp, use_tqdm=False)
        outputs.extend(chunk_out)
        print(f"    [{tag}] generated {min(i+CHUNK, len(convs))}/{len(convs)}", flush=True)
        _save(out, results, outputs, meta[:len(outputs)], model_tag, tag, k_frames)
    dt = time.time() - t0

    _save(out, results, outputs, meta, model_tag, tag, k_frames, complete=True)
    n = len(results) + len(outputs)
    print(f"  [{tag}] n={n} ({dt:.0f}s) -> {out}", flush=True)


def _save(out, prior_results, outputs, meta, model_tag, tag, k_frames, complete=False):
    new = []
    for o, m in zip(outputs, meta):
        text = o.outputs[0].text.strip()
        pred = parse_choice(text, m["n_choices"])
        ok = (m["answer_idx"] is not None and pred == m["answer_idx"])
        new.append({**m, "pred": pred, "raw": text, "correct": ok})
    results = prior_results + new
    n_correct = sum(r["correct"] for r in results)
    by_task = {}
    for r in results:
        bt = by_task.setdefault(r["task"], {"n": 0, "correct": 0})
        bt["n"] += 1; bt["correct"] += int(r["correct"])
    by_task_summary = {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                       for k, v in by_task.items()}
    meta_out = {"benchmark": "MVBench", "ordering": tag, "model": model_tag,
                "engine": "vllm", "n": len(results), "k_frames": k_frames,
                "accuracy": n_correct / max(1, len(results)),
                "by_task": by_task_summary, "complete": complete}
    json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--num-samples-per-task", type=int, default=30,
                    dest="num_samples_per_task")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_rows(args.num_samples_per_task)
    print(f"[data] {len(rows)} MVBench rows across {len(TASKS)} tasks, "
          f"{args.frames} frames/video", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, limit_images=args.frames * 2 + 1,
                   disable_mm_cache=True, model_tag=args.model)
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
        print(f"=== MVBench · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.frames,
                  args.model, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
