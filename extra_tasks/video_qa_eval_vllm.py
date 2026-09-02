"""
Open-ended video QA ordering ablation via the vLLM library -- covers MSVD-QA
(Xu et al. 2017) and TGIF-QA (Jang et al. 2017), classic short-answer video QA
benchmarks, in one harness (same test_q.json/test_a.json schema).

The cleanest video generalization of the paper's mechanism, same as NExT-QA:
"I" expands to K uniformly-sampled frames, so SITIT re-shows the whole clip.
These two add open-ended (not multiple-choice) video QA, complementing NExT-QA.

Source (HF, small): Xiaodong/MSVD_Zero_Shot_QA (1.8GB, .avi),
Xiaodong/TGIF_Zero_Shot_QA_ (1.3GB, .mp4, already extracted to
/data2/hf_cache/newtasks/{msvd_qa,tgif_qa}/).

Metric: exact-match accuracy on the normalized short answer (whole-word
containment, same convention as vqa_eval.py's official scoring, minus the
soft-accuracy weighting since these benchmarks have one canonical answer per
question, not ten human annotations).

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/video_qa_eval_vllm.py \
      --dataset msvd --orders STI,SIT,STIT,SITIT --num-samples 1000 --frames 6
"""
import argparse, glob, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import cv2
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, downscale, data_uri,
                    build_conversation, make_llm, add_echo_args, echo_tag_suffix,
                    echo_occurrence_uris)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 512
MC_SUFFIX = "\nAnswer with a single word or short phrase."

DATASET_CFG = {
    "msvd": {
        "root": "/data2/hf_cache/newtasks/msvd_qa/MSVD_Zero_Shot_QA",
        "q_file": "test_q.json", "a_file": "test_a.json",
        "video_dir": "videos", "video_ext": ".avi",
    },
    "tgif": {
        "root": "/data2/hf_cache/newtasks/tgif_qa/TGIF_Zero_Shot_QA",
        "q_file": "test_q.json", "a_file": "test_a.json",
        "video_dir": "mp4", "video_ext": ".mp4",
    },
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(s):
    # TGIF-QA's Count subtask stores answers as a JSON int, not str (confirmed
    # via a live crash: "'int' object has no attribute 'lower'" on the very
    # first TGIF row) -- MSVD-QA's answers happen to all be strings, but str()
    # here is correct for both regardless.
    return _PUNCT_RE.sub("", str(s).lower()).strip()


def load_rows(dataset, n, seed=0):
    import random
    cfg = DATASET_CFG[dataset]
    q = json.load(open(os.path.join(cfg["root"], cfg["q_file"])))
    a = json.load(open(os.path.join(cfg["root"], cfg["a_file"])))
    a_by_qid = {x["question_id"]: x["answer"] for x in a}
    rows = []
    for item in q:
        ans = a_by_qid.get(item["question_id"])
        if ans is None:
            continue
        vpath = os.path.join(cfg["root"], cfg["video_dir"],
                             item["video_name"] + cfg["video_ext"])
        if os.path.isfile(vpath):
            rows.append({"qid": item["question_id"], "video": item["video_name"],
                        "video_path": vpath, "question": item["question"],
                        "answer": ans})
    random.Random(seed).shuffle(rows)
    return rows[:n]


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


def scored_contains(pred_text, gt):
    pn, gn = normalize(pred_text), normalize(gt)
    if not gn:
        return False
    return re.search(r"(?<!\w)" + re.escape(gn) + r"(?!\w)", pn) is not None


def run_order(llm, sp, dataset, tag, letters, rows, k_frames, model_tag, log_every,
             echo_scale=None, echo_which=None):
    out = os.path.join(OUT_DIR, f"{dataset}qa_order-{tag}_{model_tag}.json")
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
        if (r["video"], r["qid"]) in done:
            continue
        frames = sample_frames(r["video_path"], k_frames)
        if not frames:
            continue
        dframes = [downscale(f, IMG_CAP) for f in frames]
        task = r["question"] + MC_SUFFIX
        if echo_scale is not None:
            occ_uris = echo_occurrence_uris(dframes, echo_scale, echo_which)
            convs.append(build_conversation(letters, task, None,
                                            per_occurrence_uris=occ_uris))
        else:
            uris = [data_uri(f) for f in dframes]
            convs.append(build_conversation(letters, task, uris))
        meta.append({"qid": r["qid"], "video": r["video"], "answer": r["answer"],
                     "question": r["question"]})

    CHUNK = 100
    t0 = time.time()
    outputs = []
    for i in range(0, len(convs), CHUNK):
        chunk_out = llm.chat(convs[i:i + CHUNK], sp, use_tqdm=False)
        outputs.extend(chunk_out)
        print(f"    [{tag}] generated {min(i+CHUNK, len(convs))}/{len(convs)}", flush=True)
        _save(out, results, outputs, meta[:len(outputs)], model_tag, dataset, tag, k_frames)
    dt = time.time() - t0

    _save(out, results, outputs, meta, model_tag, dataset, tag, k_frames, complete=True)
    n = len(results) + len(outputs)
    print(f"  [{tag}] n={n} ({dt:.0f}s) -> {out}", flush=True)


def _save(out, prior_results, outputs, meta, model_tag, dataset, tag, k_frames,
          complete=False):
    new = []
    for o, m in zip(outputs, meta):
        text = o.outputs[0].text.strip()
        ok = scored_contains(text, m["answer"])
        new.append({**m, "raw": text, "correct": ok})
    results = prior_results + new
    n_correct = sum(r["correct"] for r in results)
    meta_out = {"benchmark": dataset, "ordering": tag, "model": model_tag,
                "engine": "vllm", "n": len(results), "k_frames": k_frames,
                "accuracy": n_correct / max(1, len(results)), "complete": complete}
    json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["msvd", "tgif"])
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--num-samples", type=int, default=1000, dest="num_samples")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem",
                    help="vLLM gpu_memory_utilization; lower if another process "
                         "already holds GPU memory.")
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    add_echo_args(ap)
    args = ap.parse_args()
    if args.echo_scale is not None and not args.echo_which:
        ap.error("--echo-scale requires --echo-which {first,second}")
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_rows(args.dataset, args.num_samples)
    print(f"[data] {len(rows)} {args.dataset} rows, {args.frames} frames/video", flush=True)

    from vllm import SamplingParams
    # One engine per PROCESS, not recreated in-process per ordering: an
    # earlier in-process "del llm; gc.collect(); torch.cuda.empty_cache();
    # make_llm() again" approach (tried after a shared engine OOM'd 1400
    # chat() calls into a run -- see below) hung indefinitely reinitializing
    # the SECOND replacement engine, confirmed via a live run stuck 36+
    # minutes at the same vLLM init log line with the GPU sitting at 0%
    # util / ~350MB used. vLLM's engine core is a separate subprocess with
    # its own CUDA context; dropping the Python reference doesn't reliably
    # tear that down the way a full process exit does. So: run ONE ordering
    # per process invocation (chain `--orders X` calls in the shell, one per
    # tag) instead of looping orderings inside main() -- OS-level process
    # exit between orderings is what actually guarantees clean GPU state,
    # matching every other multi-stage chain in this session (mmvp -> blink
    # -> ... are already separate subprocess calls, not an internal loop).
    llm = make_llm(tp=args.tp, limit_images=args.frames * 2 + 1,
                   disable_mm_cache=True, model_tag=args.model, gpu_mem=args.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        if args.echo_scale is not None:
            assert tag.count("I") == 2, \
                f"--echo-scale requires a 2-image order, got {tag!r}"
            out_tag_name = echo_tag_suffix(tag, args.echo_scale, args.echo_which)
        else:
            out_tag_name = tag
        print(f"=== {args.dataset} · ordering {out_tag_name} ===", flush=True)
        run_order(llm, sp, args.dataset, out_tag_name, ORDER_LETTERS[tag], rows,
                  args.frames, args.model, args.log_every,
                  echo_scale=args.echo_scale, echo_which=args.echo_which)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
