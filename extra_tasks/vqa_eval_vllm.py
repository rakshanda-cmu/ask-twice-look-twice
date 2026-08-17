"""
Open-ended VQAv2 (val) ordering ablation via the vLLM library.

Reuses the exact official-VQA-accuracy scoring from vqa_eval.py (normalize_answer /
official_vqa_score / score_example / load_vqa / coco_image_path) so numbers are
directly comparable to any prior run of that script. Breaks results out by
answer_type (yes/no · number · other) so the paradox can be checked per modality,
not just pooled.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/vqa_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT --num-samples 3000
"""
import argparse, json, os, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
from PIL import Image
from vqa_eval import (load_vqa, coco_image_path, score_example,
                      SHORT_ANSWER_SUFFIX)
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 1024


CHUNK_SIZE = 10000  # cap on in-memory conversations (each embeds a base64 image) --
# building all ~214k VQAv2-val conversations upfront was observed to OOM-kill the
# vLLM EngineCore process at ~93GB host RAM (verified via journalctl); chunking
# keeps at most one chunk's images resident at a time. Checkpointed between
# chunks so a later crash resumes instead of restarting from zero.


def run_order(llm, sp, tag, letters, pairs, data_dir, log_every, use_suffix=True,
             out_tag=""):
    out = os.path.join(OUT_DIR, f"vqa_order-{tag}_{MODEL_TAG}{out_tag}.json")
    results, by_type, done_qids = [], {}, set()
    n_correct, score_sum = 0, 0.0
    if os.path.exists(out):
        try:
            existing = json.load(open(out))
            m = existing["meta"]
            if m.get("complete"):
                print(f"  [{tag}] cached (acc={m.get('accuracy'):.3f})", flush=True)
                return
            results = existing["results"]
            done_qids = {r["question_id"] for r in results}
            for r in results:
                n_correct += int(r["correct"]); score_sum += r["vqa_score"]
                t = r["answer_type"]
                bt = by_type.setdefault(t, {"n": 0, "correct": 0, "score_sum": 0.0})
                bt["n"] += 1; bt["correct"] += int(r["correct"]); bt["score_sum"] += r["vqa_score"]
            print(f"  [resume] {tag}: {len(done_qids)} pairs already done", flush=True)
        except Exception:
            results, by_type, done_qids = [], {}, set()
            n_correct, score_sum = 0, 0.0

    remaining = [(q, a) for q, a in pairs if q["question_id"] not in done_qids]

    def _checkpoint(dt_so_far, complete):
        by_type_summary = {t: {"n": v["n"], "accuracy": v["correct"] / v["n"],
                               "vqa_score": v["score_sum"] / v["n"]}
                           for t, v in by_type.items()}
        meta_out = {"benchmark": "VQAv2-val", "ordering": tag, "model": MODEL_TAG,
                    "engine": "vllm", "n": len(results),
                    "accuracy": n_correct / max(1, len(results)),
                    "vqa_score": score_sum / max(1, len(results)),
                    "by_answer_type": by_type_summary, "runtime_s": dt_so_far,
                    "complete": complete}
        json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)
        return meta_out

    t0 = time.time()
    for ci in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[ci:ci + CHUNK_SIZE]
        convs, meta = [], []
        for q, a in chunk:
            fname, path = coco_image_path(data_dir, q["image_id"])
            if not os.path.isfile(path):
                continue
            pil = downscale(Image.open(path).convert("RGB"), IMG_CAP)
            task = q["question"] + (SHORT_ANSWER_SUFFIX if use_suffix else "")
            convs.append(build_conversation(letters, task, data_uri(pil)))
            gts = [ans["answer"] for ans in a["answers"]]
            meta.append({"question_id": q["question_id"], "question": q["question"],
                         "answer_type": a["answer_type"], "gt_answers": gts,
                         "gt_most_common": a["multiple_choice_answer"]})

        outputs = llm.chat(convs, sp, use_tqdm=False)
        del convs

        for o, m in zip(outputs, meta):
            text = o.outputs[0].text.strip()
            correct, vqa_score, norm, matched = score_example(text, m["gt_answers"])
            n_correct += int(correct); score_sum += vqa_score
            t = m["answer_type"]
            bt = by_type.setdefault(t, {"n": 0, "correct": 0, "score_sum": 0.0})
            bt["n"] += 1; bt["correct"] += int(correct); bt["score_sum"] += vqa_score
            results.append({**m, "model_answer_raw": text, "model_answer_norm": norm,
                            "correct": correct, "vqa_score": round(vqa_score, 4)})
            if len(results) % log_every == 0:
                print(f"    [{tag}] {len(results)}/{len(pairs)} "
                      f"acc={n_correct/len(results):.3f} score={score_sum/len(results):.3f}",
                      flush=True)
        _checkpoint(time.time() - t0, complete=False)

    dt = time.time() - t0
    meta_out = _checkpoint(dt, complete=True)
    print(f"  [{tag}] n={len(results)} acc={meta_out['accuracy']:.3f} "
          f"vqa_score={meta_out['vqa_score']:.3f} ({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--num-samples", type=int, default=3000, dest="num_samples")
    ap.add_argument("--data-dir", default="/data2/hf_cache/newtasks/vqa/val2014",
                    dest="data_dir")
    ap.add_argument("--vqa-dir", default="/data2/hf_cache/newtasks/vqa",
                    dest="vqa_dir")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem",
                    help="vLLM gpu_memory_utilization; lower if another process "
                         "already holds GPU memory.")
    ap.add_argument("--log-every", type=int, default=200, dest="log_every")
    ap.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                    help="override the default (96, or 256 with --no-suffix's "
                         "typical companion of a lower budget); paper config is 16")
    ap.add_argument("--no-suffix", action="store_true", dest="no_suffix",
                    help="drop SHORT_ANSWER_SUFFIX -- matches the paper's exact "
                         "VQAv2 config (Appendix A), which our default run added")
    ap.add_argument("--out-tag", default="", dest="out_tag",
                    help="appended to the output filename's model tag, e.g. "
                         "'_paperconfig', so a diagnostic re-run doesn't "
                         "overwrite the existing production result file")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model  # rebinds the module-global MODEL_TAG other functions read at call time
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    pairs = load_vqa(args.vqa_dir)[:args.num_samples]
    print(f"[data] {len(pairs)} VQA pairs", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model, gpu_mem=args.gpu_mem)
    # 32 was fine for Qwen; Gemma-3-27B's verbose-preamble tendency (confirmed
    # on BLINK/MMVP, see mmvp_eval_vllm.py) risks truncating before the answer
    # word ever appears, even though VQA scoring itself is containment-based
    # (not exact-match) and so is otherwise tolerant of extra text around it.
    max_tokens = args.max_tokens if args.max_tokens is not None else 96
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        print(f"=== VQA · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], pairs, args.data_dir, args.log_every,
                  use_suffix=not args.no_suffix, out_tag=args.out_tag)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
