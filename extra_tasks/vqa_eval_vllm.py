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


def run_order(llm, sp, tag, letters, pairs, data_dir, log_every):
    out = os.path.join(OUT_DIR, f"vqa_order-{tag}_{MODEL_TAG}.json")
    if os.path.exists(out):
        try:
            m = json.load(open(out))["meta"]
            if m.get("complete"):
                print(f"  [{tag}] cached (acc={m.get('accuracy'):.3f})", flush=True)
                return
        except Exception:
            pass

    convs, meta = [], []
    for q, a in pairs:
        fname, path = coco_image_path(data_dir, q["image_id"])
        if not os.path.isfile(path):
            continue
        pil = downscale(Image.open(path).convert("RGB"), IMG_CAP)
        task = q["question"] + SHORT_ANSWER_SUFFIX
        convs.append(build_conversation(letters, task, data_uri(pil)))
        gts = [ans["answer"] for ans in a["answers"]]
        meta.append({"question_id": q["question_id"], "question": q["question"],
                     "answer_type": a["answer_type"], "gt_answers": gts,
                     "gt_most_common": a["multiple_choice_answer"]})

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, by_type = [], {}
    n_correct, score_sum = 0, 0.0
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
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f} score={score_sum/len(results):.3f}",
                  flush=True)

    by_type_summary = {t: {"n": v["n"], "accuracy": v["correct"] / v["n"],
                           "vqa_score": v["score_sum"] / v["n"]}
                       for t, v in by_type.items()}
    meta_out = {"benchmark": "VQAv2-val", "ordering": tag, "model": MODEL_TAG,
                "engine": "vllm", "n": len(results),
                "accuracy": n_correct / max(1, len(results)),
                "vqa_score": score_sum / max(1, len(results)),
                "by_answer_type": by_type_summary, "runtime_s": dt, "complete": True}
    json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)
    print(f"  [{tag}] n={len(results)} acc={meta_out['accuracy']:.3f} "
          f"vqa_score={meta_out['vqa_score']:.3f} ({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b"])
    ap.add_argument("--num-samples", type=int, default=3000, dest="num_samples")
    ap.add_argument("--data-dir", default="/data2/hf_cache/newtasks/vqa/val2014",
                    dest="data_dir")
    ap.add_argument("--vqa-dir", default="/data2/hf_cache/newtasks/vqa",
                    dest="vqa_dir")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=200, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model  # rebinds the module-global MODEL_TAG other functions read at call time
    if args.model == "gemma-3-27b" and args.tp != 1:
        print("  [note] forcing --tp 1 for gemma-3-27b (bnb 4-bit, single GPU only)")
        args.tp = 1

    pairs = load_vqa(args.vqa_dir)[:args.num_samples]
    print(f"[data] {len(pairs)} VQA pairs", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model)
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        print(f"=== VQA · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], pairs, args.data_dir, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
