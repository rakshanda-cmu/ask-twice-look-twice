"""
GEPA (Reflective Prompt Evolution, arxiv.org/abs/2507.19457,
github.com/gepa-ai/gepa) as an automated prompt-optimization baseline.

Evolves this repo's SYSTEM_MESSAGE text against a held-out TRAIN split of one
benchmark, using this repo's own local Qwen3-VL-8B (via vLLM) as BOTH the
task model (answering the benchmark's questions under a fixed "STI" ordering
-- GEPA optimizes the prompt TEXT, not the S/I/T ordering this repo's other
scripts study) and the reflection model (proposing improved prompt text from
text-only feedback about failures). One in-process vLLM engine serves both
roles -- no API keys, no external LiteLLM providers, per the design confirmed
against gepa's actual source (gepa.optimize_anything's `evaluator` and
`reflection_lm` are plain Python callables; see gepa/optimize_anything.py's
Evaluator/LanguageModel Protocols).

Per the task's explicit instruction, this does NOT run on RF20 -- only on the
other benchmarks (POPE first; more are added the same way, see
DATASET_ADAPTERS).

Data-split caveat: every benchmark in this repo already has a full-scale
STI/SIT/STIT/SITIT baseline covering its ENTIRE pool (e.g. POPE's 9000
samples) -- there is no untouched data left to use as GEPA's train set
without touching that pool. We carve a small, seeded, stratified TRAIN (and
VAL) subset out of the same pool GEPA optimizes against, then evaluate the
final optimized prompt on the REMAINING pool (pool minus train minus val) --
still a large N, but not identical to the full production baseline's N. This
is noted explicitly in the output meta so the comparison isn't silently
over-claimed as apples-to-apples.

Tracks, per dataset (see gepa/PROMPTS.md's item 4/5 framing):
  - training cost: wall-clock seconds, GEPAResult.total_metric_calls, total
    prompt+completion tokens for BOTH task_lm and reflection_lm calls made
    during optimization (tracked via our own callables -- GEPA itself does
    not meter tokens for callable LMs, only for LiteLLM string models)
  - inference cost: extra tokens the optimized prompt costs at inference
    time vs the baseline SYSTEM_MESSAGE (one representative call, same
    method as token_cost_analysis.py)
  - accuracy: optimized prompt evaluated on the held-out eval subset,
    reported alongside the baseline SYSTEM_MESSAGE's accuracy on the SAME
    eval subset (an apples-to-apples GEPA-vs-baseline number, even though
    the N differs from the full production baseline)

Output: ./gepa_results/<dataset>__<model>__gepa.json (new dir; does not
touch any existing STI/SIT/STIT/SITIT result file).

vLLM evaluator calls must run single-threaded (GEPA's default parallel
evaluator would call our evaluator from multiple threads onto one
synchronous vLLM engine, which is not a supported access pattern) --
EngineConfig(parallel=False) below is required, not optional.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/grg/anaconda3/envs/soft-prompt/bin/python \
    gepa_baseline.py --dataset pope --train-size 60 --val-size 40 \
    --max-metric-calls 150
"""
import argparse
import json
import os
import random
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extra_tasks"))

from constants import SYSTEM_MESSAGE  # noqa: E402
from common import downscale, data_uri, build_conversation, make_llm  # noqa: E402

OUT_DIR = os.path.join(HERE, "gepa_results")
EVAL_ORDER = "STI"  # fixed; GEPA optimizes the prompt TEXT, not ordering


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset adapters: {loader, question-builder, scorer} per benchmark.
#  Add a new dataset by adding one entry here -- everything else (GEPA
#  wiring, token/cost accounting, train/val/eval split) is dataset-agnostic.
# ═══════════════════════════════════════════════════════════════════════════

def _get_image(ex):
    """Adapters store either a decoded PIL under 'image' (small datasets like
    POPE, whose HF dataset already embeds images inline -- eager decode is
    fine at 9000 rows) or a path under 'image_path' (large datasets like VQA,
    whose 200K+-row pool would waste huge memory if eagerly decoded when only
    a few hundred/thousand rows actually get sampled into train/val/eval)."""
    if "image" in ex:
        return ex["image"]
    return Image.open(ex["image_path"]).convert("RGB")


def _pope_loader():
    from pope_eval import load_pope_samples
    rows = load_pope_samples()  # full 9000, category-stratified already
    return [{"id": r["question_id"], "image": r["image"].convert("RGB"),
             "question": r["question"], "gt": r["answer"]} for r in rows]


def _pope_task_text(ex):
    from naturalbench_eval import YESNO_SUFFIX
    return ex["question"] + YESNO_SUFFIX


def _pope_score(raw, ex):
    from naturalbench_eval import _first_yes_no
    from pope_eval import _pred_yes_no
    pred = _pred_yes_no(raw)
    correct = pred == ex["gt"]
    return correct, 1.0 if correct else 0.0, pred


def _vqa_loader():
    from vqa_eval import load_vqa, coco_image_path
    vqa_dir = "/data2/hf_cache/newtasks/vqa"
    data_dir = "/data2/hf_cache/newtasks/vqa/val2014"
    pairs = load_vqa(vqa_dir)
    rows = []
    for q, a in pairs:
        _, path = coco_image_path(data_dir, q["image_id"])
        if not os.path.isfile(path):
            continue
        rows.append({"id": q["question_id"], "image_path": path,
                     "question": q["question"],
                     "gt_answers": [x["answer"] for x in a["answers"]]})
    return rows


def _vqa_task_text(ex):
    from vqa_eval import SHORT_ANSWER_SUFFIX
    return ex["question"] + SHORT_ANSWER_SUFFIX


def _vqa_score(raw, ex):
    from vqa_eval import score_example
    correct, vqa_score, norm, _ = score_example(raw, ex["gt_answers"])
    return correct, vqa_score, norm


DATASET_ADAPTERS = {
    "pope": {"loader": _pope_loader, "task_text": _pope_task_text,
            "score": _pope_score, "img_cap": 1024},
    "vqa": {"loader": _vqa_loader, "task_text": _vqa_task_text,
           "score": _vqa_score, "img_cap": 1024},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Token-metered vLLM callables shared by task_lm and reflection_lm
# ═══════════════════════════════════════════════════════════════════════════

class MeteredEngine:
    """Wraps one vLLM engine; every .chat() call is metered (prompt + output
    token counts) into `stats`, split by role ('task' vs 'reflect') so
    training-time token cost can be reported separately from inference cost."""

    def __init__(self, llm, task_sp, reflect_sp):
        self.llm = llm
        self.task_sp = task_sp
        self.reflect_sp = reflect_sp
        self.stats = {"task_calls": 0, "task_tokens_in": 0, "task_tokens_out": 0,
                      "reflect_calls": 0, "reflect_tokens_in": 0, "reflect_tokens_out": 0}

    def _meter(self, out, role):
        self.stats[f"{role}_calls"] += 1
        self.stats[f"{role}_tokens_in"] += len(out.prompt_token_ids)
        self.stats[f"{role}_tokens_out"] += len(out.outputs[0].token_ids)

    def run_task(self, conv):
        out = self.llm.chat([conv], self.task_sp, use_tqdm=False)[0]
        self._meter(out, "task")
        return out.outputs[0].text.strip()

    def run_reflection(self, prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt  # already role/content dicts (GEPA-built)
        out = self.llm.chat([messages], self.reflect_sp, use_tqdm=False)[0]
        self._meter(out, "reflect")
        return out.outputs[0].text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  GEPA wiring
# ═══════════════════════════════════════════════════════════════════════════

def make_evaluator(engine, adapter, img_cap):
    def evaluator(candidate, example):
        pil = downscale(_get_image(example), img_cap)
        task = adapter["task_text"](example)
        conv = build_conversation(EVAL_ORDER, task, data_uri(pil),
                                  system_text=candidate["prompt"])
        raw = engine.run_task(conv)
        correct, score, pred = adapter["score"](raw, example)
        # ground truth field name varies by adapter -- POPE stores a single
        # 'gt' string, VQA stores 'gt_answers' (10 human annotations); fall
        # back to whichever is present rather than hardcoding one.
        gt_display = example.get("gt", example.get("gt_answers"))
        side_info = {
            "Question": task,
            "GroundTruth": str(gt_display),
            "ModelPrediction": str(pred),
            "ModelRawOutput": raw,
            "Correct": correct,
        }
        return score, side_info
    return evaluator


def run_gepa(dataset, model_tag, train_size, val_size, max_metric_calls,
            seed, out_dir, tp, gpu_mem, eval_size):
    import gepa
    import gepa.optimize_anything as oa

    adapter = DATASET_ADAPTERS[dataset]
    print(f"[data] loading {dataset} …", flush=True)
    pool = adapter["loader"]()
    print(f"[data] {len(pool)} total samples", flush=True)

    rng = random.Random(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    eval_idx = idx[train_size + val_size:train_size + val_size + eval_size]
    trainset = [pool[i] for i in train_idx]
    valset = [pool[i] for i in val_idx]
    evalset = [pool[i] for i in eval_idx]
    print(f"[split] train={len(trainset)} val={len(valset)} "
          f"held-out-eval={len(evalset)} (seeded, distinct index sets; pool={len(pool)})",
          flush=True)

    print(f"[load] vLLM model = {model_tag}", flush=True)
    from vllm import SamplingParams
    llm = make_llm(tp=tp, model_tag=model_tag, gpu_mem=gpu_mem)
    task_sp = SamplingParams(temperature=0.0, max_tokens=16)
    reflect_sp = SamplingParams(temperature=0.7, max_tokens=1024)
    engine = MeteredEngine(llm, task_sp, reflect_sp)

    evaluator = make_evaluator(engine, adapter, adapter["img_cap"])

    config = oa.GEPAConfig(
        engine=oa.EngineConfig(max_metric_calls=max_metric_calls, parallel=False,
                               display_progress_bar=True, seed=seed),
        reflection=oa.ReflectionConfig(reflection_lm=engine.run_reflection,
                                       reflection_minibatch_size=3),
    )

    print(f"[gepa] optimizing (max_metric_calls={max_metric_calls}) …", flush=True)
    t0 = time.time()
    result = oa.optimize_anything(
        seed_candidate={"prompt": SYSTEM_MESSAGE},
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=f"Maximize {dataset} yes/no question-answering accuracy for a "
                  f"vision-language model, given an image and a question.",
        background="The 'prompt' parameter is a SYSTEM prompt shown to the model "
                   "before the image and question. It should elicit accurate, "
                   "well-calibrated yes/no answers.",
        config=config,
    )
    train_wall_s = time.time() - t0

    best = result.best_candidate
    print(f"[gepa] done in {train_wall_s:.0f}s, {result.total_metric_calls} metric "
          f"calls, best val score = {result.val_aggregate_scores[result.best_idx]:.3f}",
          flush=True)
    print(f"[gepa] best prompt:\n{best['prompt']}\n", flush=True)

    # ── held-out eval: baseline SYSTEM_MESSAGE vs GEPA-optimized prompt ──────
    # Batched through vLLM (unlike the one-at-a-time task_lm calls GEPA's
    # evaluator makes during optimization, which must stay serial -- see
    # MeteredEngine/EngineConfig(parallel=False) above) -- this eval isn't
    # part of the metered optimization loop, so there's no reason not to use
    # vLLM's normal batched .chat(), matching every other eval script in
    # this repo (vqa_eval_vllm.py etc.) instead of looping one item at a time.
    # CHUNK_SIZE: cap on in-memory conversations (each embeds a base64 image)
    # before one llm.chat() call -- same host-RAM OOM risk documented in
    # vqa_eval_vllm.py/tallyqa_eval_vllm.py (confirmed via journalctl:
    # gemma-3-27b's full-scale VQA run OOM-killed at ~93GB anon-RSS building
    # ~214K conversations upfront). --eval-size can be scaled up to a
    # dataset's full remaining pool (tens/hundreds of thousands for VQA), so
    # this eval needs the same chunking, not just the small-N smoke-test path
    # that worked fine at eval-size<=1000.
    EVAL_CHUNK_SIZE = 5000

    def _eval_prompt(prompt_text, tag):
        n_correct = 0
        t0 = time.time()
        for ci in range(0, len(evalset), EVAL_CHUNK_SIZE):
            chunk = evalset[ci:ci + EVAL_CHUNK_SIZE]
            convs = []
            for ex in chunk:
                pil = downscale(_get_image(ex), adapter["img_cap"])
                task = adapter["task_text"](ex)
                convs.append(build_conversation(EVAL_ORDER, task, data_uri(pil),
                                                system_text=prompt_text))
            outputs = engine.llm.chat(convs, engine.task_sp, use_tqdm=False)
            del convs
            for o, ex in zip(outputs, chunk):
                raw = o.outputs[0].text.strip()
                correct, _, _ = adapter["score"](raw, ex)
                n_correct += int(correct)
            print(f"    [{tag}] {min(ci + EVAL_CHUNK_SIZE, len(evalset))}/{len(evalset)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        acc = n_correct / max(1, len(evalset))
        print(f"[eval] {tag}: acc={acc:.3f} (n={len(evalset)})", flush=True)
        return acc

    print(f"[eval] scoring baseline SYSTEM_MESSAGE and GEPA prompt on the "
          f"{len(evalset)}-sample held-out eval subset …", flush=True)
    baseline_stats_before = dict(engine.stats)
    baseline_acc = _eval_prompt(SYSTEM_MESSAGE, "baseline")
    gepa_acc = _eval_prompt(best["prompt"], "gepa-optimized")

    # ── inference-time token overhead: optimized prompt vs baseline ─────────
    from transformers import AutoProcessor, AutoConfig
    from common import MODEL_REGISTRY
    hf_id = MODEL_REGISTRY[model_tag]["hf"]
    processor = AutoProcessor.from_pretrained(hf_id)
    hf_config = AutoConfig.from_pretrained(hf_id)
    sample_img = _get_image(evalset[0] if evalset else pool[0])
    sample_task = adapter["task_text"](evalset[0] if evalset else pool[0])

    def _count_tokens(prompt_text):
        from model_manager import _build_qwen_messages
        from qwen_vl_utils import process_vision_info
        messages = _build_qwen_messages(prompt_text, sample_task,
                                        downscale(sample_img, adapter["img_cap"]),
                                        EVAL_ORDER)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           videos=video_inputs if video_inputs else None,
                           padding=True, return_tensors="pt")
        return int(inputs.input_ids.shape[1])

    baseline_tokens = _count_tokens(SYSTEM_MESSAGE)
    gepa_tokens = _count_tokens(best["prompt"])

    out = {
        "dataset": dataset, "model": model_tag, "eval_order": EVAL_ORDER,
        "seed": seed, "split": {"train": len(trainset), "val": len(valset),
                                "held_out_eval": len(evalset)},
        "training": {
            "wall_clock_s": round(train_wall_s, 1),
            "total_metric_calls": result.total_metric_calls,
            "task_calls": engine.stats["task_calls"],
            "task_tokens_in": engine.stats["task_tokens_in"],
            "task_tokens_out": engine.stats["task_tokens_out"],
            "reflect_calls": engine.stats["reflect_calls"],
            "reflect_tokens_in": engine.stats["reflect_tokens_in"],
            "reflect_tokens_out": engine.stats["reflect_tokens_out"],
            "total_tokens": (engine.stats["task_tokens_in"] + engine.stats["task_tokens_out"]
                            + engine.stats["reflect_tokens_in"] + engine.stats["reflect_tokens_out"]),
            "best_val_score": result.val_aggregate_scores[result.best_idx],
        },
        "inference_cost": {
            "baseline_prompt_tokens": baseline_tokens,
            "gepa_prompt_tokens": gepa_tokens,
            "delta_tokens": gepa_tokens - baseline_tokens,
        },
        "accuracy": {
            "baseline_system_message": {"acc": baseline_acc, "n": len(evalset)},
            "gepa_optimized": {"acc": gepa_acc, "n": len(evalset)},
            "delta": gepa_acc - baseline_acc,
        },
        "baseline_prompt": SYSTEM_MESSAGE,
        "gepa_optimized_prompt": best["prompt"],
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}__{model_tag}__gepa.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[saved] {out_path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_ADAPTERS))
    ap.add_argument("--model", default="qwen3-vl-8b", dest="model_tag")
    ap.add_argument("--train-size", type=int, default=60, dest="train_size")
    ap.add_argument("--val-size", type=int, default=40, dest="val_size")
    ap.add_argument("--max-metric-calls", type=int, default=150, dest="max_metric_calls")
    ap.add_argument("--eval-size", type=int, default=500, dest="eval_size",
                    help="held-out eval subset size (capped, not 'everything left "
                         "in the pool' -- the full pool is already used by the "
                         "production STI/SIT/STIT/SITIT baseline for this dataset).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT_DIR, dest="out_dir")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.8, dest="gpu_mem")
    args = ap.parse_args()

    run_gepa(args.dataset, args.model_tag, args.train_size, args.val_size,
             args.max_metric_calls, args.seed, args.out_dir, args.tp, args.gpu_mem,
             args.eval_size)


if __name__ == "__main__":
    main()
