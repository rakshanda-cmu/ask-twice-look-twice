"""
Run a VLM over VQA v2 (val) questions and record, per image/question, whether
the model answered correctly — and if not, the dataset's expected answer vs. the
model's predicted answer.

Pipeline
--------
1. Download metadata first:   python download_vqa.py
2. Run evaluation:            python vqa_eval.py --num-samples 5000 --model qwen3-vl-8b

Output
------
A JSON file (default: ./vqa/vqa_analysis_<model>.json) that the Streamlit app
(logit_lens_app.py) reads to browse right/wrong examples. Schema:

{
  "meta": { "model": ..., "num_samples": ..., "num_correct": ..., "accuracy": ...,
            "data_dir": ..., "vqa_dir": ..., "order": ..., "system_prompt": ... },
  "results": [
    {
      "question_id": int,
      "image_id": int,
      "image_file": "COCO_val2014_000000XXXXXX.jpg",
      "question": str,
      "question_type": str,
      "answer_type": str,                # "yes/no" | "number" | "other"
      "gt_answers": [str, ...],          # 10 raw human answers
      "gt_most_common": str,             # most-agreed (multiple_choice_answer)
      "model_answer_raw": str,           # model's full reply
      "model_answer_norm": str,          # normalized
      "matched_gt": [str, ...],          # GT answers contained in the reply
      "correct": bool,                   # any GT answer contained in the reply
      "vqa_score": float                 # official VQA accuracy (0..1)
    },
    ...
  ]
}
"""

import argparse
import json
import os
import re
import sys
import time

# ──────────────────────────────────────────────────────────────────────────────
#  Official VQA answer normalization
#  Ported from the VQA eval toolkit (vqaEval.py), trimmed to what we need.
#  https://github.com/GT-Vision-Lab/VQA
# ──────────────────────────────────────────────────────────────────────────────

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id": "I'd", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itll": "it'll", "lets": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "shant": "shan't", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodyll": "somebody'll", "someonell": "someone'll",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "wed": "we'd", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(text):
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(text):
    out = []
    for word in text.lower().split():
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            out.append(word)
    for i, word in enumerate(out):
        if word in _CONTRACTIONS:
            out[i] = _CONTRACTIONS[word]
    return " ".join(out)


def normalize_answer(ans):
    """Apply the official VQA normalization to a single answer string."""
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    ans = _process_punctuation(ans)
    ans = _process_digit_article(ans)
    return ans


# ──────────────────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────────────────

def official_vqa_score(pred_norm, gt_norm_list):
    """
    Official VQA accuracy for a single (exact) predicted answer:
        mean over the 10 leave-one-out subsets of min(#matches / 3, 1).
    """
    accs = []
    for i in range(len(gt_norm_list)):
        others = gt_norm_list[:i] + gt_norm_list[i + 1:]
        matching = sum(1 for g in others if g == pred_norm)
        accs.append(min(matching / 3.0, 1.0))
    return sum(accs) / len(accs) if accs else 0.0


def score_example(model_answer_raw, gt_answers):
    """
    VQA-style *contains* judging.

    Returns (correct, vqa_score, model_answer_norm, matched_gt).
      - correct: True if any unique GT answer string appears in the normalized
        model reply.
      - vqa_score: official VQA accuracy using the best-agreed contained GT
        answer as the prediction (0 if none contained).
    """
    model_norm = normalize_answer(model_answer_raw)
    gt_norm = [normalize_answer(a) for a in gt_answers]

    # token-boundary containment so "no" doesn't match "nobody"/"snow"
    def contained(g):
        if not g:
            return False
        return re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", model_norm) is not None

    # unique GT answers ranked by agreement (count among the 10)
    counts = {}
    for g in gt_norm:
        counts[g] = counts.get(g, 0) + 1
    ranked = sorted(counts.keys(), key=lambda g: -counts[g])

    matched = [g for g in ranked if contained(g)]
    correct = len(matched) > 0

    # prediction for the official score: best-agreed contained answer, else the
    # raw normalized reply (which generally won't match -> score 0)
    pred = matched[0] if matched else model_norm
    vqa_score = official_vqa_score(pred, gt_norm)

    return correct, vqa_score, model_norm, matched


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_vqa(vqa_dir):
    q_path = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_val2014_questions.json")
    a_path = os.path.join(vqa_dir, "v2_mscoco_val2014_annotations.json")
    for p in (q_path, a_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run `python download_vqa.py --out {vqa_dir}` first."
            )
    with open(q_path) as f:
        questions = json.load(f)["questions"]
    with open(a_path) as f:
        annotations = json.load(f)["annotations"]

    ann_by_qid = {a["question_id"]: a for a in annotations}
    # keep the dataset's natural order; pair each question with its annotation
    paired = []
    for q in questions:
        a = ann_by_qid.get(q["question_id"])
        if a is not None:
            paired.append((q, a))
    return paired


def coco_image_path(data_dir, image_id):
    fname = f"COCO_val2014_{int(image_id):012d}.jpg"
    return fname, os.path.join(data_dir, fname)


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

# Standard short-answer instruction used by LLaVA/Qwen on VQA.
SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def run_eval(args):
    from PIL import Image
    import torch
    from transformers.utils import logging as hf_logging
    from core.model_manager import ModelManager, QWEN_MODELS
    from core.utils import setup_seeds, disable_torch_init

    # Silence per-call transformers warnings (e.g. the benign
    # "Kwargs passed to `processor.__call__` …" emitted once per question by
    # Qwen's apply_chat_template) so the progress log stays readable. Errors
    # still surface.
    hf_logging.set_verbosity_error()

    setup_seeds()
    disable_torch_init()

    print(f"[load] model = {args.model}")
    mm = ModelManager(args.model)

    default_order = "IT" if args.model in QWEN_MODELS else "SIT"
    order = args.order or default_order
    system_prompt = args.system_prompt or ""

    paired = load_vqa(args.vqa_dir)
    print(f"[data] {len(paired)} val questions available; "
          f"evaluating first {args.num_samples}.")
    paired = paired[args.start: args.start + args.num_samples]

    results = []
    num_correct = 0
    score_sum = 0.0
    t0 = time.time()

    for i, (q, a) in enumerate(paired):
        image_id = q["image_id"]
        question = q["question"]
        fname, img_path = coco_image_path(args.data_dir, image_id)

        if not os.path.exists(img_path):
            print(f"  [warn] missing image {img_path}, skipping qid={q['question_id']}")
            continue

        gt_answers = [ans["answer"] for ans in a["answers"]]
        gt_most_common = a.get("multiple_choice_answer", "")

        try:
            img = Image.open(img_path).convert("RGB")
            prompt = question + SHORT_ANSWER_SUFFIX
            _, input_ids, kwargs = mm.prepare_inputs_from_pil(
                [prompt], img, system_prompt=system_prompt, order=order,
            )
            with torch.inference_mode():
                out = mm.llm_model.generate(
                    input_ids,
                    do_sample=False, num_beams=1,
                    max_new_tokens=args.max_tokens, use_cache=True,
                    **kwargs,
                )
            # Qwen-VL puts the (expanded) prompt — including vision tokens — inside
            # input_ids, so `out` is [prompt | generated]; slice off the prompt to
            # keep only the answer. LLaVA's image token expands internally (input_ids
            # length != prompt length in `out`), so decode fully and split the turn.
            if args.model in QWEN_MODELS:
                gen_ids = out[:, input_ids.shape[1]:]
                model_answer = mm.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )[0].strip()
            else:
                model_answer = mm.tokenizer.batch_decode(
                    out, skip_special_tokens=True
                )[0].strip()
                if "ASSISTANT:" in model_answer:
                    model_answer = model_answer.split("ASSISTANT:")[-1].strip()
        except Exception as e:
            print(f"  [error] qid={q['question_id']}: {e}")
            continue

        correct, vqa_score, model_norm, matched = score_example(model_answer, gt_answers)
        num_correct += int(correct)
        score_sum += vqa_score

        results.append({
            "question_id": q["question_id"],
            "image_id": image_id,
            "image_file": fname,
            "question": question,
            "question_type": a.get("question_type", ""),
            "answer_type": a.get("answer_type", ""),
            "gt_answers": gt_answers,
            "gt_most_common": gt_most_common,
            "model_answer_raw": model_answer,
            "model_answer_norm": model_norm,
            "matched_gt": matched,
            "correct": correct,
            "vqa_score": round(vqa_score, 4),
        })

        if (i + 1) % args.log_every == 0 or (i + 1) == len(paired):
            n = len(results)
            rate = (i + 1) / max(1e-9, time.time() - t0)
            print(f"  [{i+1}/{len(paired)}] "
                  f"acc(contains)={num_correct/max(1,n):.3f}  "
                  f"vqa_score={score_sum/max(1,n):.3f}  "
                  f"({rate:.2f} q/s)")

    n = len(results)
    meta = {
        "model": args.model,
        "order": order,
        "system_prompt": system_prompt,
        "num_samples": n,
        "num_correct": num_correct,
        "accuracy_contains": round(num_correct / max(1, n), 4),
        "vqa_accuracy": round(score_sum / max(1, n), 4),
        "data_dir": os.path.abspath(args.data_dir),
        "vqa_dir": os.path.abspath(args.vqa_dir),
        "max_tokens": args.max_tokens,
    }

    out_path = args.out or os.path.join(
        args.vqa_dir, f"vqa_analysis_{args.model}.json"
    )
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(meta, indent=2))
    print(f"\n[saved] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Evaluate a VLM on VQA v2 (val).")
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "qwen2.5-vl-7b", "llava-1.5"])
    ap.add_argument("--num-samples", type=int, default=5000,
                    dest="num_samples", help="How many questions to evaluate.")
    ap.add_argument("--start", type=int, default=0,
                    help="Offset into the val set (for resuming/sharding).")
    ap.add_argument("--data-dir", default="./COCO/val2014",
                    dest="data_dir", help="COCO val2014 image directory.")
    ap.add_argument("--vqa-dir", default="./vqa", dest="vqa_dir",
                    help="Dir with VQA val Questions/Annotations JSONs.")
    ap.add_argument("--order", default=None,
                    help="Section order (default: IT for Qwen, SIT for LLaVA).")
    ap.add_argument("--system-prompt", default=None, dest="system_prompt")
    ap.add_argument("--max-tokens", type=int, default=32, dest="max_tokens")
    ap.add_argument("--log-every", type=int, default=25, dest="log_every")
    ap.add_argument("--out", default=None, help="Output JSON path.")
    args = ap.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
