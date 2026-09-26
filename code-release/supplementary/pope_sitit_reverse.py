"""
POPE SITIT-with-reversed-2nd-image (S·I·T·Ī·T) — Qwen3-VL-8B or Gemma-3-27B.
Runs the SITIT ordering with the 2nd image block's patches reversed (via the shared
reverse_image_hooks), writes standard POPE result files tagged SITIT_rev. Additive;
no existing code changed.

    CUDA_VISIBLE_DEVICES=0 python pope_sitit_reverse.py --model qwen3-vl-8b
    CUDA_VISIBLE_DEVICES=1 python pope_sitit_reverse.py --model gemma-3-27b
"""
import argparse, json, os, time
import torch
from constants import SYSTEM_MESSAGE
from naturalbench_eval import YESNO_SUFFIX
from pope_eval import load_pope_samples, pope_metrics, _pred_yes_no, CATEGORIES
from reverse_image_hooks import install_reverse_hooks, REVERSE

TAG, ORDER = "SITIT_rev", "SITIT"


def _meta(model, records, max_tokens):
    return {
        "model": model, "order": TAG, "order_tag": TAG, "base_order": ORDER,
        "reverse_second_image": True, "system_prompt": SYSTEM_MESSAGE,
        "max_tokens": max_tokens, "overall": pope_metrics(records),
        "by_category": {c: pope_metrics([r for r in records if r["category"] == c])
                        for c in CATEGORIES},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--max-per-category", type=int, default=None, dest="max_per_category")
    ap.add_argument("--out-dir", default="./pope/results", dest="out_dir")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--checkpoint-every", type=int, default=200, dest="checkpoint_every")
    args = ap.parse_args()

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    setup_seeds(); disable_torch_init()

    samples = load_pope_samples(max_per_category=args.max_per_category)
    print(f"[data] {len(samples)} POPE Qs · SITIT-reverse · {args.model}", flush=True)
    mm = ModelManager(args.model)
    install_reverse_hooks(mm); REVERSE["on"] = True
    os.makedirs(args.out_dir, exist_ok=True)

    rp = os.path.join(args.out_dir, f"{mm.model_name}__{TAG}__results.json")
    records, done = [], set()
    if os.path.exists(rp):
        try:
            records = json.load(open(rp))["results"]; done = {r["question_id"] for r in records}
            print(f"  [resume] {len(done)} done")
        except Exception:
            records, done = [], set()

    def ckpt():
        json.dump({"meta": _meta(mm.model_name, records, args.max_tokens), "results": records},
                  open(rp, "w"), indent=2)

    n = len(samples); t0 = time.time()
    for si, s in enumerate(samples):
        if s["question_id"] in done:
            continue
        img = s["image"].convert("RGB")
        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [s["question"] + YESNO_SUFFIX], img, system_prompt=SYSTEM_MESSAGE, order=ORDER)
        with torch.inference_mode():
            out = mm.llm_model.generate(input_ids, do_sample=False, num_beams=1,
                                        max_new_tokens=args.max_tokens, use_cache=True, **kwargs)
        raw = mm.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        pred = _pred_yes_no(raw)
        records.append({"question_id": s["question_id"], "category": s["category"],
                        "question": s["question"], "gt": s["answer"], "model_answer_raw": raw,
                        "pred": pred, "correct": pred == s["answer"]})
        if (si + 1) % 200 == 0 or (si + 1) == n:
            m = pope_metrics(records); rate = (len(records) - len(done)) / max(1e-9, time.time() - t0)
            print(f"  [SITIT_rev] [{si+1}/{n}] acc={m['acc']:.3f} f1={m['f1']:.3f} "
                  f"yes={m['yes_ratio']:.3f} ({rate:.2f} q/s)", flush=True)
        if (si + 1) % args.checkpoint_every == 0:
            ckpt()
    ckpt(); o = pope_metrics(records)
    print(f"  [done] SITIT_rev {args.model}: acc={o['acc']:.3f} f1={o['f1']:.3f}")


if __name__ == "__main__":
    main()
