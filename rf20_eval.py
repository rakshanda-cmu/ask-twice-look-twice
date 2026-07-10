"""
Standalone RF20 (RF100-VL, 20 Roboflow detection datasets) runner — IST/STI/STIT,
reformulated as POPE-style object-presence yes/no QA.

RF20 spans 20 diverse domains (industrial, medical x-ray/dental, aerial, aquatic,
documents, sport, …) as COCO-format object detection. To fit the same
generate→judge→ordering harness as NaturalBench / POPE / Winoground we ask, for
each (image, class) pair:

    "Is there a <class> in the image?"   gt = Yes iff that class is annotated in the image

so it becomes an object-hallucination test across 20 out-of-distribution domains —
a strong generalization check for the prompt-ordering (IST/STI/STIT) effect.

Kept FULLY SEPARATE from the other pipelines — imports only two stateless,
model-agnostic helpers (yes/no parser + shared suffix). Reuses model loading + the
IST/STI/STIT ordering machinery from model_manager.

Run (resumable; checkpoints every --checkpoint-every):
    CUDA_VISIBLE_DEVICES=0 python rf20_eval.py --model qwen3-vl-8b --order IST,STI,STIT
    CUDA_VISIBLE_DEVICES=1 python rf20_eval.py --model gemma-3-27b --order IST,STI,STIT

Output (separate dir; one file per model+order):
    rf20/results/<model>__<order>__results.json
"""

import argparse
import json
import os
import time

from constants import SYSTEM_MESSAGE
from naturalbench_eval import _first_yes_no, YESNO_SUFFIX

MODEL_CHOICES = ["gemma-3-27b", "gemma-3-12b", "gemma-3-4b",
                 "qwen3-vl-8b", "qwen2.5-vl-7b", "llava-1.5"]

DATA_DIR = "/home/grg/Research/rf-20-vl-benchmark/datasets/rf100-vl-fsod"
# the 20 RF100-VL datasets → their RF20 super-category
RF20 = {
    "recode-waste": "Industrial", "defect-detection": "Industrial",
    "water-meter": "Industrial", "paper-parts": "Document",
    "all-elements": "Document", "lacrosse-object-detection": "Sport",
    "actions": "Sport", "trail-camera": "Flora/Fauna", "gwhd2021": "Flora/Fauna",
    "wb-prova": "Flora/Fauna", "aquarium-combined": "Flora/Fauna",
    "orionproducts": "Misc", "the-dreidel-project": "Misc",
    "soda-bottles": "Misc", "flir-camera-objects": "Misc",
    "new-defects-in-wood": "Misc", "wildfire-smoke": "Aerial",
    "aerial-airport": "Aerial", "dentalai": "Lab Imaging", "x-ray-id": "Lab Imaging",
}
CATEGORIES = sorted(set(RF20.values()))
SPLITS = ("test", "valid", "train")


def _pred_yes_no(raw):
    p = _first_yes_no(raw)
    if p is not None:
        return p
    low = raw.lower()
    return "no" if ("no" in low or "not" in low) else "yes"


def _article(name):
    return "an" if name[:1].lower() in "aeiou" else "a"


def _question(cls):
    return f"Is there {_article(cls)} {cls} in the image?" + YESNO_SUFFIX


def pope_metrics(records):
    tp = fp = tn = fn = yes_pred = correct = 0
    n = len(records)
    for r in records:
        gt, pred = r["gt"], r["pred"]
        if pred == "yes":
            yes_pred += 1
        if pred == gt:
            correct += 1
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n,
        "acc": round(correct / n, 4) if n else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_pred / n, 4) if n else 0.0,
    }


def _build_meta(model_name, order, system_prompt, max_tokens, records):
    by_cat = {c: pope_metrics([r for r in records if r["category"] == c])
              for c in CATEGORIES}
    by_ds = {d: pope_metrics([r for r in records if r["dataset"] == d])
             for d in sorted(set(r["dataset"] for r in records))} if records else {}
    return {
        "model": model_name,
        "order": order,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "overall": pope_metrics(records),
        "by_category": by_cat,
        "by_dataset": by_ds,
    }


def load_rf20_samples(data_dir=DATA_DIR, max_images_per_dataset=25):
    """Build presence yes/no samples across the 20 RF100-VL datasets. For each
    image we ask about every class that appears at least once in the split."""
    samples = []
    for ds, cat in RF20.items():
        ann_path = split = None
        for sp in SPLITS:
            p = os.path.join(data_dir, ds, sp, "_annotations.coco.json")
            if os.path.exists(p):
                ann_path, split = p, sp
                break
        if ann_path is None:
            print(f"  [warn] {ds}: no annotations, skipped")
            continue
        j = json.load(open(ann_path))
        id2name = {c["id"]: c["name"] for c in j["categories"]}
        present = {}                       # image_id -> set(class names)
        for a in j["annotations"]:
            present.setdefault(a["image_id"], set()).add(id2name[a["category_id"]])
        # classes actually used in this split (skip Roboflow supercategory dummies)
        used = set().union(*present.values()) if present else set()
        askable = sorted(used)
        imgs = j["images"][:max_images_per_dataset]
        for im in imgs:
            img_path = os.path.join(data_dir, ds, split, im["file_name"])
            here = present.get(im["id"], set())
            for cls in askable:
                samples.append({
                    "question_id": f"{ds}__{im['id']}__{cls}",
                    "dataset": ds, "category": cat, "cls": cls,
                    "image_path": img_path,
                    "gt": "yes" if cls in here else "no",
                })
    return samples


def run_rf20_experiment(mm, samples, order, system_prompt,
                        max_tokens=16, out_dir="./rf20/results",
                        checkpoint_every=200, log_every=100):
    import torch
    from PIL import Image

    records, done, results_path = [], set(), None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{mm.model_name}__{order}__results.json")
        if os.path.exists(results_path):
            try:
                records = json.load(open(results_path))["results"]
                done = {r["question_id"] for r in records}
                print(f"  [resume] {order}: {len(done)} samples already done")
            except Exception:
                records, done = [], set()

    def _checkpoint():
        if results_path:
            meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records)
            with open(results_path, "w") as f:
                json.dump({"meta": meta, "results": records}, f, indent=2)

    n = len(samples)
    t0 = time.time()
    for si, s in enumerate(samples):
        if s["question_id"] in done:
            continue
        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            continue
        prompt = _question(s["cls"])

        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [prompt], img, system_prompt=system_prompt, order=order,
        )
        with torch.inference_mode():
            out = mm.llm_model.generate(
                input_ids, do_sample=False, num_beams=1,
                max_new_tokens=max_tokens, use_cache=True, **kwargs,
            )
        gen = out[:, input_ids.shape[1]:]
        raw = mm.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        pred = _pred_yes_no(raw)

        records.append({
            "question_id": s["question_id"], "dataset": s["dataset"],
            "category": s["category"], "cls": s["cls"],
            "gt": s["gt"], "model_answer_raw": raw, "pred": pred,
            "correct": pred == s["gt"],
        })

        if (si + 1) % log_every == 0 or (si + 1) == n:
            m = pope_metrics(records)
            rate = (len(records) - len(done)) / max(1e-9, time.time() - t0)
            print(f"  [{order}] [{si+1}/{n}] acc={m['acc']:.3f} f1={m['f1']:.3f} "
                  f"yes={m['yes_ratio']:.3f} ({rate:.2f} q/s)", flush=True)
        if results_path and (si + 1) % checkpoint_every == 0:
            _checkpoint()

    meta = _build_meta(mm.model_name, order, system_prompt, max_tokens, records)
    if out_dir:
        _checkpoint()
    return meta, records


def main():
    ap = argparse.ArgumentParser(description="Evaluate a VLM on RF20 (IST/STI/STIT).")
    ap.add_argument("--model", default="qwen3-vl-8b", choices=MODEL_CHOICES)
    ap.add_argument("--order", default="IST", help="Comma-separated, e.g. IST,STI,STIT")
    ap.add_argument("--data-dir", default=DATA_DIR, dest="data_dir")
    ap.add_argument("--max-images-per-dataset", type=int, default=25,
                    dest="max_images_per_dataset")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--out-dir", default="./rf20/results", dest="out_dir")
    ap.add_argument("--checkpoint-every", type=int, default=200, dest="checkpoint_every")
    args = ap.parse_args()

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    orders = [o.strip().upper() for o in args.order.split(",") if o.strip()]
    print(f"[data] loading RF20 (max {args.max_images_per_dataset} img/dataset) …", flush=True)
    samples = load_rf20_samples(args.data_dir, args.max_images_per_dataset)
    yes = sum(1 for s in samples if s["gt"] == "yes")
    print(f"[data] {len(samples)} presence Qs ({yes} yes / {len(samples)-yes} no) "
          f"over {len(set(s['dataset'] for s in samples))} datasets, orders {orders}")
    print(f"[load] model = {args.model}")
    mm = ModelManager(args.model)

    for order in orders:
        print(f"\n=== {order} ===")
        meta, _ = run_rf20_experiment(
            mm, samples, order, SYSTEM_MESSAGE, max_tokens=args.max_tokens,
            out_dir=args.out_dir, checkpoint_every=args.checkpoint_every,
        )
        o = meta["overall"]
        print(f"  [done] {order}: acc={o['acc']:.3f} f1={o['f1']:.3f} "
              f"yes_ratio={o['yes_ratio']:.3f}")


if __name__ == "__main__":
    main()
