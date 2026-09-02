#!/usr/bin/env python3
"""
Token-cost Delta: how many EXTRA tokens does each ordering intervention cost,
relative to an STI baseline (1 image, 1 task, no repetition)?

Two repetition mechanisms are compared:
  - "text repetition"  (STIT): the TASK text is repeated after the image.
  - "image repetition" (SITIT): the IMAGE is echoed a second time, at
    full / half / quarter resolution for the echoed occurrence, mirroring
    rf20_eval.py's --echo-scale ablation (echo_which='second').

Only needs the HF processor + config (vision_start_token_id/vision_end_token_id/
spatial_merge_size), NOT the model weights -- so it runs on CPU and doesn't
compete with the two full-scale GPU jobs currently running. Mirrors the exact
token-counting logic in model_manager.py's prepare_qwen_vl_inputs() /
naturalbench_tokensweep.py's probe_tokens(), just without a loaded model.

Usage:
  python token_cost_analysis.py --n-images 20 --out token_cost_results.json
"""
import argparse
import glob
import json
import os
import random

from PIL import Image

from constants import SYSTEM_MESSAGE
from model_manager import _build_qwen_messages

MODEL_HF = "Qwen/Qwen3-VL-8B-Instruct"
RF20_DATA_DIR = "/home/grg/Research/rf-20-vl-benchmark/datasets/rf100-vl-fsod"
TASK_TEXT = "Is there a car in the image? Answer yes or no."


def sample_images(n, seed=0):
    hits = glob.glob(os.path.join(RF20_DATA_DIR, "*", "train", "*.jpg"))
    random.Random(seed).shuffle(hits)
    # One image per dataset (dir under RF20_DATA_DIR) for size/aspect variety,
    # capped at n.
    seen_ds, chosen = set(), []
    for p in hits:
        ds = p.split(os.sep)[-3]
        if ds in seen_ds:
            continue
        seen_ds.add(ds)
        chosen.append(p)
        if len(chosen) >= n:
            break
    return chosen


def count_tokens(processor, config, system_text, task_text, pil_image, order,
                 task2_text=None):
    """Mirrors prepare_qwen_vl_inputs()'s vision-span logic, minus model.device."""
    from qwen_vl_utils import process_vision_info

    messages = _build_qwen_messages(system_text, task_text, pil_image, order,
                                    task2_text=task2_text)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs,
                       videos=video_inputs if video_inputs else None,
                       padding=True, return_tensors="pt")

    ids = inputs.input_ids[0]
    vision_start_id = config.vision_start_token_id
    vision_end_id = config.vision_end_token_id
    starts = (ids == vision_start_id).nonzero(as_tuple=True)[0]
    ends = (ids == vision_end_id).nonzero(as_tuple=True)[0]
    n_vision_total = sum(int(e) - int(s) - 1 for s, e in zip(starts, ends))
    total = int(ids.shape[0])
    n_text = total - n_vision_total
    return {"total": total, "image": n_vision_total, "text": n_text}


def echo_variants(img, scale):
    w, h = img.size
    scaled = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.LANCZOS)
    return [img, scaled]  # echo_which='second' -- 2nd occurrence is scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=20, dest="n_images")
    ap.add_argument("--out", default="token_cost_results.json")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoConfig
    print(f"[load] {MODEL_HF} processor + config (no weights)...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_HF)
    config = AutoConfig.from_pretrained(MODEL_HF)

    paths = sample_images(args.n_images)
    print(f"[data] {len(paths)} representative RF20 images", flush=True)

    per_image = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        row = {"path": p, "size": img.size}

        row["STI"] = count_tokens(processor, config, SYSTEM_MESSAGE, TASK_TEXT,
                                  img, "STI")
        row["STIT"] = count_tokens(processor, config, SYSTEM_MESSAGE, TASK_TEXT,
                                   img, "STIT")
        row["SITIT_full"] = count_tokens(processor, config, SYSTEM_MESSAGE, TASK_TEXT,
                                         [img, img], "SITIT")
        row["SITIT_half"] = count_tokens(processor, config, SYSTEM_MESSAGE, TASK_TEXT,
                                         echo_variants(img, 0.5), "SITIT")
        row["SITIT_quarter"] = count_tokens(processor, config, SYSTEM_MESSAGE, TASK_TEXT,
                                            echo_variants(img, 0.25), "SITIT")
        per_image.append(row)
        print(f"  {os.path.basename(p)}: STI={row['STI']['total']} "
              f"STIT={row['STIT']['total']} SITIT_full={row['SITIT_full']['total']} "
              f"SITIT_half={row['SITIT_half']['total']} "
              f"SITIT_quarter={row['SITIT_quarter']['total']}", flush=True)

    variants = ["STI", "STIT", "SITIT_full", "SITIT_half", "SITIT_quarter"]
    summary = {}
    for v in variants:
        tot = sum(r[v]["total"] for r in per_image) / len(per_image)
        img_t = sum(r[v]["image"] for r in per_image) / len(per_image)
        txt_t = sum(r[v]["text"] for r in per_image) / len(per_image)
        summary[v] = {"total": tot, "image": img_t, "text": txt_t}

    base = summary["STI"]["total"]
    delta = {}
    for v in variants:
        d_total = summary[v]["total"] - base
        d_img = summary[v]["image"] - summary["STI"]["image"]
        d_txt = summary[v]["text"] - summary["STI"]["text"]
        delta[v] = {"delta_total": d_total, "delta_image": d_img, "delta_text": d_txt}

    print("\n=== Mean tokens (n=%d images), vs STI baseline ===" % len(per_image))
    for v in variants:
        s, d = summary[v], delta[v]
        print(f"  {v:14s} total={s['total']:7.1f}  image={s['image']:7.1f}  "
              f"text={s['text']:6.1f}   Delta_total={d['delta_total']:+7.1f}  "
              f"Delta_image={d['delta_image']:+7.1f}  Delta_text={d['delta_text']:+6.1f}")

    out = {"model": MODEL_HF, "n_images": len(per_image), "task_text": TASK_TEXT,
          "system_text": SYSTEM_MESSAGE, "per_image": per_image,
          "summary": summary, "delta_vs_STI": delta}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
