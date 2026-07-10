"""
RF20 DETECTION mAP eval (the official RF20 metric) under a prompt ordering.

Runs the VLM as an object detector — "Locate all of {classes} and output the
coordinates in JSON" — parses the predicted boxes, scales them from the model's
processed resolution back to original image coords, and computes COCO bbox
AP@[.50:.95] per dataset (pycocotools), then macro-averages over the 20 RF20
datasets → the RF20 mAP. Mirrors the official rf-20-vl-benchmark eval
(get_metrics.py / evaluate_qwen_local.py); coordinate handling matches Qwen's
smart_resize by deriving processed W/H from image_grid_thw.

Ordering is applied via model_manager (order=IST/STI/STIT/SITIT/SIT); --reverse
adds the 2nd-image reversal (S·I·T·Ī·T), tagged SITIT_rev.

    CUDA_VISIBLE_DEVICES=0 python rf20_map_eval.py --model qwen3-vl-8b --order STIT
    CUDA_VISIBLE_DEVICES=0 python rf20_map_eval.py --model qwen3-vl-8b --order SITIT --reverse

Output: rf20/map_results/<model>__<tag>__map.json  (mAP + per-dataset + per-category)
"""
import argparse, json, os, re, time
import numpy as np
import torch
from PIL import Image
from constants import SYSTEM_MESSAGE
from rf20_eval import RF20, CATEGORIES, DATA_DIR

DET_PROMPT = ("Locate all of the following objects: {cls} in the image and output "
              "the coordinates in JSON format like "
              '[{{"bbox_2d": [x1, y1, x2, y2], "label": "class_name"}}].')

NORM = 1000.0   # Qwen3-VL emits bbox coords normalized to a 0-1000 space


def parse_boxes(text):
    """Extract [{'bbox_2d':[x1,y1,x2,y2],'label':str}, ...] from model output."""
    if not text:
        return []
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    if "```json" in text:
        jt = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        m = re.search(r"```(?:.*\n)?([\s\S]+?)```", text)
        jt = m.group(1).strip() if m else text.strip()
    else:
        jt = text.strip()
    jt = re.sub(r",\s*(\]|\})", r"\1", jt)          # trailing commas
    try:
        obj = json.loads(jt)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass
    m = re.search(r'(\[.*\]|\{.*\})', jt, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            pass
    out = []                                         # last resort: individual {...}
    for mm_ in re.findall(r'\{[^{}]*\}', jt, re.DOTALL):
        try:
            out.append(json.loads(mm_))
        except Exception:
            pass
    return out


def _factor(mm):
    ip = mm.image_processor
    ps = getattr(ip, "patch_size", 16)
    ms = getattr(ip, "merge_size", 2)
    return ps * ms


def run_dataset(mm, ds, order, split, ann_path, data_dir, max_images, factor):
    import pycocotools.coco as _c
    j = json.load(open(ann_path))
    id2name = {c["id"]: c["name"] for c in j["categories"]}
    name2id = {v: k for k, v in id2name.items()}
    present = set(a["category_id"] for a in j["annotations"])
    askable = [id2name[c] for c in sorted(present)]          # classes used in split
    cls_prompt = ", ".join(askable)
    imgs = j["images"][:max_images]
    preds = []
    for im in imgs:
        path = os.path.join(data_dir, ds, split, im["file_name"])
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        ow, oh = pil.size
        prompt = DET_PROMPT.format(cls=cls_prompt)
        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [prompt], pil, system_prompt=SYSTEM_MESSAGE, order=order)
        with torch.inference_mode():
            out = mm.llm_model.generate(input_ids, do_sample=False, num_beams=1,
                                        max_new_tokens=1024, use_cache=True, **kwargs)
        txt = mm.tokenizer.batch_decode(out[:, input_ids.shape[1]:],
                                        skip_special_tokens=True)[0]
        for b in parse_boxes(txt):
            bb = b.get("bbox_2d"); lab = b.get("label")
            if not bb or len(bb) != 4 or lab not in name2id:
                continue
            x1, y1, x2, y2 = bb
            # Qwen3-VL emits boxes normalized to a 0-1000 space → back to original px
            x1 = x1 / NORM * ow; x2 = x2 / NORM * ow
            y1 = y1 / NORM * oh; y2 = y2 / NORM * oh
            if x2 <= x1 or y2 <= y1:
                continue
            preds.append({"image_id": im["id"], "category_id": name2id[lab],
                          "bbox": [x1, y1, x2 - x1, y2 - y1],
                          "score": 1.0, "area": (x2 - x1) * (y2 - y1)})
    # COCOeval on the (subset of) images we ran
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib
    gt = COCO(ann_path)
    ran_ids = set(im["id"] for im in imgs)
    if not preds:
        return 0.0, len(imgs)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(ran_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), len(imgs)      # AP@[.50:.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--order", default="STIT")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--data-dir", default=DATA_DIR, dest="data_dir")
    ap.add_argument("--max-images-per-dataset", type=int, default=25, dest="max_images")
    ap.add_argument("--datasets", default="", help="comma list to restrict (debug)")
    ap.add_argument("--out-dir", default="./rf20/map_results", dest="out_dir")
    args = ap.parse_args()

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl; hl.set_verbosity_error()
    setup_seeds(); disable_torch_init()

    tag = (args.order + "_rev") if args.reverse else args.order
    mm = ModelManager(args.model)
    if args.reverse:
        from reverse_image_hooks import install_reverse_hooks, REVERSE
        install_reverse_hooks(mm); REVERSE["on"] = True
    factor = _factor(mm)
    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, f"{mm.model_name}__{tag}__map.json")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or list(RF20)
    per_ds, done = {}, {}
    if os.path.exists(outp):
        try:
            done = json.load(open(outp)).get("per_dataset", {})
        except Exception:
            done = {}
    per_ds.update(done)
    print(f"[map] {args.model} · order={tag} · {len(datasets)} datasets · "
          f"{args.max_images}/dataset · factor={factor}", flush=True)
    t0 = time.time()
    for k, ds in enumerate(datasets):
        if ds in per_ds:
            continue
        split = None; ann = None
        for sp in ("test", "valid", "train"):
            p = os.path.join(args.data_dir, ds, sp, "_annotations.coco.json")
            if os.path.exists(p):
                split, ann = sp, p; break
        if ann is None:
            continue
        ap50_95, n = run_dataset(mm, ds, args.order, split, ann, args.data_dir,
                                 args.max_images, factor)
        per_ds[ds] = ap50_95 * 100
        rate = (k + 1) / max(1e-9, time.time() - t0)
        print(f"  [{k+1}/{len(datasets)}] {ds:26s} AP50-95={per_ds[ds]:5.1f} "
              f"(n={n}) ({rate:.2f} ds/s)", flush=True)
        by_cat = {c: float(np.mean([per_ds[d] for d in RF20 if RF20[d] == c and d in per_ds]))
                  for c in CATEGORIES}
        json.dump({"model": mm.model_name, "order": tag, "metric": "AP@[.50:.95] (%)",
                   "mAP": float(np.mean(list(per_ds.values()))),
                   "per_dataset": per_ds, "by_category": by_cat},
                  open(outp, "w"), indent=2)
    print(f"[done] {tag} RF20 mAP = {np.mean(list(per_ds.values())):.2f}", flush=True)


if __name__ == "__main__":
    main()
