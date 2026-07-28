"""
Prompt-ordering experiments (STI / SIT / STIT / SITIT / SITIT_rev) applied to the
DetPO detection task on RF20-VL Aerial (COCO mAP) and to RefCOCOg val (referring
[email protected]) -- run on the LOCAL Qwen3-VL-8B model so SITIT_rev's second-image patch
reversal (reverse_image_hooks) can be applied faithfully.

Orderings (S=system, T=task/question text, I=image):
  STI       S T I            question-first
  SIT       S I T            question-last
  STIT      S T I T          question echo
  SITIT     S I T I T        image echo
  SITIT_rev S I T Ī T        image echo, 2nd image block reversed (hooks)

  CUDA_VISIBLE_DEVICES=0 /home/grg/anaconda3/envs/logitlens/bin/python \
    detpo_map/ordering_eval.py --benchmark rf20   --orders STI,SIT,STIT,SITIT,SITIT_rev
  CUDA_VISIBLE_DEVICES=0 /home/grg/anaconda3/envs/logitlens/bin/python \
    detpo_map/ordering_eval.py --benchmark refcoco --orders STI,SIT,STIT,SITIT,SITIT_rev --n 0
"""
import argparse, glob, json, os, pickle, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from constants import SYSTEM_MESSAGE
from model_manager import ModelManager
from reverse_image_hooks import install_reverse_hooks, REVERSE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
DET_ROOT = "/home/grg/Research/rf-20-vl-benchmark/datasets/rf100-vl-fsod"
DETPO = "/home/grg/Research/DetPO"
DATA_INSTR = os.path.join(DETPO, "data_instr", "default")
AERIAL = ["wildfire-smoke", "aerial-airport"]
RC = "/home/grg/Research/rf-20-vl-benchmark/datasets/RefCOCO"
MODEL = "qwen3-vl-8b"
DET_CAP = 1024                      # longest-side cap for detection images (speed/mem)

# S T I letters used with prepare_inputs_from_pil; reverse flag toggles hooks.
ORDER_MAP = {
    "STI": ("STI", False), "SIT": ("SIT", False), "STIT": ("STIT", False),
    "SITIT": ("SITIT", False), "SITIT_rev": ("SITIT", True),
}

DET_PROMPT = (
    'Identify and localize all instances of "{cls}" in the image.\n'
    "Output Requirements:\n"
    "- Return valid JSON only. Do not include explanations or extra text.\n"
    "- Output a ranked list of detections sorted by confidence (highest first).\n"
    "- Include at most 20 detections.\n"
    "- If no objects are detected, return an empty list [].\n"
    'For each detection provide: "bbox_2d": [x1, y1, x2, y2] (top-left, '
    'bottom-right), "label": "{cls}", "score": float in 0..1.\n'
    "Follow these annotator instructions to improve detection accuracy:\n"
    "{instr}\n"
    'Return a JSON list like [{{"bbox_2d": [x1,y1,x2,y2], "label": "{cls}", '
    '"score": 0.95}}].')

REF_PROMPT = ('Locate "{phrase}" in the image and output its bounding box. '
              'Return valid JSON only, no extra text, in the form '
              '{{"bbox_2d": [x1, y1, x2, y2]}} where (x1,y1) is the top-left and '
              '(x2,y2) the bottom-right corner.')

NUM = re.compile(r"[-+]?\d*\.?\d+")


def downscale(pil, cap):
    w, h = pil.size
    s = cap / max(w, h)
    return pil if s >= 1.0 else pil.resize((max(1, int(w * s)), max(1, int(h * s))),
                                            Image.LANCZOS)


def gen(mm, pil, task, order_letters, reverse, max_new_tokens):
    REVERSE["on"] = reverse
    try:
        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [task], pil, system_prompt=SYSTEM_MESSAGE, order=order_letters)
        with torch.inference_mode():
            out = mm.llm_model.generate(
                input_ids, do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens, use_cache=True, **kwargs)
        text = mm.tokenizer.batch_decode(
            out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    finally:
        REVERSE["on"] = False
    return text


def parse_dets(text):
    """Return list of {'bbox':[x1,y1,x2,y2] in 0-1000, 'score':float} from output."""
    t = text.strip().strip("`")
    t = re.sub(r"^json", "", t).strip()
    out = []
    try:
        data = json.loads(t[t.index("["):t.rindex("]") + 1])
        for it in data:
            b = it.get("bbox_2d") or it.get("bbox")
            if b and len(b) >= 4:
                x1, y1, x2, y2 = map(float, b[:4])
                out.append({"bbox": _order_box(x1, y1, x2, y2),
                            "score": float(it.get("score", 1.0))})
        if out:
            return out
    except Exception:
        pass
    for m in re.finditer(r'"bbox_2d"\s*:?\s*\[([^\]]+)\]', text):
        nums = NUM.findall(m.group(1))
        if len(nums) >= 4:
            x1, y1, x2, y2 = map(float, nums[:4])
            out.append({"bbox": _order_box(x1, y1, x2, y2), "score": 1.0})
    return out


def parse_box(text):
    d = parse_dets(text)
    return d[0]["bbox"] if d else None


def _order_box(x1, y1, x2, y2):
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- RF20
def run_rf20(mm, tag, letters, reverse):
    per, ms, m50, m75 = {}, [], [], []
    for ds in AERIAL:
        ann_path = os.path.join(DET_ROOT, ds, "test", "_annotations.coco.json")
        coco_gt = COCO(ann_path)
        cats = [c for c in coco_gt.loadCats(coco_gt.getCatIds())
                if c["name"].lower() != "none"]
        instr_json = {}
        ip = os.path.join(DATA_INSTR, f"README.dataset_{ds}.json")
        if os.path.isfile(ip):
            instr_json = json.load(open(ip))
        dets = []
        imgs = coco_gt.dataset["images"]
        for k, img_info in enumerate(imgs):
            path = os.path.join(DET_ROOT, ds, "test", img_info["file_name"])
            if not os.path.isfile(path):
                continue
            pil = Image.open(path).convert("RGB")
            W, H = pil.size
            pin = downscale(pil, DET_CAP)
            for cat in cats:
                instr = instr_json.get(cat["name"], "")
                task = DET_PROMPT.format(cls=cat["name"], instr=instr)
                text = gen(mm, pin, task, letters, reverse, 1024)
                for d in parse_dets(text):
                    x1, y1, x2, y2 = d["bbox"]
                    dets.append({"image_id": img_info["id"],
                                 "category_id": cat["id"],
                                 "bbox": [x1 / 1000 * W, y1 / 1000 * H,
                                          (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H],
                                 "score": d["score"]})
            if (k + 1) % 20 == 0:
                print(f"    [{tag}] {ds} {k+1}/{len(imgs)} dets={len(dets)}", flush=True)
        if dets:
            ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
            ev.evaluate(); ev.accumulate(); ev.summarize()
            s = ev.stats
        else:
            s = [0.0] * 12
        per[ds] = {"mAP": s[0] * 100, "mAP50": s[1] * 100, "mAP75": s[2] * 100,
                   "classes": [c["name"] for c in cats], "n_images": len(imgs)}
        ms.append(s[0] * 100); m50.append(s[1] * 100); m75.append(s[2] * 100)
        print(f"  [{tag}] {ds}: mAP={s[0]*100:.1f} [email protected]={s[1]*100:.1f}", flush=True)
    mean = {"mAP": sum(ms) / len(ms), "mAP50": sum(m50) / len(m50),
            "mAP75": sum(m75) / len(m75)}
    res = {"meta": {"benchmark": "RF20-VL — Aerial", "ordering": tag,
                    "config": f"prompt ordering {tag} (default class descriptions)",
                    "model": MODEL, "per_dataset": per, "mean": mean}}
    out = os.path.join(OUT_DIR, f"rf20_aerial_order-{tag}_{MODEL}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"  [{tag}] RF20 aerial-mean mAP {mean['mAP']:.1f} -> {out}", flush=True)


# ------------------------------------------------------------------------ RefCOCO
def run_refcoco(mm, tag, letters, reverse, n):
    refs = [r for r in pickle.load(open(f"{RC}/refcocog/refs(umd).p", "rb"))
            if r["split"] == "val"]
    inst = json.load(open(f"{RC}/refcocog/instances.json"))
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {im["id"]: im for im in inst["images"]}
    if n and n < len(refs):
        refs = refs[:n]
    out = os.path.join(OUT_DIR, f"refcocog_val_order-{tag}_{MODEL}.json")
    per, done = [], set()
    if os.path.exists(out):
        try:
            per = json.load(open(out)).get("per_sample", [])
            done = {p["ref_id"] for p in per}
        except Exception:
            per, done = [], set()
    correct = sum(p["ok"] for p in per)
    parsed = sum(p.get("parsed", 1) for p in per)
    t0 = time.time()
    for i, r in enumerate(refs):
        if r["ref_id"] in done:
            continue
        gt = ann[r["ann_id"]]["bbox"]
        im_info = img[r["image_id"]]
        W, H = im_info["width"], im_info["height"]
        phrase = r["sentences"][0]["sent"]
        try:
            pil = Image.open(os.path.join(RC, "train2014", im_info["file_name"])).convert("RGB")
            text = gen(mm, pil, REF_PROMPT.format(phrase=phrase), letters, reverse, 96)
        except Exception as e:
            text = ""
            print(f"    [{tag}] err {i}: {e}", flush=True)
        box = parse_box(text)
        ok = False
        if box is not None:
            parsed += 1
            x1, y1, x2, y2 = box
            pred = [x1 / 1000 * W, y1 / 1000 * H, (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H]
            ok = iou_xywh(pred, gt) >= 0.5
        correct += int(ok)
        per.append({"ref_id": r["ref_id"], "ok": ok, "parsed": 1 if box else 0})
        if (i + 1) % 50 == 0:
            acc = correct / len(per)
            print(f"    [{tag}] {i+1}/{len(refs)} [email protected]={acc:.3f} "
                  f"{len(per)/(time.time()-t0+1e-9):.2f}/s", flush=True)
        if (i + 1) % 100 == 0 or i + 1 == len(refs):
            _save_ref(out, tag, refs, per, correct, parsed)
    _save_ref(out, tag, refs, per, correct, parsed)
    print(f"  [{tag}] RefCOCOg [email protected] {correct/max(1,len(per))*100:.1f} -> {out}", flush=True)


def _save_ref(out, tag, refs, per, correct, parsed):
    n = len(per)
    meta = {"benchmark": "RefCOCOg", "variant": "umd", "split": "val",
            "metric": "[email protected]", "ordering": tag, "model": MODEL,
            "n": n, "n_total_split": len(refs), "parsed": parsed,
            "correct": correct, "acc": correct / n if n else 0.0}
    json.dump({"meta": meta, "per_sample": per}, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["rf20", "refcoco"], required=True)
    ap.add_argument("--orders", default="STI,SIT,STIT,SITIT,SITIT_rev")
    ap.add_argument("--n", type=int, default=0, help="RefCOCO ref cap (0=all)")
    args = ap.parse_args()

    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    setup_seeds(); disable_torch_init()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[load] ModelManager({MODEL})", flush=True)
    mm = ModelManager(MODEL)
    install_reverse_hooks(mm)
    REVERSE["on"] = False

    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    for tag in orders:
        if tag not in ORDER_MAP:
            print(f"  skip unknown ordering {tag}"); continue
        letters, reverse = ORDER_MAP[tag]
        print(f"=== {args.benchmark} · ordering {tag} (letters={letters} "
              f"reverse={reverse}) ===", flush=True)
        if args.benchmark == "rf20":
            run_rf20(mm, tag, letters, reverse)
        else:
            run_refcoco(mm, tag, letters, reverse, args.n)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
