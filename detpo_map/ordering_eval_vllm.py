"""
RF20-VL detection ordering experiments via the **vLLM library** (offline `LLM` API),
for the four hook-free orderings STI / SIT / STIT / SITIT.  SITIT_rev is intentionally
NOT run here: it reverses the 2nd image block's patches inside the model's hidden
states (reverse_image_hooks.py), which vLLM cannot express -- it stays on the local
HF path (detpo_map/ordering_eval.py) and is parked for now.

Multi-class prompting (all classes per image, DetPO-paper protocol). Writes the SAME
per-dataset result files as the HF runner so the website reads them unchanged:
    detpo_map/results/rf20ds_<ds>_order-<TAG>_qwen3-vl-8b.json   (meta.engine="vllm")

Run (all 20 RF20 datasets, 4 orderings) in the vLLM env:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python detpo_map/ordering_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT
"""
import argparse, base64, io, json, os, re, time

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
DET_ROOT = "/home/grg/Research/rf-20-vl-benchmark/datasets/rf100-vl-fsod"
DATA_INSTR = "/home/grg/Research/DetPO/data_instr/default"
# Model registry: tag -> (HF repo id, vLLM quantization arg or None). See
# extra_tasks/common.py's MODEL_REGISTRY docstring for the no-NVLink /
# bnb-4-bit-single-GPU rationale (same box, same constraint).
MODEL_REGISTRY = {
    "qwen3-vl-8b": {"hf": "Qwen/Qwen3-VL-8B-Instruct", "quantization": None},
    "gemma-3-27b": {"hf": "google/gemma-3-27b-it", "quantization": "bitsandbytes"},
}
MODEL_HF = MODEL_REGISTRY["qwen3-vl-8b"]["hf"]
MODEL_TAG = "qwen3-vl-8b"
DET_CAP = 1024

SYSTEM_MESSAGE = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  "answers to the human's questions.")

# All 20 RF20-VL datasets (aerial included so the whole 4-ordering table is one engine).
RF20_DATASETS = [
    "wildfire-smoke", "aerial-airport", "paper-parts", "all-elements",
    "trail-camera", "gwhd2021", "wb-prova", "aquarium-combined", "recode-waste",
    "defect-detection", "water-meter", "dentalai", "x-ray-id", "orionproducts",
    "the-dreidel-project", "soda-bottles", "flir-camera-objects",
    "new-defects-in-wood", "lacrosse-object-detection", "actions",
]

# Ordering tag -> S/T/I letter layout (content-part order in the single user turn).
ORDER_LETTERS = {"STI": "STI", "SIT": "SIT", "STIT": "STIT", "SITIT": "SITIT",
                 "STITI": "STITI"}

DET_PROMPT_MULTI = (
    "Detect every object in the image that belongs to any of these classes: "
    "{classes}.\n"
    "Output Requirements:\n"
    "- Return valid JSON only. Do not include explanations or extra text.\n"
    "- A single ranked list of detections sorted by confidence (highest first).\n"
    "- At most 50 detections. If none, return an empty list [].\n"
    'For each detection provide: "bbox_2d": [x1, y1, x2, y2] (top-left, '
    'bottom-right), "label": exactly one of the class names above, "score": float '
    "in 0..1.\n"
    "Per-class annotator guidance:\n"
    "{instr}\n"
    'Return a JSON list like [{{"bbox_2d": [x1,y1,x2,y2], "label": "<class>", '
    '"score": 0.95}}].')

NUM = re.compile(r"[-+]?\d*\.?\d+")


def _order_box(x1, y1, x2, y2):
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_dets(text):
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
                            "score": float(it.get("score", 1.0)),
                            "label": it.get("label")})
        if out:
            return out
    except Exception:
        pass
    for m in re.finditer(r'"bbox_2d"\s*:?\s*\[([^\]]+)\]', text):
        nums = NUM.findall(m.group(1))
        if len(nums) >= 4:
            x1, y1, x2, y2 = map(float, nums[:4])
            out.append({"bbox": _order_box(x1, y1, x2, y2), "score": 1.0, "label": None})
    return out


def _match_label(label, name2id, norm2id):
    if label is None:
        return None
    if label in name2id:
        return name2id[label]
    return norm2id.get(re.sub(r"[^a-z0-9]", "", str(label).lower()))


def downscale(pil, cap):
    w, h = pil.size
    s = cap / max(w, h)
    return pil if s >= 1.0 else pil.resize((max(1, int(w * s)), max(1, int(h * s))),
                                            Image.LANCZOS)


def data_uri(pil):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_conversation(letters, task, du):
    parts = []
    for c in letters:
        if c == "S":
            parts.append({"type": "text", "text": SYSTEM_MESSAGE})
        elif c == "T":
            parts.append({"type": "text", "text": task})
        elif c == "I":
            parts.append({"type": "image_url", "image_url": {"url": du}})
    return [{"role": "user", "content": parts}]


def run_dataset(llm, sp, tag, letters, ds):
    out = os.path.join(OUT_DIR, f"rf20ds_{ds}_order-{tag}_{MODEL_TAG}.json")
    if os.path.exists(out):
        try:
            m = json.load(open(out))["meta"]
            if m.get("complete") and m.get("engine") == "vllm":
                print(f"  [{tag}] {ds}: cached ({m.get('mAP'):.1f})", flush=True)
                return
        except Exception:
            pass
    ann_path = os.path.join(DET_ROOT, ds, "test", "_annotations.coco.json")
    coco_gt = COCO(ann_path)
    cats = [c for c in coco_gt.loadCats(coco_gt.getCatIds())
            if c["name"].lower() != "none"]
    name2id = {c["name"]: c["id"] for c in cats}
    norm2id = {re.sub(r"[^a-z0-9]", "", c["name"].lower()): c["id"] for c in cats}
    instr_json = {}
    ip = os.path.join(DATA_INSTR, f"README.dataset_{ds}.json")
    if os.path.isfile(ip):
        instr_json = json.load(open(ip))
    classes = ", ".join(c["name"] for c in cats)
    instr = "\n".join(f"- {c['name']}: {instr_json.get(c['name'], '')}".rstrip()
                      for c in cats)
    task = DET_PROMPT_MULTI.format(classes=classes, instr=instr)

    imgs = coco_gt.dataset["images"]
    convs, meta_img = [], []
    for img_info in imgs:
        path = os.path.join(DET_ROOT, ds, "test", img_info["file_name"])
        if not os.path.isfile(path):
            continue
        pil = downscale(Image.open(path).convert("RGB"), DET_CAP)
        convs.append(build_conversation(letters, task, data_uri(pil)))
        meta_img.append((img_info["id"], img_info["width"], img_info["height"]))

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    dets = []
    for o, (img_id, W, H) in zip(outputs, meta_img):
        text = o.outputs[0].text
        for d in parse_dets(text):
            cid = _match_label(d.get("label"), name2id, norm2id)
            if cid is None and len(cats) == 1:
                cid = cats[0]["id"]
            if cid is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            dets.append({"image_id": img_id, "category_id": cid,
                         "bbox": [x1 / 1000 * W, y1 / 1000 * H,
                                  (x2 - x1) / 1000 * W, (y2 - y1) / 1000 * H],
                         "score": d["score"]})
    if dets:
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
    else:
        s = [0.0] * 12
    meta = {"benchmark": "RF20-VL", "dataset": ds, "ordering": tag,
            "model": MODEL_TAG, "engine": "vllm", "prompting": "multi-class",
            "mAP": float(s[0]) * 100, "mAP50": float(s[1]) * 100,
            "mAP75": float(s[2]) * 100, "classes": [c["name"] for c in cats],
            "n_images": len(meta_img), "n_dets": len(dets), "complete": True}
    json.dump({"meta": meta}, open(out, "w"), indent=2)
    print(f"  [{tag}] {ds}: mAP={meta['mAP']:.1f} AP50={meta['mAP50']:.1f} "
          f"({len(meta_img)} imgs in {dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="STI,SIT,STIT,SITIT")
    ap.add_argument("--datasets", default=",".join(RF20_DATASETS))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model
    if args.model == "gemma-3-27b" and args.tp != 1:
        print("  [note] forcing --tp 1 for gemma-3-27b (bnb 4-bit, single GPU only)")
        args.tp = 1
    cfg = MODEL_REGISTRY[args.model]

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(model=cfg["hf"], trust_remote_code=True,
                      max_model_len=24096, tensor_parallel_size=args.tp,
                      gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 2})
    if cfg["quantization"]:
        llm_kwargs["quantization"] = cfg["quantization"]
        llm_kwargs["load_format"] = cfg["quantization"]
    else:
        llm_kwargs["dtype"] = "float16"
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for tag in orders:
        if tag not in ORDER_LETTERS:
            print(f"  skip {tag} (not a hook-free ordering)"); continue
        letters = ORDER_LETTERS[tag]
        print(f"=== vLLM · ordering {tag} (letters={letters}) ===", flush=True)
        for ds in datasets:
            run_dataset(llm, sp, tag, letters, ds)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
