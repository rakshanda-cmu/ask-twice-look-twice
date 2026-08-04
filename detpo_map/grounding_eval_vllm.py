"""
Generalized referring-grounding ordering ablation via the vLLM library --
covers RefCOCO, RefCOCO+, and RefCOCOg in one harness (RefCOCOg's val-split
STI..SITIT_rev numbers already exist from detpo_map/ordering_eval.py; this adds
STITI there and adds RefCOCO / RefCOCO+ on testA+testB, matching the paper's
standard reporting splits).

Metric: referring accuracy at IoU >= 0.5 (one predicted box per expression),
same convention as the existing RefCOCOg results.

Orderings here are the hook-free set (STI/SIT/STIT/SITIT/STITI); SITIT_rev
needs the local HF patch-reversal hooks (detpo_map/ordering_eval.py) and is
queued separately, same pattern as RF20.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python detpo_map/grounding_eval_vllm.py \
      --datasets refcoco,refcoco+ --splits testA,testB \
      --orders STI,SIT,STIT,SITIT,STITI
  # RefCOCOg STITI only (other orderings already exist from the HF run):
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python detpo_map/grounding_eval_vllm.py \
      --datasets refcocog --splits val --orders STITI
"""
import argparse, base64, io, json, os, pickle, re, sys, time

from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "extra_tasks"))
from common import load_hf_chat_engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
RC = "/home/grg/Research/rf-20-vl-benchmark/datasets/RefCOCO"
# coord_scale: see ordering_eval_vllm.py's MODEL_REGISTRY docstring -- Qwen
# emits boxes normalized to 0-1000 by training convention; Gemma-3 has no such
# convention and its raw box coords are pixels on the FIXED 896x896 square
# canvas its Gemma3ImageProcessor resizes every image to (confirmed via
# preprocessor_config.json and empirically via an out-of-bounds box on a
# 640x480 test image), not a 0-1000 normalized space.
#
# gemma-4-31b's coord_scale is INTENTIONALLY None: Gemma4ImageProcessor has no
# fixed square canvas at all (patch_size=16, variable soft-token budget per
# image, confirmed via its processor_config.json) -- there is no reason to
# assume its raw box output follows the SAME convention as either Qwen's 1000
# or Gemma-3's 896, and guessing wrong here would silently corrupt every
# gemma-4-31b RF20/grounding number the same way the original 1000-for-everyone
# bug did for gemma-3-27b. Fill this in only after empirically capturing a raw
# gemma-4-31b bbox_2d output against a known image size (see the debug script
# used to find Gemma-3's 896 -- same method). The None deliberately makes
# run_order() below crash with a clear error if this model is used for
# grounding before that verification happens, rather than silently guessing.
MODEL_REGISTRY = {
    "qwen3-vl-8b": {"hf": "Qwen/Qwen3-VL-8B-Instruct", "quantization": None,
                    "coord_scale": 1000, "box_order": "xyxy", "engine": "vllm"},
    "gemma-3-27b": {"hf": "google/gemma-3-27b-it", "quantization": "bitsandbytes",
                    "coord_scale": 896, "box_order": "xyxy", "engine": "vllm"},
    # coord_scale=1000, box_order="yxyx" -- both confirmed empirically, see
    # ordering_eval_vllm.py's MODEL_REGISTRY comment for the verification
    # method. engine="local_hf": vLLM 0.19.1 cannot load this model (see
    # extra_tasks/common.py's MODEL_REGISTRY comment for the upstream bug).
    "gemma-4-31b": {"hf": "google/gemma-4-31B-it", "quantization": "bitsandbytes",
                    "coord_scale": 1000, "box_order": "yxyx", "engine": "local_hf"},
}
MODEL_HF = MODEL_REGISTRY["qwen3-vl-8b"]["hf"]
MODEL_TAG = "qwen3-vl-8b"
IMG_CAP = 1024

ORDER_LETTERS = {"STI": "STI", "SIT": "SIT", "STIT": "STIT", "SITIT": "SITIT",
                 "STITI": "STITI"}

SYSTEM_MESSAGE = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  "answers to the human's questions.")

REF_PROMPT = ('Locate "{phrase}" in the image and output its bounding box. '
              'Return valid JSON only, no extra text, in the form '
              '{{"bbox_2d": [x1, y1, x2, y2]}} where (x1,y1) is the top-left and '
              '(x2,y2) the bottom-right corner.')

NUM = re.compile(r"[-+]?\d*\.?\d+")

# Dataset-specific config: pickle filename, split key used inside the pickle,
# and the image directory each dataset's file_name resolves against.
DATASET_CFG = {
    "refcoco":   {"pfile": "refs(unc).p", "img_dir": "train2014"},
    "refcoco+":  {"pfile": "refs(unc).p", "img_dir": "train2014"},
    "refcocog":  {"pfile": "refs(umd).p", "img_dir": "train2014"},
}


def downscale(pil, cap):
    w, h = pil.size
    s = cap / max(w, h)
    return pil if s >= 1.0 else pil.resize((max(1, int(w * s)), max(1, int(h * s))),
                                            Image.LANCZOS)


def data_uri(pil, quality=90):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_conversation(letters, task_text, image_uri):
    parts = []
    for c in letters:
        if c == "S":
            parts.append({"type": "text", "text": SYSTEM_MESSAGE})
        elif c == "T":
            parts.append({"type": "text", "text": task_text})
        elif c == "I":
            parts.append({"type": "image_url", "image_url": {"url": image_uri}})
    return [{"role": "user", "content": parts}]


def _order_box(x1, y1, x2, y2):
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _maybe_swap_yx(x1, y1, x2, y2, box_order):
    # See ordering_eval_vllm.py's _maybe_swap_yx docstring -- gemma-4-31b
    # emits "bbox_2d" as [y1, x1, y2, x2] regardless of the prompted order,
    # confirmed empirically by drawing predicted boxes on real images.
    if box_order == "yxyx":
        return y1, x1, y2, x2
    return x1, y1, x2, y2


def parse_box(text, box_order="xyxy"):
    t = text.strip().strip("`")
    t = re.sub(r"^json", "", t).strip()
    try:
        i, j = t.index("{"), t.rindex("}")
        d = json.loads(t[i:j + 1])
        b = d.get("bbox_2d") or d.get("bbox")
        if b and len(b) >= 4:
            x1, y1, x2, y2 = map(float, b[:4])
            x1, y1, x2, y2 = _maybe_swap_yx(x1, y1, x2, y2, box_order)
            return _order_box(x1, y1, x2, y2)
    except Exception:
        pass
    m = re.search(r'"bbox_2d"\s*:?\s*\[([^\]]+)\]', text)
    if m:
        nums = NUM.findall(m.group(1))
        if len(nums) >= 4:
            x1, y1, x2, y2 = map(float, nums[:4])
            x1, y1, x2, y2 = _maybe_swap_yx(x1, y1, x2, y2, box_order)
            return _order_box(x1, y1, x2, y2)
    return None


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_split(dataset, split):
    cfg = DATASET_CFG[dataset]
    refs = [r for r in pickle.load(open(f"{RC}/{dataset}/{cfg['pfile']}", "rb"))
            if r["split"] == split]
    inst = json.load(open(f"{RC}/{dataset}/instances.json"))
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {im["id"]: im for im in inst["images"]}
    return refs, ann, img, cfg["img_dir"]


def run_order(llm, sp, dataset, split, tag, letters, refs, ann, img, img_dir):
    out = os.path.join(OUT_DIR, f"{dataset}_{split}_order-{tag}_{MODEL_TAG}.json")
    if os.path.exists(out):
        try:
            m = json.load(open(out))["meta"]
            if m.get("complete"):
                print(f"  [{dataset}/{split}/{tag}] cached (acc={m.get('acc'):.3f})",
                      flush=True)
                return
        except Exception:
            pass

    convs, meta = [], []
    for r in refs:
        gt = ann[r["ann_id"]]["bbox"]
        im_info = img[r["image_id"]]
        W, H = im_info["width"], im_info["height"]
        phrase = r["sentences"][0]["sent"]
        path = os.path.join(RC, img_dir, im_info["file_name"])
        if not os.path.isfile(path):
            continue
        pil = downscale(Image.open(path).convert("RGB"), IMG_CAP)
        convs.append(build_conversation(letters, REF_PROMPT.format(phrase=phrase),
                                        data_uri(pil)))
        meta.append({"ref_id": r["ref_id"], "gt": gt, "W": W, "H": H})

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, correct, parsed = [], 0, 0
    for o, m in zip(outputs, meta):
        text = o.outputs[0].text.strip()
        box = parse_box(text, box_order=MODEL_REGISTRY[MODEL_TAG].get("box_order", "xyxy"))
        ok = False
        if box is not None:
            parsed += 1
            x1, y1, x2, y2 = box
            cs = MODEL_REGISTRY[MODEL_TAG]["coord_scale"]
            assert cs is not None, (
                f"{MODEL_TAG} has no verified coord_scale -- fill in "
                "MODEL_REGISTRY before running grounding/detection with it")
            pred = [x1 / cs * m["W"], y1 / cs * m["H"],
                    (x2 - x1) / cs * m["W"], (y2 - y1) / cs * m["H"]]
            ok = iou_xywh(pred, m["gt"]) >= 0.5
        correct += int(ok)
        results.append({"ref_id": m["ref_id"], "ok": ok, "parsed": box is not None})

    n = len(results)
    meta_out = {"benchmark": dataset, "split": split, "ordering": tag,
                "model": MODEL_TAG, "engine": MODEL_REGISTRY[MODEL_TAG]["engine"],
                "metric": "ref_acc_iou0.5",
                "n": n, "parsed": parsed, "correct": correct,
                "acc": correct / max(1, n), "runtime_s": dt, "complete": True}
    json.dump({"meta": meta_out, "results": results}, open(out, "w"), indent=2)
    print(f"  [{dataset}/{split}/{tag}] n={n} acc={meta_out['acc']*100:.1f}% "
          f"({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="refcoco,refcoco+")
    ap.add_argument("--splits", default="testA,testB")
    ap.add_argument("--orders", default="STI,SIT,STIT,SITIT,STITI")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1
    cfg = MODEL_REGISTRY[args.model]

    from vllm import SamplingParams
    if cfg["engine"] == "local_hf":
        llm = load_hf_chat_engine(args.model)
    else:
        from vllm import LLM
        llm_kwargs = dict(model=cfg["hf"], trust_remote_code=True,
                          max_model_len=24096, tensor_parallel_size=args.tp,
                          gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 2})
        if cfg["quantization"]:
            llm_kwargs["quantization"] = cfg["quantization"]
            llm_kwargs["load_format"] = cfg["quantization"]
        else:
            llm_kwargs["dtype"] = "float16"
        llm = LLM(**llm_kwargs)
    # 64 was fine for Qwen (terse compliance) but risks truncating Gemma-3-27B
    # before its JSON, given the same verbose-preamble behavior confirmed on
    # BLINK/MMVP (see extra_tasks/mmvp_eval_vllm.py's max_tokens comment).
    sp = SamplingParams(temperature=0.0, max_tokens=150)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    for dataset in datasets:
        for split in splits:
            refs, ann, img, img_dir = load_split(dataset, split)
            print(f"[data] {dataset}/{split}: {len(refs)} refs", flush=True)
            for tag in orders:
                if tag not in ORDER_LETTERS:
                    print(f"  skip {tag} (not hook-free)"); continue
                run_order(llm, sp, dataset, split, tag, ORDER_LETTERS[tag],
                          refs, ann, img, img_dir)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
