"""
RefCOCOg (umd) val referring-expression grounding eval via the DetPO vLLM server.

For each val referring expression we ask the served Qwen3-VL model for ONE bounding
box, scale it from Qwen3-VL's 0-1000 normalised space to pixels (same convention as
DetPO: x_abs = x/1000 * orig_dim), and score it correct if IoU with the ground-truth
box of the referred object is >= 0.5. Reports referring [email protected].

  /home/grg/anaconda3/envs/qwen-vllm-env/bin/python refcocog_eval.py \
      --model Qwen/Qwen3-VL-30B-A3B-Instruct --n 2573 \
      --out results/refcocog_val_refacc_qwen3vl-30b-a3b.json
"""
import argparse, base64, json, os, pickle, re, time
from io import BytesIO

from PIL import Image
from openai import OpenAI

RC = "/home/grg/Research/rf-20-vl-benchmark/datasets/RefCOCO"
REFS = os.path.join(RC, "refcocog", "refs(umd).p")
INSTANCES = os.path.join(RC, "refcocog", "instances.json")
IMG_DIR = os.path.join(RC, "train2014")            # RefCOCOg images live in COCO train2014


def data_url(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


BOX_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_box(text):
    """Pull the first 4 numbers (bbox_2d / [x1,y1,x2,y2]) from the model output."""
    m = re.search(r'"bbox_2d"\s*:?\s*\[([^\]]+)\]', text)
    frag = m.group(1) if m else text
    nums = BOX_RE.findall(frag)
    if len(nums) < 4:
        nums = BOX_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = map(float, nums[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


PROMPT = ('Locate "{phrase}" in the image and output its bounding box. '
          'Return valid JSON only, no extra text, in the form '
          '{{"bbox_2d": [x1, y1, x2, y2]}} where (x1,y1) is the top-left and '
          '(x2,y2) the bottom-right corner.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--server_url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0, help="0 = all refs in the split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = OpenAI(base_url=args.server_url, api_key="EMPTY")

    refs = pickle.load(open(REFS, "rb"))
    refs = [r for r in refs if r["split"] == args.split]
    inst = json.load(open(INSTANCES))
    ann_by_id = {a["id"]: a for a in inst["annotations"]}
    img_by_id = {im["id"]: im for im in inst["images"]}
    if args.n and args.n < len(refs):
        refs = refs[:args.n]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    per, correct, parsed, t0 = [], 0, 0, time.time()
    for i, r in enumerate(refs):
        gt = ann_by_id[r["ann_id"]]["bbox"]                 # [x,y,w,h] pixels
        im_info = img_by_id[r["image_id"]]
        path = os.path.join(IMG_DIR, im_info["file_name"])
        W, H = im_info["width"], im_info["height"]
        phrase = r["sentences"][0]["sent"]
        try:
            img = Image.open(path).convert("RGB")
            resp = client.chat.completions.create(
                model=args.model, temperature=0.0, max_tokens=128,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT.format(phrase=phrase)},
                    {"type": "image_url", "image_url": {"url": data_url(img)}}]}])
            out = resp.choices[0].message.content or ""
        except Exception as e:
            out = ""
            print(f"  [{i}] error: {e}", flush=True)
        box = parse_box(out)
        ok = False
        if box is not None:
            parsed += 1
            x1, y1, x2, y2 = box
            pred = [x1 / 1000.0 * W, y1 / 1000.0 * H,
                    (x2 - x1) / 1000.0 * W, (y2 - y1) / 1000.0 * H]
            ok = iou_xywh(pred, gt) >= 0.5
        correct += int(ok)
        per.append({"ref_id": r["ref_id"], "sent": phrase, "ok": ok, "raw": out[:200]})
        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(refs)}  [email protected]={acc:.3f}  parsed={parsed}/{i+1}  "
                  f"{rate:.2f} img/s", flush=True)
        if (i + 1) % 100 == 0 or i + 1 == len(refs):
            json.dump({"meta": _meta(args, refs, correct, parsed, i + 1),
                       "per_sample": per}, open(args.out, "w"), indent=2)

    meta = _meta(args, refs, correct, parsed, len(refs))
    json.dump({"meta": meta, "per_sample": per}, open(args.out, "w"), indent=2)
    print("[done]", json.dumps(meta), flush=True)


def _meta(args, refs, correct, parsed, n):
    return {"benchmark": "RefCOCOg", "variant": "umd", "split": args.split,
            "metric": "[email protected]", "model": args.model, "n": n,
            "n_total_split": len(refs), "parsed": parsed,
            "correct": correct, "acc": correct / n if n else 0.0}


if __name__ == "__main__":
    main()
