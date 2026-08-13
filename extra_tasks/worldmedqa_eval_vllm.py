"""
WorldMedQA-V ordering ablation via the vLLM library.

WorldMedQA-V (He et al. 2024) is a multilingual/multimodal medical exam QA
benchmark (Brazil, Israel, Japan, Spain), each question paired with a
clinical image (ECG, X-ray, skin lesion, etc.) and 4 lettered options. This
run uses the **English-translated** question text for all 4 countries only
(the "_local" native-language TSVs are skipped -- this repo's other harnesses
are English-only, and mixing untranslated prompts would confound the
ordering comparison with a language effect).

Source: WorldMedQA/V (HF), {country}_english_processed.tsv. Ground truth is
the "correct_option" column ("answer" is frequently NaN in this release --
verified empirically, not used). Images are base64-encoded strings.
Metric: multiple-choice exact match on the "(X)" letter, pooled + per-country.

Run:
  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python extra_tasks/worldmedqa_eval_vllm.py \
      --orders STI,SIT,STIT,SITIT
"""
import argparse, base64, io, json, os, re, sys, time

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
from PIL import Image
from common import (ORDER_LIST, ORDER_LETTERS, MODEL_TAG, downscale, data_uri,
                    build_conversation, make_llm)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
IMG_CAP = 1024
COUNTRIES = ["brazil", "israel", "japan", "spain"]
MC_SUFFIX = "\nAnswer with the option letter in parentheses, e.g. (A)."
LETTER_PAREN = re.compile(r"\(([A-Za-z])\)")


def load_worldmedqa():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    for country in COUNTRIES:
        p = hf_hub_download("WorldMedQA/V", f"{country}_english_processed.tsv",
                            repo_type="dataset")
        df = pd.read_csv(p, sep="\t")
        for _, r in df.iterrows():
            gt = r.get("correct_option")
            if gt is None or str(gt) == "nan" or r.get("image") is None:
                continue
            opts = "\n".join(f"({L}) {r[L]}" for L in ("A", "B", "C", "D")
                             if r.get(L) is not None and str(r[L]) != "nan")
            rows.append({"index": int(r["index"]), "country": country,
                        "question": f"{r['question']}\n{opts}",
                        "answer": str(gt).strip(), "image_b64": r["image"]})
    return rows


def parse_letter(text):
    m = LETTER_PAREN.search(text.strip())
    return m.group(1).upper() if m else None


def run_order(llm, sp, tag, letters, rows, log_every):
    out = os.path.join(OUT_DIR, f"worldmedqa_order-{tag}_{MODEL_TAG}.json")
    if os.path.exists(out):
        try:
            m = json.load(open(out))["meta"]
            if m.get("complete"):
                print(f"  [{tag}] cached (acc={m.get('accuracy'):.3f})", flush=True)
                return
        except Exception:
            pass

    convs = []
    for r in rows:
        img_bytes = base64.b64decode(r["image_b64"])
        pil = downscale(Image.open(io.BytesIO(img_bytes)).convert("RGB"), IMG_CAP)
        task = r["question"] + MC_SUFFIX
        convs.append(build_conversation(letters, task, data_uri(pil)))

    t0 = time.time()
    outputs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0

    results, n_correct, by_country = [], 0, {}
    for o, r in zip(outputs, rows):
        text = o.outputs[0].text.strip()
        pred = parse_letter(text)
        gt = r["answer"].upper()
        correct = pred == gt
        n_correct += int(correct)
        bc = by_country.setdefault(r["country"], {"n": 0, "correct": 0})
        bc["n"] += 1; bc["correct"] += int(correct)
        results.append({"index": r["index"], "country": r["country"], "gt": gt,
                        "pred": pred, "raw": text, "correct": correct})
        if len(results) % log_every == 0:
            print(f"    [{tag}] {len(results)}/{len(convs)} "
                  f"acc={n_correct/len(results):.3f}", flush=True)

    by_country_summary = {c: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                          for c, v in by_country.items()}
    meta = {"benchmark": "WorldMedQA-V", "ordering": tag, "model": MODEL_TAG,
            "engine": "vllm", "n": len(results),
            "accuracy": n_correct / max(1, len(results)), "by_country": by_country_summary,
            "runtime_s": dt, "complete": True}
    json.dump({"meta": meta, "results": results}, open(out, "w"), indent=2)
    print(f"  [{tag}] n={len(results)} acc={meta['accuracy']:.3f} "
          f"({dt:.0f}s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(ORDER_LIST))
    ap.add_argument("--model", default="qwen3-vl-8b",
                    choices=["qwen3-vl-8b", "gemma-3-27b", "gemma-4-31b"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem",
                    help="vLLM gpu_memory_utilization; lower this if another "
                         "process (e.g. the r1-1.5b server) already holds GPU0.")
    ap.add_argument("--log-every", type=int, default=50, dest="log_every")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    global MODEL_TAG
    MODEL_TAG = args.model
    if args.model in ("gemma-3-27b", "gemma-4-31b") and args.tp != 1:
        print(f"  [note] forcing --tp 1 for {args.model} (bnb 4-bit, single GPU only)")
        args.tp = 1

    rows = load_worldmedqa()
    print(f"[data] {len(rows)} WorldMedQA-V rows", flush=True)

    from vllm import SamplingParams
    llm = make_llm(tp=args.tp, model_tag=args.model, gpu_mem=args.gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    for tag in [o.strip() for o in args.orders.split(",") if o.strip()]:
        if tag not in ORDER_LIST:
            print(f"  skip {tag} (not hook-free)"); continue
        print(f"=== WorldMedQA-V · ordering {tag} ===", flush=True)
        run_order(llm, sp, tag, ORDER_LETTERS[tag], rows, args.log_every)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
