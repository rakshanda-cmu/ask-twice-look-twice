"""
Run the DetPO baseline (zero-shot, default class descriptions) COCO-mAP evaluation
on the RF20-VL *Aerial* domain (wildfire-smoke, aerial-airport) against a running
vLLM server, and aggregate the per-dataset mAP into one result JSON for the website.

Requires the DetPO vLLM server up at --server_url. Run from anywhere; paths are
absolute.

  HF_HOME=/data2/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    /home/grg/anaconda3/envs/qwen-vllm-env/bin/python detpo_map/rf20_aerial_eval.py \
      --model Qwen3-VL-30B-A3B-Instruct
"""
import argparse, json, os, subprocess, sys

DETPO = "/home/grg/Research/DetPO"
ROOT = "/home/grg/Research/rf-20-vl-benchmark/datasets/rf100-vl-fsod"
DATA_INSTR = os.path.join(DETPO, "data_instr", "default", "README.dataset")
AERIAL = ["wildfire-smoke", "aerial-airport"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")


def dataset_info(ds):
    ann = json.load(open(os.path.join(ROOT, ds, "test", "_annotations.coco.json")))
    classes = [c["name"] for c in ann.get("categories", []) if c["name"].lower() != "none"]
    return len(ann["images"]), classes


def run_one(args, ds, work_dir):
    """Invoke DetPO run_evaluation for one dataset; return its COCO 'model' stats."""
    env = dict(os.environ)
    env["PYTHONPATH"] = DETPO + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", "/data2/hf_cache")
    env.setdefault("HF_HUB_CACHE", "/data2/hf_cache/hub")
    cmd = [sys.executable, "-m", "detpo.run_evaluation",
           "--model_name", args.model,
           "--root_path", ROOT,
           "--dataset_path", ds,
           "--data_instr_path", DATA_INSTR,
           "--output_dir", work_dir,
           "--server_url", args.server_url,
           # DetPO's eval generator iterates all_detections["ranking"], which is
           # None unless --rank_rescore is set (latent bug in the zero-shot path).
           # The baseline mAP we report is the raw-confidence "model" eval_type;
           # this flag only additionally populates the "ranking" eval_type.
           "--rank_rescore"]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=DETPO, env=env, check=True)
    ev = os.path.join(work_dir, "evaluations", "default", f"evaluation_{ds}.json")
    stats = json.load(open(ev)).get("model", [0.0] * 12)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--server_url", default="http://localhost:8000/v1")
    ap.add_argument("--datasets", default=",".join(AERIAL))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    work_dir = os.path.join(OUT_DIR, "_detpo_run", args.model.replace("/", "_"))
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    per, ms, m50s, m75s = {}, [], [], []
    for ds in datasets:
        n_img, classes = dataset_info(ds)
        stats = run_one(args, ds, work_dir)
        mAP, mAP50, mAP75 = stats[0] * 100, stats[1] * 100, stats[2] * 100
        per[ds] = {"mAP": mAP, "mAP50": mAP50, "mAP75": mAP75,
                   "classes": classes, "n_images": n_img}
        ms.append(mAP); m50s.append(mAP50); m75s.append(mAP75)
        print(f"  [{ds}] mAP={mAP:.1f}  [email protected]={mAP50:.1f}  [email protected]={mAP75:.1f}", flush=True)

    mean = {"mAP": sum(ms) / len(ms), "mAP50": sum(m50s) / len(m50s),
            "mAP75": sum(m75s) / len(m75s)} if ms else {}
    result = {"meta": {
        "benchmark": "RF20-VL — Aerial",
        "config": "baseline (default class descriptions, zero-shot)",
        "model": args.model, "per_dataset": per, "mean": mean}}
    out = os.path.join(OUT_DIR, f"rf20_aerial_baseline_{args.model.replace('/', '_')}.json")
    json.dump(result, open(out, "w"), indent=2)
    print("[done] wrote", out, "| aerial-mean mAP",
          f"{mean.get('mAP', 0):.1f}", flush=True)


if __name__ == "__main__":
    main()
