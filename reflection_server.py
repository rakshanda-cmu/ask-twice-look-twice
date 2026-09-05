"""
Standalone reflection-model server for gepa_baseline.py.

Original design flaw being fixed here: gepa_baseline.py used the SAME
Qwen3-VL-8B instance as BOTH task_lm and reflection_lm. This goes against
GEPA's own intended design (arxiv.org/abs/2507.19457) -- a genuinely
stronger/different reflector analyzing a cheaper task model's failures and
proposing improved prompts. Reusing the task model for reflection produced
consistently generic, boilerplate proposals that never won GEPA's
strict-improvement acceptance test in either POPE or VQA runs (both
converged back to ~the original SYSTEM_MESSAGE with ~0 accuracy delta).

Fix: load Gemma-3-27B (already used elsewhere in this repo, via vLLM's
normal path -- MODEL_REGISTRY marks it "engine": "vllm", not the bnb-4-bit
local_hf special case that's only needed for gemma-4-31b) as the reflector,
on its OWN GPU. This has to be a SEPARATE OS process from gepa_baseline.py:
vLLM's Qwen task-model engine and this Gemma reflector engine each want
their own physical GPU via CUDA_VISIBLE_DEVICES, and one Python process
can't cleanly give two vLLM/HF model instances two different CUDA_VISIBLE_
DEVICES scopes. Communication is simple file-based IPC (a request json in,
a response json out) rather than a real network server -- GEPA makes only a
handful of reflection calls per run (order of 10-20), so polling latency is
a non-issue, and this avoids adding an HTTP framework dependency.

Run (in the background, BEFORE launching gepa_baseline.py --reflection-model
gemma-3-27b):
  CUDA_VISIBLE_DEVICES=1 /home/grg/anaconda3/envs/qwen-vllm-env/bin/python \
    reflection_server.py --dir ./gepa_reflection_ipc
"""
import argparse
import glob
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./gepa_reflection_ipc")
    ap.add_argument("--model", default="gemma-3-27b")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85, dest="gpu_mem")
    ap.add_argument("--poll-interval", type=float, default=0.3, dest="poll_interval")
    args = ap.parse_args()

    req_dir = os.path.join(args.dir, "requests")
    resp_dir = os.path.join(args.dir, "responses")
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_tasks"))
    from common import make_llm
    from vllm import SamplingParams

    print(f"[load] {args.model} for reflection …", flush=True)
    llm = make_llm(tp=args.tp, model_tag=args.model, gpu_mem=args.gpu_mem)
    print(f"[ready] watching {req_dir} for reflection requests …", flush=True)

    seen = set()
    n_served = 0
    stop_path = os.path.join(args.dir, "STOP")
    while not os.path.exists(stop_path):
        for req_path in sorted(glob.glob(os.path.join(req_dir, "*.json"))):
            req_id = os.path.basename(req_path)[:-5]
            if req_id in seen:
                continue
            resp_path = os.path.join(resp_dir, f"{req_id}.json")
            if os.path.exists(resp_path):
                seen.add(req_id)
                continue
            try:
                req = json.load(open(req_path))
            except Exception:
                continue  # writer may still be mid-write; retry next poll
            messages = req["messages"]
            sp = SamplingParams(temperature=req.get("temperature", 0.7),
                                max_tokens=req.get("max_tokens", 1024))
            out = llm.chat([messages], sp, use_tqdm=False)[0]
            text = out.outputs[0].text.strip()
            resp = {"text": text, "tokens_in": len(out.prompt_token_ids),
                    "tokens_out": len(out.outputs[0].token_ids)}
            tmp = resp_path + ".tmp"
            json.dump(resp, open(tmp, "w"))
            os.rename(tmp, resp_path)  # atomic -- client never sees a partial write
            seen.add(req_id)
            n_served += 1
            print(f"  [served] {req_id} ({n_served} total)", flush=True)
        time.sleep(args.poll_interval)
    print("[stop] STOP file seen, shutting down", flush=True)


if __name__ == "__main__":
    main()
