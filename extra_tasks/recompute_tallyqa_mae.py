"""
One-off: recompute TallyQA's MAE from the already-saved raw model text, applying
the same implausible-count filter now in parse_count() (PLAUSIBLE_MAX_COUNT=200),
without re-running generation. See tallyqa_eval_vllm.py's parse_count docstring
for why this is needed (a single degenerate "1000000000" output inflated the
naive MAE by ~6 orders of magnitude on the STI run).

Run: /home/grg/anaconda3/envs/logitlens/bin/python extra_tasks/recompute_tallyqa_mae.py
"""
import glob, json, os
from tallyqa_eval_vllm import parse_count

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

for path in sorted(glob.glob(os.path.join(RESULTS, "tallyqa_order-*.json"))):
    d = json.load(open(path))
    old_mae = d["meta"].get("mae")
    n_correct, abs_err_sum, n_parsed = 0, 0.0, 0
    for r in d["results"]:
        pred = parse_count(r["raw"])
        r["pred"] = pred
        r["correct"] = pred == r["gt"]
        n_correct += int(r["correct"])
        if pred is not None:
            n_parsed += 1
            abs_err_sum += abs(pred - r["gt"])
    new_mae = abs_err_sum / max(1, n_parsed)
    d["meta"]["mae"] = new_mae
    d["meta"]["parsed"] = n_parsed
    d["meta"]["accuracy"] = n_correct / max(1, len(d["results"]))
    json.dump(d, open(path, "w"), indent=2)
    tag = d["meta"]["ordering"]
    print(f"{tag}: mae {old_mae} -> {new_mae:.3f}  (parsed {n_parsed}/{len(d['results'])})")
