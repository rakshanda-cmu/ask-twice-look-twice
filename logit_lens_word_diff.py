"""
STI vs IST logit-lens word-diff (item 6 of the resolution-sweep/GEPA/logit-lens
task set).

For a sample of (image, question) pairs, runs the model's ANSWER generation
under both STI (System-Task-Image) and IST (Image-System-Task) orderings,
extracting the top-1 predicted word at each (layer, generation-step) position
via logit_lens_overlay.compute_text_logit_lens's 'generated' section -- the
one part of the existing logit-lens machinery that is directly comparable
1-to-1 between the two orderings (same greedy decoding, same left-to-right
generation steps), unlike the 'before'/'after' INPUT-text sections, whose
token content and position differ entirely between STI's and IST's prompt
layouts (there is no "layer, position" pair spanning the two orderings that
means the same input token, since the whole sequence is reshuffled) -- so
only 'generated' supports a meaningful position-aligned diff.

Reports only genuine word CHANGES: per (layer, step), if STI's and IST's
top-1 predicted word are identical OR synonyms (via WordNet synsets), it is
not a real behavioral difference and is dropped. Only surviving word PAIRS
(genuinely different words, not spelling/synonym variants) are kept.

Output: ./logit_lens_word_diff_results.json (new file; no existing result
touched). A per-example list of surviving (layer, step, STI_word, IST_word)
diffs, plus an aggregate "which layers diverge most" summary.

Needs the GPU (runs the actual model forward passes via model_manager's
ModelManager, the same machinery the Streamlit logit-lens page uses).

Run:
  CUDA_VISIBLE_DEVICES=0 /home/grg/anaconda3/envs/soft-prompt/bin/python \
    logit_lens_word_diff.py --n-samples 20 --model qwen3-vl-8b
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _normalize(w):
    return re.sub(r"^\W+|\W+$", "", (w or "").strip().lower())


def is_same_or_synonym(w1, w2):
    """True if w1/w2 are identical (after normalizing case/punctuation) or
    are WordNet synonyms of each other -- either direction counts, since
    lemma sets aren't symmetric for near-synonyms."""
    from nltk.corpus import wordnet as wn
    a, b = _normalize(w1), _normalize(w2)
    if not a or not b:
        return a == b
    if a == b:
        return True
    syns_a = {l.name().lower().replace("_", " ") for s in wn.synsets(a) for l in s.lemmas()}
    syns_b = {l.name().lower().replace("_", " ") for s in wn.synsets(b) for l in s.lemmas()}
    return b in syns_a or a in syns_b


def diff_generated_words(words_sti, words_ist, layer_range):
    """words_*: list[list[str]], shape (n_layers, n_gen_tokens) from
    compute_text_logit_lens()'s 'generated'/'words'. Returns a list of
    surviving diffs (genuine word changes, synonyms filtered out)."""
    n_layers = min(len(words_sti), len(words_ist))
    diffs = []
    for li in range(n_layers):
        row_sti, row_ist = words_sti[li], words_ist[li]
        n_steps = min(len(row_sti), len(row_ist))
        for si in range(n_steps):
            w_sti, w_ist = row_sti[si], row_ist[si]
            if not is_same_or_synonym(w_sti, w_ist):
                diffs.append({"layer": layer_range[li], "step": si,
                             "sti_word": w_sti, "ist_word": w_ist})
    return diffs


def run_one(mm, img, query, system_prompt, layer_range, top_k, max_tokens):
    """Runs STI and IST, returns (answer_sti, answer_ist, diffs)."""
    import torch
    from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper
    from logit_lens_overlay import compute_text_logit_lens

    logits_warper = TopKLogitsWarper(top_k=top_k, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])

    def _run(order):
        _, input_ids, kwargs = mm.prepare_inputs_from_pil(
            [query], img, system_prompt=system_prompt, order=order)
        with torch.inference_mode():
            outputs = mm.llm_model.generate(
                input_ids, do_sample=False, num_beams=1,
                max_new_tokens=max_tokens, use_cache=True,
                output_hidden_states=True, return_dict_in_generate=True,
                **kwargs)
        gen = outputs["sequences"][:, input_ids.shape[1]:]
        answer = mm.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        text_data = compute_text_logit_lens(
            mm.llm_model, mm.tokenizer, input_ids, outputs,
            mm.img_start_idx, mm.img_end_idx, layer_range,
            logits_warper, logits_processor)
        return answer, text_data["generated"]["words"]

    ans_sti, words_sti = _run("STI")
    ans_ist, words_ist = _run("IST")
    diffs = diff_generated_words(words_sti, words_ist, layer_range)
    return ans_sti, ans_ist, diffs


def load_samples(n, seed=0):
    """Reuses RF20's already-downloaded images (broad visual variety across 20
    datasets, no new download needed -- confirmed working in
    token_cost_analysis.py) -- just (image, one question) pairs, no GT
    needed since this is a mechanistic probe, not an accuracy eval."""
    import random
    from PIL import Image
    from rf20_eval import load_rf20_samples, _question
    rows = load_rf20_samples(max_images_per_dataset=5)
    random.Random(seed).shuffle(rows)
    samples = []
    for r in rows:
        if len(samples) >= n:
            break
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            continue
        samples.append({"image": img, "query": _question(r["cls"])})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--n-samples", type=int, default=20, dest="n_samples")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,27",
                    help="comma-separated layer indices to probe")
    ap.add_argument("--top-k", type=int, default=50, dest="top_k")
    ap.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    ap.add_argument("--out", default="logit_lens_word_diff_results.json")
    args = ap.parse_args()
    layer_range = [int(x) for x in args.layers.split(",")]

    from model_manager import ModelManager
    from utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hf_logging
    from constants import SYSTEM_MESSAGE
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    print(f"[data] loading {args.n_samples} samples …", flush=True)
    samples = load_samples(args.n_samples)
    print(f"[load] model = {args.model}", flush=True)
    mm = ModelManager(args.model)

    per_example, layer_counts = [], {l: 0 for l in layer_range}
    for i, s in enumerate(samples):
        ans_sti, ans_ist, diffs = run_one(mm, s["image"], s["query"], SYSTEM_MESSAGE,
                                          layer_range, args.top_k, args.max_tokens)
        for d in diffs:
            layer_counts[d["layer"]] += 1
        per_example.append({"query": s["query"], "answer_sti": ans_sti,
                            "answer_ist": ans_ist, "diffs": diffs})
        print(f"  [{i+1}/{len(samples)}] \"{s['query'][:50]}\" "
              f"STI=\"{ans_sti[:30]}\" IST=\"{ans_ist[:30]}\" "
              f"{len(diffs)} genuine word-diffs", flush=True)

    total_diffs = sum(len(e["diffs"]) for e in per_example)
    out = {"model": args.model, "n_samples": len(per_example),
          "layer_range": layer_range, "max_tokens": args.max_tokens,
          "total_genuine_diffs": total_diffs,
          "diffs_by_layer": layer_counts,
          "per_example": per_example}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[done] {total_diffs} genuine word-diffs across {len(per_example)} "
          f"examples. By layer: {layer_counts}", flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
