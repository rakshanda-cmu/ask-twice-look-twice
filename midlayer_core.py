"""
Core logic for the Middle-Layer Analysis page.

For a selected NaturalBench (image, question) pair we run the logit lens under
BOTH prompt orderings (IST and STI), save every layer frame to disk (so the UI
drill-down is instant), and track the examples in a manifest with a FIFO buffer
of 5 per scenario (IST✓/STI✗ and STI✓/IST✗).

Shared by:
  • precompute_midlayer.py  (batch-run the curated examples, CLI)
  • midlayer_browser.py      (Streamlit page: view + add new examples)
"""

import json
import os
import time

from constants import SYSTEM_MESSAGE
from naturalbench_eval import PAIRS, answer_suffix  # noqa: F401 (PAIRS for ref)

NB_DIR = "./naturalbench"
RESULTS_DIR = os.path.join(NB_DIR, "results")
MID_DIR = os.path.join(NB_DIR, "midlayer")
MANIFEST = os.path.join(MID_DIR, "manifest.json")

ORDERS = ("IST", "STI")
VARIANTS = ("normal", "space")   # normal = Tab 3, space = Tab 4 (20 space tokens)
N_SPACES = 20
BUFFER_PER_SCENARIO = 45   # FIFO cap per scenario for UI-added examples

# scenario keys
SCEN_A = "IST_right_STI_wrong"   # IST ✓ / STI ✗
SCEN_B = "STI_right_IST_wrong"   # STI ✓ / IST ✗
SCEN_BOTH_RIGHT = "both_correct"
SCEN_BOTH_WRONG = "both_wrong"

SCENARIO_LABEL = {
    SCEN_A: "IST ✓ / STI ✗",
    SCEN_B: "STI ✓ / IST ✗",
    SCEN_BOTH_RIGHT: "Both ✓",
    SCEN_BOTH_WRONG: "Both ✗",
}


def scenario_of(ist_correct, sti_correct):
    if ist_correct and not sti_correct:
        return SCEN_A
    if sti_correct and not ist_correct:
        return SCEN_B
    if ist_correct and sti_correct:
        return SCEN_BOTH_RIGHT
    return SCEN_BOTH_WRONG


def example_code(group_index, image_index, question_index):
    return f"g{group_index}_i{image_index}_q{question_index}"


# ──────────────────────────────────────────────────────────────────────────────
#  Layers
# ──────────────────────────────────────────────────────────────────────────────

def get_layer_range(model_manager):
    """All transformer-block layers (0 .. n_layers-1)."""
    n = model_manager.num_layers
    return list(range(n))


# ──────────────────────────────────────────────────────────────────────────────
#  Building an example record from existing NaturalBench results
# ──────────────────────────────────────────────────────────────────────────────

def _results_path(model, order):
    return os.path.join(RESULTS_DIR, f"{model}__{order}__results.json")


def _load_results(model, order):
    p = _results_path(model, order)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run naturalbench_eval first.")
    with open(p) as f:
        return json.load(f)["results"]


def build_example_record(model, group_index, image_index, question_index):
    """
    Look up a (group, image, question) pair in the IST & STI result files and
    return a metadata dict (question text, gold answer, each ordering's answer,
    scenario). Raises if not found.
    """
    ist = {g["index"]: g for g in _load_results(model, "IST")}
    sti = {g["index"]: g for g in _load_results(model, "STI")}
    gi, gs = ist.get(group_index), sti.get(group_index)
    if gi is None or gs is None:
        raise ValueError(f"Group {group_index} not found in results.")

    def find_pair(g):
        for p in g["pairs"]:
            if p["image_index"] == image_index and p["question_index"] == question_index:
                return p
        raise ValueError(f"Pair i{image_index} q{question_index} not in group {group_index}.")

    pi, ps = find_pair(gi), find_pair(gs)
    scen = scenario_of(pi["correct"], ps["correct"])
    return {
        "code": example_code(group_index, image_index, question_index),
        "model": model,
        "group_index": group_index,
        "image_index": image_index,
        "question_index": question_index,
        "image_rel": gi[f"image_{image_index}"],
        "question": pi["question"],
        "question_type": gi["question_type"],
        "source": gi["source"],
        "expected": pi["gt_answer"],
        "ist_answer": pi["model_answer_raw"],
        "sti_answer": ps["model_answer_raw"],
        "ist_correct": pi["correct"],
        "sti_correct": ps["correct"],
        "scenario": scen,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Running the logit lens for an example and saving frames
# ──────────────────────────────────────────────────────────────────────────────

def _variant_dir(code, variant, order):
    return os.path.join(MID_DIR, code, variant, order)


def decode_readable(tokenizer, ids, space_id=220):
    """
    Decode a token id sequence into a human-readable prompt string: image-pad
    runs are collapsed to ⟨N×image_pad⟩ and runs of space tokens are shown as ␣.
    Special tokens are kept (e.g. <|im_start|>) so the exact structure is visible.
    """
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    try:
        img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    except Exception:
        img_pad = None
    out, i, n = [], 0, len(ids)
    while i < n:
        t = ids[i]
        if img_pad is not None and t == img_pad:
            j = i
            while j < n and ids[j] == img_pad:
                j += 1
            out.append(f"⟨{j - i}×image_pad⟩")
            i = j
        elif t == space_id:
            j = i
            while j < n and ids[j] == space_id:
                j += 1
            out.append("␣" * (j - i))
            i = j
        else:
            out.append(tokenizer.decode([t]))
            i += 1
    return "".join(out)


def capture_prompt_info(model_manager, rec):
    """
    Build the exact inputs for both variants × orderings (no generation/render)
    and return a structured record of *what is passed to the model*:
    decoded prompt, sequence length, #vision tokens, grid, #spaces.
    """
    from PIL import Image
    img = Image.open(os.path.join(NB_DIR, rec["image_rel"])).convert("RGB")
    query = rec["question"] + answer_suffix(rec["question_type"])
    info = {}
    for variant in VARIANTS:
        info[variant] = {}
        for order in ORDERS:
            if variant == "space":
                _, ids, _ = model_manager.prepare_inputs_with_spaces(
                    [query], img, system_prompt=SYSTEM_MESSAGE, order=order,
                    n_spaces=N_SPACES)
            else:
                _, ids, _ = model_manager.prepare_inputs_from_pil(
                    [query], img, system_prompt=SYSTEM_MESSAGE, order=order)
            info[variant][order] = {
                "order": order,
                "seq_len": int(ids.shape[1]),
                "n_vision_tokens": int(model_manager.grid_h * model_manager.grid_w),
                "grid": f"{model_manager.grid_h}x{model_manager.grid_w}",
                "n_spaces": N_SPACES if variant == "space" else 0,
                "system_prompt": SYSTEM_MESSAGE,
                "task_prompt": query,
                "decoded_prompt": decode_readable(model_manager.tokenizer, ids[0]),
            }
    return info


def process_example(model_manager, rec, resolution=640, alpha=0.55,
                    max_tokens=16, top_k=50, progress_cb=None, ts=None,
                    variants=VARIANTS, show_text_lens=True):
    """
    Run the logit lens for `rec` under both orderings AND both variants:
      - 'normal' : standard prompt
      - 'space'  : 20 space tokens after Task (IST) / image (STI)
    Renders include the full token grid (show_text_lens). Saves per-layer PNGs,
    a final.png and anim.gif under naturalbench/midlayer/<code>/<variant>/<order>/.
    Returns rec augmented with dirs[variant][order] + answers_generated.
    """
    import imageio
    import numpy as np
    from PIL import Image
    from logit_lens_runner import run_logit_lens

    layer_range = get_layer_range(model_manager)
    code = rec["code"]
    img = Image.open(os.path.join(NB_DIR, rec["image_rel"])).convert("RGB")
    query = rec["question"] + answer_suffix(rec["question_type"])

    rec = dict(rec)
    rec["dirs"] = {v: {} for v in variants}
    rec["answers_generated"] = {v: {} for v in variants}
    rec["final_layer"] = layer_range[-1]
    rec["num_layers"] = len(layer_range)
    rec["n_spaces"] = N_SPACES
    rec["ts"] = ts if ts is not None else time.time()

    for variant in variants:
        n_spaces = N_SPACES if variant == "space" else 0
        for order in ORDERS:
            if progress_cb:
                progress_cb(f"{code}: {variant}/{order} …")
            answer, frames = run_logit_lens(
                model_manager, img, query,
                system_prompt=SYSTEM_MESSAGE, order=order,
                layer_range=layer_range, top_k=top_k, max_tokens=max_tokens,
                resolution=resolution, alpha=alpha,
                show_text_lens=show_text_lens, n_spaces=n_spaces,
            )
            odir = _variant_dir(code, variant, order)
            os.makedirs(odir, exist_ok=True)
            for li, frame in zip(layer_range, frames):
                frame.save(os.path.join(odir, f"layer_{li:02d}.png"))
            frames[-1].save(os.path.join(odir, "final.png"))
            imageio.mimsave(
                os.path.join(odir, "anim.gif"),
                [np.array(f) for f in frames],
                format="GIF", duration=400, loop=0,
            )
            rec["dirs"][variant][order] = os.path.relpath(odir, ".")
            rec["answers_generated"][variant][order] = answer

    return rec


# ──────────────────────────────────────────────────────────────────────────────
#  Manifest + FIFO buffer
# ──────────────────────────────────────────────────────────────────────────────

def _img_ok(path):
    """True if the file exists and is a readable image (not a partial write)."""
    if not os.path.exists(path):
        return False
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def is_example_complete(code, num_layers, variants=VARIANTS):
    """True if every variant × ordering is fully and validly rendered on disk."""
    for variant in variants:
        for order in ORDERS:
            odir = _variant_dir(code, variant, order)
            if not os.path.isdir(odir):
                return False
            layers = [f for f in os.listdir(odir)
                      if f.startswith("layer_") and f.endswith(".png")]
            if len(layers) < num_layers:
                return False
            # validate the final frame and the gif are readable (catch the
            # case where a kill truncated the last write -> corrupt file)
            if not _img_ok(os.path.join(odir, "final.png")):
                return False
            if not _img_ok(os.path.join(odir, "anim.gif")):
                return False
    return True


def attach_existing(rec, num_layers, ts=None, variants=VARIANTS):
    """
    Build manifest fields for an example already rendered on disk (resume),
    without re-running the model. Generated answers fall back to the recorded
    experiment answers for the normal variant.
    """
    rec = dict(rec)
    rec["dirs"] = {v: {o: _variant_dir(rec["code"], v, o) for o in ORDERS}
                   for v in variants}
    rec["final_layer"] = num_layers - 1
    rec["num_layers"] = num_layers
    rec["n_spaces"] = N_SPACES
    rec["answers_generated"] = {
        v: {"IST": rec["ist_answer"], "STI": rec["sti_answer"]} for v in variants
    }
    rec["ts"] = ts if ts is not None else time.time()
    return rec


def load_manifest():
    if not os.path.exists(MANIFEST):
        return {"examples": []}
    with open(MANIFEST) as f:
        return json.load(f)


def save_manifest(man):
    os.makedirs(MID_DIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)


def _delete_example_dir(code):
    import shutil
    d = os.path.join(MID_DIR, code)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


def add_example(man, rec, buffer_per_scenario=BUFFER_PER_SCENARIO):
    """
    Add `rec` to the manifest, replacing any existing example with the same code,
    and FIFO-evicting the oldest example in the same scenario if the buffer for
    that scenario exceeds the limit. Mutates and returns `man`.
    """
    exs = [e for e in man["examples"] if e["code"] != rec["code"]]
    # also remove the on-disk dir of a same-code re-run handled by process_example
    exs.append(rec)

    # FIFO eviction within this scenario (only buffer A and B)
    scen = rec["scenario"]
    if scen in (SCEN_A, SCEN_B):
        same = [e for e in exs if e["scenario"] == scen]
        same_sorted = sorted(same, key=lambda e: e.get("ts", 0))
        while len(same_sorted) > buffer_per_scenario:
            oldest = same_sorted.pop(0)
            exs = [e for e in exs if e["code"] != oldest["code"]]
            _delete_example_dir(oldest["code"])

    man["examples"] = exs
    return man
