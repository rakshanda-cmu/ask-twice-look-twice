# Prompt templates & ordering construction (DetPO detection + RefCOCO grounding)

These are the exact templates used by `detpo_map/ordering_eval.py` for the
**Qwen3-VL-8B** prompt-ordering experiments (STI / SIT / STIT / SITIT / SITIT_rev).
The tokens rearranged by the orderings are **S** (system), **T** (task/question
text) and **I** (image).

> The **30B baseline** rows are produced by DetPO's own `run_evaluation` on the
> served Qwen3-VL-30B-A3B and use DetPO's native detection prompt, *not* the
> templates below. The templates here define the **8B ordering** rows.

---

## S — system message (the "S" token)

```
A chat between a curious user and an artificial intelligence assistant. The
assistant gives helpful, detailed, and polite answers to the human's questions.
```

Passed as the chat `system` message. Present in every ordering (all start with S).

---

## T — RF20-VL detection task (multi-class, one prompt per image)

Matches the DetPO paper protocol ("detect all target classes simultaneously").
`{classes}` is the comma-separated list of the dataset's class names; `{instr}` is
one `- <class>: <annotator instruction>` line per class, taken from
`DetPO/data_instr/default/README.dataset_<dataset>.json`.

```
Detect every object in the image that belongs to any of these classes: {classes}.
Output Requirements:
- Return valid JSON only. Do not include explanations or extra text.
- A single ranked list of detections sorted by confidence (highest first).
- At most 50 detections. If none, return an empty list [].
For each detection provide: "bbox_2d": [x1, y1, x2, y2] (top-left, bottom-right),
"label": exactly one of the class names above, "score": float in 0..1.
Per-class annotator guidance:
{instr}
Return a JSON list like [{"bbox_2d": [x1,y1,x2,y2], "label": "<class>", "score": 0.95}].
```

Predicted `label` strings are matched back to category ids (exact, then
alphanumeric-normalized). A detection whose label matches no class is dropped
(unless the dataset has a single class, in which case it is assigned to it).

---

## T — RefCOCOg referring-grounding task (one prompt per expression)

`{phrase}` is the referring expression (first sentence of each RefCOCOg umd
annotation).

```
Locate "{phrase}" in the image and output its bounding box. Return valid JSON only,
no extra text, in the form {"bbox_2d": [x1, y1, x2, y2]} where (x1,y1) is the
top-left and (x2,y2) the bottom-right corner.
```

---

## Ordering construction (how S / T / I are arranged)

Content parts are placed into a single chat `user` turn in the letter order of the
tag (system message always first, as the `system` role):

| Tag | Layout | Meaning |
|-----|--------|---------|
| STI | S · T · I | question-first |
| SIT | S · I · T | question-last |
| STIT | S · T · I · T | question echo (T before and after the image) |
| SITIT | S · I · T · I · T | image echo (image included twice) |
| SITIT_rev | S · I · T · Ī · T | image echo, **2nd image block reversed** |

- For **SITIT**, the *same* image is inserted twice (two `image` content parts).
- **SITIT_rev** uses the identical SITIT layout; the reversal is **not** a change to
  the prompt or a visual flip. `reverse_image_hooks.py` reverses the **second image
  block's vision patch order together with its 2D M-RoPE positions** inside the
  model's hidden states (validated to reproduce stock output when disabled). This
  is why the ordering runs on the local HF model rather than the vLLM server.

---

## Coordinates & decoding

- The model emits boxes as `[x1, y1, x2, y2]` normalized to **0–1000** (Qwen3-VL
  convention, confirmed in the DetPO supplement). They are converted to pixels by
  `x_px = x/1000 * W`, `y_px = y/1000 * H`.
- Images are downscaled so the longest side is ≤ **1024** px before inference.
- Decoding is **greedy** (`do_sample=False, num_beams=1`); `max_new_tokens` = 1024
  for detection, 96 for grounding.
- Model: `Qwen/Qwen3-VL-8B-Instruct` (local Hugging Face), correctness of the box
  metric: COCO **mAP** for RF20 detection, **referring [email protected]** (IoU ≥ 0.5) for
  RefCOCOg.
