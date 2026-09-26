"""
Model-agnostic "reverse the 2nd image block's patches" hooks, for the SITIT-reverse
(S·I·T·Ī·T) ablation. Toggle with REVERSE['on'].

  Qwen3-VL : reverse the 2nd image's pooler embeds + DeepStack features + 2D M-RoPE
             positions together (each patch keeps its true content/position, read
             back-to-front). Validated to reproduce stock output when off.
  Gemma-3  : reverse the 2nd image's projected tokens. Gemma uses plain 1D positions
             and has no DeepStack, so nothing else needs reversing.

get_image_features()'s return shape is both transformers-version- and
architecture-dependent: older versions returned a ModelOutput-style object
(`.pooler_output`/`.deepstack_features` attributes); the version this repo
now runs against returns plain tuples/tensors instead --
Qwen3VLModel.get_image_features -> `(image_embeds, deepstack_image_embeds)`
where image_embeds is itself a per-image-split tuple (DeepStack is a
Qwen3-VL-only architecture feature); Qwen2_5_VLModel.get_image_features ->
just the per-image-split tuple directly, no DeepStack wrapper at all (qwen2.5-vl-7b
has no `deepstack_merger_list` on its visual tower -- naively unpacking its
2-image output as `(image_embeds, deepstack) = out` silently assigns image 1's
embeds to `deepstack`, corrupting shapes downstream); Gemma3Model.get_image_features
-> a bare [num_images, tokens, hidden] tensor. All three shapes are handled
below so this survives either transformers API era or Qwen generation.

Nothing here modifies existing modules; the runners import install_reverse_hooks +
REVERSE and flip REVERSE['on'] = True.
"""
import torch

REVERSE = {"on": False}


def _install_qwen(mm):
    base = mm.llm_model.model
    orig_feat = base.get_image_features
    orig_rope = base.get_rope_index
    merge = base.visual.spatial_merge_size ** 2
    has_deepstack = hasattr(base.visual, "deepstack_merger_list")

    def flip2(seq):
        if seq is None or len(seq) < 2:
            return seq
        lst = list(seq); lst[1] = torch.flip(lst[1], dims=[0])
        return tuple(lst) if isinstance(seq, tuple) else lst

    def _flip_deepstack(ds, grid):
        if ds is None or grid is None:
            return ds
        sizes = (grid.prod(-1) // merge).tolist()
        new = []
        for layer in ds:
            if torch.is_tensor(layer):
                parts = list(torch.split(layer, sizes))
                if len(parts) >= 2:
                    parts[1] = torch.flip(parts[1], dims=[0])
                new.append(torch.cat(parts, dim=0))
            else:
                new.append(flip2(layer))
        return type(ds)(new)

    def patched_feat(*a, **k):
        out = orig_feat(*a, **k)
        if REVERSE["on"]:
            grid = k.get("image_grid_thw", a[1] if len(a) > 1 else None)
            is_bare_tuple = isinstance(out, tuple) and not hasattr(out, "pooler_output")
            if is_bare_tuple and has_deepstack:
                # Qwen3-VL: (image_embeds, deepstack_image_embeds)
                image_embeds, deepstack_image_embeds = out
                image_embeds = flip2(image_embeds)
                deepstack_image_embeds = _flip_deepstack(deepstack_image_embeds, grid)
                return image_embeds, deepstack_image_embeds
            if is_bare_tuple:
                # Qwen2.5-VL (and earlier): just the per-image-split tuple,
                # no DeepStack wrapper -- do NOT unpack as a 2-tuple.
                return flip2(out)
            out.pooler_output = flip2(out.pooler_output)
            if has_deepstack:
                out.deepstack_features = _flip_deepstack(
                    getattr(out, "deepstack_features", None), grid)
        return out

    def patched_rope(*a, **k):
        pos, delta = orig_rope(*a, **k)
        if REVERSE["on"]:
            mmids = k.get("mm_token_type_ids", a[1] if len(a) > 1 else None)
            if mmids is not None:
                idx = torch.nonzero((mmids == 1)[0], as_tuple=True)[0]
                if len(idx):
                    runs, start = [], idx[0].item()
                    for x, y in zip(idx[:-1].tolist(), idx[1:].tolist()):
                        if y != x + 1:
                            runs.append((start, x)); start = y
                    runs.append((start, idx[-1].item()))
                    if len(runs) >= 2:
                        s, e = runs[1]
                        pos[:, :, s:e + 1] = torch.flip(pos[:, :, s:e + 1], dims=[2])
        return pos, delta

    base.get_image_features = patched_feat
    base.get_rope_index = patched_rope


def _install_gemma(mm):
    base = mm.llm_model.model
    orig_feat = base.get_image_features

    def patched_feat(*a, **k):
        out = orig_feat(*a, **k)
        if REVERSE["on"]:
            # current transformers: bare [num_images, tokens_per_image, hidden]
            # tensor; older versions wrapped it as out.pooler_output.
            is_bare_tensor = torch.is_tensor(out)
            po = out if is_bare_tensor else out.pooler_output
            if torch.is_tensor(po) and po.shape[0] >= 2:
                po = po.clone()
                po[1] = torch.flip(po[1], dims=[0])   # reverse 2nd image's tokens
                if is_bare_tensor:
                    return po
                out.pooler_output = po
        return out

    base.get_image_features = patched_feat


def install_reverse_hooks(mm):
    name = mm.model_name.lower()
    if "qwen" in name:
        _install_qwen(mm)
    elif "gemma" in name:
        _install_gemma(mm)
    else:
        raise ValueError(f"reverse hooks not implemented for {name}")
    return REVERSE
