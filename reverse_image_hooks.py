"""
Model-agnostic "reverse the 2nd image block's patches" hooks, for the SITIT-reverse
(S·I·T·Ī·T) ablation. Toggle with REVERSE['on'].

  Qwen3-VL : reverse the 2nd image's pooler embeds + DeepStack features + 2D M-RoPE
             positions together (each patch keeps its true content/position, read
             back-to-front). Validated to reproduce stock output when off.
  Gemma-3  : reverse the 2nd image's projected tokens. Gemma uses plain 1D positions
             and has no DeepStack, so nothing else needs reversing.

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

    def flip2(seq):
        if seq is None or len(seq) < 2:
            return seq
        lst = list(seq); lst[1] = torch.flip(lst[1], dims=[0])
        return tuple(lst) if isinstance(seq, tuple) else lst

    def patched_feat(*a, **k):
        out = orig_feat(*a, **k)
        if REVERSE["on"]:
            grid = k.get("image_grid_thw", a[1] if len(a) > 1 else None)
            out.pooler_output = flip2(out.pooler_output)
            ds = getattr(out, "deepstack_features", None)
            if ds is not None and grid is not None:
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
                out.deepstack_features = type(ds)(new)
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
            po = out.pooler_output          # [num_images, tokens_per_image, hidden]
            if torch.is_tensor(po) and po.shape[0] >= 2:
                po = po.clone()
                po[1] = torch.flip(po[1], dims=[0])   # reverse 2nd image's tokens
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
