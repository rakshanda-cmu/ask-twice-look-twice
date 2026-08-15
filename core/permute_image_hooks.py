"""
Model-agnostic "randomly permute the 2nd image block's patches" hooks, for the
SITIT-permute control. This is the same manipulation as the SITIT-reverse ablation
(reverse_image_hooks.py) except the 2nd image block is reordered by a fixed random
permutation instead of reversed. Each patch keeps its true content/position pairing
(features and 2D M-RoPE positions are permuted by the SAME permutation), so only the
causal sequence order of the 2nd copy is scrambled.

Purpose: reverse is one specific reordering. If a random reordering does as well as
reverse (and both track plain forward repetition), then the gain is repetition
(whole-image context), not the reversed scan order or 1D locality. Toggle PERMUTE['on'].

Qwen3-VL only (the causal-decoder case the ablation is about).
"""
import torch

PERMUTE = {"on": False}
_SEED = 1234


def _perm(length, device):
    """Deterministic permutation of `length`, seeded by length (so the feature and
    position hooks produce the SAME permutation within one forward pass)."""
    g = torch.Generator().manual_seed(_SEED * 100003 + int(length))
    return torch.randperm(int(length), generator=g).to(device)


def _install_qwen(mm):
    base = mm.llm_model.model
    orig_feat = base.get_image_features
    orig_rope = base.get_rope_index
    merge = base.visual.spatial_merge_size ** 2

    def perm2(seq):
        if seq is None or len(seq) < 2:
            return seq
        lst = list(seq)
        t = lst[1]
        lst[1] = t[_perm(t.shape[0], t.device)]
        return tuple(lst) if isinstance(seq, tuple) else lst

    def patched_feat(*a, **k):
        out = orig_feat(*a, **k)
        if PERMUTE["on"]:
            grid = k.get("image_grid_thw", a[1] if len(a) > 1 else None)
            out.pooler_output = perm2(out.pooler_output)
            ds = getattr(out, "deepstack_features", None)
            if ds is not None and grid is not None:
                sizes = (grid.prod(-1) // merge).tolist()
                new = []
                for layer in ds:
                    if torch.is_tensor(layer):
                        parts = list(torch.split(layer, sizes))
                        if len(parts) >= 2:
                            p = parts[1]
                            parts[1] = p[_perm(p.shape[0], p.device)]
                        new.append(torch.cat(parts, dim=0))
                    else:
                        new.append(perm2(layer))
                out.deepstack_features = type(ds)(new)
        return out

    def patched_rope(*a, **k):
        pos, delta = orig_rope(*a, **k)
        if PERMUTE["on"]:
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
                        p = _perm(e - s + 1, pos.device)
                        pos[:, :, s:e + 1] = pos[:, :, s:e + 1][:, :, p]
        return pos, delta

    base.get_image_features = patched_feat
    base.get_rope_index = patched_rope


def _install_gemma(mm):
    """Gemma-3: permute the 2nd image's projected tokens (mirrors the reverse hook,
    which flips them). Gemma uses plain 1D positions and no DeepStack, so only the
    pooler tokens are reordered."""
    base = mm.llm_model.model
    orig_feat = base.get_image_features

    def patched_feat(*a, **k):
        out = orig_feat(*a, **k)
        if PERMUTE["on"]:
            po = out.pooler_output
            if torch.is_tensor(po) and po.shape[0] >= 2:
                po = po.clone()
                po[1] = po[1][_perm(po[1].shape[0], po[1].device)]
                out.pooler_output = po
        return out

    base.get_image_features = patched_feat


def install_permute_hooks(mm):
    name = mm.model_name.lower()
    if "qwen" in name:
        _install_qwen(mm)
    elif "gemma" in name:
        _install_gemma(mm)
    else:
        raise ValueError(f"permute hooks not implemented for {name}")
    return PERMUTE
