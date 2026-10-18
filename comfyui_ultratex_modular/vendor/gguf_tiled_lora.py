"""Opt-in tiled FP32 LoRA materialization for GGUF Linear layers.

Each GGUF layer still dequantizes its original packed Q4 weight using its own
backend. Standard additive 2-D LoRA updates are applied in small FP32 row tiles
instead of allocating a dense FP32 delta for the full matrix.

ComfyUI's original get_weight is retained for unsupported formats, or whenever
UTX_TILED_LORA=0. No weights, camera logic, token counts or diffusion schedule change.
This monkeypatch is scoped to a single UltraTex sampling/probe context and restored
on all exits; ComfyUI normally serializes prompt execution. Not for concurrent runs.
"""
from __future__ import annotations

import contextlib
import logging
import os

import torch

_LOG = logging.getLogger("UltraTex")
_LOGGED_FALLBACKS = set()


def enabled():
    return os.environ.get("UTX_TILED_LORA", "1").strip().lower() not in ("0", "false", "off", "no")


def rows_per_tile():
    try:
        val = int(os.environ.get("UTX_LORA_TILE_ROWS", "4096"))
    except ValueError:
        val = 4096
    return max(256, min(val, 8192))


def _owner(model):
    """Find GGUF's actual GGMLLayer class, not an assumed import name."""
    for layer in model.modules():
        for cls in type(layer).__mro__:
            if cls.__name__ == "GGMLLayer" and callable(cls.__dict__.get("get_weight")):
                return cls
    return None


def _extract(tensor, device, dtype, patch_dtype):
    """Prevalidate EVERY patch before altering the packed GGUF execution path.

    Unsupported patch types, nonstandard transforms, DoRA, conv LoRA and offsets
    other than output rows return None and use GGUF/ComfyUI's original routine.
    """
    if patch_dtype is not None or dtype not in (torch.float16, torch.bfloat16):
        return None
    groups = getattr(tensor, "patches", None)
    if not groups:
        return None
    entries = []
    try:
        for patches, key in groups:
            for p in patches:
                if not isinstance(p, (tuple, list)) or len(p) != 5:
                    return None
                strength, adapter, model_strength, offset, function = p
                if float(model_strength) != 1.0 or function is not None:
                    return None
                if type(adapter).__name__ != "LoRAAdapter" or not hasattr(adapter, "weights"):
                    return None
                up, down, alpha, mid, dora, reshape = adapter.weights
                if mid is not None or dora is not None or reshape is not None:
                    return None
                if not (isinstance(up, torch.Tensor) and isinstance(down, torch.Tensor)
                        and up.ndim == down.ndim == 2 and up.shape[1] == down.shape[0]):
                    return None
                if offset is None:
                    start, length = 0, up.shape[0]
                else:
                    if not (isinstance(offset, (tuple, list)) and len(offset) == 3 and
                            offset[0] == 0 and int(offset[1]) >= 0):
                        return None
                    start, length = int(offset[1]), int(offset[2])
                if start + length > tensor.shape[0] or up.shape[0] != length or down.shape[1] != tensor.shape[1]:
                    return None
                factor = float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
                entries.append((up, down, factor, start, length))
    except (ValueError, TypeError, AttributeError, IndexError):
        return None
    return entries if entries else None


def tiled_get_weight(self, tensor, dtype, original, *, tile_rows=None, metrics=None):
    """Call-compatible GGMLLayer.get_weight replacement (only compatible LoRAs)."""
    patches = _extract(tensor, tensor.device, dtype, getattr(self, "patch_dtype", None))
    if not patches:
        if metrics is not None and getattr(tensor, "patches", None):
            metrics["fallback_casts"] += 1
        return original(self, tensor, dtype)

    # Resolve the EXACT installed GGUF dequant function, without reimplementing Q4.
    dequantize = original.__globals__.get("dequantize_tensor")
    ggml_tensor = original.__globals__.get("GGMLTensor")
    if not callable(dequantize):
        if metrics is not None:
            metrics["fallback_casts"] += 1
        return original(self, tensor, dtype)

    # Pre-transfer the small LoRA factors; keep these out of the timed dense cast.
    import comfy.model_management as mm
    factors = []
    for up, down, factor, start, length in patches:
        up32 = mm.cast_to_device(up, tensor.device, torch.float32)
        down32 = mm.cast_to_device(down, tensor.device, torch.float32)
        factors.append((up32, down32, factor, start, length))

    weight = dequantize(tensor, dtype, self.dequant_dtype)
    if ggml_tensor is not None and isinstance(weight, ggml_tensor):
        weight = torch.Tensor(weight)
    # If the installed backend returns an unexpected shape/dtype, use its native code.
    if (weight.ndim != 2 or tuple(weight.shape) != tuple(tensor.shape)
            or weight.dtype not in (torch.float16, torch.bfloat16)):
        if metrics is not None:
            metrics["fallback_casts"] += 1
        del weight, factors
        return original(self, tensor, dtype)

    rows = rows_per_tile() if tile_rows is None else int(tile_rows)
    # FP32 matmul + FP32 scaling -> cast to weight dtype before add, the same
    # numerical sequence as ComfyUI's LoRAAdapter.calculate_weight.
    for up, down, factor, start, length in factors:
        if factor == 0:
            continue
        for row in range(0, length, rows):
            end = min(row + rows, length)
            delta = torch.mm(up[row:end], down)
            delta.mul_(factor)
            weight[start + row:start + end].add_(delta.to(weight.dtype))
            del delta
    if metrics is not None:
        metrics["tiled_casts"] += 1
    return weight


@contextlib.contextmanager
def patch_gguf_lora(model, *, mode=None, metrics=None):
    """Temporarily replace one backend class method only while UTX executes."""
    if mode is None:
        mode = enabled()
    if metrics is None:
        metrics = {"tiled_casts": 0, "fallback_casts": 0}
    if not mode:
        yield metrics
        return
    owner = _owner(model)
    if owner is None:
        _LOG.warning("[UTX tiled LoRA] GGMLLayer backend not found; running original GGUF path")
        yield metrics
        return
    original = owner.get_weight

    def wrapped(self, tensor, dtype):
        return tiled_get_weight(self, tensor, dtype, original, metrics=metrics)

    owner.get_weight = wrapped
    try:
        yield metrics
    finally:
        owner.get_weight = original
