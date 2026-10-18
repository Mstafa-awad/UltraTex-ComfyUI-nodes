"""Read-only GGUF dequantization vs LoRA-patching memory diagnostics.

Nothing in this module is called by image generation. The only temporary mutation
is removing a selected GGMLTensor's `.patches` to measure the same quantized
weight without the LoRA; it is always restored before this function returns.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_LOG = logging.getLogger("UltraTex")
_GIB = 1024 ** 3


def _is_quantized_ggml_linear(layer):
    return (
        isinstance(layer, torch.nn.Linear)
        and any(c.__name__ == "GGMLLayer" for c in type(layer).__mro__)
        and callable(getattr(layer, "is_ggml_quantized", None))
        and layer.is_ggml_quantized()
        and callable(getattr(layer, "cast_bias_weight", None))
    )


def _bytes(tensor):
    return int(tensor.numel() * tensor.element_size()) if isinstance(tensor, torch.Tensor) else 0


def _measure_cast(layer, template, *, device):
    """Peak live allocated VRAM *above the before-cast baseline*.

    Does not run FLUX or an image-generation operation. CUDA measurements
    synchronize so results include async dequant/patch work.
    """
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        before = None
    started = time.perf_counter()
    weight = bias = None
    try:
        weight, bias = layer.cast_bias_weight(template)
        if cuda:
            torch.cuda.synchronize(device)
        result = {
            "output_dtype": str(weight.dtype),
            "output_shape": list(weight.shape),
            "expanded_weight_mib": round(_bytes(weight) / 2**20, 3),
            "bias_mib": round(_bytes(bias) / 2**20, 3),
            "cast_seconds": round(time.perf_counter() - started, 5),
            "peak_extra_allocated_mib": (
                round(max(0, torch.cuda.max_memory_allocated(device) - before) / 2**20, 3)
                if cuda else None
            ),
            "live_extra_allocated_mib": (
                round((torch.cuda.memory_allocated(device) - before) / 2**20, 3)
                if cuda else None
            ),
        }
        return result
    finally:
        del weight, bias
        if cuda:
            torch.cuda.synchronize(device)


@torch.inference_mode()
def _one_layer(layer, device, dtype):
    """Compare SAME Q4 layer with/without adapter, without executing a diffusion step."""
    raw_weight = getattr(layer, "weight", None)
    patches = getattr(raw_weight, "patches", None)
    if not isinstance(patches, (list, tuple)) or len(patches) == 0:
        return None
    template = torch.empty((1, 1, int(layer.in_features)), device=device, dtype=dtype)
    try:
        # GGUF attaches adapter patches to the quantized parameter. Temporarily
        # remove that list, NOT the underlying Q4 weights or the patcher state.
        try:
            raw_weight.patches = []
            baseline = _measure_cast(layer, template, device=device)
        finally:
            raw_weight.patches = patches
        patched = _measure_cast(layer, template, device=device)
        # Repeat the same measurement with tiled LoRA enabled (only this cast).
        # Comparing independent peaks avoids mistaking one tensor's live memory for
        # the complete model footprint.
        from ..vendor.gguf_tiled_lora import patch_gguf_lora
        with patch_gguf_lora(layer, mode=True) as tiled_metrics:
            tiled = _measure_cast(layer, template, device=device)
        difference = (
            round(patched["peak_extra_allocated_mib"] - baseline["peak_extra_allocated_mib"], 3)
            if baseline["peak_extra_allocated_mib"] is not None and patched["peak_extra_allocated_mib"] is not None
            else None
        )
        return {
            "patch_groups": len(patches),
            "gguf_storage_dtype": str(raw_weight.dtype),
            "gguf_storage_mib": round(_bytes(raw_weight) / 2**20, 3),
            "baseline_no_lora": baseline,
            "with_lora": patched,
            "with_lora_tiled": tiled,
            "tiled_patch_counts": tiled_metrics.copy(),
            "tiled_peak_savings_mib": (
                round(patched["peak_extra_allocated_mib"] - tiled["peak_extra_allocated_mib"], 3)
                if patched["peak_extra_allocated_mib"] is not None and tiled["peak_extra_allocated_mib"] is not None else None
            ),
            "extra_peak_from_lora_mib": difference,
        }
    finally:
        del template


def _select_layers(dm, max_layers):
    """Find largest QUANTIZED Linear layers targeted by the installed adapter."""
    candidates = []
    for name, layer in dm.named_modules():
        if not _is_quantized_ggml_linear(layer):
            continue
        raw = layer.weight
        if not getattr(raw, "patches", None):
            continue
        # Never trust packed GGUF shape for output dimensions.
        expanded_size = int(layer.in_features) * int(layer.out_features)
        candidates.append((expanded_size, name, layer))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    # Favor both single and double blocks if adapter targets both; otherwise
    # take the largest few layers. No need to process hundreds of layers.
    selected = candidates[:int(max_layers)]
    if len(selected) > 1 and all("single_blocks" in item[1] for item in selected):
        other = next((c for c in candidates if "double_blocks" in c[1]), None)
        if other is not None:
            selected[-1] = other
    return selected, len(candidates)


@torch.inference_mode()
def probe_gguf_lora(model, lora_path, task, strength, *, max_layers=3, output_dir):
    """Run a controlled A/B cast on the real installed GGUF adapter.

    The loader/ComfyUI manager are used just as generation would use them.
    This function intentionally does not modify the sampling/LoRA math.
    """
    import comfy.model_management as mm
    from .models import apply_lora, validate_backbone

    validate_backbone(model, require_gguf=True)
    if strength <= 0:
        raise ValueError("Choose LoRA strength > 0 for a meaningful A/B comparison")
    patched, model_info, lora_report = apply_lora(model, lora_path, task, strength)
    device = mm.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("This VRAM diagnostic needs CUDA; no GPU memory result is possible on CPU")

    _LOG.info("[UTX GGUF probe] Loading GGUF + LoRA through normal ComfyUI model manager. No diffusion will run.")
    # Memory requested is only for one expanded layer at a time, not a 2048 run.
    # The manager decides which Q4 weights remain on the GPU.
    mm.load_models_gpu([patched], memory_required=512 * 2**20)
    dm = patched.model.diffusion_model
    dtype = getattr(patched.model, "manual_cast_dtype", None) or patched.model.get_dtype()
    if dtype not in (torch.bfloat16, torch.float16):
        dtype = torch.bfloat16
    layers, total = _select_layers(dm, max_layers)
    if not layers:
        raise RuntimeError(
            "No patched GGML Linear layers found after ComfyUI loaded the LoRA. "
            "Check that the GGUF loader is used and the patcher is attaching adapters."
        )

    report = {
        "stage": "gguf_lora_cast_memory_probe",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "model": "FLUX.2 Klein GGUF (attached model)",
        "task": task,
        "lora_strength": strength,
        "matched_adapter_layers": lora_report.get("matched_layers"),
        "patched_quantized_linears": total,
        "device": str(device),
        "compute_dtype": str(dtype),
        "cuda_total_vram_gib": round(torch.cuda.get_device_properties(device).total_memory / _GIB, 3),
        "note": (
            "Only the weight-casting operation is measured. This is NOT a full 2048 diffusion VRAM benchmark. "
            "Each layer runs sequentially, so do NOT sum layer peaks. These values show transient live "
            "PyTorch allocation, not total model memory or Windows shared-GPU paging."
        ),
        "layers": [],
    }
    _LOG.info("[UTX GGUF probe] %d patched GGUF Linear layers; testing %d largest representatives.", total, len(layers))
    for size, name, layer in layers:
        _LOG.info("[UTX GGUF probe] Testing %s (%d x %d)", name, layer.out_features, layer.in_features)
        gc.collect()
        result = _one_layer(layer, device, dtype)
        result.update(name=name, dense_elements=size, dense_bf16_mib=round(size * 2 / 2**20, 3))
        report["layers"].append(result)
        _LOG.info("[UTX GGUF probe] %s: Q4->%s with NO LoRA: %.1f MiB peak; WITH LoRA: %.1f MiB peak; delta %+.1f MiB",
                  name, result['with_lora']['output_dtype'],
                  result['baseline_no_lora']['peak_extra_allocated_mib'],
                  result['with_lora']['peak_extra_allocated_mib'],
                  result['extra_peak_from_lora_mib'])

    biggest = max(report["layers"], key=lambda r: r["extra_peak_from_lora_mib"])
    biggest_delta = biggest["extra_peak_from_lora_mib"]
    report["largest_observed_lora_extra_mib"] = biggest_delta
    report["largest_observed_tiled_saving_mib"] = max((x["tiled_peak_savings_mib"] for x in report["layers"]
                                                       if x["tiled_peak_savings_mib"] is not None), default=None)
    report["tiled_mode_note"] = (
        "Tiled applies standard 2D LoRA using FP32 row matrices. Unsupported adapters fall back. "
        "These are independent layer CAST peaks; not whole-image peak VRAM."
    )
    if biggest_delta > 256:
        diagnosis = (
            "LoRA adds a large TEMPORARY allocation in at least one tested layer. "
            "Further investigation of GGUF patch application is justified."
        )
    else:
        diagnosis = (
            "LoRA was NOT responsible for a >=256 MiB extra peak in any sampled layer. "
            "Q4 weight expansion still occurs WITHOUT LoRA. The 2048 OOM may be dominated by "
            "attention/MLP activations and/or model residency."
        )
    report["interpretation"] = diagnosis
    directory = Path(output_dir) / "ultratex_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    # Timestamp plus microseconds avoids overwriting previous comparisons.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    outpath = directory / f"gguf_lora_probe_{stamp}.json"
    outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    message = (
        f"GGUF Q4 -> {dtype} memory test | {total} patched layers, {len(layers)} sampled\n"
        + "\n".join(
            f"{r['name']}: no LoRA {r['baseline_no_lora']['peak_extra_allocated_mib']:.1f} MiB, "
            f"with LoRA {r['with_lora']['peak_extra_allocated_mib']:.1f} MiB, "
            f"tiled {r['with_lora_tiled']['peak_extra_allocated_mib']:.1f} MiB, "
            f"saved {r['tiled_peak_savings_mib']:+.1f} MiB, "
            f"tile/fallback {r['tiled_patch_counts']['tiled_casts']}/{r['tiled_patch_counts']['fallback_casts']}; "
            f"extra {r['extra_peak_from_lora_mib']:+.1f} MiB"
            for r in report["layers"]
        ) + f"\n\n{diagnosis}\nFull report: {outpath}"
    )
    _LOG.info("[UTX GGUF probe] %s", message)
    return message, str(outpath), report
