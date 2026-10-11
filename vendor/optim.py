"""Speed / memory optimisations for UltraTex sampling at high resolution.

* Block-Sparse Attention: UltraTex's SLA top-k kernel (the attention it was trained with), installed as
  ComfyUI's `optimized_attention_override` for the duration of one sampling call only.
* Token chunking: the DiT is token-wise outside attention, so the big MLP activations are computed in
  chunks of tokens. Single blocks are replaced through `patches_replace["dit"]` (linear1 runs once per
  chunk: its qkv part is buffered, its MLP part activated immediately); double-block image MLPs are
  wrapped while sampling. Results are identical to the unchunked forward up to matmul rounding.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

import torch

log = logging.getLogger("UltraTex")

_SLA = {}


class Profiler:
    """Opt-in (ULTRATEX_PROFILE=1) per-step timing of DiT sections; CUDA-synchronised, so it slows sampling."""

    def __init__(self):
        self.enabled = os.environ.get("ULTRATEX_PROFILE", "0") == "1"
        self.times = {}

    @contextlib.contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t

    def report(self, step: int, step_time: float):
        if self.enabled:
            parts = ", ".join(f"{k} {v:.1f}s" for k, v in sorted(self.times.items()))
            log.info("UltraTex profile step %d (%.1fs): %s", step, step_time, parts)
            self.times.clear()


PROF = Profiler()


def timing_attention_override():
    """Pass-through override that only times ComfyUI's own attention (profiling)."""

    def override(func, *args, **kwargs):
        with PROF.section("attention"):
            return func(*args, **kwargs)

    return override


def _sla(topk: float):
    if topk not in _SLA:
        from .sparse_attention import SparseLinearAttention

        _SLA[topk] = SparseLinearAttention(head_dim=128, topk=topk, BLKQ=128, BLKK=64)
    return _SLA[topk]


def sparse_attention_override(topk: float):
    """ComfyUI attention override running SLA when shapes allow it, the default backend otherwise."""
    sla = _sla(topk)

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        usable = (
            skip_reshape
            and mask is None
            and q.ndim == 4
            and q.shape[-1] == 128
            and q.shape[2] == k.shape[2] >= 1024
            and q.is_cuda
        )
        if not usable:
            with PROF.section("attention"):
                return func(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape, **kwargs)
        with PROF.section("attention"):
            out = sla(q, k, v)  # (B, H, L, D)
        if skip_output_reshape:
            return out
        b, h, n, d = out.shape
        return out.transpose(1, 2).reshape(b, n, h * d)

    return override


def check_sparse_attention() -> str | None:
    """Return an error message when the Triton kernel cannot run here."""
    try:
        q = torch.randn(1, 2, 2048, 128, device="cuda", dtype=torch.bfloat16)
        _sla(0.2)(q, q, q)
        return None
    except Exception as exc:  # triton missing / unsupported GPU
        return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- chunked blocks
@contextlib.contextmanager
def linear_chunks(layer, template):
    """Reuse one GGUF layer's native dequantized/patched weights across token chunks.

    Added by UltraTex Modular. Nothing is cached beyond this scope. The installed
    GGUF cast API applies the same adapter patches as its normal linear forward.
    Unknown layers or module hooks retain the original execution path.
    """
    compatible = (isinstance(layer, torch.nn.Linear)
                  and any(t.__name__ == "GGMLLayer" for t in type(layer).__mro__)
                  and callable(getattr(layer, "cast_bias_weight", None))
                  and callable(getattr(layer, "is_ggml_quantized", None))
                  and layer.is_ggml_quantized()
                  and not layer._forward_hooks and not layer._forward_pre_hooks)
    if not compatible:
        yield layer
        return
    weight, bias = layer.cast_bias_weight(template)
    try:
        yield lambda value: torch.nn.functional.linear(value, weight, bias)
    finally:
        # Clear closure cells too, even if the caller retains the temporary callable.
        weight = bias = None


def chunked_single_block(block, chunk: int):
    """patches_replace entry for one comfy.ldm.flux.layers.SingleStreamBlock."""
    from comfy.ldm.flux.layers import apply_mod, modulated_norm
    from comfy.ldm.flux.math import attention
    from comfy.ldm.modules.attention import AttentionTensorContainer

    def run(args, extra):
        x, vec, pe = args["img"], args["vec"], args["pe"]
        topts = args.get("transformer_options", {})
        if args.get("attn_mask") is not None or topts.get("patches", {}).get("attn1_patch") or topts.get("patches", {}).get("attn1_output_patch"):
            return extra["original_block"](args)  # unusual setups: keep ComfyUI's implementation
        mod = block.modulation(vec)[0] if block.modulation else vec
        hs = block.hidden_size
        b, n, _ = x.shape
        # Separate storage lets Q/K normalization release its input buffers.
        # A view of an interleaved QKV buffer kept all three alive through V.
        q = torch.empty(b, block.num_heads, n, hs // block.num_heads, dtype=x.dtype, device=x.device)
        k, v = torch.empty_like(q), torch.empty_like(q)
        mlp = torch.empty(b, n, block.mlp_hidden_dim, dtype=x.dtype, device=x.device)
        with PROF.section("single.linear1"), linear_chunks(block.linear1, x) as linear1:
            for s in range(0, n, chunk):
                out = linear1(modulated_norm(x[:, s : s + chunk], block.pre_norm, mod.scale, mod.shift))
                chunk_qkv = out[..., : 3 * hs].reshape(b,-1,3,block.num_heads,hs//block.num_heads).permute(2,0,3,1,4)
                q[:,:,s:s+chunk] = chunk_qkv[0]
                k[:,:,s:s+chunk] = chunk_qkv[1]
                v[:,:,s:s+chunk] = chunk_qkv[2]
                m = out[..., 3 * hs :]
                if block.yak_mlp:
                    m = block.mlp_act(m[..., block.mlp_hidden_dim_first // 2 :]) * m[..., : block.mlp_hidden_dim_first // 2]
                else:
                    m = block.mlp_act(m)
                mlp[:, s : s + chunk] = m
                del out, m, chunk_qkv
        with PROF.section("single.qknorm_rope_attn"):
            q, k = block.norm(q, k, v)
            attn = attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                             pe=pe, mask=None, transformer_options=topts, preferred_attention=block.comfy_attention)
            del q, k, v
        with PROF.section("single.linear2"), linear_chunks(block.linear2, x) as linear2:
            for s in range(0, n, chunk):
                out = linear2(torch.cat((attn[:, s : s + chunk], mlp[:, s : s + chunk]), 2))
                x[:, s : s + chunk] += apply_mod(out, mod.gate, None, None)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return {"img": x}

    return run


def _chunked_forward(original_forward, chunk: int):
    def forward(x):
        if x.shape[1] <= chunk:
            return original_forward(x)
        return torch.cat([original_forward(x[:, s : s + chunk]) for s in range(0, x.shape[1], chunk)], dim=1)

    return forward


def chunked_mlp(mlp, chunk: int):
    """Reuse two linear layers across MLP chunks without retaining chunk outputs.

    Only the native Linear/activation/Linear sequential layout is specialized.
    Layer hooks and native quantization behavior still go through linear_chunks.
    The temporary weights live for one MLP call, never across diffusion steps.
    """
    original=mlp.forward
    if not (isinstance(mlp,torch.nn.Sequential) and len(mlp)==3
            and isinstance(mlp[0],torch.nn.Linear) and isinstance(mlp[2],torch.nn.Linear)
            and not any(layer._forward_hooks or layer._forward_pre_hooks for layer in mlp.modules())):
        return _chunked_forward(original,chunk)
    def forward(x):
        if x.shape[1]<=chunk:
            return original(x)
        result=torch.empty((*x.shape[:-1],mlp[2].out_features),device=x.device,dtype=x.dtype)
        with linear_chunks(mlp[0],x) as first, linear_chunks(mlp[2],x) as last:
            for s in range(0,x.shape[1],chunk):
                result[:,s:s+chunk]=last(mlp[1](first(x[:,s:s+chunk])))
        return result
    return forward


_SINGLE_ATTRS = ("linear1", "linear2", "pre_norm", "norm", "mlp_act", "mlp_hidden_dim", "mlp_hidden_dim_first",
                 "yak_mlp", "modulation", "hidden_size", "num_heads", "comfy_attention")


@contextlib.contextmanager
def chunked_model(dm, transformer_options: dict, chunk: int):
    """Install token chunking on `dm` (diffusion model) for the duration of the block.

    Module objects and their names are left untouched (ComfyUI's patcher / dynamic VRAM key weights by
    module path): single blocks go through patches_replace, double-block MLPs get an instance-level
    forward that is removed afterwards.
    """
    topts = dict(transformer_options)
    replace = {k: dict(v) for k, v in topts.get("patches_replace", {}).items()}
    dit = replace.setdefault("dit", {})
    if PROF.enabled:
        for i in range(len(dm.double_blocks)):
            def timed(args, extra):
                with PROF.section("double_blocks (incl. attention)"):
                    return extra["original_block"](args)
            dit.setdefault(("double_block", i), timed)
        topts["patches_replace"] = replace
    if chunk <= 0:
        yield topts
        return
    singles = 0
    for i, block in enumerate(dm.single_blocks):
        key = ("single_block", i)
        if key in dit:
            raise RuntimeError(f"UltraTex cannot guarantee token chunking on single block {i}: "
                               "another custom patch already replaces it. Resolve the conflict first.")
        if all(hasattr(block, a) for a in _SINGLE_ATTRS):
            dit[key] = chunked_single_block(block, chunk)
            singles += 1
    if singles != len(dm.single_blocks):
        # Old versions silently fell back to full activation buffers, triggering
        # RAM spilling even when the user selected a small chunk size.
        raise RuntimeError(f"UltraTex token chunking cannot patch "
                           f"{len(dm.single_blocks)-singles}/{len(dm.single_blocks)} single blocks "
                           "with this ComfyUI block layout. Update the node/ComfyUI "
                           "rather than silently running an unchunked 2K sampler.")
    topts["patches_replace"] = replace
    wrapped = []
    for block in dm.double_blocks:
        mlp = getattr(block, "img_mlp", None)
        if isinstance(mlp, torch.nn.Module) and "forward" not in mlp.__dict__:
            mlp.forward = chunked_mlp(mlp, chunk)
            wrapped.append(mlp)
    try:
        yield topts
    finally:
        for mlp in wrapped:
            del mlp.forward
