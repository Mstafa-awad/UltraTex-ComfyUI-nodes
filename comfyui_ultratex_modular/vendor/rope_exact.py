"""Paired RoPE with separate native FP32 products and addition.

The installed Kitchen Triton expression contracts a different multiplication on
the tested runtime. Disabled contraction preserves the tested native eager
arithmetic, without materializing full FP32 Q/K intermediates.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope(Q, K, P, OQ, OK, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, pair = i // 64, i % 64
    pos = tl.program_id(1) * N * 128 + token * 128 + pair * 2
    mask = token < N
    a = tl.load(P + i * 4, mask, other=0)
    b = tl.load(P + i * 4 + 1, mask, other=0)
    c = tl.load(P + i * 4 + 2, mask, other=0)
    d = tl.load(P + i * 4 + 3, mask, other=0)
    q0 = tl.load(Q + pos, mask, other=0).to(tl.float32)
    q1 = tl.load(Q + pos + 1, mask, other=0).to(tl.float32)
    k0 = tl.load(K + pos, mask, other=0).to(tl.float32)
    k1 = tl.load(K + pos + 1, mask, other=0).to(tl.float32)
    tl.store(OQ + pos, a * q0 + b * q1, mask)
    tl.store(OQ + pos + 1, c * q0 + d * q1, mask)
    tl.store(OK + pos, a * k0 + b * k1, mask)
    tl.store(OK + pos + 1, c * k0 + d * k1, mask)


def apply_rope(q, k, pe):
    if (torch.is_grad_enabled() or not q.is_cuda or not k.is_cuda or not pe.is_cuda
            or q.device != k.device or q.device != pe.device
            or q.shape != k.shape or q.ndim != 4 or q.shape[0] != 1
            or q.shape[-1] != 128 or q.shape[2] < 16384
            or q.dtype != k.dtype or q.dtype not in (torch.bfloat16, torch.float16)
            or not q.is_contiguous() or not k.is_contiguous()
            or pe.dtype != torch.float32 or not pe.is_contiguous()
            or tuple(pe.shape) != (1, 1, q.shape[2], 64, 2, 2)):
        return None
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    with torch.cuda.device(q.device):
        _rope[(triton.cdiv(q.shape[2] * 64, 1024), q.shape[1])](
            q, k, pe, oq, ok, q.shape[2], 1024, enable_fp_fusion=False)
    return oq, ok
