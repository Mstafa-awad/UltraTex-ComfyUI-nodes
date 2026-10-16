""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import os
import torch
import triton
import triton.language as tl


@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    # A masked Triton load without ``other`` leaves masked lanes undefined.
    # Those lanes participate in the block reduction below, so they must be
    # zero for a partial final block.
    x = tl.load(
        X + x_offset + offs_l[:, None] * D + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    # Triton 3.x removed the ``dtype=`` reduction argument. Cast explicitly so
    # accumulation remains fp32, matching the original SLA implementation.
    x_mean = tl.sum(x.to(tl.float32), axis=0) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))



@triton.jit
def compress_centered_kernel(
    X, XM, CENTER,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Center K and pool in one GPU read; do not allocate (B,H,L,D) centered K.

    The subtraction is rounded back to the input dtype BEFORE the fp32
    reduction, reproducing the original ``(k - k.mean(...)).to(k.dtype)``
    intermediate as closely as Triton permits.
    """
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)
    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    valid = offs_l[:, None] < L
    x = tl.load(X + idx_bh * L * D + offs_l[:, None] * D + offs_d[None, :],
                mask=valid, other=0.)
    center = tl.load(CENTER + idx_bh * D + offs_d)
    centered = (x.to(tl.float32) - center[None, :].to(tl.float32)).to(x.dtype)
    centered = tl.where(valid, centered, tl.full((BLOCK_L, D), 0, x.dtype))
    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    pooled = tl.sum(centered.to(tl.float32), axis=0) / nx
    blocks = (L + BLOCK_L - 1) // BLOCK_L
    tl.store(XM + idx_bh * blocks * D + idx_l * D + offs_d,
             pooled.to(XM.dtype.element_ty))


def mean_pool_centered(x, BLK):
    """SLA key centering fused into its pooling kernel for inference.

    The existing path is retained for unsupported layouts, training and
    UTX_SLA_FUSED_POOL=0. Uses the same key mean and BF16/F16 rounding
    as the old full-size tensor subtraction.
    """
    if (not x.is_cuda or torch.is_grad_enabled() or not x.is_contiguous()
            or os.environ.get("UTX_SLA_FUSED_POOL", "1").strip().lower()
            in ("0", "off", "false", "no")):
        return mean_pool(x - torch.mean(x, dim=-2, keepdim=True), BLK)
    B, H, L, D = x.shape
    # D=128 in UltraTex, so Triton can load one contiguous feature vector.
    if D != 128:
        return mean_pool(x - torch.mean(x, dim=-2, keepdim=True), BLK)
    center = torch.mean(x, dim=-2).contiguous()
    blocks = (L + BLK - 1) // BLK
    output = torch.empty((B, H, blocks, D), device=x.device, dtype=x.dtype)
    compress_centered_kernel[(blocks, B * H)](x, output, center, L, D, BLK)
    return output

def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    # Match SLA's smooth-k key centering, without a full-sequence temporary.
    # The fused operation only changes the storage strategy, not K tokens.
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool_centered(k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk
