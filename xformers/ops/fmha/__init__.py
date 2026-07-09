# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# The fmha implementation has moved to the mslk package. This package and its
# submodules re-export mslk symbols to preserve the xformers.ops.fmha API.
import importlib.util

import torch

if importlib.util.find_spec("mslk"):
    # flake8: noqa
    from mslk.attention.fmha import (
        _deserialize_bias,
        _detect_lse_packed_or_raise,
        _fMHA,
        _memory_efficient_attention,
        _memory_efficient_attention_backward,
        _memory_efficient_attention_forward,
        _memory_efficient_attention_forward_requires_grad,
        _memory_efficient_attention_forward_torch_wrapper,
        _memory_efficient_attention_forward_torch_wrapper_meta,
        _memory_efficient_attention_forward_torch_wrapper_with_bias,
        _memory_efficient_attention_forward_torch_wrapper_with_bias_meta,
        _OPS_LOOKUP,
        _serialize_op,
        _unserialize_op,
        ALL_BW_OPS,
        ALL_FW_OPS,
        AttentionBias,
        AttentionBwOpBase,
        AttentionFwOpBase,
        AttentionOp,
        AttentionOpBase,
        BlockDiagonalMask,
        dispatch,
        Inputs,
        LowerTriangularMask,
        memory_efficient_attention,
        memory_efficient_attention_backward,
        memory_efficient_attention_forward,
        memory_efficient_attention_forward_requires_grad,
        memory_efficient_attention_partial,
        MemoryEfficientAttentionCkOp,
        MemoryEfficientAttentionCutlassBlackwellOp,
        MemoryEfficientAttentionCutlassFwdFlashBwOp,
        MemoryEfficientAttentionCutlassOp,
        MemoryEfficientAttentionFlashAttentionOp,
        MemoryEfficientAttentionSplitKCkOp,
        merge_attentions,
    )

    from mslk.attention.fmha.dispatch import (
        _dispatch_bw,
        _dispatch_fw,
        _ensure_op_supports_or_raise,
        _get_use_fa3,
        _set_use_fa3,
    )

    from . import (
        attn_bias,
        ck,
        ck_splitk,
        common,
        cutlass,
        cutlass_blackwell,
        flash,
        flash3,
        triton_splitk,
    )

    torch.library.define(
        "xformer::memory_efficient_attention_forward",
        "(Tensor q, Tensor k, Tensor v, Tensor? b = None, float? p = 0.0, float? scale = None) -> Tensor",
    )

    torch.library.impl(
        "xformer::memory_efficient_attention_forward",
        "Meta",
        _memory_efficient_attention_forward_torch_wrapper_meta,
    )
    torch.library.impl(
        "xformer::memory_efficient_attention_forward",
        "CUDA",
        _memory_efficient_attention_forward_torch_wrapper,
    )

    torch.library.define(
        "xformer::memory_efficient_attention_forward_with_bias",
        "(Tensor q, Tensor k, Tensor v, Tensor b, float? p = 0.0, float? scale = None) -> Tensor",
    )

    torch.library.impl(
        "xformer::memory_efficient_attention_forward_with_bias",
        "Meta",
        _memory_efficient_attention_forward_torch_wrapper_with_bias_meta,
    )

    torch.library.impl(
        "xformer::memory_efficient_attention_forward_with_bias",
        "CUDA",
        _memory_efficient_attention_forward_torch_wrapper_with_bias,
    )
elif hasattr(torch.version, "hip") and torch.version.hip is not None:
    # ROCm fallback: when the mslk package is unavailable, dispatch
    # memory_efficient_attention to aiter's flash attention kernel
    # (AMD-native, GQA-native) instead of falling through to PyTorch
    # SDPA, which does not support the MX-FP4 K/V cache path on gfx950.
    import os
    # aiter's gluon (paged-attention) kernels require triton>=3.6.0; the
    # ROCm 7.2 base image ships triton 3.3.x.  Downgrade the version gate
    # to a warning so aiter imports cleanly.  The gluon path is unused by
    # the fmha dispatch -- only the hand-written gfx9 ASM flash_attn_func
    # (fmha_v3) path is exercised, which is compatible with triton 3.3.x.
    os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")
    _AITER_AVAILABLE = importlib.util.find_spec("aiter") is not None

    # MX-FP4 K/V cache attention op. Selected with kernel-priority over the
    # BF16 aiter flash path when gcnArchName == "gfx950" AND the K/V tensors
    # carry the torch.float4_e2m1 (MX-FP4) dtype, routing through
    # aiter.ops.attention_mxfp4_kv instead of PyTorch SDPA.
    from .amd_mxfp4 import OpAmdMxfp4Fwd  # noqa: E402

    class AttentionBias:  # noqa: E701
        """Minimal AttentionBias base for the ROCm/aiter fallback path."""

    class LowerTriangularMask(AttentionBias):  # noqa: E701
        """Causal (lower-triangular) attention mask."""

    class AttentionOpBase:  # noqa: E701
        """Minimal op base for API compatibility."""

    class AttentionOp:  # noqa: E701
        def __init__(self, fw, bw):
            self.op = fw
            self.op_bw = bw

    def memory_efficient_attention(
        query, key, value, attn_bias=None, p=0.0, scale=None, op=None,
        k_scale=None, v_scale=None,
    ):
        # Kernel-priority MX-FP4 K/V path: gfx950 + torch.float4_e2m1 K/V.
        # OpAmdMxfp4Fwd.supports gates on gcnArchName == "gfx950" AND
        # kv_dtype == torch.float4_e2m1; otherwise fall through to BF16.
        if OpAmdMxfp4Fwd.supports(query, key, value, attn_bias=attn_bias):
            return OpAmdMxfp4Fwd.apply(
                query, key, value, attn_bias=attn_bias, p=p, scale=scale,
                k_scale=k_scale, v_scale=v_scale,
            )
        if not _AITER_AVAILABLE:
            # Fall through to PyTorch SDPA when aiter is unavailable.
            # This preserves the upstream AMD behavior (SDPA fall-through)
            # and keeps BF16 attention functional without aiter. The MX-FP4
            # K/V path (OpAmdMxfp4Fwd above) still requires aiter.
            import torch.nn.functional as _F

            causal = isinstance(attn_bias, LowerTriangularMask)
            if scale is None:
                scale = 1.0 / (query.shape[-1] ** 0.5)
            # Handle GQA: expand K/V heads to match Q heads for SDPA.
            q_heads = query.shape[1]
            kv_heads = key.shape[1]
            if q_heads != kv_heads:
                rep = q_heads // kv_heads
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            return _F.scaled_dot_product_attention(
                query, key, value, is_causal=causal, scale=scale,
            )
        import aiter.ops.mha as _aiter_mha

        # xformers convention: [B, H, S, D] (BHSD)
        # aiter flash_attn_func convention: [B, S, H, D] (BSHD)
        causal = isinstance(attn_bias, LowerTriangularMask)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if scale is None:
            scale = 1.0 / (q.shape[-1] ** 0.5)
        out = _aiter_mha.flash_attn_func(
            q, k, v, softmax_scale=scale, causal=causal
        )
        # BSHD -> BHSD
        return out.transpose(1, 2).contiguous()

    def memory_efficient_attention_forward(q, k, v, attn_bias=None, scale=None):
        return memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=scale)

    # Forward op registry. OpAmdMxfp4Fwd is listed first so dispatch picks
    # the MX-FP4 K/V kernel on gfx950 + torch.float4_e2m1 K/V before the
    # BF16 aiter flash fallback.
    ALL_FW_OPS = (OpAmdMxfp4Fwd,)
    ALL_BW_OPS = ()
