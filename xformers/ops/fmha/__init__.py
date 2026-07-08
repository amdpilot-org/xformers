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
    _AITER_AVAILABLE = importlib.util.find_spec("aiter") is not None

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
        query, key, value, attn_bias=None, p=0.0, scale=None, op=None
    ):
        if not _AITER_AVAILABLE:
            raise RuntimeError(
                "memory_efficient_attention on ROCm requires either the mslk "
                "package or aiter; neither is available."
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
