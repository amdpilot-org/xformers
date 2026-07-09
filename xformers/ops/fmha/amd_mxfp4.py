# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""AMD MX-FP4 K/V cache attention for ``memory_efficient_attention`` on gfx950.

This module implements the forward attention path that consumes an MX-FP4
(``torch.float4_e2m1``) K/V cache via AITER's ``attention_mxfp4_kv`` kernel.

It is selected only on MI355X (gfx950) when the K/V tensors carry the
``torch.float4_e2m1`` dtype -- the on-wire format of an MX-FP4 K/V cache
tile. On every other device / dtype the dispatcher in ``fmha.__init__``
falls through to the regular BF16 path, so importing this module is always
safe even on builds without aiter or without the float4_e2m1 dtype.
"""

import importlib.util

import torch

# AITER ships the MX-FP4 K/V attention kernel. It is optional at import time
# so that xformers remains importable on builds without aiter.
AITER_AVAILABLE = importlib.util.find_spec("aiter") is not None

# MX-FP4 K/V uses the float4_e2m1 packed dtype. Recent PyTorch builds expose
# it as ``torch.float4_e2m1``; reference it defensively so this module imports
# cleanly on builds that lack the dtype (the op simply never gets selected).
FLOAT4_E2M1 = getattr(torch, "float4_e2m1", None)


def is_gfx950() -> bool:
    """Return True iff the active HIP device is gfx950 (MI355X)."""
    if not (hasattr(torch.version, "hip") and torch.version.hip is not None):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "") or ""
        return "gfx950" in arch.lower()
    except Exception:
        return False


def kv_is_mxfp4(key: torch.Tensor, value: torch.Tensor) -> bool:
    """Return True iff K/V carry the MX-FP4 (float4_e2m1) dtype."""
    if FLOAT4_E2M1 is None:
        return False
    return key.dtype == FLOAT4_E2M1 and value.dtype == FLOAT4_E2M1


class OpAmdMxfp4Fwd:
    """Forward attention op backed by AITER's MX-FP4 K/V kernel on gfx950.

    Selection contract (enforced by the dispatcher in ``fmha.__init__``):

      * ``gcnArchName == "gfx950"``  (MI355X / CDNA4), AND
      * ``kv_dtype == torch.float4_e2m1``  (MX-FP4 K/V cache tile).

    When selected, the op delegates to ``aiter.ops.attention_mxfp4_kv`` which
    fuses the MX-FP4 dequantization with the attention math, keeping the K/V
    traffic at 4-bit throughout the HBM read. This is the path that lets
    xformers' ``memory_efficient_attention`` serve MX-FP4-quantized models
    (e.g. ``amd/Llama-3.3-70B-Instruct-MXFP4``) on MI355X instead of falling
    through to PyTorch SDPA, which has no MX-FP4 K/V support.
    """

    #: whether the op can be selected on this build
    AVAILABLE = AITER_AVAILABLE and FLOAT4_E2M1 is not None

    @staticmethod
    def supports(query, key, value, attn_bias=None, **kwargs) -> bool:
        """True iff the MX-FP4 K/V path should be taken for these inputs."""
        if not OpAmdMxfp4Fwd.AVAILABLE:
            return False
        if not is_gfx950():
            return False
        return kv_is_mxfp4(key, value)

    @staticmethod
    def apply(
        query,
        key,
        value,
        attn_bias=None,
        p=0.0,
        scale=None,
        k_scale=None,
        v_scale=None,
        **kwargs,
    ):
        """Run MX-FP4 K/V attention via ``aiter.ops.attention_mxfp4_kv``.

        Args:
            query: BF16 tensor, xformers BHSD layout ``[B, H, S, D]``.
            key:   MX-FP4 (float4_e2m1) packed K cache, ``[B, KV_H, S, D]``.
            value: MX-FP4 (float4_e2m1) packed V cache, ``[B, KV_H, S, D]``.
            k_scale: optional per-block BF16 scale tensor for K.
            v_scale: optional per-block BF16 scale tensor for V.
            attn_bias: only ``LowerTriangularMask`` (causal) is honoured.
            scale: softmax scale; defaults to ``1 / sqrt(D)``.
        """
        if not AITER_AVAILABLE:
            raise RuntimeError(
                "OpAmdMxfp4Fwd requires the aiter package for the MX-FP4 K/V "
                "attention kernel (aiter.ops.attention_mxfp4_kv)."
            )
        if FLOAT4_E2M1 is None:
            raise RuntimeError(
                "OpAmdMxfp4Fwd requires torch.float4_e2m1 (MX-FP4 dtype), "
                "which is not available in this PyTorch build."
            )

        # Import lazily so the module stays importable without aiter.
        from aiter.ops import attention_mxfp4_kv

        # xformers convention: [B, H, S, D] (BHSD)
        # aiter convention:    [B, S, H, D] (BSHD)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if scale is None:
            scale = 1.0 / (q.shape[-1] ** 0.5)

        # Detect causal attention from the xformers bias type. The ROCm
        # fallback path only defines LowerTriangularMask as a causal marker.
        causal = False
        if attn_bias is not None:
            causal = type(attn_bias).__name__ == "LowerTriangularMask"

        # AITER's attention_mxfp4_kv fuses MX-FP4 dequant + attention. The
        # per-block scales (k_scale / v_scale) are passed through when the
        # caller supplies them; otherwise aiter derives them from the packed
        # tile metadata.
        call_kwargs = dict(softmax_scale=scale, causal=causal)
        if k_scale is not None:
            call_kwargs["k_scale"] = k_scale.transpose(1, 2).contiguous()
        if v_scale is not None:
            call_kwargs["v_scale"] = v_scale.transpose(1, 2).contiguous()

        out = attention_mxfp4_kv(q, k, v, **call_kwargs)
        # BSHD -> BHSD
        return out.transpose(1, 2).contiguous()
