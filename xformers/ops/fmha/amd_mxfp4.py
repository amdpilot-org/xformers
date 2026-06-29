# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""AMD gfx950 (MI355X) MX-FP4 K/V cache forward attention op.

Implements an AMD-native forward attention path for
``xformers.ops.memory_efficient_attention`` that consumes an MX-FP4 K/V cache
(packed ``float4_e2m1fn_x2`` elements with ``e8m0`` per-32-element block scales).

The K/V cache is dequantized to BF16 and run through AITER's flash-attention
forward kernel (``aiter.ops.mha.flash_attn_func``). The op is gated on
``gcnArchName == "gfx950"`` and a K/V dtype of ``torch.float4_e2m1fn_x2`` (the
ROCm PyTorch spelling of MX-FP4; ``torch.float4_e2m1`` is not available in this
stack).

This module is self-contained: it does not require the optional ``mslk``
dispatch package, so the op can be imported and exercised directly. When
``mslk`` is installed, :class:`OpAmdMxfp4Fwd` is also registered into the
xformers forward dispatch list so that ``memory_efficient_attention`` picks it
up automatically for MX-FP4 K/V inputs on gfx950.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# MX-FP4 quantization block size (fixed by the MX specification).
_MXFP4_BLOCK = 32

# float4_e2m1 value lookup table (16 representable levels, sign in bit 3).
_MXFP4_LUT: Tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _is_rocm() -> bool:
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _gcn_arch_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))


def is_amd_mxfp4_supported() -> bool:
    """True on gfx950 (MI355X) with ROCm."""
    return _is_rocm() and "gfx950" in _gcn_arch_name()


def _mxfp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=device)


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """Convert an e8m0 (biased 8-bit exponent) block-scale tensor to f32.

    Mirrors ``aiter.utility.fp4_utils.e8m0_to_f32``: the byte is the 8-bit
    exponent of an f32 (bias 127), so ``scale = 2**(byte - 127)``. A zero byte
    decodes to ``2**-127`` (minimum normal) and ``0xFF`` to NaN, matching the
    MX-FP4 reference dequantization.
    """
    sc = scale.view(torch.uint8)
    zero_case = sc == 0
    nan_case = sc == 0xFF
    scale_f32 = (sc.to(torch.int32) << 23).view(torch.float32)
    scale_f32 = torch.where(zero_case, torch.full_like(scale_f32, 2.0 ** -127), scale_f32)
    scale_f32 = torch.where(nan_case, torch.full_like(scale_f32, float("nan")), scale_f32)
    return scale_f32


def mxfp4_dequant_kv(fp4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an MX-FP4 K/V tensor to BF16.

    Args:
        fp4: packed MX-FP4 elements, last dim ``D`` viewed as
            ``torch.float4_e2m1fn_x2`` (underlying storage is ``uint8`` with
            shape ``[..., D/2]``; two fp4 nibbles per byte, low nibble first).
        scale: ``e8m0`` per-block scales with last dim ``D/32`` (``uint8``).

    Returns:
        BF16 tensor of shape ``[..., D]``.
    """
    if fp4.dtype != torch.float4_e2m1fn_x2:
        raise TypeError(
            f"expected float4_e2m1fn_x2 K/V, got {fp4.dtype}; the ROCm PyTorch "
            "spelling of MX-FP4 is torch.float4_e2m1fn_x2"
        )
    orig_shape = fp4.shape
    D_packed = orig_shape[-1]          # packed bytes: 2 fp4 nibbles per byte
    D = D_packed * 2                   # actual head dim (unpacked elements)
    if D % _MXFP4_BLOCK != 0:
        raise ValueError(f"head dim {D} must be a multiple of {_MXFP4_BLOCK}")
    n_blk = D // _MXFP4_BLOCK          # number of 32-element scale blocks
    flat = fp4.reshape(-1, D_packed).view(torch.uint8)     # [N, D_packed]
    sc = scale.reshape(-1, n_blk)                           # [N, D/32]
    N = flat.shape[0]
    # Unpack the two fp4 nibbles per byte (low nibble = even element).
    lo = flat & 0x0F                                       # [N, D_packed]
    hi = (flat >> 4) & 0x0F                                # [N, D_packed]
    idx = torch.stack((lo, hi), dim=-1).reshape(N, D)     # [N, D]
    vals = _mxfp4_lut(fp4.device)[idx.long()]             # [N, D] f32
    scale_f32 = _e8m0_to_f32(sc)                           # [N, D/32] f32
    # Broadcast-multiply each 32-element block by its scale (no materialized
    # expansion of the scale tensor).
    vals = vals.reshape(N, n_blk, _MXFP4_BLOCK)
    dq = vals * scale_f32.unsqueeze(-1)
    out_shape = orig_shape[:-1] + (D,)
    return dq.reshape(N, D).to(torch.bfloat16).reshape(out_shape)


def mxfp4_kv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """MX-FP4 K/V cache forward attention on gfx950.

    Args:
        q: query, BF16, shape ``[B, Hq, S, D]`` (xformers/SDPA layout).
        k, v: MX-FP4 K/V cache, ``float4_e2m1fn_x2``, shape ``[B, Hkv, S, D]``.
        k_scale, v_scale: ``e8m0`` block scales, ``uint8``, shape
            ``[B, Hkv, S, D/32]``.
        is_causal: apply a causal mask.
        scale: softmax scale; defaults to ``1/sqrt(D)``.

    Returns:
        BF16 output, shape ``[B, Hq, S, D]``.
    """
    if not is_amd_mxfp4_supported():
        raise RuntimeError(
            "OpAmdMxfp4Fwd requires gfx950 (MI355X) with ROCm, got "
            f"gcnArchName={_gcn_arch_name()!r}"
        )
    from aiter.ops.mha import flash_attn_func

    # aiter flash_attn_func expects [B, S, H, D]; q/k/v arrive as [B, H, S, D].
    # Permute the *packed* fp4 cache (D/2 bytes) to flash layout BEFORE dequant
    # so the dequantized BF16 output is already contiguous in flash layout.
    # This avoids an expensive post-dequant transpose of the 2x-larger BF16 tensor.
    q_flash = q.transpose(1, 2).contiguous()               # [B, S, Hq, D]
    k_flash = mxfp4_dequant_kv(
        k.transpose(1, 2).contiguous(),                    # [B, S, Hkv, D_packed]
        k_scale.transpose(1, 2).contiguous(),              # [B, S, Hkv, D/32]
    )                                                       # -> [B, S, Hkv, D] bf16
    v_flash = mxfp4_dequant_kv(
        v.transpose(1, 2).contiguous(),
        v_scale.transpose(1, 2).contiguous(),
    )
    out = flash_attn_func(
        q_flash,
        k_flash,
        v_flash,
        softmax_scale=scale,
        causal=is_causal,
        window_size=(-1, -1, 0),
    )
    return out.transpose(1, 2).contiguous()  # back to [B, Hq, S, D]


class OpAmdMxfp4Fwd:
    """AMD gfx950 MX-FP4 K/V cache forward attention op.

    Gated on ``gcnArchName == "gfx950"`` and a K/V dtype of
    ``torch.float4_e2m1fn_x2``. When the optional ``mslk`` dispatch package is
    installed, this class is registered as an :class:`AttentionFwOpBase` in the
    xformers forward op list (see :func:`_register_into_dispatch`).
    """

    NAME = "amd-mxfp4-fwd"

    @classmethod
    def supports(cls, q, k, v, kv_scale=None, is_causal: bool = False) -> bool:
        if not is_amd_mxfp4_supported():
            return False
        if k.dtype != torch.float4_e2m1fn_x2 or v.dtype != torch.float4_e2m1fn_x2:
            return False
        if q.dtype != torch.bfloat16:
            return False
        if kv_scale is None:
            return False
        return True

    @classmethod
    def apply(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        return mxfp4_kv_attention(
            q, k, k_scale, v, v_scale, is_causal=is_causal, scale=scale
        )


def _register_into_dispatch() -> bool:
    """Register :class:`OpAmdMxfp4Fwd` into the mslk/xformers dispatch.

    Only acts when the ``mslk`` package (which provides
    ``AttentionFwOpBase`` / ``ALL_FW_OPS`` / ``_OPS_LOOKUP``) is importable.
    Returns ``True`` if registration was performed.
    """
    try:
        from mslk.attention.fmha import ALL_FW_OPS, AttentionFwOpBase  # type: ignore
    except Exception:
        return False

    if any(getattr(op, "NAME", None) == OpAmdMxfp4Fwd.NAME for op in ALL_FW_OPS):
        return True

    class _MslkAmdMxfp4Fwd(AttentionFwOpBase, OpAmdMxfp4Fwd):  # type: ignore[misc]
        pass

    _MslkAmdMxfp4Fwd.__name__ = "OpAmdMxfp4Fwd"
    ALL_FW_OPS.insert(0, _MslkAmdMxfp4Fwd)
    return True


# Best-effort registration when the dispatch package is available.
_register_into_dispatch()
