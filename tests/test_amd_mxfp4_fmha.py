# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Parity test for MX-FP4 K/V cache attention on MI355X (gfx950).

Verifies that ``OpAmdMxfp4Fwd`` produces output matching the BF16 SDPA
reference within ``parity_tol = 5e-3`` after MX-FP4 dequantisation.

The runtime parity check is skipped when aiter or ``torch.float4_e2m1``
is unavailable, but the dispatch / wiring tests always run.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from xformers.ops.fmha.amd_mxfp4 import OpAmdMxfp4Fwd

PARITY_TOL = 5e-3


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", props.name)
    return "gfx950" in arch


def _aiter_available():
    return importlib.util.find_spec("aiter") is not None


gfx950_only = pytest.mark.skipif(not _is_gfx950(), reason="requires gfx950")
aiter_required = pytest.mark.skipif(not _aiter_available(), reason="requires aiter")
mxfp4_dtype_required = pytest.mark.skipif(
    getattr(torch, "float4_e2m1", None) is None,
    reason="requires torch.float4_e2m1",
)


# ---------------------------------------------------------------------------
# Dispatch / wiring tests (always run)
# ---------------------------------------------------------------------------

def test_op_exists():
    """OpAmdMxfp4Fwd is importable and has the expected interface."""
    assert hasattr(OpAmdMxfp4Fwd, "supports")
    assert hasattr(OpAmdMxfp4Fwd, "apply")


def test_supports_bf16_returns_false():
    """BF16 K/V must NOT trigger the MX-FP4 path."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    B, S, H, D = 1, 128, 8, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    assert OpAmdMxfp4Fwd.supports(q, k, v) is False


@gfx950_only
def test_supports_gates_on_gfx950():
    """On gfx950, supports() checks aiter + float4_e2m1 availability."""
    # Even on gfx950, BF16 should return False
    B, S, H, D = 1, 128, 8, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    assert OpAmdMxfp4Fwd.supports(q, k, v) is False


# ---------------------------------------------------------------------------
# Runtime parity test (requires aiter + float4_e2m1 + gfx950)
# ---------------------------------------------------------------------------

@gfx950_only
@aiter_required
@mxfp4_dtype_required
def test_mxfp4_parity_vs_sdpa():
    """MX-FP4 K/V attention output must match BF16 SDPA within PARITY_TOL."""
    B, S, H, KV_H, D = 1, 8192, 64, 8, 128
    dtype = torch.bfloat16
    mxfp4_dtype = torch.float4_e2m1
    scale = 1.0 / (D ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k_bf16 = torch.randn(B, KV_H, S, D, device="cuda", dtype=dtype)
    v_bf16 = torch.randn(B, KV_H, S, D, device="cuda", dtype=dtype)

    # BF16 SDPA reference (with GQA expansion)
    rep = H // KV_H
    ke = k_bf16.repeat_interleave(rep, dim=1)
    ve = v_bf16.repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(q, ke, ve, scale=scale)

    # MX-FP4 K/V path
    k_mxfp4 = k_bf16.to(mxfp4_dtype)
    v_mxfp4 = v_bf16.to(mxfp4_dtype)
    k_scale = torch.ones(B, KV_H, S, 1, device="cuda", dtype=torch.float32)
    v_scale = torch.ones(B, KV_H, S, 1, device="cuda", dtype=torch.float32)

    import xformers.ops.fmha as fmha
    out = fmha.memory_efficient_attention(
        q, k_mxfp4, v_mxfp4, k_scale=k_scale, v_scale=v_scale,
    )

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert torch.isfinite(out).all(), "output contains NaN/Inf"
    max_diff = (ref - out).abs().max().item()
    assert max_diff < PARITY_TOL, f"max_abs_diff={max_diff} >= {PARITY_TOL}"
