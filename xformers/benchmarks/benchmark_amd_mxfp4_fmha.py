# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark MX-FP4 K/V cache attention vs BF16 SDPA baseline on MI355X (gfx950).

Reports:
  - ``bf16_kv_baseline_ms``: per-iteration ms for F.scaled_dot_product_attention
    at the target shapes (the current AMD fall-through path).
  - ``mxfp4_ms``: per-iteration ms for the MX-FP4 K/V path via
    ``xformers.ops.fmha.memory_efficient_attention`` when aiter +
    ``torch.float4_e2m1`` are available; ``N/A`` otherwise.

Usage::

    python3 -m xformers.benchmarks.benchmark_amd_mxfp4_fmha \
        --seq 8192 --q-heads 64 --kv-heads 8 --dim 128 \
        --warmup 3 --iters 10
"""

import argparse
import importlib.util
import os
import sys

# Make xformers importable when running this script directly (not via -m).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# aiter's gluon kernels require triton>=3.6.0; the ROCm 7.2 base image ships
# triton 3.3.x.  Downgrade the version gate to a warning so aiter imports
# cleanly.  The gluon path is unused by the fmha dispatch.
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch
import torch.nn.functional as F


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--q-heads", type=int, default=64)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    return p.parse_args()


def _time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _repeat_kv(t, n_rep):
    return t.repeat_interleave(n_rep, dim=1)


def main():
    args = _parse_args()
    B, S, H, KV_H, D = 1, args.seq, args.q_heads, args.kv_heads, args.dim
    scale = 1.0 / (D ** 0.5)
    dtype = torch.bfloat16
    device = "cuda"

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", props.name)
    print(f"[bench] device arch={arch}  torch={torch.__version__}")
    print(f"[bench] shapes: B={B} S={S} H={H} KV_H={KV_H} D={D} dtype={dtype}")

    # --- BF16 SDPA baseline (current AMD fall-through) ---
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, KV_H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, KV_H, S, D, device=device, dtype=dtype)
    rep = H // KV_H

    def run_sdpa():
        ke = _repeat_kv(k, rep)
        ve = _repeat_kv(v, rep)
        return F.scaled_dot_product_attention(q, ke, ve, scale=scale)

    bf16_ms = _time_fn(run_sdpa, args.warmup, args.iters)
    out_ref = run_sdpa()
    print(f"[bench] bf16_kv_baseline_ms = {bf16_ms:.4f}  (N={args.iters})")
    print(f"bf16_kv_baseline_ms: {bf16_ms:.4f}")
    print(f"[bench]   ref output finite={torch.isfinite(out_ref).all().item()}")

    # --- BF16 xformers path (aiter flash_attn_func when aiter available) ---
    aiter_available = importlib.util.find_spec("aiter") is not None
    if aiter_available:
        import xformers.ops.fmha as fmha

        def run_xformers():
            return fmha.memory_efficient_attention(q, k, v, scale=scale)

        xf_ms = _time_fn(run_xformers, args.warmup, args.iters)
        out_xf = run_xformers()
        max_diff_xf = (out_ref - out_xf).abs().max().item()
        print(f"[bench] bf16_xformers_ms = {xf_ms:.4f}  (N={args.iters})")
        print(f"[bench]   max_abs_diff vs SDPA = {max_diff_xf:.6f}")
        print(f"[bench]   speedup vs SDPA = {bf16_ms / xf_ms:.2f}x")
    else:
        print("[bench] bf16_xformers_ms = N/A  (aiter not installed)")

    # --- MX-FP4 K/V path via xformers ---
    mxfp4_dtype = getattr(torch, "float4_e2m1", None)

    if not aiter_available or mxfp4_dtype is None:
        reason = "aiter not installed" if not aiter_available else "torch.float4_e2m1 not available"
        print(f"[bench] mxfp4_ms = N/A  ({reason})")
        print(f"[bench] MX-FP4 path is wired but cannot run: {reason}")
        print(f"[bench] aiter_available={aiter_available}  mxfp4_dtype={mxfp4_dtype}")
    else:
        # Build MX-FP4 K/V tensors (packed).  In production these come from
        # the model's quantised cache; here we quantise from BF16 for the
        # benchmark.
        k_mxfp4 = k.to(mxfp4_dtype)
        v_mxfp4 = v.to(mxfp4_dtype)
        k_scale_t = torch.ones(B, KV_H, S, 1, device=device, dtype=torch.float32)
        v_scale_t = torch.ones(B, KV_H, S, 1, device=device, dtype=torch.float32)

        def run_mxfp4():
            return fmha.memory_efficient_attention(
                q, k_mxfp4, v_mxfp4, k_scale=k_scale_t, v_scale=v_scale_t,
            )

        mxfp4_ms = _time_fn(run_mxfp4, args.warmup, args.iters)
        out_mxfp4 = run_mxfp4()
        max_diff = (out_ref - out_mxfp4).abs().max().item()
        print(f"[bench] mxfp4_ms = {mxfp4_ms:.4f}  (N={args.iters})")
        print(f"[bench]   max_abs_diff vs SDPA = {max_diff:.6f}")
        print(f"[bench]   speedup = {bf16_ms / mxfp4_ms:.2f}x")

    print("[bench] DONE")


if __name__ == "__main__":
    main()
