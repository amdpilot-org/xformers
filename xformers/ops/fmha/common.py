# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# When the `mslk` package is installed (the private home of the xformers
# FMHA dispatch, relocated to fairinternal/mslk), re-export the canonical
# implementations. Otherwise (e.g. inside amdpilot's ROCm container, where the
# PyPI `mslk` is an empty 0.0.0 template), provide a minimal non-mslk
# fallback that exposes the names imported by `xformers/ops/fmha/__init__.py` and
# `amd_mxfp4.py`. The fallback intentionally supports only the BF16 SDPA
# forward path and the AMD MX-FP4 K/V path; backward and varlen paths raise
# `NotImplementedError`.
import importlib.util

if importlib.util.find_spec("mslk"):
    # flake8: noqa
    from mslk.attention.fmha.common import (  # noqa: E402, F401
        _attn_bias_apply,
        AttentionBwOpBase,
        AttentionFwOpBase,
        AttentionOp,
        AttentionOpBase,
        bmk2bmhk,
        bmhk2bmhk,
        bmk2bmk,
        check_lastdim_alignment_stride1,
        Context,
        Gradients,
        Inputs,
        pack_fp8_tensorwise_per_head,
        ScaledTensor,
    )
else:
    import math
    from dataclasses import dataclass
    from typing import (
        Any,
        Callable,
        Iterable,
        List,
        Optional,
        Set,
        Tuple,
        Type,
        Union,
    )

    import torch

    from ..common import BaseOperator
    from .attn_bias import (
        AttentionBias,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalMask,
        BlockDiagonalPaddedKeysMask,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
        PagedBlockDiagonalGappyKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
    )

    @dataclass
    class Context:
        lse: torch.Tensor
        out: torch.Tensor
        op_bw: Optional[Type["AttentionBwOpBase"]] = None
        rng_state: Optional[Any] = None
        qkv_share_storage: bool = False

    @dataclass
    class Gradients:
        dq: torch.Tensor
        dk: torch.Tensor
        dv: torch.Tensor
        db: Optional[torch.Tensor] = None

    @dataclass
    class Inputs:
        query: torch.Tensor
        key: torch.Tensor
        value: torch.Tensor
        attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None
        p: float = 0.0
        scale: Optional[float] = None
        output_dtype: Optional[torch.dtype] = None
        is_partial: bool = False

        @property
        def device(self) -> torch.device:
            return self.query.device

        @property
        def scale_float(self) -> float:
            return self.query.shape[-1] ** (-0.5) if self.scale is None else self.scale

        def get_qkv_in_bmghk(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.query.ndim == 5:
                return self.query, self.key, self.value
            if self.query.ndim == 4:
                return (
                    self.query.unsqueeze(2),
                    self.key.unsqueeze(2),
                    self.value.unsqueeze(2),
                )
            return (
                self.query[:, :, None, None],
                self.key[:, :, None, None],
                self.value[:, :, None, None],
            )

        def normalize_bmhk(self) -> Tuple[int, ...]:
            if self.query.ndim not in (3, 4, 5):
                raise ValueError(
                    f"Invalid shape for query: {self.query.shape}. "
                    "Expected shape [batch, seqlen, head_groups, num_heads_per_group, K]"
                    ", [batch, seqlen, num_heads, K], or [batch, seqlen, K]."
                )
            if self.query.ndim == 5:
                output_shape = (
                    self.query.shape[0],
                    self.query.shape[1],
                    self.query.shape[2] * self.query.shape[3],
                    self.value.shape[-1],
                )
            elif self.value.dtype == torch.int32:
                # Quantized K/V cache case (the last dims of Q and K differ).
                output_shape = tuple(self.query.shape)
            else:
                output_shape = (self.query.shape[:-1]) + (self.value.shape[-1],)
            if self.query.ndim == 3:
                self.query = self.query.unsqueeze(2)
                self.key = self.key.unsqueeze(2)
                self.value = self.value.unsqueeze(2)
            return output_shape

        def validate_inputs(self) -> None:
            qkv = (self.query, self.key, self.value)
            if self.query.ndim not in (3, 4, 5) or any(
                x.ndim != self.query.ndim for x in qkv
            ):
                raise ValueError(
                    f"Query/Key/Value should all have BMGHK, BMHK or BMK shape.\n"
                    f"  query.shape: {self.query.shape}\n"
                    f"  key.shape  : {self.key.shape}\n"
                    f"  value.shape: {self.value.shape}"
                )
            if any(x.device != self.query.device for x in qkv):
                raise ValueError("Query/Key/Value should all be on the same device")
            if self.p < 0.0 or self.p > 1.0:
                raise ValueError(f"Invalid dropout probability: p={self.p}")

    class AttentionOpBase(BaseOperator):
        """Base class for any attention operator (non-mslk fallback)"""
        OPERATOR: Any = None
        SUPPORTED_DEVICES: Set[str]
        CUDA_MINIMUM_COMPUTE_CAPABILITY: Tuple[int, int] = (5, 0)
        CUDA_MAXIMUM_COMPUTE_CAPABILITY: Optional[Tuple[int, int]] = None
        SUPPORTED_DTYPES: Set[torch.dtype]
        SUPPORTED_MAX_K: float
        SUPPORTED_MIN_K: int = 0
        SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (type(None),)
        SUPPORTS_DROPOUT: bool
        SUPPORTS_CUSTOM_SCALE: bool = False
        SUPPORTS_DIFFERENT_VALUE_EMBED: bool = False
        SUPPORTS_OUTPUT_DTYPE: bool = False
        SUPPORTS_PARTIAL: bool = False
        IS_DETERMINISTIC: bool = True
        SUPPORTS_BMGHK: bool = False
        NAME: str
        OPERATOR_CATEGORY = "memory_efficient_attention"
        VARLEN_LSE_PACKED: bool = True
        _TEST_BATCH_SIZES: List[int] = [1, 300]
        _TEST_K: List[int] = [32, 128]

        @classmethod
        def supports(cls, d: Inputs) -> bool:
            return not cls.not_supported_reasons(d)

        @classmethod
        def shape_not_supported_reasons(
            cls, Mq: int, Mkv: int, K: int, Kv: int
        ) -> List[str]:
            reasons = []
            if not cls.SUPPORTS_DIFFERENT_VALUE_EMBED and K != Kv:
                reasons.append("query.shape[-1] != value.shape[-1]")
            if max(K, Kv) > cls.SUPPORTED_MAX_K:
                reasons.append(
                    f"max(query.shape[-1], value.shape[-1]) > {cls.SUPPORTED_MAX_K}"
                )
            if min(K, Kv) < cls.SUPPORTED_MIN_K:
                reasons.append(
                    f"min(query.shape[-1], value.shape[-1]) < {cls.SUPPORTED_MIN_K}"
                )
            return reasons

        @classmethod
        def not_supported_reasons(cls, d: Inputs) -> List[str]:
            query_shape = d.query.shape
            reasons = cls.shape_not_supported_reasons(
                Mq=query_shape[1],
                Mkv=d.key.shape[1],
                K=query_shape[-1],
                Kv=d.value.shape[-1] if d.value.dtype == torch.int32 else d.value.shape[-1],
            )
            device_type = d.query.device.type
            dtype = d.query.dtype
            if device_type not in cls.SUPPORTED_DEVICES:
                reasons.append(f"device={device_type} (supported: {cls.SUPPORTED_DEVICES})")
            if dtype not in cls.SUPPORTED_DTYPES:
                reasons.append(f"dtype={dtype} (supported: {cls.SUPPORTED_DTYPES})")
            if type(d.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES:
                reasons.append(f"attn_bias type is {type(d.attn_bias)}")
            if (d.p != 0.0) and not cls.SUPPORTS_DROPOUT:
                reasons.append("dropout > 0.0")
            if d.scale is not None and not cls.SUPPORTS_CUSTOM_SCALE:
                reasons.append("has custom scale")
            if not cls.is_available():
                reasons.append("operator not available")
            return reasons

    class AttentionFwOpBase(AttentionOpBase):
        @classmethod
        def apply(
            cls, inp: Inputs, needs_gradient: bool
        ) -> Tuple[torch.Tensor, Optional[Context]]:
            raise NotImplementedError()

    class AttentionBwOpBase(AttentionOpBase):
        SUPPORTS_ATTN_BIAS_GRAD = False
        SUPPORTS_PARTIAL = True

        @classmethod
        def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
            raise NotImplementedError()

    AttentionOp = Tuple[
        Optional[Type[AttentionFwOpBase]], Optional[Type[AttentionBwOpBase]]
    ]

    def bmk2bmhk(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor
        return tensor.reshape(
            [tensor.shape[0] // num_heads, num_heads, tensor.shape[1], tensor.shape[2]]
        ).permute((0, 2, 1, 3))

    def bmhk2bmhk(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        # No-op for contiguous 4D BMHK.
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected BMHK (4D) tensor, got {tensor.ndim}D"
            )
        return tensor

    def bmk2bmk(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        # No-op for 3D BMK.
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected BMK (3D) tensor, got {tensor.ndim}D"
            )
        return tensor

    def check_lastdim_alignment_stride1(
        reasons: List[str], name: str, x: torch.Tensor, alignment: int
    ) -> None:
        if x.shape[-1] % alignment != 0:
            reasons.append(f"{name}.shape[-1] % {alignment} != 0")
        elif x.stride(-2) % alignment != 0:
            reasons.append(
                f"{name}.stride(-2) % {alignment} != 0 ({name}.stride() = {x.stride()})"
            )
        if x.stride(-1) > 1:
            reasons.append(
                f"{name}.stride(-1) > 1 ({name}.stride() = {x.stride()})"
            )

    def _attn_bias_apply(
        attn_bias: Optional[Union[torch.Tensor, AttentionBias]],
        op: Callable[[torch.Tensor], torch.Tensor],
    ) -> Optional[Union[torch.Tensor, AttentionBias]]:
        if isinstance(attn_bias, torch.Tensor):
            return op(attn_bias)
        if isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
            return LowerTriangularMaskWithTensorBias(op(attn_bias._bias))
        return attn_bias

    def _is_bias_type_supported_in_BMK(attn_bias_type: Any) -> bool:
        if isinstance(None, attn_bias_type):
            return True
        return attn_bias_type in [LowerTriangularMask, torch.Tensor]

    def pack_fp8_tensorwise_per_head(
        x: torch.Tensor,
        scale: Union[torch.Tensor, float],
        original_dtype,
    ) -> torch.Tensor:
        # Simplified placeholder of the mslk ScaledTensor path. The
        # non-mslk fallback is forward-only and does not pack a ScaledTensor
        # itself; the AMD MX-FP4 op dequantizes K/V before the attention
        # call, so this helper is left as a direct (un-scaled) tensor.
        return x

    class ScaledTensor(torch.Tensor):
        __slots__ = ["scale", "dequant_func", "original_dtype"]

        __torch_function__ = torch._C._disabled_torch_function_impl

        def __new__(
            cls,
            data: torch.Tensor,
            scale: torch.Tensor,
            dequant_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            original_dtype: torch.dtype,
            require_grad: bool = False,
        ) -> "ScaledTensor":
            instance = torch.Tensor._make_subclass(cls, data, require_grad=require_grad)
            instance.scale = scale  # type: ignore
            instance.dequant_func = dequant_func  # type: ignore
            instance.original_dtype = original_dtype  # type: ignore
            return instance

        def dequantize(self) -> torch.Tensor:
            data = torch.Tensor(self.float())
            dequantized_data = self.dequant_func(data, self.scale)  # type: ignore
            return dequantized_data.to(self.original_dtype)  # type: ignore

        def unpack(self) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.data, self.scale  # type: ignore

        def __repr__(self) -> str:
            return (
                f"ScaledTensor(data={self.data}, "
                f"scale={self.scale}, original_dtype={self.original_dtype})"
            )
