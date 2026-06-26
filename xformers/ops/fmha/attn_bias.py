# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# When the `mslk` package is installed, re-export the canonical attention
# bias implementations from it. Otherwise, provide a minimal non-mslk fallback
# that exposes the names imported by `xformers/ops/fmha/__init__.py` and
# `xformers/ops/fmha/common.py`. The fallback covers only the simple
# `None` bias plus the small set of LowerTriangular/BlockDiagonal names; it
# raises `NotImplementedError` for anything it does not implement (e.g. varlen
# paged keys) rather than silently misbehaving.
import importlib.util

if importlib.util.find_spec("mslk"):
    # flake8: noqa
    from mslk.attention.fmha.attn_bias import (  # noqa: E402, F401
        AttentionBias,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalLocalAttentionFromBottomRightGappyKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalMask,
        BlockDiagonalPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
        PagedBlockDiagonalCausalLocalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalGappyKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
        VARLEN_BIASES,
    )
else:
    from typing import Any, Iterable, Optional, Tuple, Union

    import torch

    class AttentionBias:
        """Base class for a custom attention bias (non-mslk fallback).

        The non-mslk fallback supports the no-bias path (`None`) only.
        Materialization here is intentionally a stub; it raises so that any
        downstream caller that requires bias materialization will fail loudly.
        """

        def materialize(
            self,
            shape: Tuple[int, ...],
            dtype: torch.dtype = torch.float32,
            device: Union[str, torch.device] = "cpu",
        ) -> torch.Tensor:
            raise NotImplementedError(
                "Non-mslk fallback does not materialize attention bias"
            )

    class LowerTriangularMask(AttentionBias):
        pass

    class LowerTriangularMaskWithTensorBias(LowerTriangularMask):
        def __init__(self, bias: torch.Tensor) -> None:
            self._bias = bias

        def materialize(
            self,
            shape: Tuple[int, ...],
            dtype: torch.dtype = torch.float32,
            device: Union[str, torch.device] = "cpu",
        ) -> torch.Tensor:
            bias = self._bias.to(dtype=dtype, device=device)
            mask = torch.zeros(shape, dtype=dtype, device=device)
            return bias.unsqueeze(0).log().add(mask)

    class LowerTriangularFromBottomRightMask(LowerTriangularMask):
        pass

    class LowerTriangularFromBottomRightLocalAttentionMask(LowerTriangularMask):
        pass

    class LocalAttentionFromBottomRightMask(AttentionBias):
        pass

    class BlockDiagonalMask(AttentionBias):
        pass

    class BlockDiagonalPaddedKeysMask(BlockDiagonalMask):
        pass

    class BlockDiagonalGappyKeysMask(BlockDiagonalMask):
        pass

    class PagedBlockDiagonalPaddedKeysMask(BlockDiagonalMask):
        pass

    class PagedBlockDiagonalGappyKeysMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalFromBottomRightMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalWithOffsetGappyKeysMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalWithOffsetPaddedKeysMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalLocalAttentionMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalLocalAttentionFromBottomRightMask(BlockDiagonalMask):
        pass

    class BlockDiagonalCausalLocalAttentionPaddedKeysMask(BlockDiagonalMask):
        pass

    class BlockDiagonalLocalAttentionFromBottomRightGappyKeysMask(BlockDiagonalMask):
        pass

    class BlockDiagonalLocalAttentionPaddedKeysMask(BlockDiagonalMask):
        pass

    class PagedBlockDiagonalCausalLocalPaddedKeysMask(BlockDiagonalMask):
        pass

    class PagedBlockDiagonalCausalWithOffsetGappyKeysMask(BlockDiagonalMask):
        pass

    class PagedBlockDiagonalCausalWithOffsetPaddedKeysMask(BlockDiagonalMask):
        pass

    VARLEN_BIASES: Iterable[type] = ()  # type: ignore
