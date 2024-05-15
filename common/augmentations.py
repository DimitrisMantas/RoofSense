from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import kornia.augmentation as K
import torch
from kornia.geometry import hflip, vflip
from overrides import override
from torch import Tensor


class MinMaxScaling(K.IntensityAugmentationBase2D):
    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1, same_on_batch=True)
        self.flags = {"mins": mins.view(1, -1, 1, 1), "maxs": maxs.view(1, -1, 1, 1)}
        self.delta = 1e-10

    # noinspection PyShadowingBuiltins
    @override
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        mins = torch.as_tensor(flags["mins"], device=input.device, dtype=input.dtype)
        maxs = torch.as_tensor(flags["maxs"], device=input.device, dtype=input.dtype)
        return (input - mins) / (maxs - mins + self.delta)


class RandomDiagonalFlip(K.GeometricAugmentationBase2D):
    def __init__(self, diag: Literal["main", "antid"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = {"diag": diag}

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        if input.shape[-1] != input.shape[-2]:
            raise RuntimeError(
                "Flipping along the image diagonal is only applicable to square inputs."
            )
        if self.flags["diag"] == "main":
            return input.transpose(-1, -2).contiguous()
        else:
            return vflip(hflip(input))


class RandomSharpness(K.RandomSharpness):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        input[:, :3, ...] = super().apply_transform(
            input[:, :3, ...], params, flags, transform
        )
        return input

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        return torch.flipud(torch.fliplr(input))


class ColorJiggle(K.ColorJiggle):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        input[:, :3, ...] = super().apply_transform(
            input[:, :3, ...], params, flags, transform
        )
        return input
