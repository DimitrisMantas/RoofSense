from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

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
    def __init__(self, diag: Literal["main", "anti"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = {"diag": diag}

    def compute_transformation(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]
    ) -> Tensor:
        flip_mat: Tensor = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=input.device, dtype=input.dtype
        )

        return flip_mat.expand(input.shape[0], 3, 3)

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
            return hflip(input.rot90(dims=(-1, -2)))
        else:
            return vflip(input.rot90(dims=(-1, -2)))

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(
                f"Expected the `transform` be a Tensor. Got {type(transform)}."
            )
        return self.apply_transform(
            input,
            params=self._params,
            transform=torch.as_tensor(
                transform, device=input.device, dtype=input.dtype
            ),
            flags=flags,
        )


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
