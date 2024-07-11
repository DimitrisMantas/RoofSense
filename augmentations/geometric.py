from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import kornia.augmentation as K
import torch
from kornia.geometry import hflip, vflip
from overrides import override
from torch import Tensor


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
