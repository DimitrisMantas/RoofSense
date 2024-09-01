from typing import Any, Literal

import torch
from kornia.augmentation import GeometricAugmentationBase2D
from kornia.geometry import hflip, vflip
from torch import Tensor
from typing_extensions import override


class RandomDiagonalFlip(GeometricAugmentationBase2D):
    def __init__(
        self,
        diag: Literal["main", "anti"],
        p: float = 0.5,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ):
        super().__init__(p, p_batch, same_on_batch, keepdim)
        self.flags = {"diag": diag}

    @override
    def compute_transformation(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]
    ) -> Tensor:
        flip_mat: Tensor = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=input.device, dtype=input.dtype
        )
        return flip_mat.expand(input.shape[0], 3, 3)

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        if input.shape[-1] != input.shape[-2]:
            raise RuntimeError(
                "Flipping along the image diagonal is only applicable to square inputs."
            )
        if self.flags["diag"] == "main":
            return hflip(input.rot90(dims=(-1, -2)))
        else:
            return vflip(input.rot90(dims=(-1, -2)))

    @override
    def inverse_transform(
        self,
        input: Tensor,
        flags: dict[str, Any],
        transform: Tensor | None = None,
        size: tuple[int, int] | None = None,
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
