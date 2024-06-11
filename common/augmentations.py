from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

import kornia
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


class AppendGreenness(K.IntensityAugmentationBase2D):
    """Basically NDVI for RGB (https://research.tudelft.nl/files/96224757/1_s2.0_S0924271621001854_main.pdf). rgb data must be 0..1"""

    def __init__(
        self, index_red: int = 0, index_green: int = 1, index_blue: int = 2
    ) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: reference band channel index
            index_green: difference band channel index of component 1
            index_blue: difference band channel index of component 2
        """
        super().__init__(p=1)
        self.flags = {
            "index_red": index_red,
            "index_green": index_green,
            "index_blue": index_blue,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_red = input[..., flags["index_red"], :, :]
        band_green = input[..., flags["index_green"], :, :]
        band_blue = input[..., flags["index_blue"], :, :]

        ag = band_green - 0.39 * band_red - 0.61 * band_blue
        # NOTE: This is [-1,1] (the range is known because the index is computed after the scaling step) so we need to scale it to [0,1]
        ag = (ag + 1) / 2

        # add the channel dim to be able to cat
        ag = ag.unsqueeze(dim=1)

        input = torch.cat((input, ag), dim=1)
        return input


class AppendHSV(K.IntensityAugmentationBase2D):
    """rgb data must be 0..1"""

    def __init__(self, *args, **kwargs):
        super().__init__(p=1, *args, **kwargs)

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        hsv = kornia.color.rgb_to_hsv(input[:, :3, ...])
        # scale the H channel to from [0,2pi] to [0,1]
        hsv[:, 0, ...] = hsv[:, 0, ...] / (2 * math.pi)

        input = torch.cat((input, hsv), dim=1)
        return input


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


class NormalizeRGB(K.Normalize):
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


class RandomEqualize(K.RandomEqualize):
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


class RandomPosterize(K.RandomPosterize):
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


class RandomGaussianBlur(K.RandomGaussianBlur):
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
