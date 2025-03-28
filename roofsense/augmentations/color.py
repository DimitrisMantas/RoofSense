from __future__ import annotations

import math
from typing import Any

import kornia
import torch
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from typing_extensions import override


class AppendHSV(IntensityAugmentationBase2D):
    r"""Append the triangular greenness index to each batch sample."""

    def __init__(self, r_idx: int = 0, g_idx: int = 1, b_idx: int = 2) -> None:
        r"""Initialize the augmentation.

        Args:
            r_idx:
                The index of the red band in the input samples.
            g_idx:
                The index of the green band in the input samples.
            b_idx:
                The index of the blue band in the input samples.

        Notes:
            Each sample must contain at least three bands in the RGB configuration.
            The default band order is [0, 1, 2].
            Each band must  be scaled to the interval :math:`\left[0, 1\right]`.
        """
        super().__init__(p=1)
        self.flags = {"r_idx": r_idx, "g_idx": g_idx, "b_idx": b_idx}

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        slice = [flags["r_idx"], flags["g_idx"], flags["b_idx"]]

        hsv = kornia.color.rgb_to_hsv(input[:, slice, ...])
        # Scale channel from [0, 2π] to [0, 1].
        hsv[:, 0, ...] = hsv[:, 0, ...] / (2 * math.pi)

        return torch.cat((input, hsv), dim=1)


class AppendLab(IntensityAugmentationBase2D):
    r"""Append the triangular greenness index to each batch sample."""

    def __init__(self, r_idx: int = 0, g_idx: int = 1, b_idx: int = 2) -> None:
        r"""Initialize the augmentation.

        Args:
            r_idx:
                The index of the red band in the input samples.
            g_idx:
                The index of the green band in the input samples.
            b_idx:
                The index of the blue band in the input samples.

        Notes:
            Each sample must contain at least three bands in the RGB configuration.
            The default band order is [0, 1, 2].
            Each band must  be scaled to the interval :math:`\left[0, 1\right]`.
        """
        super().__init__(p=1)
        self.flags = {"r_idx": r_idx, "g_idx": g_idx, "b_idx": b_idx}

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        slice = [flags["r_idx"], flags["g_idx"], flags["b_idx"]]

        lab = kornia.color.rgb_to_lab(input[:, slice, ...])
        # Scale channel from [0, 100] to [0, 1].
        lab[:, 0, ...] = lab[:, 0, ...] / 100
        # Scale channel from [-128, 127] to [0, 1].
        lab[:, 1, ...] = (lab[:, 1, ...] + 128) / 255
        lab[:, 2, ...] = (lab[:, 2, ...] + 128) / 255

        return torch.cat((input, lab), dim=1)
