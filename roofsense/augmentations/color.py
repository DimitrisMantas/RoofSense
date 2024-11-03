from __future__ import annotations

import math
from typing import Any

import kornia
import torch
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from typing_extensions import override


class AppendHSV(IntensityAugmentationBase2D):
    """Append the triangular greenness index to each batch sample."""

    def __init__(self):
        r"""Configure the appender.

        Notes:
            The channels involved in the pertinent calculations are expected to
            contain values which are in the interval :math:`\left[0, 1\right]`.
        """
        super().__init__(p=1)

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        hsv = kornia.color.rgb_to_hsv(input[:, :3, ...])
        # Scale channel from [0, 2π] to [0, 1].
        hsv[:, 0, ...] = hsv[:, 0, ...] / (2 * math.pi)

        return torch.cat((input, hsv), dim=1)
