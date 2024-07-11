from __future__ import annotations

import math
from typing import Any, Dict, Optional

import kornia
import kornia.augmentation as K
import torch
from overrides import override
from torch import Tensor


class RGBAugmentation(K.IntensityAugmentationBase2D):
    def __init__(
        self, base: type[K.IntensityAugmentationBase2D], *args, **kwargs
    ) -> None:
        super().__init__()
        self.base = base(*args, **kwargs)

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # TODO: Check whether the upstream arguments discarded.
        input[:, :3, ...] = self.base(input[:, :3, ...])
        return input


class AppendHSV(K.IntensityAugmentationBase2D):
    """rgb data must be 0..1"""

    def __init__(self):
        super().__init__(p=1)

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
