from __future__ import annotations

from typing import Optional

import kornia.augmentation as K
import torch
from torch import Tensor
from typing_extensions import override


# TODO: Find out why the mask is not normalised and whether it should be fixed.
class MinMaxScaling(K.IntensityAugmentationBase2D):
    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1,
                         same_on_batch=True)
        self.flags = {"mins": mins.view(1, -1, 1, 1),
                      "maxs": maxs.view(1, -1, 1, 1)}
        self.delta = 1e-10

    @override
    def apply_transform(
        self,
        input:     Tensor,
        params:    dict[str,
                        Tensor],
        flags:     dict[str,
                        int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        mins = torch.as_tensor(flags["mins"], device=input.device, dtype=input.dtype)
        maxs = torch.as_tensor(flags["maxs"], device=input.device, dtype=input.dtype)
        return (input - mins) / (maxs - mins + self.delta)
