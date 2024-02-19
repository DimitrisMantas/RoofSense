from __future__ import annotations

from typing import Optional, Any

import kornia.augmentation as K
from torch import Tensor


# TODO: Find out why the mask is not normalised and whether it should be fixed.
class MinMaxNormalization(K.IntensityAugmentationBase2D):
    def __init__(self) -> None:
        # NOTE: This approach ensures that the input is always normalised.
        super().__init__(p=1)

    # noinspection PyShadowingBuiltins
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Compute the minimum cell values across the input width.
        mins = input.min(-1).values
        # Compute the minimum cell values across the input width.
        mins = mins.min(-1).values
        # NOTE: The corresponding values always appear first in the underlying array.
        maxs = input.max(-1)[0].max(-1)[0]

        # Reintroduce the reduced dimensions.
        mins = mins[:, :, None, None]
        maxs = maxs[:, :, None, None]
        # Pad the denominator.
        # NOTE: This approach ensures that division-by-zero errors when processing
        #       constant-valued inputs are avoided.
        return (input - mins) / (maxs - mins + 1e-12)
