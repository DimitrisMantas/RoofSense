from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from typing_extensions import override


class AppendTGI(IntensityAugmentationBase2D):
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
        r = input[:, flags["r_idx"], ...]
        g = input[:, flags["g_idx"], ...]
        b = input[:, flags["b_idx"], ...]

        tgi = g - 0.39 * r - 0.61 * b
        # Scale channel from [-1, 1] to [0, 1].
        tgi = 0.5 * (tgi + 1)
        tgi = tgi.unsqueeze(dim=1)

        return torch.cat((input, tgi), dim=1)
