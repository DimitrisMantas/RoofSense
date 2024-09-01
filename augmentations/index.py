from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from typing_extensions import override


class AppendTGI(IntensityAugmentationBase2D):
    """Compute and append the triangular greenness index to each batch sample."""

    def __init__(self, red_idx: int = 0, green_idx: int = 1, blue_idx: int = 2) -> None:
        r"""Configure the appender.

        Args:
            red_idx:
                The index of the red input channel.
            green_idx:
                The index of the green input channel.
            blue_idx:
                The index of the blue input channel.

        Notes:
            Tje RGB values are expected to be in the interval :math:`\left[0, 1\right]`.
        """
        super().__init__(p=1)
        self.flags = {"red_idx": red_idx, "green_idx": green_idx, "blue_idx": blue_idx}

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        red = input[:, flags["red_idx"], ...]
        green = input[:, flags["green_idx"], ...]
        blue = input[:, flags["blue_idx"], ...]

        tgi = green - 0.39 * red - 0.61 * blue
        # Rescale band from [-1, 1] to [0, 1].
        tgi = 0.5 * (tgi + 1)
        tgi = tgi.unsqueeze(dim=1)

        return torch.cat((input, tgi), dim=1)
