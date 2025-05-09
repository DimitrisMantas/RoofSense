from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from typing_extensions import override


class MinMaxScaling(IntensityAugmentationBase2D):
    r"""Scale each band of each batch sample to the interval :math:`\left[0, 1\right]`."""

    def __init__(self, mins: Tensor, maxs: Tensor, eps: float = 1e-12) -> None:
        r"""Initialize the augmentation.

        Args:
            mins:
                An 1D tensor of shape :math:`\left(C,\right)` containing the global  minimum of each input  band.
            maxs:
                An 1D tensor of shape :math:`\left(C,\right)` containing the global  maximum of each input  band.
            eps:
                A small positive value to avoid division-by-zero errors when the minimum and the maximum values of an input band are equal.

        Notes:
            The scale tensors are automatically transferred to the batch device and cast to the corresponding data type.
        """
        super().__init__(p=1, same_on_batch=True)
        self.flags = {
            "mins": mins.view(1, -1, 1, 1),
            "maxs": maxs.view(1, -1, 1, 1),
            "eps": eps,
        }

    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        mins = torch.as_tensor(flags["mins"], device=input.device, dtype=input.dtype)
        maxs = torch.as_tensor(flags["maxs"], device=input.device, dtype=input.dtype)

        return (input - mins) / (maxs - mins + flags["eps"])
