from enum import UNIQUE, StrEnum, auto, verify
from typing import Literal

import monai.losses
import torch
from torch import Tensor


@verify(UNIQUE)
class DistribLoss(StrEnum):
    CROSS = auto()
    FOCAL = auto()


@verify(UNIQUE)
class RegionLoss(StrEnum):
    DICE = auto()
    JACC = auto()


# TODO: Add single-component loss support.
class CompoundLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        this: DistribLoss,
        that: RegionLoss,
        ignore_background: bool = True,
        weight: Tensor | None = None,
        reduction: Literal["mean", "none", "sum"] = "mean",
        this_kwargs: dict[str, bool | float] | None = None,
        that_kwargs: dict[str, bool | float] | None = None,
        this_lambda: float = 1,
        that_lambda: float = 1,
    ):
        super().__init__(reduction=reduction)

        if weight is not None:
            # NOTE: MONAI does not normalize class weights.
            weight /= weight.sum()

        common_kwargs = {"weight": weight, "reduction": reduction}
        variable_kwargs = this_kwargs if this_kwargs is not None else {}

        if this == DistribLoss.CROSS:
            self.this = torch.nn.CrossEntropyLoss(
                ignore_index=0 if ignore_background else -100,
                **common_kwargs,
                **variable_kwargs,
            )
        elif this == DistribLoss.FOCAL:
            self.this = monai.losses.FocalLoss(
                include_background=not ignore_background,
                to_onehot_y=True,
                use_softmax=True,
                **common_kwargs,
                **variable_kwargs,
            )
        else:
            raise ValueError

        common_kwargs = {
            "include_background": not ignore_background,
            "to_onehot_y": True,
            "softmax": True,
            "reduction": reduction,
            "weight": weight[1:]
            if ignore_background and weight is not None
            else weight,
        }
        variable_kwargs = that_kwargs if that_kwargs is not None else {}
        if that == RegionLoss.DICE:
            self.that = monai.losses.DiceLoss(
                jaccard=False, **common_kwargs, **variable_kwargs
            )
        elif that == RegionLoss.JACC:
            self.that = monai.losses.DiceLoss(
                jaccard=True, **common_kwargs, **variable_kwargs
            )

        self.this_lambda = this_lambda
        self.that_lambda = that_lambda

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # NOTE: Target tensors are of shape BxHxW be MONAI requires it to be Bx1xHxW.
        target = torch.unsqueeze(target, dim=1)

        return self.this_lambda * self.this(
            input,
            torch.squeeze(target, dim=1)
            if isinstance(self.this, torch.nn.CrossEntropyLoss)
            else target,
        ) + self.that_lambda * self.that(input, target)
