from enum import UNIQUE, Enum, auto, verify
from typing import Literal

import monai.losses
import torch
from torch import Tensor


@verify(UNIQUE)
class DistribBasedLoss(Enum):
    CROSS = auto()
    FOCAL = auto()


@verify(UNIQUE)
class RegionBasedLoss(Enum):
    DICE = auto()
    JACC = auto()


# TODO: Add single-component loss support.
class CompoundLoss(
    # NOTE: This is technically a weighted loss, but inheriting from its parent
    # avoids registering an unnecessary weight buffer.
    torch.nn.modules.loss._Loss
):
    """Composite loss function composed of a distribution- and, optionally, a region-based component."""

    def __init__(
        self,
        this: DistribBasedLoss,
        that: RegionBasedLoss,
        ignore_background: bool = True,
        weight: Tensor | None = None,
        reduction: Literal["mean", "sum"] = "mean",
        this_kwargs: dict[str, bool | float] | None = None,
        that_smooth: bool = False,
        that_kwargs: dict[str, bool | float] | None = None,
        this_lambda: float = 1,
        that_lambda: float = 1,
    ):
        super().__init__(reduction=reduction)

        if weight is not None:
            # NOTE: MONAI does not normalize class weights.
            weight: Tensor
            weight /= weight.sum()

        self.weight = weight

        common_kwargs = {"reduction": reduction}
        variable_kwargs = this_kwargs if this_kwargs is not None else {}

        if this == DistribBasedLoss.CROSS:
            self.this = torch.nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=0 if ignore_background else -100,
                **common_kwargs,
                **variable_kwargs,
            )
        elif this == DistribBasedLoss.FOCAL:
            self.this = monai.losses.FocalLoss(
                include_background=not ignore_background,
                to_onehot_y=True,
                # weight=weight[1:]
                # if ignore_background and weight is not None
                # else weight,
                use_softmax=True,
                **common_kwargs,
                **variable_kwargs,
            )
            # to be able to load checkpoints
            # self.this.class_weight = (
            #     weight if weight is None else weight[1:].view((-1, 1, 1))
            # )
        else:
            raise ValueError

        common_kwargs = {
            "include_background": not ignore_background,
            "to_onehot_y": True,
            "other_act": lambda pred: pred.log_softmax(dim=1).exp(),
            "reduction": reduction,
            # "weight": weight[1:]
            # if ignore_background and weight is not None
            # else weight,
        }
        variable_kwargs = that_kwargs if that_kwargs is not None else {}

        if that == RegionBasedLoss.DICE:
            self.that = monai.losses.DiceLoss(
                jaccard=False, **common_kwargs, **variable_kwargs
            )
        elif that == RegionBasedLoss.JACC:
            self.that = monai.losses.DiceLoss(
                jaccard=True, **common_kwargs, **variable_kwargs
            )
        else:
            raise ValueError

        # to be able to load checkpoints
        # self.that.class_weight = weight if weight is None else weight[1:]

        self.that_smooth = that_smooth

        self.this_lambda = this_lambda
        self.that_lambda = that_lambda

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # NOTE: Target tensors are of shape BxHxW be MONAI requires it to be Bx1xHxW.
        target = torch.unsqueeze(target, dim=1)

        this_val: Tensor = self.this(
            input,
            torch.squeeze(target, dim=1)
            if isinstance(self.this, torch.nn.CrossEntropyLoss)
            else target,
        )

        that_val: Tensor = self.that(input, target)
        if self.that_smooth:
            that_val = that_val.cosh().log()

        # x = monai.metrics.compute_iou(
        #     input.log_softmax(dim=1).exp(),
        #     monai.losses.dice.one_hot(target,num_classes=9),
        #     include_background=self.that.include_background,
        # )
        # x[torch.isnan(x)] = 0.0
        # x=x.mean()

        return self.this_lambda * this_val + self.that_lambda * that_val
