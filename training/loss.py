import warnings
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
    # NOTE: This is technically a weighted loss, but inheriting from the
    # corresponding parent class avoids registering an unnecessary weight buffer.
    # This is because MONAI losses maintain their own weight buffers.
    torch.nn.modules.loss._WeightedLoss
):
    def __init__(
        self,
        this: DistribBasedLoss,
        that: RegionBasedLoss,
        ignore_background: bool = True,
        weight: Tensor | None = None,
        reduction: Literal["mean", "sum"] = "mean",
        this_kwargs: dict[str, bool | float] | None = None,
        that_kwargs: dict[str, bool | float] | None = None,
        this_lambda: float = 1,
        that_lambda: float = 1,
    ):
        if weight is not None:
            # NOTE: MONAI does not normalize class weights.
            weight: Tensor
            weight /= weight.sum()

        super().__init__(weight=weight, reduction=reduction)

        common_kwargs = {
            "include_background": not ignore_background,
            "to_onehot_y": True,
        }
        variable_kwargs = this_kwargs if this_kwargs is not None else {}

        if this == DistribBasedLoss.CROSS:
            self.this = torch.nn.CrossEntropyLoss(
                # NOTE: PyTorch expects the ignored index to always be specified.
                ignore_index=0 if ignore_background else -100,
                # NOTE: PyTorch works as expected regarding class weights and loss
                # reductions.
                weight=weight,
                reduction=reduction,
                **variable_kwargs,
            )
        elif this == DistribBasedLoss.FOCAL:
            # TODO: Adjust FocalLoss to return the same results as CrossEntropyLoss when
            #  the corresponding modulating factor is equal to one.
            warnings.warn(
                "This loss is experimental and its implementation may not "
                "be correct!",
                UserWarning,
            )
            self.this = monai.losses.FocalLoss(
                reduction=reduction,
                use_softmax=True,
                **common_kwargs,
                **variable_kwargs,
            )
        else:
            raise ValueError

        common_kwargs |= {
            # NOTE: This activation function is more numerically stable than standard
            # softmax in that it helps avoid vanishing gradients.
            "other_act": lambda pred: pred.log_softmax(dim=1).exp(),
            # NOTE: MONAI expects the weight tensor to not include the background
            # class when it is excluded.
            # "weight": weight if weight is None else weight[1:],
            # NOTE: MONAI computes weighted loss sums instead of averages, so any
            # reduction must be applied manually to the classwise results.
            "reduction": "none",
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
        # Broadcast to BxC.
        that_val = that_val.view(*that_val.shape[:2])

        # Apply the class weights.
        if self.weight is not None:
            that_val *= self.weight[1:]

        # Perform the reduction.
        if self.reduction == "mean":
            # Sum along the class dimension to take the weighted loss mean and
            # average the result across all batches.
            that_val = that_val.sum(dim=1).mean()
        else:
            that_val = that_val.sum()

        return self.this_lambda * this_val + self.that_lambda * that_val
