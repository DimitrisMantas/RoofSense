from enum import UNIQUE, StrEnum, auto, verify
from typing import Literal

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss
from torchseg.losses import (MULTICLASS_MODE,
                             DiceLoss,
                             JaccardLoss,
                             LovaszLoss,
                             TverskyLoss, )


@verify(UNIQUE)
class DistributionLoss(StrEnum):
    CROSS_ENTROPY = auto()
    FOCAL = auto()


@verify(UNIQUE)
class RegionLoss(StrEnum):
    DICE = auto()
    JACCARD = auto()
    LOVASZ = auto()
    TVERSKY = auto()


class FocalLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        gamma: float = 2,
        reduction: Literal["mean", "none", "sum"] = "mean",
    ):
        super().__init__(weight=weight, reduction=reduction)

        self.base = CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce = self.base(input, target)
        pt = torch.exp(-ce)

        ls = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            self.weight: Tensor
            # TODO: Figure out why the weights must be gathered.
            at = self.weight.div(self.weight.sum()).gather(0, target.data.view(-1))
            ls *= at

        if self.reduction == "mean":
            return ls.mean()
        elif self.reduction == "sum":
            return ls.sum()
        else:
            return ls


class CompoundLoss(_WeightedLoss):
    def __init__(
        self,
        base: DistributionLoss,
        other: RegionLoss,
        weight: Tensor | None = None,
        ignore_index: int | None = None,
        reduction: Literal["mean", "none", "sum"] = "mean",
        base_kwargs: dict[str, float] | None = None,
        other_kwargs: dict[str, bool | float] | None = None,
        base_lambda: float = 1,
        other_lambda: float = 1,
    ):
        super().__init__(weight, reduction=reduction)

        common_kwargs = {
            "weight": weight,
            # NOTE: Region-based losses require this parameter to be specified.
            "ignore_index": -100 if ignore_index is None else ignore_index,
            "reduction": reduction,
        }
        optional_kwargs = base_kwargs if base_kwargs is not None else {}

        if base == DistributionLoss.CROSS_ENTROPY:
            self.base = CrossEntropyLoss(**common_kwargs, **optional_kwargs)
        elif base == DistributionLoss.FOCAL:
            self.base = FocalLoss(**common_kwargs, **optional_kwargs)
        else:
            raise TypeError(
                f"Failed to parse base loss: {base!r}.Expected {DistributionLoss.__class__.__name__} or string equivalent, but got {type(base)!r} instead. Invalid type."
            )

        common_kwargs = {"mode": MULTICLASS_MODE, "ignore_index": ignore_index}
        optional_kwargs = other_kwargs if other_kwargs is not None else {}
        if other == RegionLoss.DICE:
            self.other = DiceLoss(**common_kwargs, **optional_kwargs)
        elif other == RegionLoss.JACCARD:
            self.other = JaccardLoss(**common_kwargs, **optional_kwargs)
        elif other == RegionLoss.LOVASZ:
            self.other = LovaszLoss(**common_kwargs, **optional_kwargs)
        elif other == RegionLoss.TVERSKY:
            self.other = TverskyLoss(**common_kwargs, **optional_kwargs)
            raise TypeError(
                f"Failed to parse base loss: {base!r}.Expected {RegionLoss.__class__.__name__} or string equivalent, but got {type(base)!r} instead. Invalid type."
            )

        self.base_lambda = base_lambda
        self.other_lambda = other_lambda

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.base_lambda * self.base(
            input, target
        ) + self.other_lambda * self.other(input, target)
