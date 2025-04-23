from collections.abc import Iterable
from functools import partial
from types import ModuleType
from typing import Literal, Any, cast
from typing_extensions import override
import monai
import torch
from monai.losses import (
    DiceLoss,
    FocalLoss,
    GeneralizedDiceLoss,
    HausdorffDTLoss,
    TverskyLoss,
)
from torch import Tensor
from torch.nn import CrossEntropyLoss

Loss = (
    CrossEntropyLoss
    | DiceLoss
    | GeneralizedDiceLoss
    | FocalLoss
    | TverskyLoss
    | HausdorffDTLoss
)


class CrossEntropyBasedCompositeLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_background: bool = True,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        complement: str | type[torch.nn.modules.loss._Loss] | None = None,
        # The complement loss name.
        complement_cfg: dict[str, Any] = None,
        # The compliment loss configuration.
        lambdas: Iterable[float] = (1, 1),
        # The compliment loss weight in the total loss calculations.
        loss_reduction: Literal["mean", "mul", "sum"] = "sum",
        # The loss component reduction strategy.
    ) -> None:
        super().__init__(
            weight,
            ignore_index=0 if ignore_background else -100,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

        if complement is None:
            return

        self.is_monai_used: bool = False
        try:
            complement_cls = getattr(monai.losses, complement)
        except AttributeError:
            complement_cls = complement
        else:
            self.is_monai_used = True

        complement_cls = cast(complement_cls.__base__, complement_cls)

        self.complement = complement_cls(
            **complement_cfg if complement_cfg is not None else {}
        )

        self.lambdas = lambdas
        self.loss_reduction = loss_reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        this = super().forward(input, target)

        if self.is_monai_used:
            # NOTE: Target tensors are of shape BxHxW but MONAI requires it to be BxCxHxW.
            target = torch.unsqueeze(target, dim=1)
        that=self.complement(input, target)

        if self.loss_reduction == "mean":
            pass
        elif self.loss_reduction == "mul":
            pass
        elif self.loss_reduction == "sum":
            pass

class CompoundLoss(torch.nn.modules.loss._Loss):
    """Compound loss function."""

    losses = {
        loss.__name__.lower(): loss
        for loss in [
            CrossEntropyLoss,
            DiceLoss,
            GeneralizedDiceLoss,
            FocalLoss,
            TverskyLoss,
            HausdorffDTLoss,
        ]
    }
    """The supported loss constituents."""

    def __init__(
        self,
        names: Loss | Iterable[Loss],
        lambdas: float | Iterable[float] | None = None,
        weight: Tensor | None = None,
        include_background: bool = True,
        reduction: Literal["mean", "sum"] = "mean",
        label_smoothing: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(reduction=reduction)

        self.names: list[Loss] = [names] if isinstance(names, str) else names
        for name in self.names:
            name = name.lower()
            loss = self.losses.get(name, None)
            if loss is None:
                raise ValueError(
                    f"Expected name in {list(self.losses.keys())!r}, but got {name!r}."
                )
            if loss.__module__.split(".", maxsplit=1)[0] == "torch":
                # Cross-Entropy Loss
                setattr(
                    self,
                    name,
                    loss(
                        weight=weight,
                        ignore_index=-100 if include_background else 0,
                        reduction=reduction,
                        label_smoothing=label_smoothing,
                    ),
                )
            else:
                setattr(
                    self,
                    name,
                    init_monai_loss(loss, include_background, reduction, **kwargs),
                )

        self.lambdas = lambdas if lambdas is not None else [1] * len(self.names)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # NOTE: Target tensors are of shape BxHxW but MONAI requires it to be BxCxHxW.
        target = torch.unsqueeze(target, dim=1)

        loss = torch.tensor([0], dtype=input.dtype, device=input.device)
        for name, lambda_ in zip(self.names, self.lambdas, strict=True):
            component = getattr(self, name)

            temp = (
                component(
                    input,
                    torch.squeeze(target, dim=1)
                    if isinstance(component, torch.nn.CrossEntropyLoss)
                    else target,
                )
                * lambda_
            )
            loss += temp

        return loss


def init_monai_loss(
    loss: type[
        DiceLoss | GeneralizedDiceLoss | FocalLoss | TverskyLoss | HausdorffDTLoss
    ],
    include_background: bool = True,
    reduction: Literal["mean", "sum"] = "mean",
    **kwargs,
) -> DiceLoss | GeneralizedDiceLoss | FocalLoss | TverskyLoss | HausdorffDTLoss:
    params = {
        "include_background": include_background,
        "to_onehot_y": True,
        "other_act": lambda x: x.log_softmax(dim=1).exp(),
        "reduction": reduction,
    }
    try:
        out = loss(**params, **kwargs)
    except TypeError as e:
        if loss.__name__ == "FocalLoss":
            params["use_softmax"] = params.pop("other_act")
            out = loss(**params, **kwargs)
        else:
            raise e
    return out
