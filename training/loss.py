from collections.abc import Iterable
from typing import Literal

import torch
from monai.losses import (DiceLoss,
                          FocalLoss,
                          GeneralizedDiceLoss,
                          HausdorffDTLoss,
                          TverskyLoss, )
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


class CompoundLoss1(torch.nn.modules.loss._Loss):
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

        self.names: Iterable[Loss] = [names] if isinstance(names, str) else names
        for name in names:
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

        self.lambdas = lambdas if lambdas is not None else [1] * len(names)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # NOTE: Target tensors are of shape BxHxW but MONAI requires it to be BxCxHxW.
        target = torch.unsqueeze(target, dim=1)

        loss = torch.tensor([0], dtype=input.dtype, device=input.device)
        for name, lambda_ in zip(self.names, self.lambdas, strict=True):
            component = getattr(self, name)
            loss += (
                component(
                    input,
                    torch.squeeze(target, dim=1)
                    if isinstance(component, torch.nn.CrossEntropyLoss)
                    else target,
                )
                * lambda_
            )

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
