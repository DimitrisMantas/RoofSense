from typing import Literal

from segmentation_models_pytorch.losses import JaccardLoss
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss


class CrossEntropyJaccardLoss(_WeightedLoss):
    def __init__(
        self,
        # Input
        num_classes: int,
        ignore_index: int = -100,
        # CE
        class_weights: Tensor | None = None,
        reduction: Literal["mean", "sum"] = "mean",
        label_smoothing: float = 0,
        # IoU
        log_iou: bool = False,
        # Output
        nll_activation: float = 1,
        iou_activation: float = 1,
    ):
        super().__init__(class_weights, reduction=reduction)

        self.nll = CrossEntropyLoss(
            class_weights,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.iou = JaccardLoss(
            "multiclass",
            classes=[i for i in range(num_classes) if i != ignore_index],
            log_loss=log_iou,
            smooth=label_smoothing,
        )

        self.nll_activation = nll_activation
        self.iou_activation = iou_activation

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.nll_activation * self.nll(
            input, target
        ) + self.iou_activation * self.iou(input, target)
