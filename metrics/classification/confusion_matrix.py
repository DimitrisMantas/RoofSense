from typing import Any, Literal

import torchmetrics.classification
from torch import Tensor
from torchmetrics.utilities.plot import _AX_TYPE, _CMAP_TYPE, _PLOT_OUT_TYPE
from typing_extensions import override

from metrics.utilities.plot import plot_confusion_matrix


class MulticlassConfusionMatrix(torchmetrics.classification.MulticlassConfusionMatrix):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        normalize: Literal["none", "true", "pred", "all"] | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, ignore_index, normalize, validate_args, **kwargs)

    @override
    def plot(
        self,
        val: Tensor | None = None,
        ax: _AX_TYPE | None = None,
        add_text: bool = True,
        labels: list[str] | None = None,
        cmap: _CMAP_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        val = val if val is not None else self.compute()
        if not isinstance(val, Tensor):
            raise TypeError(f"Expected val to be a single tensor but got {val}")
        fig, ax = plot_confusion_matrix(
            val, ax=ax, add_text=add_text, labels=labels, cmap=cmap
        )
        return fig, ax
