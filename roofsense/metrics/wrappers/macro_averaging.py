from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric


class MacroAverageWrapper(WrapperMetric):
    # TODO: Find out what this property does.
    full_state_update = True

    mean: torch.Tensor

    def __init__(
        self, base_metric: Metric, ignore_index: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected base metric to be an instance of `torchmetrics.Metric` but received {base_metric}."
            )
        self._base_metric = base_metric
        self.ignore_index = ignore_index
        self.mean = torch.tensor(torch.nan)

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._base_metric.update(*args, **kwargs)

    def compute(self) -> dict[str, Tensor]:
        val = self._base_metric.compute()
        if not self._is_suitable_val(val):
            raise RuntimeError(
                f"Returned value from base metric should be a float or scalar tensor, but got {val}."
            )
        if self.ignore_index is not None:
            val[self.ignore_index] = torch.nan
        self.mean = val.nanmean()
        return self.mean

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super(WrapperMetric, self).forward(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._base_metric.reset()

    @staticmethod
    def _is_suitable_val(val: float | Tensor) -> bool:
        if isinstance(val, (int, float)):
            return False
        if isinstance(val, Tensor):
            return True
        return False

    def plot(
        self, val: Tensor | Sequence[Tensor] | None = None, ax: _AX_TYPE | None = None
    ) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)
