from collections.abc import Sequence
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric


class MacroAverageWrapper(WrapperMetric):
    """Compute the macroscopic average of an unreduced metric."""

    # TODO: Find out what this property does.
    full_state_update = True

    mean: torch.Tensor

    def __init__(self, base_metric: Metric, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected base metric to be an instance of `torchmetrics.Metric` but received {base_metric}."
            )
        self._base_metric = base_metric

        self.mean = torch.tensor(torch.nan)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the underlying metric."""
        self._base_metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        """Compute the underlying metric as well as max and min values for this metric.

        Returns a dictionary that consists of the computed value (``raw``), as well as the minimum (``min``) and maximum
        (``max``) values.

        """
        val = self._base_metric.compute()
        if not self._is_suitable_val(val):
            raise RuntimeError(
                f"Returned value from base metric should be a float or scalar tensor, but got {val}."
            )
        self.mean = val.nanmean()
        return self.mean

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the original forward method of the base metric class."""
        return super(WrapperMetric, self).forward(*args, **kwargs)

    def reset(self) -> None:
        """Set ``max_val`` and ``min_val`` to the initialization bounds and resets the base metric."""
        super().reset()
        self._base_metric.reset()

    @staticmethod
    def _is_suitable_val(val: Union[float, Tensor]) -> bool:
        """Check whether min/max is a scalar value."""
        if isinstance(val, (int, float)):
            return False
        if isinstance(val, Tensor):
            return True
        return False

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.wrappers import MinMaxMetric
            >>> from torchmetrics.classification import BinaryAccuracy
            >>> metric = MinMaxMetric(BinaryAccuracy())
            >>> metric.update(torch.randint(2, (20,)), torch.randint(2, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.wrappers import MinMaxMetric
            >>> from torchmetrics.classification import BinaryAccuracy
            >>> metric = MinMaxMetric(BinaryAccuracy())
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.randint(2, (20,)), torch.randint(2, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
