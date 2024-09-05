from collections.abc import Generator
from itertools import product
from typing import Any, Union, no_type_check

import numpy as np
from torch import Tensor
from torchmetrics.utilities.imports import (_LATEX_AVAILABLE,
                                            _MATPLOTLIB_AVAILABLE,
                                            _SCIENCEPLOT_AVAILABLE, )
from torchmetrics.utilities.plot import _get_col_row_split, _get_text_color, trim_axs

if _MATPLOTLIB_AVAILABLE:
    import matplotlib
    import matplotlib.axes
    import matplotlib.pyplot as plt

    _PLOT_OUT_TYPE = tuple[plt.Figure, matplotlib.axes.Axes | np.ndarray]
    _AX_TYPE = matplotlib.axes.Axes
    _CMAP_TYPE = Union[matplotlib.colors.Colormap, str]

    style_change = plt.style.context
else:
    _PLOT_OUT_TYPE = tuple[object, object]  # type: ignore[misc]
    _AX_TYPE = object
    _CMAP_TYPE = object  # type: ignore[misc]

    from contextlib import contextmanager

    @contextmanager
    def style_change(*args: Any, **kwargs: Any) -> Generator:
        """No-ops decorator if matplotlib is not installed."""
        yield


if _SCIENCEPLOT_AVAILABLE:
    import scienceplots  # noqa: F401

    _style = ["science", "no-latex"]

_style = ["science"] if _SCIENCEPLOT_AVAILABLE and _LATEX_AVAILABLE else ["default"]


def _error_on_missing_matplotlib() -> None:
    """Raise error if matplotlib is not installed."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ModuleNotFoundError(
            "Plot function expects `matplotlib` to be installed. Please install with `pip install matplotlib`"
        )


@style_change(_style)
@no_type_check
def plot_confusion_matrix(
    confmat: Tensor,
    ax: _AX_TYPE | None = None,
    add_text: bool = True,
    labels: list[int | str] | None = None,
    cmap: _CMAP_TYPE | None = None,
) -> _PLOT_OUT_TYPE:
    """Plot an confusion matrix.

    Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/confusion_matrix.py.
    Works for both binary, multiclass and multilabel confusion matrices.

    Args:
        confmat: the confusion matrix. Either should be an [N,N] matrix in the binary and multiclass cases or an
            [N, 2, 2] matrix for multilabel classification
        ax: Axis from a figure. If not provided, a new figure and axis will be created
        add_text: if text should be added to each cell with the given value
        labels: labels to add the x- and y-axis
        cmap: matplotlib colormap to use for the confusion matrix
            https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Returns:
        A tuple consisting of the figure and respective ax objects (or array of ax objects) of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed

    """
    _error_on_missing_matplotlib()

    if confmat.ndim == 3:  # multilabel
        nb, n_classes = confmat.shape[0], 2
        rows, cols = _get_col_row_split(nb)
    else:
        nb, n_classes, rows, cols = 1, confmat.shape[0], 1, 1

    if labels is not None and confmat.ndim != 3 and len(labels) != n_classes:
        raise ValueError(
            "Expected number of elements in arg `labels` to match number of labels in confmat but "
            f"got {len(labels)} and {n_classes}"
        )
    if confmat.ndim == 3:
        fig_label = labels or np.arange(nb)
        labels = list(map(str, range(n_classes)))
    else:
        fig_label = None
        labels = labels or np.arange(n_classes).tolist()

    fig, axs = (
        plt.subplots(nrows=rows, ncols=cols,figsize=(4,4), constrained_layout=True)
        if ax is None
        else (ax.get_figure(), ax)
    )
    axs = trim_axs(axs, nb)
    for i in range(nb):
        ax = axs[i] if rows != 1 and cols != 1 else axs
        if fig_label is not None:
            ax.set_title(f"Label {fig_label[i]}")
        im = ax.imshow(
            confmat[i].cpu().detach() if confmat.ndim == 3 else confmat.cpu().detach(),
            cmap=cmap,
        )
        if i // cols == rows - 1:  # bottom row only
            ax.set_xlabel("Predicted Class")
        if i % cols == 0:  # leftmost column only
            ax.set_ylabel("True Class")
        ax.minorticks_off()
        ax.set_xticks(list(range(n_classes)))
        ax.set_yticks(list(range(n_classes)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        if add_text:
            for ii, jj in product(range(n_classes), range(n_classes)):
                val = confmat[i, ii, jj] if confmat.ndim == 3 else confmat[ii, jj]
                patch_color = im.cmap(im.norm(val.item()))
                c = _get_text_color(patch_color)
                ax.text(
                    jj, ii, str(round(val.item(), 2)), ha="center", va="center", color=c
                )

    return fig, axs
