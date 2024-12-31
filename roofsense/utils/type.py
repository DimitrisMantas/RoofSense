# TODO - Reformat, finalize function and variable names, and add documentation.
from collections.abc import Sequence
from typing import Literal, Required, TypedDict

BoundingBoxLike = Sequence[float]


####################################
class MetricKwargs(TypedDict, total=False):
    """Performance metric keyword arguments."""

    num_classes: Required[int]
    average: Literal["micro", "macro", "weighted", "none"] | None
    ignore_index: Required[int | None]
    zero_division: float
