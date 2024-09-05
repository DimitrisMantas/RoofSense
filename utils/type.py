# TODO - Reformat, finalize function and variable names, and add documentation.
from collections.abc import Callable, Sequence
from typing import Any, Literal, Required, TypedDict, Union

import requests

BAG3TileData = dict
BAG3DTileIndexJSON = dict
BAG3DTileIndexData = dict
JSONLike = dict[str, Any]
BoundingBoxLike = Sequence[float]
Timeout = Union[float, tuple[float, float], tuple[float, None]]
HookCallback = Callable[[requests.Response, ...], Any]
AssetManifest = dict[str, list[str]]


####################################
class MetricKwargs(TypedDict, total=False):
    """Performance metric keyword arguments."""

    num_classes: Required[int]
    average: Literal["micro", "macro", "weighted", "none"] | None
    ignore_index: Required[int | None]
    zero_division: float
