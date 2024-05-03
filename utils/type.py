# TODO - Reformat, finalize function and variable names, and add documentation.
from collections.abc import Callable, Sequence
from typing import Any, Union

import requests

BAG3TileData = dict
BAG3DTileIndexJSON = dict
BAG3DTileIndexData = dict
JSONLike = dict[str, Any]
BoundingBoxLike = Sequence[float]
Timeout = Union[float, tuple[float, float], tuple[float, None]]
HookCallback = Callable[[requests.Response, ...], Any]
AssetManifest = dict[str, list[str]]
