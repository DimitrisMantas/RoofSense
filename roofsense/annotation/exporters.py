import os
from typing import Any

import numpy as np
import rasterio.plot


def to_png(
    data: np.ndarray[[Any, Any, Any], np.dtype[np.number]],
    filepath: str | bytes | os.PathLike,
) -> None:
    """Export the RGB bands of the input raster to a PNG file.

    Args:
        data:
            The input raster. Must contain at least three bands in the RGB configuration.
        filepath:
            The destination filepath.

    Returns:
        None
    """
    with rasterio.open(
        filepath,
        mode="w",
        count=3,
        height=data.shape[1],
        width=data.shape[2],
        dtype=np.uint8,
    ) as f:
        f.write(data[:3, ...])
