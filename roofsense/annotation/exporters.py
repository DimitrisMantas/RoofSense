import os
from typing import Any

import numpy as np
import rasterio


def to_png(
    data: np.ndarray[[Any, Any, Any], np.dtype[np.number]],
    filepath: str | bytes | os.PathLike[str],
    r_idx: int = 0,
    g_idx: int = 1,
    b_idx: int = 2,
) -> None:
    """Export the RGB bands of the input raster to a PNG file.

    Args:
        data:
            The input raster. Must contain at least three bands in the RGB configuration.
            The default band order is [0, 1, 2].
        filepath:
            The destination filepath.
        r_idx:
            The index of the red band in the input raster.
        g_idx:
            The index of the green band in the input raster.
        b_idx:
            The index of the blue band in the input raster.

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
        f.write(data[[r_idx, g_idx, b_idx], ...])
