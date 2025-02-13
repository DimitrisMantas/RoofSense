import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import rasterio

from roofsense.utils.file import confirm_write_op


def to_clr(
    class_names: dict[int, str],
    class_colors: dict[int, Iterable[int]],
    filepath: str | bytes | os.PathLike[str],
    overwrite: bool = False,
    ignore_background: bool = True,
    invert_bw: bool = True,
) -> None:
    """Export the input class names and colors to a QGIS color map.

    Args:
        class_names:
            The input names, keyed sequentially by class index starting from zero.
            Must be able to be joined to the input colors.
        class_colors:
            The input colors, keyed sequentially by class index starting from zero.
            Each color is defined using the RGBA model.
            Must be able to be joined to the input names.
        filepath:
            The destination filepath.
        overwrite:
            'True' to overwrite any previous output if it exists; 'False' otherwise.
        ignore_background:
            'True' to not specify a color for the background class; 'False' otherwise.
        invert_bw:
            'True' to map white to black and vice versa; 'False' otherwise.
    """
    if not confirm_write_op(filepath, overwrite):
        return

    lines = []
    for i, (name, color) in enumerate(
        zip(class_names.values(), class_colors.values(), strict=True)
    ):
        if ignore_background and i == 0:
            continue
        final_color = [c for c in color[:3]]
        if invert_bw:
            s = sum(final_color)
            if s == 0:
                final_color = [255, 255, 255]
            elif s == 255 * 3:
                final_color = [0, 0, 0]
        final_color.append(color[-1])
        lines.append(f"{i} {' '.join(map(str, final_color))} {name}\n")

        with open(filepath, mode="w") as f:
            f.writelines(lines)
            f.write("\n")


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
