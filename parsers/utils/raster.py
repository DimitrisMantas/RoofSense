from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Optional

import numpy as np
import rasterio
import rasterio.fill
import rasterio.mask
import rasterio.merge
import rasterio.windows


class Raster:
    def __init__(
        self,
        size: float,
        bbox: Sequence[float],
        meta: Optional[rasterio.profiles.Profile] = None,
    ) -> None:
        if meta is None:
            self._meta = DefaultProfile()

        self._size = size

        self._bbox = bbox
        self._lenx = math.ceil((self._bbox[2] - self._bbox[0]) / size)
        self._leny = math.ceil((self._bbox[3] - self._bbox[1]) / size)

        # TOSELF: Specify the data type explicitly?
        self._data = np.full((self._leny, self._lenx), self._meta["nodata"])

    # TODO: Add type hints to this method.
    def __getitem__(self, key):
        return self._data[key]

    # TODO: Add type hints to this method.
    def __setitem__(self, key, val):
        self._data[key] = val

    @property
    def width(self) -> int:
        return self._lenx

    @property
    def height(self) -> int:
        return self._leny

    # TODO: Add type hints to this function.
    def xy(self):
        # Place the origin of the grid at its bottom left corner.
        # NOTE: This simplifies the following step.
        rows, cols = np.mgrid[self.height - 1 : -1 : -1, 0 : self.width]

        # Transform the image to world coordinates.
        x = self._bbox[0] + self._size * (cols + 0.5)
        y = self._bbox[1] + self._size * (rows + 0.5)

        return np.column_stack([x.ravel(), y.ravel()])

    def slope(self, degrees: bool = True) -> Raster:
        r = copy.deepcopy(self)
        r._data = np.arctan(np.hypot(*np.gradient(self._data, self._size)))
        if degrees:
            r._data = np.degrees(r._data)
        return r

    def fill(self) -> None:
        mask = ~np.ma.getmaskarray(np.ma.masked_invalid(self._data))
        self._data = rasterio.fill.fillnodata(self._data, mask)

    def save(self, filename: str) -> None:
        count = _get_num_bands(self._data)
        transform = rasterio.transform.from_origin(
            self._bbox[0], self._bbox[3], self._size, self._size
        )

        f: rasterio.io.DatasetWriter
        with rasterio.open(
            filename,
            "w",
            width=self._lenx,
            height=self._leny,
            count=count,
            transform=transform,
            **self._meta,
        ) as f:
            f.write(self._data, indexes=1 if count == 1 else None)


class DefaultProfile(rasterio.profiles.Profile):
    defaults = {  # TODO: See which is the most suitable. no-data value.
        "nodata": np.nan,  # TODO: See which is the most suitable data type.
        "dtype": np.float32,
        # NOTE: Tiled images can be efficiently split into patches by exploiting their
        #       internal data block mechanism.
        "tiled": True,  # TODO: Read the block size from an environment variable.
        "blockxsize": 1024,
        "blockysize": 1024,
        "compress": "LZW",
    }


class MultiBandProfile(rasterio.profiles.Profile):
    defaults = {
        "compress": "LZW",
        # NOTE: The default photometric interpretation of the BM5 imagery is not
        #       compatible with lossless compression.
        "photometric": "RGB",
    }


class SingleBandProfile(rasterio.profiles.Profile):
    defaults = {
        "compress": "LZW",
        # NOTE: The default photometric interpretation of the BM5 imagery is not
        #       compatible with single-band rasters.
        "photometric": "MINISBLACK",
    }


def _get_num_bands(data: np.ndarray) -> int:
    return data.shape[0] if len(data.shape) == 3 else 1
