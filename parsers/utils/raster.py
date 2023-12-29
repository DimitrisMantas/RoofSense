from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pyproj
import rasterio


class Raster:
    def __init__(
        self,
        resolution: float,
        extents: Sequence[float],
        profile: Optional[dict[str, str]] = None,
    ) -> None:
        # grid = raster.Raster(CELL_SIZE, dt.get_bbox())
        #
        # # FIXME: Integrate this block into the Raster initializer.
        # # Construct the grid.
        # # TODO: Add documentation.
        # rows, cols = np.mgrid[grid.len_y - 1:-1:-1, 0:grid.len_x]
        # # TODO: Add documentation.
        # xx = grid.bbox[0] + CELL_SIZE * (cols + 0.5)
        # yy = grid.bbox[1] + CELL_SIZE * (rows + 0.5)
        # # TODO: Add documentation.
        # cells = np.column_stack([xx.ravel(), yy.ravel()])

        if profile is None:
            self.profile = Profiles.AHN3

        self.__cell_size = resolution

        self.bbox = extents
        self.len_x = math.ceil((self.bbox[2] - self.bbox[0]) / resolution)
        self.len_y = math.ceil((self.bbox[3] - self.bbox[1]) / resolution)

        self.__data = np.full((self.len_y, self.len_x), self.profile["nodata"])

    def xy(self, index: Sequence[int]) -> tuple[float, float]:
        x = self.bbox[0] + self.__cell_size * (index[0] + 0.5)
        y = self.bbox[1] + self.__cell_size * (index[1] + 0.5)
        return x, y

    def __getitem__(self, idx: int | Sequence[int]) -> None:
        if isinstance(idx, int):
            return self.__data[divmod()]

    def __setitem__(self, idx: int | Sequence[int], val: float) -> None:
        self.__data[idx] = val

    @property
    def width(self):
        return self.len_x

    @property
    def height(self):
        return self.len_y

    def crop(self, extents: Sequence[float]) -> Raster:
        return self

    def save(self, filename: str) -> None:
        # Define an appropriate transformation to map the position of each cell from raster space to real-world
        # coordinates.
        # noinspection PyUnresolvedReferences
        ij_to_xy = rasterio.transform.from_origin(
            self.bbox[0], self.bbox[3], self.__cell_size, self.__cell_size
        )

        with rasterio.open(
            filename,
            "w",
            width=self.len_x,
            height=self.len_y,
            transform=ij_to_xy,
            **self.profile,
        ) as f:
            f.write(self.__data, 1)


class Profiles:
    """
    A collection of named raster profiles.
    """

    # A[n] [tiled, band-interleaved,] LZW-compressed, 32-bit floating-point GeoTIFF raster which is compatible with
    # corresponding AHN3 datasets.
    AHN3 = {  # "tiled": True,
        # "blockxsize": 256,
        # "blockysize": 256,
        # "interleave": "band",
        "compress": "lzw",
        "dtype": "float32",
        "driver": "GTiff",
        "count": 4,
        "crs": pyproj.CRS("EPSG:28992"),
        "nodata": 3.4028234663852886e38,
    }
