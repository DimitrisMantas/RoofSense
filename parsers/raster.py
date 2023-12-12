import math
from typing import Optional, Sequence, Union

import numpy as np
import pyproj
import rasterio

import utils


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

        "count": 1, "crs": pyproj.CRS("EPSG:28992"), "nodata": 3.4028234663852886e+38}


class Raster:
    """
    A square-cell raster.
    """

    # This suppression is so that PyCharm does not complain about Iterable not declaring a concrete implementation of
    # __getitem__.
    # noinspection PyUnresolvedReferences
    def __init__(self,
                 resolution: float,
                 bbox: Sequence[float],
                 profile: Optional[dict[str, str]] = None) -> None:
        """
        Creates an empty raster of a given resolution and profile.

        The value of each cell of the raster is initially set to the ``nodata`` value defined in its profile.

        Args:
            resolution:
                A non-negative number representing the side length (i.e., the size) of each cell of the raster in the
                units used by the coordinate reference system defined in its profile.
            bbox:
                An iterable containing at least four real numbers defining the axis-aligned minimum bounding rectangle
                of the raster in the order: ``(min_x, min_y, max_x, max_y)``.
            profile:
                A string-keyed dictionary defining the profile of the raster. See :class:`Profiles<raster.Profiles>` for
                more information.
        """
        if profile is None:
            self.profile = Profiles.AHN3

        self.__cell_size = resolution

        self.bbox = bbox
        self.len_x = math.ceil((self.bbox[2] - self.bbox[0]) / resolution)
        self.len_y = math.ceil((self.bbox[3] - self.bbox[1]) / resolution)

        self.__data = np.full([self.len_y, self.len_x], self.profile["nodata"])

    def __getitem__(self, idx: Union[int, Sequence[int]]) -> None:
        return self.__data[idx]

    def __setitem__(self, idx: Union[int, Sequence[int]], val: float) -> None:
        self.__data[idx] = val

    def save(self, filename: str) -> None:
        """
        Saves the raster to a file.

        Args:
            filename:
                A string representing the relative or absolute system path to the output file.
        """

        # Create the required directory to store the output file.
        utils.mkdirs(filename)

        # Define an appropriate transformation to map the position of each cell from raster space to real-world
        # coordinates.
        # noinspection PyUnresolvedReferences
        ij_to_xy = rasterio.transform.from_origin(self.bbox[0],
                                                  self.bbox[3],
                                                  self.__cell_size,
                                                  self.__cell_size)

        with rasterio.open(filename,
                           "w",
                           width=self.len_x,
                           height=self.len_y,
                           transform=ij_to_xy,
                           **self.profile) as f:
            f.write(self.__data, 1)
