from __future__ import annotations

import copy
import math
import os
from os import PathLike
from typing import Optional

import numpy as np
import rasterio
import rasterio.fill
import rasterio.mask
import rasterio.merge
import rasterio.windows

import config
from utils.type import BoundingBoxLike

# TODO: Find a way to not rely on the configuration file being initialised for this
#       module to be imported.
config.config()


class Raster:
    def __init__(
        self,
        resol: float,
        bbox: BoundingBoxLike,
        meta: Optional[
            rasterio.profiles.Profile
        ] = None,
    ) -> None:
        self._resol = resol
        self._bbox = bbox
        self._lenx = math.ceil((self._bbox[2] - self._bbox[0]) / self._resol)
        self._leny = math.ceil((self._bbox[3] - self._bbox[1]) / self._resol)
        self._meta = meta if meta is not None else DefaultProfile()
        self.data = np.full((self._leny, self._lenx), self._meta["nodata"],dtype=self._meta["dtype"])

    def __len__(self)->int:
        return self.data.size

    # TODO: Add type hints to this method.
    def __getitem__(self, key):
        return self.data[key]

    # TODO: Add type hints to this method.
    def __setitem__(self, key, val):
        self.data[key] = val
    @property
    def bbox(self):
        return self._bbox
    @property
    def meta(self):
        return self._meta

    @property
    def res(self):
        return self._resol
    @property
    def width(self) -> int:
        return self._lenx

    @property
    def height(self) -> int:
        return self._leny

    def xy(self):
        rows, cols = np.mgrid[0 : self.height, 0 : self.width]

        cells_x = self.bbox[0] + (cols + 0.5) * self.res
        cells_y = self.bbox[3] - (rows + 0.5) * self.res

        return np.vstack([cells_x.ravel(), cells_y.ravel()]).transpose()

    def slope(self, degrees: bool = True) -> Raster:
        r = copy.deepcopy(self)
        r.data = np.arctan(np.hypot(*np.gradient(self.data, self._resol)))
        if degrees:
            r.data = np.degrees(r.data)
        return r

    def fill(self, radius: float = 100, smoothing_iters: int = 0) -> None:
        # TODO: Check whether there is a more general-purpose method of computing the
        #       no-data mask.
        mask = np.logical_not(np.ma.getmaskarray(np.ma.masked_invalid(self.data)))
        self.data = rasterio.fill.fillnodata(
            self.data,
            mask,
            max_search_distance=radius,
            smoothing_iterations=smoothing_iters,
        )

    def save(self, path: str | PathLike) -> None:
        num_bands = self.data.shape[0] if len(self.data.shape) == 3 else 1
        # TODO: Check whether this statement is true.
        # NOTE: The default output transform is overriden if included in the raster
        #       metadata.
        transform = rasterio.transform.from_origin(
            self._bbox[0], self._bbox[3], self._resol, self._resol
        )
        f: rasterio.io.DatasetWriter
        with rasterio.open(
            path,
            "w",
            width=self._lenx,
            height=self._leny,
            count=num_bands,
            transform=transform,
            **self._meta,
        ) as f:
            f.write(self.data, indexes=1 if num_bands == 1 else None)


class DefaultProfile(rasterio.profiles.Profile):
    defaults = {
        "dtype": np.float32,
        "nodata": np.nan,
        "crs": config.var("CRS"),
        # NOTE: Tiled rasters can be efficiently split into patches by exploiting their
        #       internal data block structure.
        "tiled": True,
        "blockxsize": config.var("BLOCK_SIZE"),
        "blockysize": config.var("BLOCK_SIZE"),
        "compress": config.var("COMPRESSION"),
        "num_threads": os.cpu_count(),
    }


class MultiBandProfile(rasterio.profiles.Profile):
    defaults = {
        "num_threads": os.cpu_count()
    }
