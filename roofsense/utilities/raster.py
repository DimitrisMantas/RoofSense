from __future__ import annotations

import copy
import math
import os
import warnings
from os import PathLike
from typing import Any

import numpy as np
import rasterio.fill
import rasterio.mask
import rasterio.merge
import rasterio.windows

from roofsense.utilities.types import BoundingBoxLike


class DefaultProfile(rasterio.profiles.Profile):
    defaults = {
        "interleave": "BAND",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "LZW",
        "num_threads": os.cpu_count(),
    }

    def __init__(
        self,
        crs: str | None = None,
        dtype: type[np.integer] | type[np.floating] | None = None,
        nodata: float | None = None,
    ) -> None:
        """Profile for single- or multi-chanel band-interleaved,
        512x512-pixel--tiled, LZW-compressed rasters.

        Args:
            crs:
                The EPSG identifier of the destination coordinate reference system.
                Set to ``None`` to inherit the CRS of a source raster when updating
                its profile or leave this field unspecified.
            dtype:
                The destination data type. Set to ``None`` to inherit the data type
                of a source raster when updating its profile.
            nodata:
                The destination no-data value. Set to ``None`` to inherit the no-data
                value of a source raster when updating its profile or leave this
                field unspecified.
        """
        options = {"crs": crs, "dtype": dtype, "nodata": nodata}

        # copy the dict items to not change dict size during iteration
        items = tuple(options.items())
        for name, option in items:
            if option is None:
                options.pop(name)

        predictor = (
            2
            if np.issubdtype(dtype, np.integer)
            else 3
            if np.issubdtype(dtype, np.floating)
            else 1
        )
        if predictor == 1:
            warnings.warn(
                f"Could not infer data type: {dtype!r} as 8-, 16-, 32-, or 64-bit "
                f"integer or floating-point number . Cannot provide compression "
                f"predictor unless one is inherited from a source raster."
            )

        super().__init__(predictor=predictor, **options)


class Raster:
    def __init__(
        self,
        resol: float,
        bbox: BoundingBoxLike,
        meta: rasterio.profiles.Profile = DefaultProfile(),
    ) -> None:
        self._resol = resol
        self._bbox = bbox
        self._lenx = math.ceil((self._bbox[2] - self._bbox[0]) / self._resol)
        self._leny = math.ceil((self._bbox[3] - self._bbox[1]) / self._resol)
        self._meta = meta
        self.data = np.full(
            (self._leny, self._lenx), self._meta["nodata"], dtype=self._meta["dtype"]
        )
        self._transform = rasterio.transform.from_origin(
            self._bbox[0], self._bbox[3], self._resol, self._resol
        )

    @property
    def transform(self):
        return self._transform

    def __len__(self) -> int:
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

    def fill(self, radius: int = 100, smoothing_iters: int = 0) -> None:
        nodata = self.meta["nodata"]

        if nodata is None:
            msg = (
                "Unable to identify valid cells when the no-data value is "
                "unspecified. Missing values cannot be filled."
            )
            warnings.warn(msg, UserWarning)
            return

        self.data = rasterio.fill.fillnodata(
            self.data,
            ~np.isclose(self.data, nodata)
            if np.isfinite(nodata)
            else np.isfinite(self.data),
            radius,
            smoothing_iters,
        )

    def save(self, path: str | PathLike) -> None:
        num_bands = self.data.shape[0] if len(self.data.shape) == 3 else 1
        # TODO: Check whether this statement is true.
        # NOTE: The default output transform is overriden if included in the raster
        #       metadata.
        f: rasterio.io.DatasetWriter
        with rasterio.open(
            path,
            "w",
            width=self._lenx,
            height=self._leny,
            count=num_bands,
            transform=self.transform,
            **self._meta,
        ) as f:
            f.write(self.data, indexes=1 if num_bands == 1 else None)


def get_raster_metadata(filepath: str, attr: str, *attrs: str) -> list[Any]:
    src: rasterio.io.DatasetReader
    with rasterio.open(filepath) as src:
        return [getattr(src, attr) for attr in [attr] + list(attrs)]
