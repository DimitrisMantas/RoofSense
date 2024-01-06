from __future__ import annotations

import math
import typing

import numpy as np
import rasterio
import rasterio.windows
import shapely


class Raster:
    def __init__(
        self,
        size: float,
        bbox: typing.Sequence[float],
        meta: typing.Optional[rasterio.profiles.Profile] = None,
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
    def width(self) -> float:
        return self._lenx

    @property
    def height(self) -> float:
        return self._leny

    def save(self, filename: str) -> None:
        count = _count(self._data)
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
    defaults = {
        "driver": "GTiff",  # FIXME: How to select the optimal data type?
        "dtype": np.float32,
        "nodata": np.nan,  # FIXME: Hardcoded value!
        "crs": "EPSG:28992",  # FIXME: Determine the optimal patch size for sampling.
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": True,  # TODO: Check the other available settings.
        #       https://gdal.org/drivers/raster/gtiff.html
        "interleave": "pixel",
    }


def crop(inname: str, outname: str, bbox: typing.Sequence[float], bands=None):
    f: rasterio.io.DatasetReader
    with rasterio.open(inname) as f:
        window = rasterio.windows.from_bounds(*bbox, transform=f.transform)
        data = f.read(indexes=bands, window=window)

    # NOTE: The crop area may extend beyond the bounds of the raster and so may need to
    #       be "cropped" to them.
    valid_bbox = shapely.intersection(_geom(bbox), _geom(f.bounds))

    count = _count(data)
    transform = rasterio.transform.from_bounds(
        *valid_bbox.bounds, width=data.shape[-1], height=data.shape[-2]
    )

    g: rasterio.io.DatasetWriter
    with rasterio.open(
        outname,
        "w",
        width=data.shape[-1],
        height=data.shape[-2],
        count=count,
        transform=transform,
        **DefaultProfile(),
    ) as g:
        g.write(data, indexes=bands)


def merge(filenames):
    raise NotImplementedError


def _count(data: np.ndarray) -> int:
    return data.shape[0] if len(data.shape) == 3 else 1


# TODO: Add type hints to this function.
def _geom(bbox) -> shapely.Polygon:
    return shapely.Polygon(
        (
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
            (bbox[0], bbox[1]),
        )
    )
