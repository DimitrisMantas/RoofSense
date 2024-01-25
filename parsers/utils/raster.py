from __future__ import annotations

import copy
import math
import typing
from collections.abc import Sequence

import numpy as np
import rasterio
import rasterio.fill
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


# FIXME: Parallelize this function.
# TODO: Add type hints to this function.
# TODO: Optimize the search radius, power, and filler hyperparameters.
def rasterize(
    pc, scalars: str | Sequence[str], size: typing.Optional[float] = 0.25
) -> Raster | dict[str, Raster]:
    if isinstance(scalars, str):
        scalars = [scalars]
    # FIXME: Refactor ``xy()`` into a module function because doesn't make sense to
    #        create an empty raster just for the sake of using its accessors.
    r = Raster(size, pc.bbox())

    # TODO: The point cloud should ensure that its spatial index has been initialized
    #       before this call.
    neighbors, distances = pc.index.query_radius(r.xy(), r=size, return_distance=True)

    rasters = {scalar: Raster(size, pc.bbox()) for scalar in scalars}
    for cell_id, cell_nb in enumerate(neighbors):
        if len(cell_nb) == 0:
            continue
        attribs = {scalar: [] for scalar in scalars}
        for scalar in scalars:
            attribs[scalar].append(pc[cell_nb][scalar])
        row, col = np.divmod(cell_id, r.width)
        if np.any(distances[cell_id] == 0):
            # The cell center belongs to the point cloud.
            nb_id = cell_nb[np.argsort(distances[cell_id])[0]]
            for scalar in scalars:
                rasters[scalar][row, col] = pc[[nb_id]][scalar].scaled_array()
        else:
            weights = distances[cell_id] ** -2
            for scalar in scalars:
                rasters[scalar][row, col] = np.sum(attribs[scalar] * weights) / np.sum(
                    weights
                )

    for raster in rasters.values():
        raster.fill()

    return rasters


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
