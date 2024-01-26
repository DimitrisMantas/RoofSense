from __future__ import annotations

import json
import pathlib

from typing_extensions import override

import config
import utils
from parsers.base import DataParser


class LiDARDataParser(DataParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update_fields(obj_id)

        """
        TODO:
        1. *Merge the point clouds,*
            1.1 Memory requirements.
            1.2 How to deal with the overlap?
        2. Crop the resulting product to the extents of the buffered surface boundary.
        3.  Rasterize the elevation and reflectance fields.
            3.1 *The spatial index of the point cloud must be constructed automatically."
        4. Compute the slope of the former raster.
        5. Save the reflectance snd slope rasters.
        """

    # FIXME: Write an asset data parser to avoid duplicate methods in the parser module.
    @override
    def _update_fields(self, obj_id: str) -> None:
        path = (
            f"{config.var('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('ASSET_MANIFEST_EXTENSION')}"
            f"{config.var('JSON')}"
        )
        with pathlib.Path(path).open() as f:
            self._data = json.load(f)

        self._surfs = utils.geom.buffer(utils.geom.read_surfaces(obj_id))
