import os
from typing import Optional

import cjio.cityjson
import cjio.cjio
# noinspection PyDeprecation
import cjio.models
import shapely
from typing_extensions import override

import config
import utils.file
from parsers.base import DataParser


class BAG3DParser(DataParser):
    def __init__(self) -> None:
        super().__init__()
        self._cjson: Optional[cjio.cityjson.CityJSON] = None
        self._surfs: Optional[dict[str, list[shapely.Polygon]]] = None

    @override
    def parse(self, obj_id: str) -> None:
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
            f"{config.var('GEOPACKAGE')}"
        )
        if utils.file.exists(out_path):
            return

        self._update(obj_id)

        # noinspection PyDeprecation
        buildings: dict[str, cjio.models.CityObject]
        buildings = self._cjson.get_cityobjects(type="building")
        for building in buildings.values():
            self._parse_building_parts(building)

        config.default_data_tabl(self._surfs).to_file(out_path)

    @override
    def _update(self, obj_id: str) -> None:
        # noinspection PyDeprecation
        self._cjson = cjio.cityjson.load(
            f"{config.env('TEMP_DIR')}{obj_id}{config.var('CITY_JSON')}"
        )
        self._surfs = config.default_data_dict()

    # noinspection PyDeprecation
    def _parse_building_parts(self, building: cjio.models.CityObject) -> None:
        parts: dict[str, cjio.models.CityObject]
        parts = self._cjson.get_cityobjects(id=building.children)
        for part in parts.values():
            self._parse_surfaces(building, part)

    # noinspection PyDeprecation
    def _parse_surfaces(
        self, building: cjio.models.CityObject, building_part: cjio.models.CityObject
    ) -> None:
        # noinspection PyDeprecation
        part_geom: cjio.models.Geometry
        # Parse the LoD 2.2 surfaces.
        # NOTE: The LoD 1.1, 1.2, and 2.2 representations appear first, second,
        #       and third in the corresponding array, respectively.
        part_geom = building_part.geometry[2]
        # Parse the roof surfaces.
        # NOTE: The wall and roof surfaces appear first and second in the corresponding
        #       array, respectively.
        part_surfs = part_geom.surfaces[1]
        part_surfs = part_geom.get_surface_boundaries(part_surfs)
        for surf in part_surfs:
            # Parse the exterior surface boundary.
            surf = shapely.force_2d(shapely.Polygon(surf[0]))

            self._surfs[os.environ["DEFAULT_ID_FIELD_NAME"]].append(building.id)
            self._surfs[os.environ["DEFAULT_GM_FIELD_NAME"]].append(surf)
