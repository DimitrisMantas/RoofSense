import os

import cjio.cityjson
import cjio.cjio
# noinspection PyDeprecation
import cjio.models
import shapely
from typing_extensions import override

import config
import utils.file
from parsers.base import DataParser


class BAG3DDataParser(DataParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
            f"{config.var('GEOPACKAGE')}"
        )
        if utils.file.exists(path):
            return

        self._update_fields(obj_id)
        # noinspection PyDeprecation
        buildings: dict[str, cjio.models.CityObject]
        buildings = self._data.get_cityobjects(type="building")

        for building in buildings.values():
            self._parse_building_parts(building)

        config.default_data_tabl(self._surfs).to_file(path)

    @override
    def _update_fields(self, obj_id: str) -> None:
        # noinspection PyDeprecation
        self._data = cjio.cityjson.load(
            f"{config.env('TEMP_DIR')}{obj_id}{config.var('CITY_JSON')}"
        )
        self._surfs = config.default_data_dict()

    # noinspection PyDeprecation
    def _parse_building_parts(self, item: cjio.models.CityObject) -> None:
        parts: dict[str, cjio.models.CityObject]
        parts = self._data.get_cityobjects(id=item.children)
        for part in parts.values():
            self._parse_surfaces(item, part)

    def _parse_surfaces(self, item, building_part) -> None:
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

            self._surfs[os.environ["DEFAULT_ID_FIELD_NAME"]].append(item.id)
            self._surfs[os.environ["DEFAULT_GM_FIELD_NAME"]].append(surf)
