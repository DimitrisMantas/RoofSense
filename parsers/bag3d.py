import os

import cjio.cityjson
import cjio.cjio
# noinspection PyDeprecation
import cjio.models
import shapely

import config
from parsers._base import DataParser


# TODO: Do not parse previously processed tiles.
# TODO: Rewrite the data parsers so that they can be reused across multiple object.
# noinspection PyDeprecation
class BAG3DDataParser(DataParser):
    def __init__(self, obj_id: str) -> None:
        super().__init__(obj_id)

        # TODO: Harmonize this variable name.
        pathnam = f"{config.env('TEMP_DIR')}{self._obj_id}{config.var('CITY_JSON')}"
        self._data = cjio.cityjson.load(pathnam)

        self._surf = config.default_data_dict()

    # TODO: Find out why the override decorator cannot be imported from the typing
    #       module.
    def parse(self) -> None:
        items: dict[str, cjio.models.CityObject]
        items = self._data.get_cityobjects(type="building")

        for item in items.values():
            self._parse_parts(item)

        # TODO: Harmonize this variable name.
        pathnam = (
            f"{config.env('TEMP_DIR')}"
            f"{self._obj_id}"
            f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
            f"{config.var('GEOPACKAGE')}"
        )
        config.default_data_tabl(self._surf).to_file(pathnam)

    def _parse_parts(self, item: cjio.models.CityObject) -> None:
        item_parts: dict[str, cjio.models.CityObject]
        item_parts = self._data.get_cityobjects(id=item.children)
        for part in item_parts.values():
            self._parse_surfaces(item, part)

    def _parse_surfaces(self, item, building_part) -> None:
        part_geom: cjio.models.Geometry
        # Parse the LoD 2.2 surfaces.
        # NOTE: The LoD 1.1, 1.2, and 2.2 geometry representations appear first,
        #       second, and third in the underlying array, respectively.
        part_geom = building_part.geometry[2]

        # Parse the roof surfaces.
        # NOTE: Wall and roof surfaces appear first and second in the underlying
        #       array, respectively.
        part_surf = part_geom.surfaces[1]
        part_surf = part_geom.get_surface_boundaries(part_surf)
        for surface in part_surf:
            # Parse the exterior surface boundary.
            surface = shapely.force_2d(shapely.Polygon(surface[0]))

            self._surf[os.environ["DEFAULT_ID_FIELD_NAME"]].append(item.id)
            self._surf[os.environ["DEFAULT_GM_FIELD_NAME"]].append(surface)
