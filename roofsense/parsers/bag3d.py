import cjio.cityjson
import cjio.cjio
# noinspection PyDeprecation
import cjio.models
import geopandas as gpd
import shapely

from roofsense import config
from roofsense.utils.file import exists


class BAG3DParser:
    def __init__(self) -> None:
        super().__init__()
        self._data: cjio.cityjson.CityJSON | None = None
        self._surfs: dict[str, list[shapely.Polygon]] | None = None

    def parse(self, obj_id: str) -> None:
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
            f"{config.var('GEOPACKAGE')}"
        )
        if exists(out_path):
            return
        self._update(obj_id)
        # noinspection PyDeprecation
        buildings: dict[str, cjio.models.CityObject]
        buildings = self._data.get_cityobjects(type="building")
        for building in buildings.values():
            self._parse_building_parts(building)
        gpd.GeoDataFrame(self._surfs, crs=config.var("CRS")).to_file(out_path)

    def _update(self, obj_id: str) -> None:
        # noinspection PyDeprecation
        self._data = cjio.cityjson.load(
            f"{config.env('TEMP_DIR')}{obj_id}{config.var('CITY_JSON')}"
        )
        self._surfs = {
            config.var("DEFAULT_ID_FIELD_NAME"): [],
            # Median roof elevation.
            "b3_h_dak_50p": [],
            config.var("DEFAULT_GM_FIELD_NAME"): [],
        }

    # noinspection PyDeprecation
    def _parse_building_parts(self, building: cjio.models.CityObject) -> None:
        parts: dict[str, cjio.models.CityObject]
        parts = self._data.get_cityobjects(id=building.children)
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
            self._surfs[config.var("DEFAULT_ID_FIELD_NAME")].append(building.id)
            self._surfs["b3_h_dak_50p"].append(building.attributes["b3_h_dak_50p"])
            self._surfs[config.var("DEFAULT_GM_FIELD_NAME")].append(surf)
