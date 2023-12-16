import os

import cjio.cityjson
import cjio.cjio
import cjio.models
import requests
import shapely

import config
import utils


# TODO - Reformat, finalize function and variable names, and add documentation.


class DataParser:
    # TOSELF - Rewrite this dictionary as a module- or class-private enumeration?
    LEVEL_OF_DETAIL = {"1.2": 0, "1.3": 1, "2.2": 2}

    def __init__(self, id_: str) -> None:
        self.id = id_

        path = f"{config.env('TEMP_DIR')}{self.id}{config.var('CITY_JSON')}"
        self._data = cjio.cityjson.load(path)

        self._building_boundaries = config.default_data_dict()
        self._surfaces_boundaries = config.default_data_dict()

    def parse(self):
        # NOTE: Do not remove type hints!
        items: dict[str, cjio.models.CityObject]
        items = self._data.get_cityobjects(type="building")

        for item in items.values():
            # TOSELF - Parse the item.
            self._parse_sitem(item)
            # TOSELF - Parse its parts.
            self._parse_pitem(item)

        building_path = f"{config.env('TEMP_DIR')}{self.id}{config.var('DEFAULT_BUILDING_FOOTPRINT_FILE_ID')}{config.var('GEOPACKAGE')}"
        surfaces_path = f"{config.env('TEMP_DIR')}{self.id}{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}{config.var('GEOPACKAGE')}"
        config.default_data_tabl(self._building_boundaries).to_file(building_path)
        config.default_data_tabl(self._surfaces_boundaries).to_file(surfaces_path)

    def _parse_sitem(self, item: cjio.models.CityObject) -> None:
        # NOTE: Do not remove type hints!
        geometry: cjio.models.Geometry
        # NOTE - Parent City Objects always contain a single Geometry Object in the corresponding array
        #        (i.e., `geometry`) in the 3DBAG data files.
        geometry = item.geometry[0]

        boundary: list
        # Get the exterior `Polygon` of the first and naturally single `Surface` in the corresponding `MultiSurface`.
        # NOTE - The 3DBAG do not contain geometric primitives with interior members.
        boundary = geometry.boundaries[0][0]
        boundary = shapely.force_2d(shapely.Polygon(boundary))

        # TOSELF - Rewrite the default dataframe into its own class so that its fields can be accessed without
        #          needing to look up environment variables, which reduces readability?
        self._building_boundaries[os.environ["DEFAULT_ID_FIELD_NAME"]].append(item.id)
        self._building_boundaries[os.environ["DEFAULT_GM_FIELD_NAME"]].append(boundary)

    def _parse_pitem(self, item: cjio.models.CityObject) -> None:
        # NOTE: Do not remove type hints!
        building_parts: dict[str, cjio.models.CityObject]
        building_parts = self._data.get_cityobjects(id=item.children)
        # TOSELF - Does every `Building` parents exactly one `BuildingPart`?
        for part in building_parts.values():
            self._parse_surfaces(item, part)

    def _parse_surfaces(self, item, building_part) -> None:
        # TOSELF - If `BuildingPart` instances contain the actual building geometry, what is the purpose of the
        #       `Building` class?
        # TOSELF - All three LoDs contain semantic surfaces.
        # NOTE: Do not remove type hints!
        part_geometry: cjio.models.Geometry
        part_geometry = building_part.geometry[DataParser.LEVEL_OF_DETAIL["2.2"]]

        # NOTE - The `RoofSurface` Semantic Object (SO) always appears second in corresponding arrays
        #        (i.e., `surfaces`) in the 3DBAG data files.
        surfaces = part_geometry.surfaces[1]
        surfaces = part_geometry.get_surface_boundaries(surfaces)
        for surface in surfaces:
            # Get the exterior `Polygon` of the corresponding `Surface`.
            surface = shapely.force_2d(shapely.Polygon(surface[0]))

            self._surfaces_boundaries[os.environ["DEFAULT_ID_FIELD_NAME"]].append(
                item.id
            )
            self._surfaces_boundaries[os.environ["DEFAULT_GM_FIELD_NAME"]].append(
                surface
            )


class DataType:
    ITEM, TILE = range(2)


def download(id_: str, type_: DataType = DataType.TILE) -> None:
    if type_ == DataType.ITEM:
        partial_path = f"{config.env('TEMP_DIR')}{id_}"
        _download_item_data(id_, partial_path)
    elif type_ == DataType.TILE:
        partial_path = f"{config.env('TEMP_DIR')}{id_}"
        _download_tile_data(id_, partial_path)
    else:
        raise ValueError(f"No such data type: {type_}")


def _download_item_data(id_: str, partial_path: str) -> None:
    url = f"{config.var('BAG3D_API_BASE_URL')}{id_}"
    filename = f"{partial_path}{config.var('CITY_JSON')}"

    with requests.Session() as session:
        utils.file.BlockingFileDownloader(
            url, filename, session=session, callbacks=utils.cjio.to_jsonl
        ).download()


def _download_tile_data(id_: str, partial_path: str) -> None:
    # Compile static, compound environment variables.
    # FIXME - Do not recompile this constant at every function call.
    _BASE_TILE_DATA_URL = (
        f"{config.var('BAG3D_TILE_URL')}{config.var('BAG3D_VER')}/tiles/"
    )

    url = f"{_BASE_TILE_DATA_URL}{id_.replace('-', '/')}/{id_}{config.var('CITY_JSON')}"
    filename = f"{partial_path}{config.var('CITY_JSON')}"

    with requests.Session() as session:
        utils.file.BlockingFileDownloader(url, filename, session=session).download()
