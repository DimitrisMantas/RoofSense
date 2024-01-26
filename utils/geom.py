import geopandas as gpd

import config
from utils.type import BoundingBoxLike


# TODO: Harmonize this variable name.
def read_surfaces(obj_id: str) -> gpd.GeoDataFrame:
    name = (
        f"{config.env('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
        f"{config.var('GEOPACKAGE')}"
    )
    return gpd.read_file(name)


def buffer(geom: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom["geometry"] = geom["geometry"].buffer(float(config.var("BUFFER_DISTANCE")))
    return geom


def is_bboxlike(obj_id: str):
    return (
        isinstance(obj_id, BoundingBoxLike.__origin__)
        and len(obj_id) == 4
        and all([(isinstance(i, int) or isinstance(i, float)) for i in obj_id])
    )
