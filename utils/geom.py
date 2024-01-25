import geopandas as gpd

import config


# TODO: Harmonize this variable name.
def read(obj_id: str) -> gpd.GeoDataFrame:
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
