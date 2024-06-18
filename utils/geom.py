import geopandas as gpd

import config


def read_surfaces(tile_id: str) -> gpd.GeoDataFrame:
    name = (
        f"{config.env('TEMP_DIR')}"
        f"{tile_id}"
        f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
        f"{config.var('GEOPACKAGE')}"
    )
    return gpd.read_file(name)
