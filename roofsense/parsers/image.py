from typing import Iterable

import numpy as np
import rasterio.merge

from roofsense.bag3d import BAG3DTileStore
from roofsense.parsers.base import BAG3DTileAssetParser, BAG3DTileAssetParsingStage
from roofsense.utils import raster
from roofsense.utils.file import confirm_write_op


def merge_images(parser: BAG3DTileAssetParser, tile_id: str, overwrite: bool) -> None:
    # TODO: Consider refactoring this block into a separate method.
    dst_path = parser.resolve_filepath(tile_id + ".rgb.tif")
    if not confirm_write_op(dst_path, overwrite=overwrite):
        return

    src_paths = [
        parser.resolve_filepath(id_ + ".tif") for id_ in parser.manifest.image.tid
    ]
    rasterio.merge.merge(
        src_paths,
        bounds=parser.surfaces.total_bounds.tolist(),
        target_aligned_pixels=True,
        dst_path=dst_path,
        dst_kwds=raster.DefaultProfile(
            # NOTE: The data type is specified in order for a descriptor to be
            # able to be assigned to the output profile.
            dtype=np.uint8
        ),
    )


class ImageParser(BAG3DTileAssetParser):
    """Parser for aerial imagery of the Dutch National Imagery Program."""

    def __init__(
        self,
        tile_store: BAG3DTileStore,
        callbacks: BAG3DTileAssetParsingStage
        | Iterable[BAG3DTileAssetParsingStage]
        | None = (merge_images,),
    ) -> None:
        super().__init__(tile_store, callbacks)
