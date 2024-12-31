import numpy as np
import rasterio.merge
from typing_extensions import override

from roofsense.parsers.base import AssetParser
from roofsense.utils import raster
from roofsense.utils.file import confirm_write_op


class ImageParser(AssetParser):
    """BM5 Tile Parser."""

    @override
    def parse(self, tile_id: str, overwrite: bool = False) -> None:
        """Parse the BM5 data corresponding to a particular 3DBAG tile.

        This method merges the input data to the tile bounds using a reverse
        painter's algorithm.
        The local coordinates of the output cell centers are guaranteed to be integer
        multiples of the respective spatial resolution such that the constituent
        images are aligned.
        This means that the resulting raster bounds are at least equal to those of
        the underlying geometry.
        See ``rasterio.merge.merge`` for more implementation details on raster merging.

        Warnings:
            This method assumes that the provided tile has already been parsed and
            that the corresponding asset manifest and roof surfaces exist in the
            specified data directory.

        Args:
            tile_id:
                The tile ID.
            overwrite:
                A flag indicating whether to overwrite any previous output.
        """
        self._update(tile_id)

        # TODO: Consider refactoring this block into a separate method.
        dst_path = self.resolve_filepath(tile_id + ".rgb.tif")
        if not confirm_write_op(dst_path, overwrite=overwrite):
            return

        src_paths = [
            self.resolve_filepath(id_ + ".tif") for id_ in self.manifest.image.tid
        ]
        rasterio.merge.merge(
            src_paths,
            bounds=self.surfaces.total_bounds.tolist(),
            target_aligned_pixels=True,
            dst_path=dst_path,
            dst_kwds=raster.DefaultProfile(
                # NOTE: The data type is specified in order for a descriptor to be
                # able to be assigned to the output profile.
                dtype=np.uint8
            ),
        )
