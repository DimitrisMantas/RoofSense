import geopandas as gpd

from roofsense.bag3d import BAG3DTileStore
from roofsense.preprocessing.parsers.image import ImageParser
from roofsense.preprocessing.parsers.lidar import LiDARParser
from roofsense.preprocessing.stack import RasterStackBuilder
from roofsense.preprocessing.tile_splitters import split


class BAG3DSampler:
    def __init__(
        self,
        seeds_filepath: str,
        image_index_filepath: str,
        lidar_index_filepath: str,
        bag3d_store: BAG3DTileStore = BAG3DTileStore(),
        image_parser_cls: type[ImageParser] = ImageParser,
        lidar_parser_cls: type[LiDARParser] = LiDARParser,
    ) -> None:
        self.bag3d_store = bag3d_store
        self.image_parser = image_parser_cls(tile_store=self.bag3d_store)
        self.lidar_parser = lidar_parser_cls(tile_store=self.bag3d_store)
        self._seeds = gpd.read_file(seeds_filepath)
        self.image_sheet_filepath = image_index_filepath
        self.lidar_sheet_filepath = lidar_index_filepath

    def sample(self, size: int, background_cutoff: float) -> list[str]:
        num_im = 0
        sample = []
        while num_im < size:
            tile_ids = self.bag3d_store.sample_tile(self._seeds)
            for tile_id in tile_ids:
                if tile_id in sample:
                    continue
                # Discard tiles whose surface are is larger than 100 ha.
                # NOTE: This limit is padded by 10% to account for any discrepancies in
                #       the underlying sheet index.
                # NOTE: This ensures that at most four AHN4 tiles are downloaded and
                #       parsed per tile,
                #       and thus a minimum level of service is maintained during the
                #       preprocessing stage.
                # NOTE: The selected tile IDs begin with 9 or 10.
                if (
                    self.bag3d_store.index.loc[
                        self.bag3d_store.index["tile_id"] == tile_id, "geometry"
                    ].area.iat[0]
                    > 1.1e6
                ):
                    continue
                # -----
                # Process the tile.
                print(f"Processing tile {tile_id}...")
                # Download the corresponding 3DBAG data.
                self.bag3d_store.download_tile(tile_id)

                # Download the corresponding assets.
                self.bag3d_store.asset_manifest(
                    tile_id,
                    image_index=gpd.read_file(self.image_sheet_filepath),
                    lidar_index=gpd.read_file(self.lidar_sheet_filepath),
                ).downl(self.bag3d_store.dirpath).save(self.bag3d_store.dirpath)

                # Parse the data.
                self.image_parser.parse(tile_id)
                self.lidar_parser.parse(tile_id)

                # Create the raster stack.
                RasterStackBuilder(store=self.bag3d_store).merge(tile_id)

                # Prepare the stacks for annotation.
                print(f"Requesting at most {size - num_im} chips.")
                new = split(
                    self.bag3d_store, tile_id, background_cutoff, limit=size - num_im
                )
                print(f"Got {new} chips.")
                num_im += new
                # -----
                sample.append(tile_id)
        return sample
