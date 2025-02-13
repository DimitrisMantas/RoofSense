from roofsense.bag3d import BAG3DTileStore, LevelOfDetail
from roofsense.inference.obia import MaskGeneralizer


def main():
    # TODO: Use the tile store to infer the tile ID.
    tile_id = "10-280-560"

    generalizer = MaskGeneralizer(tile_store=BAG3DTileStore(dirpath=r"data/store"))
    for name, lod in zip(
        ["lod12", "lod13", "lod22"],
        [LevelOfDetail.LoD12, LevelOfDetail.LoD13, LevelOfDetail.LoD22],
    ):
        generalizer.generalize(
            src_filepath=rf"data/maps/pixel/{tile_id}.map.pixel.tif",
            dst_filepath=rf"data/maps/obia/{tile_id}.map.{name}.tif",
            lod=lod,
        )


if __name__ == "__main__":
    main()
