from terratorch.tasks.tiled_inference import TiledInferenceParameters

from roofsense.bag3d import BAG3DTileStore
from roofsense.inference.tile import TiledInferenceEngine


def main():
    tile_id = "10-280-560"

    TiledInferenceEngine(
        ckpt_path=r"data/model/roofsense.paper.ckpt",
        tile_store=BAG3DTileStore(dirpath=r"data/store"),
        model_params={"encoder_params": {"block_args": {"attn_layer": "eca"}}},
    ).run(
        tile_id=tile_id,
        dst_filepath=rf"data/maps/pixel/{tile_id}.map.pixel.tif",
        params=TiledInferenceParameters(
            h_crop=512, h_stride=256, w_crop=512, w_stride=256
        ),
    )


if __name__ == "__main__":
    main()
