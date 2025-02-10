from terratorch.tasks.tiled_inference import TiledInferenceParameters

from roofsense.inference.tile import TiledInferenceEngine


def main():
    tile_id = "9-272-552"
    TiledInferenceEngine(
        ckpt_path=r"C:\Documents\RoofSense\logs\optimization_random_search_round_3_tpe\version_20\ckpts\last.ckpt",
        model_params={
            "encoder_output_stride": 8,
            "encoder_params": {"block_args": {"attn_layer": "eca"}},
        },
    ).run(
        tile_id=tile_id,
        dst_filename=rf"C:\Documents\RoofSense\dataset\infer\{tile_id}.map.tpe.stride16.terratorch.tif",
        params=TiledInferenceParameters(
            h_crop=512, h_stride=256, w_crop=512, w_stride=256
        ),
    )


if __name__ == "__main__":
    main()
