import os.path
from typing import IO

import lightning
import numpy as np
import rasterio
import rasterio.mask
import torch
import torch.utils.data
import torchgeo.datasets
import torchgeo.samplers
import torchgeo.transforms
from kornia.constants import DataKey, Resample
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference

import roofsense.training.task
from roofsense.augmentations.color import AppendLab
from roofsense.augmentations.scale import MinMaxScaling
from roofsense.bag3d import BAG3DTileStore, LevelOfDetail
from roofsense.utilities.file import confirm_write_op
from roofsense.utilities.raster import DefaultProfile


# TODO: Add support for changing the current model.
class TiledInferenceEngine:
    """Tiled inference engine."""

    def __init__(
        self,
        checkpoint_path: _PATH | IO,
        map_location: _MAP_LOCATION_TYPE = None,
        tile_store: BAG3DTileStore = BAG3DTileStore(),
        seed: int | None = None,
        tf32: bool = True,
        **kwargs,
    ) -> None:
        self._tile_store = tile_store

        lightning.pytorch.seed_everything(seed, workers=True, verbose=False)
        if tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True

        # Initialize the model.
        self._model: roofsense.training.task.TrainingTask = (
            roofsense.training.task.TrainingTask.load_from_checkpoint(
                checkpoint_path, map_location=map_location, **kwargs
            )
        )
        self._model.freeze()

        # Initialize the norm.
        scales = torch.from_numpy(
            np.fromfile(
                # FIXME: Expose this parameter to the user.
                r"C:\Documents\RoofSense\roofsense\dataset\scales.bin"
            )
        )
        self._norm = torchgeo.transforms.AugmentationSequential(
            *[MinMaxScaling(*torch.tensor_split(scales, 2)), AppendLab()],
            **{
                "data_keys": ["image"],
                "extra_args": {
                    # NOTE: We choose to always resample with bilinear interpolation to
                    # preserve the scaled value range of the stack.
                    # This is important because both reflectance and slope values have a
                    # physical interpretation.
                    # NOTE; Interpolation with aligned corners may disturb the spatial
                    # inductive biases of the model.
                    # See https://discuss.pytorch.org/t/what-we-should-use-align-corners
                    # -false/22663/5 for more information.
                    DataKey.IMAGE: {
                        "resample": Resample.BILINEAR,
                        "align_corners": False,
                    }
                },
            },
        )

    def run(
        self, tile_id: str, dst_filepath: str, params: TiledInferenceParameters
    ) -> None:
        """Generate the pixelwise roofing material map of a given 3DBAG tile.

        Args:
            tile_id:
                The ID of the input tile.
            dst_filepath:
                The path to the output map.
            params:
                The chipping parameters to split the tile into individual patches.

        Warnings:
            The output of multiple consecutive executions using the same input parameters may vary slightly in case the underlying model contains normalisation layers.
        """
        if not confirm_write_op(dst_filepath):
            return

        surfs = self._tile_store.read_tile(tile_id, lod=LevelOfDetail.LoD22).dissolve()

        path = os.path.join(self._tile_store.dirpath, f"{tile_id}.stack.tif")
        if not os.path.isfile(path):
            msg = f"Found no raster stack corresponding to tile with ID: {tile_id}."
            raise ValueError(msg)
        src: rasterio.io.DatasetReader
        with rasterio.open(path) as src:
            data, _ = rasterio.mask.mask(src, shapes=surfs["geometry"], nodata=0)
            meta = src.meta

        batch = dict(image=torch.from_numpy(data).to(torch.float32))
        batch = self._norm(batch)
        image = batch["image"].to(self._model.device)

        probs = tiled_inference(
            self._model,
            input_batch=image,
            out_channels=self._model.hparams["num_classes"],
            inference_parameters=params,
        )
        preds = (
            probs[
                # There is only one sample in the batch.
                0, ...
            ]
            .argmax(dim=0)
            .cpu()
            .numpy()
        )

        # Write the map.
        meta.update(count=1, nodata=0)
        meta.update(
            # Specify the output data type here to get the appropriate predictor
            DefaultProfile(dtype=np.uint8)
        )
        dst: rasterio.io.DatasetWriter
        with rasterio.open(dst_filepath, mode="w+", **meta) as dst:
            dst.write(preds, indexes=1)
            # Remask the map
            preds, _ = rasterio.mask.mask(dst, shapes=surfs["geometry"], indexes=1)
            dst.write(preds, indexes=1)
