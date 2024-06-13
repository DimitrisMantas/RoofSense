import os

import lightning
import numpy as np
import rasterio
import torch
import torch.utils.data
import torchgeo.datasets
import torchgeo.samplers
import torchgeo.transforms
import tqdm

import common.augmentations
import inference.dataset
import training.task


# TODO: Add support for changing the current model and tile.
class InferenceEngine:
    def __init__(
        self,
        data_root: str,
        model_ckpt: str,
        seed: int | None = None,
        tf32: bool = True,
    ) -> None:
        lightning.pytorch.seed_everything(seed, workers=True)

        if tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True

        self.dataset = inference.dataset.InferenceDataset(data_root)
        self.augmet = torchgeo.transforms.AugmentationSequential(
            common.augmentations.MinMaxScaling(
                # TODO: Expose these parameters in the initializer.
                mins=torch.tensor([0, 0, 0, 0, 0]),
                maxs=torch.tensor([255, 255, 255, 1, 90]),
            ),
            data_keys=["image"],
        )

        # NOTE: This is required to detect the best available accelerator.
        self.trainer = lightning.Trainer(
            # TODO: Find out whether this is a global setting.
            benchmark=True,
            logger=False,
        )

        self.model: training.task.TrainingTask = (
            training.task.TrainingTask.load_from_checkpoint(
                model_ckpt, map_location=self.trainer.strategy.root_device.type
            )
        )
        self.model.eval()
        self.model.freeze()

        self.sampler = torchgeo.samplers.GridGeoSampler(
            self.dataset,
            # TODO: Expose these parameters in the initializer.
            size=512,
            stride=256,
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            # TODO: Expose these parameters in the initializer.
            batch_size=16,
            sampler=self.sampler,
            num_workers=max(os.cpu_count() // 2, 1),
            collate_fn=torchgeo.datasets.stack_samples,
            pin_memory=True,
            generator=torch.Generator().manual_seed(0),
            persistent_workers=True,
        )

        # TODO: Expose these parameters in the initializer.
        self.size = torchgeo.samplers.utils._to_tuple(512)
        self.strd = torchgeo.samplers.utils._to_tuple(256)

        self.size = (self.size[0] * self.dataset.res, self.size[1] * self.dataset.res)
        self.strd = (self.strd[0] * self.dataset.res, self.strd[1] * self.dataset.res)

        self.rows, self.cols = torchgeo.samplers.utils.tile_to_chips(
            self.dataset.bounds, self.size, self.strd
        )

        self.width = round(
            (self.dataset.bounds.maxx - self.dataset.bounds.minx) / self.dataset.res
        )
        self.height = round(
            (self.dataset.bounds.maxy - self.dataset.bounds.miny) / self.dataset.res
        )
        # TODO: Expose these parameters in the initializer.
        self.buffer = np.zeros(
            (
                self.model.hparams["num_classes"],
                256 * (self.rows + 1),
                256 * (self.cols + 1),
            ),
            dtype=np.float32,
        )
        self.images = np.zeros(
            (256 * (self.rows + 1), 256 * (self.cols + 1)), dtype=np.uint8
        )

    def run(self,filename:str)->None:
        img_idx = 0
        for batch in tqdm.tqdm(
            self.loader,
            total=len(self.loader),
            desc=self.__class__.__name__.replace("Engine", ""),
        ):
            # NOTE: Augmentations are performed following the batch sampling step to
            # avoid broadcasting them into 5D, which is the default behavior of
            # MinMaxScaling.
            batch: dict[
                str,
                list[rasterio.crs.CRS]
                | list[torchgeo.datasets.BoundingBox]
                | torch.Tensor,
            ] = self.augmet(batch)
            images: torch.Tensor = batch["image"].to(
                self.trainer.strategy.root_device.type
            )

            with torch.inference_mode():
                probs: torch.Tensor = self.model(images).softmax(dim=1).cpu().numpy()

            for i in range(images.shape[0]):
                row_idx, col_idx = divmod(img_idx, self.cols)
                # NOTE: The chip sampler starts at the southeast corner of the stack.
                row_off, col_off = (self.buffer.shape[1] - row_idx * 256, col_idx * 256)

                self.buffer[:, row_off - 512 : row_off, col_off : col_off + 512] += (
                    probs[i]
                )
                self.images[row_off - 512 : row_off, col_off : col_off + 512] += 1

                img_idx += 1

        # Blend
        preds = self.buffer / self.images
        # Predict
        preds = preds.argmax(axis=0)
        # Trim the buffer to the stack dimensions.
        preds = preds[
            self.buffer.shape[1] - self.height : self.buffer.shape[1], : self.width
        ]

        dst: rasterio.io.DatasetWriter
        with rasterio.open(
            filename,
            mode="w",
            width=self.width,
            height=self.height,
            count=1,
            crs=self.dataset.crs,
            transform=rasterio.transform.from_origin(
                west=self.dataset.bounds.minx,
                north=self.dataset.bounds.maxy,
                xsize=self.dataset.res,
                ysize=self.dataset.res,
            ),
            dtype=np.uint8,
            nodata=0,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="DEFLATE",
            num_threads=os.cpu_count(),
            predictor=2,
        ) as dst:
            dst.write(preds, indexes=1)


if __name__ == "__main__":
    engine = InferenceEngine(
        data_root="../dataset/infer", model_ckpt="../logs/training/base-tgi/ckpts/best.ckpt",
    )
    engine.run("../dataset/infer/9-284-556.map.base.tgi.tif")
