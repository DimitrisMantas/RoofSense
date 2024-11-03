import glob
import os
import warnings
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal, Never

import rasterio
import torch.utils.data
from matplotlib.figure import Figure
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor
from torch.utils.data import Dataset

from roofsense.enums.band import Band


class ChipDataset(Dataset[Tensor]):
    """Raster Stack Chip Dataset."""

    filename_glob: str = ".tif"
    """The  file extension used to glob for chip paths and verify dataset integrity 
    during its initialization phase."""

    def __init__(
        self, dirpath: str, bands: Iterable[Band] = Band.ALL, cache: bool = True
    ) -> None:
        """Configure the dataset.

        Args:
            dirpath:
                The path to the data directory.
            bands:
                The list of bands to load.
            cache:
                A flag indicating whether to cache previous data queries.
                This feature is implemented using an LRU cache with a size of 128 items.
        """
        self.bands = bands
        self.cache = cache

        self._dirpath = dirpath
        self._filepaths = self._verify_integrity()

    @property
    def dirpath(self) -> str:
        """The path to the data directory."""
        return self._dirpath

    @property
    def filepaths(self) -> list[str]:
        """The chip paths found in the specified data directory."""
        return self._filepaths

    # TODO: Implement this method.
    def plot(self, batch: dict[Literal["image"], Tensor]) -> Figure | None:
        """Plot a single chip.

        Notes:
            The current implementation of this method supports only single- or at
            least three-band images.

        Args:
            batch:
                The chip.

        Returns:
            A ``matplotlib.figure.Figure`` instance visualizing the chip or ``None``
            if it has exactly two bands.
        """
        image = batch["image"]

        num_channels = image.size(dim=1)
        if num_channels == 1:
            return self._plot_single_channel(image)
        elif num_channels >= 3:
            return self._plot_rgb(image)

        msg = (
            f"Expected single or at least three-band (RGB) image, but got two "
            f"channels: {self.bands!r}."
            f" "
            f"Unable to infer valid band configuration for visualization."
        )
        warnings.warn(msg, UserWarning)

    def _plot_single_channel(self, image: Tensor) -> Figure:
        raise NotImplementedError

    def _plot_rgb(self, image: Tensor) -> Figure:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> dict[Literal["image"], Tensor]:
        if self.cache:
            return self._getitem_cached_impl(idx)
        else:
            return self._getitem_normal_impl(idx)

    @lru_cache
    def _getitem_cached_impl(self, idx: int) -> dict[Literal["image"], Tensor]:
        return self._getitem_normal_impl(idx)

    def _getitem_normal_impl(self, idx: int) -> dict[Literal["image"], Tensor]:
        src: rasterio.io.DatasetReader
        with warnings.catch_warnings(action="ignore", category=NotGeoreferencedWarning):
            with rasterio.open(self.filepaths[idx]) as src:
                data = src.read(self.bands)
        image = torch.from_numpy(data).to(torch.float32)
        return {"image": image}

    def _verify_integrity(self) -> list[str] | Never:
        files = glob.glob(os.path.join(self.dirpath, "*" + self.filename_glob))
        files.sort()

        if not files:
            msg = (
                f"Found no ({self.filename_glob!r}) files in specified data "
                f"directory: {self.dirpath!r}."
                f" "
                f"Dataset is empty."
            )
            raise ValueError(msg)

        return files
