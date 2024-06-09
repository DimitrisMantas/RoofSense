from collections.abc import Callable, Sequence
from typing import Any

from torchgeo.datasets import BoundingBox, RasterDataset
from typing_extensions import override


class InferenceDataset(RasterDataset):
    # GeoDataset
    filename_glob = "*.stack.mask.tif"
    # RasterDataset
    # TODO: Fill these fields in.
    all_bands: list[str] = []
    rgb_bands: list[str] = []
    cmap: dict[int, tuple[int, int, int, int]] = {}

    def __init__(
        self,
        # TODO: Provide the tile 3D or its location?
        root: str,
        tile_id: str | None = None,
        download: bool = False,
        checksum: bool = False,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        self.root = root

        # TODO: Support multiple stacks, just process them one at a time.
        if tile_id is not None:
            raise NotImplementedError("The root must contain a single stack.")

        # TODO: Build the stack if it does not already exist.
        if download or checksum:
            raise NotImplementedError("The raster stack must already exist.")

        super().__init__(root, bands=bands, transforms=transforms, cache=cache)

        base_err_msg = (
            f"Expected a single stack at location: {self.root!r},"
            f" "
            f"but got {len(self)} instead."
        )
        if len(self) == 0:
            raise ValueError(base_err_msg + " " + "Empty dataset root path.")
        elif len(self) > 1:
            raise ValueError(base_err_msg + " " + "Invalid dataset root path.")

    @override
    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        sample: dict[str, Any] = super().__getitem__(query)
        # simulate boundless read
        sample["image"] = sample["image"].nan_to_num(0)
        return sample


if __name__ == "__main__":
    dataset = InferenceDataset("../dataset/infer")
    x = 1
