import os
from collections.abc import Callable

from torch import Tensor
from torchgeo.datasets import Potsdam2D


class Potsdam2DRBG(Potsdam2D):
    filenames = ["2_Ortho_RGB.zip", "5_Labels_all.zip"]
    image_root = "2_Ortho_RGB"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        checksum: bool = False,
    ) -> None:
        super().__init__(root, split, transforms, checksum)

        assert split in self.splits
        self.root = root
        self.split = split
        self.checksum = checksum

        self._verify()

        self.files = []
        for name in self.splits[split]:
            image = os.path.join(root, self.image_root, name) + "_RGB.tif"
            mask = os.path.join(root, name) + "_label.tif"
            if os.path.exists(image) and os.path.exists(mask):
                self.files.append(dict(image=image, mask=mask))
