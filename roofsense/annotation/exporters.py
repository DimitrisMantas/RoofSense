import numpy as np
import rasterio.plot


def to_png(data: np.ndarray, path: str):
    with rasterio.open(
        path,
        mode="w",
        count=3,
        height=data.shape[1],
        width=data.shape[2],
        dtype=np.uint8,
    ) as f:
        f.write(data[:3, ...])
