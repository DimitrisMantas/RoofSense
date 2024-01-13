from __future__ import annotations

from typing import Sequence

import laspy
import numpy as np


class PointCloud:
    def __init__(self, filename: str) -> None:
        self.index = None
        with laspy.open(filename) as f:
            self.las = f.read()

    @property
    def header(self):
        return self.las.header

    @property
    def points(self):
        return self.las.points

    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.header.x_min,
            self.header.y_min,
            self.header.x_max,
            self.header.y_max,
        )

    def crop(self, bbox: Sequence[float]) -> PointCloud:
        min_x = (bbox[0] - self.header.x_offset) / self.header.x_scale
        min_y = (bbox[1] - self.header.y_offset) / self.header.y_scale
        max_x = (bbox[2] - self.header.x_offset) / self.header.x_scale
        max_y = (bbox[3] - self.header.y_offset) / self.header.y_scale
        self.las.points = self.las.points[
            np.logical_and(
                np.logical_and(min_x <= self.las.X, self.las.X <= max_x),
                np.logical_and(min_y <= self.las.Y, self.las.Y <= max_y),
            )
        ]
        return self

    def save(self, filename: str) -> None:
        with laspy.open(filename, "w", header=self.header) as f:
            f.write_points(self.las.points)

        # Compute the

    def __len__(self) -> int:
        return len(self.points)

    # TODO: Add type hints to this method.
    def __getitem__(self, key):
        if isinstance(key, int):
            # NOTE: This indexing notation results in the point being returned in a
            #       familiar format.
            k = [key]
        else:
            k = key
        return self.points[k]
