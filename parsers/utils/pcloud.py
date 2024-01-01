from __future__ import annotations

import collections
import math
from typing import Sequence

import geopandas as gpd
import laspy
import numpy as np
import startinpy

from feature.utils import timing


class PointCloud:
    @timing
    def __init__(self, filename: str) -> None:
        self.dt = None
        with laspy.open(filename) as f:
            self.las = f.read()

    @timing
    def crop(self, bbox: Sequence[float]) -> PointCloud:
        min_x = (bbox[0] - self.las.header.x_offset) / self.las.header.x_scale
        min_y = (bbox[1] - self.las.header.y_offset) / self.las.header.y_scale
        max_x = (bbox[2] - self.las.header.x_offset) / self.las.header.x_scale
        max_y = (bbox[3] - self.las.header.y_offset) / self.las.header.y_scale
        self.las.points = self.las.points[
            np.logical_and(
                np.logical_and(min_x <= self.las.X, self.las.X <= max_x),
                np.logical_and(min_y <= self.las.Y, self.las.Y <= max_y),
            )
        ]
        return self

    @timing
    def save(self, filename: str) -> None:
        with laspy.open(filename, "w", header=self.las.header, do_compress=True) as f:
            f.write_points(self.las.points)

    @timing
    def triangulate(self):
        pts = self.las.xyz

        self.dt = startinpy.DT()
        # NOTE: The snap tolerance cannot be set to zero (i.e., disabled) so the
        # nearest distinct value is used instead.
        self.dt.snap_tolerance = math.ulp(0)

        # Maintain a reference to the duplicate PC vertices.

        # NOTE: Two or more vertices are considered to be duplicate when their
        # two-dimensional projections on the Euclidean plane are identical. However.
        # they can obviously differ in n-dimensional space. In this context these
        # vertices are rechecked after the DT has been constructed so that the one
        # with the highest elevation is actually inserted.
        tentative_pts: dict[
            int,  # NOTE: There may be more than one duplicate points.
            list[int],
        ] = collections.defaultdict(list)
        # Maintain a reference to the finalized PC vertices.
        finalized_pts: dict[int, int] = {}

        candidate_id: int  # The ID of a candidate vertex in the PC.
        tentative_id: int  # The ID of a candidate vertex in the DT.
        finalized_id: int  # The ID of a candidate vertex in the DT.

        tentative_id = 1
        for candidate_id, pt in enumerate(pts):
            finalized_id = self.dt.insert_one_pt(*pt)
            if finalized_id == tentative_id:
                finalized_pts[finalized_id] = candidate_id
                tentative_id += 1
            else:
                tentative_pts[finalized_id].append(candidate_id)

        # NOTE: This array is compiled on demand.
        dt_pts = self.dt.points
        for finalized_id, candidate_ids in tentative_pts.items():
            for candidate_id in candidate_ids:
                if dt_pts[finalized_id][2] > pts[candidate_id][2]:
                    self.dt.remove(finalized_id)
                    self.dt.insert_one_pt(*pts[candidate_id])
                    # Replace the previous ID of the vertex in the PC with the
                    # current one.
                    finalized_pts[finalized_id] = candidate_id

        # Confirm that the resulting lookup table is correct.
        # TODO: Remove before final release.
        assert np.allclose(self.dt.points[1:], pts[list(finalized_pts.values())])

    @timing
    def intersect(self, objs: gpd.GeoDataFrame) -> np.ndarray:
        return self.to_gdf().overlay(objs)["id_1"].to_numpy()

    def to_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "id": np.arange(len(self)),
                "geometry": gpd.points_from_xy(self.las.x, self.las.y),
            },
            crs="EPSG:28992",
        )

    def slope(self):
        """Compute the slope field."""
        if self.dt is None:
            self.triangulate()

        # Compute the

    def __len__(self) -> int:
        return len(self.las.points)

    def __getitem__(self, key: int):
        ...
