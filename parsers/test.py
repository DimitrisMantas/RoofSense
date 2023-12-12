import time

import laspy
import numpy as np


class PointCloud:
    def __init__(self, filename: str) -> None:
        with laspy.open(filename) as f:
            self.data = f.read()

    def extract_buildings(self) -> laspy.PackedPointRecord:
        pts = self.data.points[np.logical_or(self.data.classification == 6,
                                             self.data.classification == 26)]
        PointCloud.ensure_contiguous(pts)
        return pts

    @staticmethod
    def ensure_contiguous(pts: laspy.PackedPointRecord) -> None:
        if not pts.array.flags['C_CONTIGUOUS']:
            pts.array = np.ascontiguousarray(pts.array)


t0 = time.perf_counter()
pc = PointCloud("C:/Documents/RoofSense/data/temp/37EN2_11.LAZ")
t1 = time.perf_counter()
print(t1 - t0)
t0 = time.perf_counter()
x = pc.extract_buildings()
x = 1
t1 = time.perf_counter()
print(t1 - t0)
