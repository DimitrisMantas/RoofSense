# %%
import collections
import math

import geopandas as gpd
import laspy
import numpy as np
import startinpy

from feature.parsers.raster import Raster


# %%
class PointCloud:
    def __init__(self, filename: str) -> None:
        with laspy.open(filename) as f:
            self.las = f.read()

    def __len__(self) -> int:
        return len(self.las.points)

    def __getitem__(self, key: int):
        ...


# FIXME: Rewrite this function as a PointCloud method.


def to_gdf(pc: PointCloud) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": np.arange(len(pc)), "geometry": gpd.points_from_xy(pc.las.x, pc.las.y)},
        crs="EPSG:28992",
    )


# FIXME: Rewrite this function as a PointCloud method.


def intersect(pc: PointCloud, objs: gpd.GeoDataFrame) -> np.ndarray:
    return to_gdf(pc).overlay(objs)["id_1"].to_numpy()


# %%
# class BufferableGeoDataFrame(gpd.GeoDataFrame):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()
#
#     def buffer(self, *args, **kwargs) -> None:
#         ...


def __init__(filename: str) -> gpd.GeoDataFrame:
    return gpd.read_file(filename)


# TODO: Review the various buffer options.
# TODO: Rewrite this function as an appropriate class method.
def buffer(objs: gpd.GeoDataFrame, dist: float = 10) -> None:
    objs["geometry"] = objs["geometry"].buffer(dist)


# %% md
## Step 1

# %%
pc = PointCloud("C:/Documents/RoofSense/feature/aula.laz")
# %% md
## Step 2

# %%
fps = __init__("C:/Documents/RoofSense/data/temp/9-284-556.buildings.gpkg")
# %% md
## Step 3

# %%
buffer(fps)
# %% md
## Step 4

# %%
ids = intersect(pc, fps)
# %% md

# %%
# TOSELF: Overwrite the original point cloud?
pts = pc.las.points[ids]
pts = np.vstack(
    (
        pts.x,
        pts.y,  # NOTE: The point elevation must be used instead of e.g., reflectance
        #       because it is required in the subsequent steps.
        pts.z,
    )
).transpose()
# %% md
## Step 5

# %%
# FIXME: Rewrite this block as a PointCloud method and improve its documentation.

dt = startinpy.DT()
# NOTE: The snap tolerance cannot be set to zero (i.e., disabled) so the nearest
#       distinct value is used instead.
dt.snap_tolerance = math.ulp(0)

# Maintain a reference to the duplicate PC vertices.
# NOTE: Two or more vertices are considered to be duplicate when their two-dimensional
#       projections on the Euclidean plane are identical. However. they can obviously
#       differ in n-dimensional space. In this context these vertices are rechecked
#       after the DT has been constructed so that the one with the highest elevation is
#       actually inserted.
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
    finalized_id = dt.insert_one_pt(*pt)
    if finalized_id == tentative_id:
        finalized_pts[finalized_id] = candidate_id
        tentative_id += 1
    else:
        tentative_pts[finalized_id].append(candidate_id)

# NOTE: This array is compiled on demand.
dt_pts = dt.points
for finalized_id, candidate_ids in tentative_pts.items():
    for candidate_id in candidate_ids:
        if dt_pts[finalized_id][2] > pts[candidate_id][2]:
            dt.remove(finalized_id)
            dt.insert_one_pt(*pts[candidate_id])
            # Replace the previous ID of the vertex in the PC with the current one.
            finalized_pts[finalized_id] = candidate_id
# %% md

# %%
np.allclose(dt.points[1:], pts[list(finalized_pts.values())])
# %% md
## Step 6


# %% md

# %%
# TOSELF: Promote this argument to an environment variable?
CELL_SIZE = 0.25

grid = Raster(CELL_SIZE, dt.get_bbox())

# FIXME: Integrate this block into the Raster initializer.
# Construct the grid.
# TODO: Add documentation.
rows, cols = np.mgrid[grid.len_y - 1 : -1 : -1, 0 : grid.len_x]
# TODO: Add documentation.
xx = grid.bbox[0] + CELL_SIZE * (cols + 0.5)
yy = grid.bbox[1] + CELL_SIZE * (rows + 0.5)
# TODO: Add documentation.
cells = np.column_stack([xx.ravel(), yy.ravel()])
# %% md

# %%
# TODO: Speed up this block.
# TOSELF: Compute the slope for all faces to avoid additional calls to dt.locate?
valid_cells = []
valid_faces = []
for i, center in enumerate(cells):
    try:
        valid_faces.append(dt.locate(*center))
    except Exception:
        continue
    valid_cells.append(i)


# TOSELF: Discard duplicate faces?
# valid_faces = np.unique(valid_faces, axis=0)
# %%


def locate(cells):
    valid_cells = []
    valid_faces = []
    for i, center in enumerate(cells):
        try:
            # NOTE: The DT must be initialized in the global scope when this function
            #       is called.
            valid_faces.append(dt.locate(*center))
        except Exception:
            continue
        valid_cells.append(i)
    return valid_cells, valid_faces


import concurrent.futures
import os

valid_faces_mp = []
valid_cells_mp = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for c, f in executor.map(locate, np.array_split(cells, os.cpu_count())):
        valid_faces_mp.extend(f)
        valid_cells_mp.extend(c)

print(f"cells check: {valid_cells == valid_cells_mp}")
print(f"cells check: {valid_faces == valid_faces_mp}")
x = 1
# %%
# TODO: Add documentation.

valid_faces = np.array(valid_faces)
v1 = dt_pts[valid_faces[:, 0]]
v2 = dt_pts[valid_faces[:, 1]]
v3 = dt_pts[valid_faces[:, 2]]

u = v2 - v1
v = v3 - v1

n = np.cross(u, v)
n = n / np.linalg.norm(n, axis=1)[:, None]

z = np.array([0, 0, 1])
s = np.degrees(np.arccos(np.dot(n, z)))
# %%
grid._Raster__data[np.divmod(valid_cells, grid.len_x)] = s
grid.save("../aula_slpe.tiff")
# %% md
## Step 6

# %%
# Interpolate the field values at the cell centers.
vals = dt.interpolate({"method": "Laplace"}, cells)

# Populate the raster.
grid._Raster__data = vals.reshape((grid.len_y, grid.len_x))

# Save the raster.
grid.save("../aula_elev.tiff")
