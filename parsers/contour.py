import collections
import itertools
import multiprocessing

import fiona
import numpy as np
import pyproj
import rasterio
import startinpy

import utils

Edge = collections.namedtuple("Edge", "e1 e2")


def read_dtm(filename: str) -> np.ndarray:
    f: rasterio.DatasetReader
    with rasterio.open(filename) as f:
        data = f.read(1)

        pts = []
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                x = f.transform[2] + (col * f.transform[0]) + (0.5 * f.transform[0])
                y = f.transform[5] + (row * f.transform[4]) + (0.5 * f.transform[4])
                z = data[row][col]

                if z != f.nodatavals:
                    pts.append([x, y, z])

    return np.array(pts)


def isolines(filename: str, isovalue: float):
    reader: rasterio.DatasetReader
    # Read the DTM into memory.
    pts = read_dtm(filename)

    # Compute the Delaunay triangulation (DT) of the DTM.
    dt = startinpy.DT()
    # This suppression is so that PyCharm does not complain about unexpected arguments being passed to
    # startinpy.DT.insert.
    # noinspection PyArgumentList
    dt.insert(pts, insertionstrategy="BBox")

    segments = []
    for simplex in dt.triangles:
        # Check how the elevation of the vertices of each simplex compares to the isovalue.
        below = [v for v in simplex if dt.get_point(v)[2] < isovalue]
        above = [v for v in simplex if dt.get_point(v)[2] >= isovalue]
        # The contour lines do not intersect the simplex.
        if len(below) == 0 or len(below) == 3:
            continue

        # One vertex of the simplex is either below or above the isovalue, as defined above, with the opposite being
        # true for the remaining two.

        # Find these vertices.
        minority = above if len(above) < len(below) else below
        majority = above if len(above) > len(below) else below

        # Find the edges of the DT which intersect the contour lines. These edges are not properly oriented and might be
        # computed more than once since any two simplices share exactly one edge.
        edges = [Edge(minority[0], majority[0]), Edge(minority[0], majority[1])]

        endpts = []
        for edge in edges:
            # Compute the location of the intersection of each edge and the contour lines along its length.
            p = (isovalue - dt.get_point(edge.e2)[2]) / (
                        dt.get_point(edge.e1)[2] - dt.get_point(edge.e2)[2])
            # Compute the intersection.
            intersection = (
            (1 - p) * dt.get_point(edge.e2)[0] + p * dt.get_point(edge.e1)[0],
            (1 - p) * dt.get_point(edge.e2)[1] + p * dt.get_point(edge.e1)[1])

            endpts.append(intersection)

        # Populate the line segments corresponding to the isolines.
        segments.append(Edge(endpts[0], endpts[1]))

    # Discard any duplicate segments.
    segments = set(segments)

    # Map each segment endpoint to the segment it corresponds to.
    endpt_to_segment = collections.defaultdict(set)
    for segment in segments:
        endpt_to_segment[segment.e1].add(segment)
        endpt_to_segment[segment.e2].add(segment)

    # TODO - Rename this to contours.
    contour_lines = []
    while segments:
        # Get a random segment and enqueue it.
        segment = collections.deque(segments.pop())
        while True:
            # TODO - Code Review.
            candidate_tails = endpt_to_segment[segment[-1]].intersection(segments)
            if candidate_tails:
                tail = candidate_tails.pop()
                segment.append(tail.e1 if tail.e2 == segment[-1] else tail.e2)
                segments.remove(tail)
            candidate_heads = endpt_to_segment[segment[0]].intersection(segments)
            if candidate_heads:
                head = candidate_heads.pop()
                segment.appendleft(head.e1 if head.e2 == segment[0] else head.e2)
                segments.remove(head)
            if not candidate_tails and not candidate_heads:
                # There are no more segments touching this line, so we're done with it.
                contour_lines.append(list(segment))
                break

    return contour_lines


if __name__ == "__main__":
    # interval = 1
    # # Compute the elevation of each contour line.
    # isovalues = np.arange(np.ceil(np.min(pts[:, 2])), np.floor(np.max(pts[:, 2])) + interval, interval)

    filename = "dtm_0_5_cls"
    isovalues = [16, 18, 20, 22]

    with multiprocessing.Pool() as pool:
        contours = pool.starmap(isolines,
                                zip(itertools.repeat(filename + ".tif"), isovalues))

    # contours = isolines("data/test/dtm/0_5/test_dtm_0_5_cls.tif", 22)

    schema = {"geometry": "MultiLineString", "properties": [["ELEV", "float"]]}

    # Create the required directory to store the output file.
    utils.mkdirs(filename + ".gpkg")

    with fiona.open(filename + ".gpkg", "w", crs=pyproj.CRS("EPSG:28992").to_wkt(),
                    driver="GPKG",
                    schema=schema) as f:
        for i, contour in enumerate(contours):
            output = {
                "geometry": {'type': 'MultiLineString', 'coordinates': contour},
                "properties": {"ELEV": isovalues[i]}}
            f.write(output)
