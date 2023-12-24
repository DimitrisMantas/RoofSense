# -- my_code_hw01.py
# -- geo1015.2023.hw01
# -- Hugo Ledoux
# -- 666
import math

import numpy as np
import startinpy

"""
You can add any new function to this unit.

Do not modify the functions given or change the signature of the functions.

You can use any packages from the Python standard Library 
(therefore nothing with pip install, except those allowed for hw01). 

You need to complete the 3 functions below:
  1. `get_voronoi_edges()`
  1. `get_area_voronoi_cell()`
  1. `interpolate_tin()`

"""


class Tin:
    def __init__(self):
        self.dt = startinpy.DT()

    def number_of_vertices(self):
        return self.dt.number_of_vertices()

    def number_of_triangles(self):
        return self.dt.number_of_triangles()

    def insert_one_pt(self, x, y, z):
        self.dt.insert_one_pt(x, y, z)

    def info(self):
        print(self.dt.points)

    def get_delaunay_vertices(self):
        return self.dt.points

    def get_delaunay_edges(self):
        pts = self.dt.points
        edges = []
        for tr in self.dt.triangles:
            a = pts[tr[0]]
            b = pts[tr[1]]
            c = pts[tr[2]]
            edges.append(a)
            edges.append(b)
            edges.append(a)
            edges.append(c)
            edges.append(b)
            edges.append(c)
        return edges

    def get_voronoi_edges(self):
        """
        !!! TO BE COMPLETED !!!

        Function that returns all the Voronoi edges of the bounded
        Voronoi cells in the dataset.

        Input:
            none
        Output:
            edges: an array of points (a point is an array with 2 values [x, y]).
                   each consecutive pair forms an edge.
                   if edges = [ [0., 0.], [1., 0.], [1., 1.], [0., 1.] ] then 2 edges
                   will be drawn: (0,0)->(1,0) + (1,1)->(0,1)
                   (function get_delaunay_edges() uses the same principle)
        """
        pts = self.dt.points
        edges = []
        for tr in self.dt.triangles:
            p0 = circumcentre(pts[tr[0]], pts[tr[1]], pts[tr[2]])
            adjs = self.dt.adjacent_triangles_to_triangle(tr)
            for each in adjs:
                edges.append(p0)
                edges.append(circumcentre(pts[each][0], pts[each][1], pts[each][2]))
        return edges

    def interpolate_tin(self, x, y):
        """
        !!! TO BE COMPLETED !!!

        Function that interpolates linearly in a TIN.

        Input:
            x:      x-coordinate of the interpolation location
            y:      y-coordinate of the interpolation location
        Output:
            z: the estimation of the height value,
               numpy.nan if outside the convex hull
               (NaN: Not a Number https://numpy.org/devdocs/reference/constants.html#numpy.nan)
        """
        try:
            expect = self.dt.interpolate({"method": "Laplace"}, [[x, y]], strict=True)[
                0
            ]
            actual = self.laplace(x, y)
            print(f"{expect =}")
            print(f"{actual =}")
            return expect
        except:
            return np.nan

    def laplace(self, x: float, y: float) -> tuple[list[int], list[float]]:
        # // -- cannot interpolate if no TIN
        # if dt.is_init == false {
        #     re.push(Err(StartinError::EmptyTriangulation));
        #     continue;
        # }

        try:
            self.dt.locate(x, y)
        except Exception:
            raise ValueError("The point is outside the convex hull.")

        idx = self.dt.insert_one_pt(x, y, 0)
        # TODO: Add a case where the point is already in the DT.
        if self.dt.is_vertex_convex_hull(idx):
            self.dt.remove(idx)
            raise ValueError("The point is outside the convex hull.")

        pts = self.dt.points
        trs = self.dt.incident_triangles_to_vertex(idx)

        centers = []
        contrib = []
        for tr in trs:
            centers.append(circumcentre(pts[tr[0]], pts[tr[1]], pts[tr[2]]))
            contrib.append(tr[2])
        centers.append(centers[-1])

        weights = []
        for i in range(len(centers) - 1):
            dx = centers[i][0] - centers[i + 1][0]
            dy = centers[i][1] - centers[i + 1][1]
            e = math.hypot(dx, dy)

            dx = pts[idx][0] - pts[contrib[i]][0]
            dy = pts[idx][1] - pts[contrib[i]][1]
            w = math.hypot(dx, dy)

            weights.append(e / w)

        self.dt.remove(idx)

        # return contrib, weights
        con = []
        for c in contrib:
            con.append(pts[c][2])
        weights = np.array(weights)
        con = np.array(con)
        return np.sum(con * weights) / np.sum(weights)

    def get_area_voronoi_cell(self, vi):
        """
        !!! TO BE COMPLETED !!!

        Function that obtain the area of one Voronoi cells.

        Input:
            vi:     the position of the vertex in the TIN structure to display
        Output:
            z: the area of vi Voronoi cell,
               return numpy.inf if the cell is unbounded
               (infinity https://numpy.org/devdocs/reference/constants.html#numpy.inf)
        """
        if self.dt.is_vertex_convex_hull(vi) == True:
            return np.inf
        elif self.dt.number_of_triangles() < 1:
            return np.inf
        pts = self.dt.points
        trs = self.dt.incident_triangles_to_vertex(vi)
        ccs = []
        for t in trs:
            ccs.append(circumcentre(pts[t[0]], pts[t[1]], pts[t[2]]))
        ccs.append(ccs[0])
        area = 0.0
        for i, cc in enumerate(ccs[:-1]):
            a = pts[vi]
            b = ccs[i]
            c = ccs[i + 1]
            area += det3x3(a[0], a[1], 1, b[0], b[1], 1, c[0], c[1], 1) / 2.0
        return area


# -- nicked from http://www.ambrsoft.com/trigocalc/circle3d.htm
def circumcentre(a, b, c):
    A = det3x3(a[0], a[1], 1, b[0], b[1], 1, c[0], c[1], 1)
    B = det3x3(
        a[0] * a[0] + a[1] * a[1],
        a[1],
        1,
        b[0] * b[0] + b[1] * b[1],
        b[1],
        1,
        c[0] * c[0] + c[1] * c[1],
        c[1],
        1,
    )
    C = det3x3(
        a[0] * a[0] + a[1] * a[1],
        a[0],
        1,
        b[0] * b[0] + b[1] * b[1],
        b[0],
        1,
        c[0] * c[0] + c[1] * c[1],
        c[0],
        1,
    )
    x = B / (2 * A)
    y = -C / (2 * A)
    return (x, y)


def det3x3(ax, ay, az, bx, by, bz, cx, cy, cz):
    temp1 = ax * (by * cz - bz * cy)
    temp2 = ay * (bx * cz - bz * cx)
    temp3 = az * (bx * cy - by * cx)
    return temp1 - temp2 + temp3
