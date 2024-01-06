import math

import numpy as np
import startinpy


def laplace(dt: startinpy.DT, x: float, y: float) -> tuple[list[int], list[float]]:
    num = dt.number_of_vertices()
    if not num:
        raise RuntimeError("The triangulation does not contain any vertices.")

    try:
        dt.locate(x, y)
    except Exception:
        raise ValueError(
            f"The interpolation point ({x, y}) is located outside the convex hull of "
            f"the triangulation."
        )

    idx = dt.insert_one_pt(x, y, 0)
    # The point was inserted at the same location as an existing vertex.
    if dt.number_of_vertices() == num:
        return [dt.points[idx]], [1]

    if dt.is_vertex_convex_hull(idx):
        dt.remove(idx)
        raise ValueError(
            f"The interpolation point ({x, y}) is located on the boundary of the "
            f"convex hull of the triangulation."
        )

    pts = dt.points
    trs = dt.incident_triangles_to_vertex(idx)

    centers = []
    contrib = []
    for tr in trs:
        centers.append(circumcentre(pts[tr[0]], pts[tr[1]], pts[tr[2]]))
        contrib.append(tr[2])
    centers.append(centers[0])

    weights = []
    for i in range(len(centers) - 1):
        dx = centers[i][0] - centers[i + 1][0]
        dy = centers[i][1] - centers[i + 1][1]
        e = math.hypot(dx, dy)

        dx = pts[idx][0] - pts[contrib[i]][0]
        dy = pts[idx][1] - pts[contrib[i]][1]
        w = math.hypot(dx, dy)

        weights.append(e / w)

    dt.remove(idx)
    # return contrib,weights
    con = []
    for c in contrib:
        con.append(pts[c][2])
    weights = np.array(weights)
    con = np.array(con)
    return np.sum(con * weights) / np.sum(weights)


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
