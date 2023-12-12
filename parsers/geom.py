from typing import Optional, Union

import numpy as np


class Point:
    """A two-dimensional point in the Cartesian coordinate system."""

    def __init__(self, x: Optional[float] = 0, y: Optional[float] = 0) -> None:
        self.x = x
        self.y = y


class BoundingBox:
    """An axis-aligned rectangle."""

    def __init__(self,
                 origin: Point,
                 len_x: float,
                 len_y: Optional[Union[float, None]] = None) -> None:
        """
        Creates a new axis-aligned rectangle.

        Args:
            origin:
                A :class:`Point<raster.Profiles>` instance representing the lower left (i.e., southwest) vertex of the
                rectangle.
            len_x:
                A non-negative number representing the side length of the rectangle along the X-axis.
            len_y:
                A non-negative number representing the side length of the rectangle along the Y-axis. If this parameter
                is not provided, the rectangle is then degenerated to a square of side length ``len_x``.
        """
        if len_y is None:
            len_y = len_x

        self.origin = origin

        self.len_x = len_x
        self.len_y = len_y

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """
        Intersects a point set with the bounding box.

        Args:
            pts:
                An ``N x M`` NumPy array containing the ``M``-dimensional point set of length ``N`` to filter. The ``X``
                and ``Y`` dimensions of this set are defined by the first and second column of the array, respectively.

        Returns:
            An ``N x 1`` NumPy array containing a boolean value corresponding to each point in the set, whose value
            depends on whether it is contained inside the bounding box (``True``) or not (``False``).

        Raises:
            IndexError:
                If the point set is one-dimensional i.e., contains only one point.
        """
        return np.logical_and(np.logical_and(self.origin.x <= pts[:, 0],
                                             pts[:, 0] <= self.origin.x + self.len_x),
                              np.logical_and(self.origin.y <= pts[:, 1],
                                             pts[:, 1] <= self.origin.y + self.len_y))
