from __future__ import annotations

from enum import IntEnum, auto
from operator import itemgetter


def _add_all(cls: Band) -> Band:
    cls._member_map_["ALL"] = tuple(cls)
    return cls


def _add_rgb(cls: Band) -> Band:
    cls._member_map_["RGB"] = itemgetter(*[0, 1, 2])(list(cls))
    return cls


@_add_all
@_add_rgb
class Band(IntEnum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    REFLECTANCE = auto()
    SLOPE = auto()
    nDRM = auto()
    DENSITY = auto()
