#          Copyright © 2023 Dimitris Mantas
#
#          This file is part of RoofSense.
#
#          This program is free software: you can redistribute it and/or modify
#          it under the terms of the GNU General Public License as published by
#          the Free Software Foundation, either version 3 of the License, or
#          (at your option) any later version.
#
#          This program is distributed in the hope that it will be useful,
#          but WITHOUT ANY WARRANTY; without even the implied warranty of
#          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#          GNU General Public License for more details.
#
#          You should have received a copy of the GNU General Public License
#          along with this program.  If not, see <https://www.gnu.org/licenses/>.

import enum


class TileResolution(enum.Enum):
    LOW, HIGH = range(2)


def _low_resolution_tile_index(point: tuple[float, float]) -> tuple[float, float]:
    pass


def _high_resolution_tile_index(point: tuple[float, float]) -> tuple[float, float]:
    pass


_TILE_RESOLUTION_T0_INDEX = {
    TileResolution.LOW: _low_resolution_tile_index,
    TileResolution.HIGH: _high_resolution_tile_index, }


class TileIndex:
    # NOTE - Is it OK to hardcode strings?
    _BASE_URL = "https://ns_hwh.fundaments.nl/hwh-ortho/"

    def __init__(self, year: int, resolution: TileResolution) -> None:
        self.__year = year
        self.__resolution = resolution
        self.__init_url()

        self.__index = _TILE_RESOLUTION_T0_INDEX[self.__resolution]

    @property
    def year(self) -> int:
        return self.__year

    @year.setter
    def year(self, value: int) -> None:
        # NOTE - Is this check worth it?
        if self.__year == value:
            return
        self.__year = value
        self.__init_url()

    @property
    def resolution(self) -> TileResolution:
        return self.__resolution

    @resolution.setter
    def resolution(self, value: TileResolution) -> None:
        # NOTE - Is this check worth it?
        if self.__resolution == value:
            return
        self.__resolution = value
        self.__index = _TILE_RESOLUTION_T0_INDEX[self.__resolution]

    def __init_url(self) -> None:
        self.__url = TileIndex._BASE_URL
        # NOTE - Is it OK to hardcode strings?
        if self.__resolution == TileResolution.LOW:
            self.__url += f"{self.__year}/Ortho/6/CIR_tif/"
        else:
            # TODO - Find out how HR URLs work. If the missing information is tile-dependent, use placeholders and
            #        handle them in `__build_url`.
            self.__url += f"{self.__year}/Ortho/{0}/{1}/beelden_tif_tegels/"

    def build_url(self, point: tuple[float, float]) -> str:
        pass
