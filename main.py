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

import config
import downloaders


def main():
    # Configure the program runtime.
    config.config()

    # Fetch a tile id.
    # NOTE - The program can operate on a building-by-building or tile-by-tile basis. The default is the latter since
    #        the 3DBGAG tile identifiers are contained in the corresponding index, and thus the classification process
    #        can continue automatically.
    id_ = "9-284-556"

    # Download the corresponding 3DBAG data.
    downloaders.bag3d.download(id_)

    # Parse the tile.
    # FIXME - Do not parse previously processed tiles.
    # FIXME - Move the data parser out of the downloader module.
    downloaders.bag3d.DataParser(id_).parse()

    # Download the corresponding AHN4 and BM5 data.
    # NOTE - Load the index now so that it does not have to be reloaded when processing a different tile.
    # FIXME - Move the index loader out of the downloader module.
    index = downloaders.ahn34.load_index()
    downloaders.ahn34.download(id_, index)


if __name__ == "__main__":
    main()
