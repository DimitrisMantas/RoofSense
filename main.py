import config
import downloaders
from parsers.bag3d import BAG3DDataParser
from parsers.lidar import LidarDataParser
from parsers.ortho import OrthoDataParser


def main():
    # Configure the program runtime.
    config.config()

    # Fetch a tile id.

    # NOTE - The program can operate on a building-by-building or tile-by-tile basis.
    #        The default is the latter since the 3DBGAG tile identifiers are contained
    #        in the corresponding index, and thus the classification process can
    #        continue automatically.
    obj_id = "9-284-556"

    # Download the corresponding 3DBAG data.
    downloaders.bag3d.download(obj_id)

    # Parse the tile.
    # FIXME: Do not parse previously processed tiles.
    BAG3DDataParser(obj_id).parse()

    # Download the corresponding AHN and BM data.
    # NOTE - Load the index now so that it does not have to be reloaded when processing
    #        a different tile.
    # FIXME - Aggregate all index loaders and move them to a separate file inside a
    #         data module.
    index1 = downloaders.lidar.load_index()
    downloaders.lidar.download(obj_id, index1)

    index2 = downloaders.ortho.load_index()
    downloaders.ortho.download(obj_id, index2)

    LidarDataParser("").parse()
    OrthoDataParser("").parse()


if __name__ == "__main__":
    main()
