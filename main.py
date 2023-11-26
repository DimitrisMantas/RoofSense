import config
import download


def main():
    # Configure the program runtime.
    config.config()

    # Fetch a tile id.
    # NOTE - The program can operate on a building-by-building or tile-by-tile basis. The default is the latter since
    #        the 3DBGAG tile identifiers are contained in the corresponding index, and thus the classification process
    #        can continue automatically.
    id_ = "9-284-556"

    # Download the corresponding 3DBAG data.
    download.bag3d.download(id_)

    # Parse the tile.
    # FIXME - Do not parse previously processed tiles.
    # FIXME - Move the data parser out of the download module.
    download.bag3d.DataParser(id_).parse()

    # Download the corresponding AHN4 and BM5 data.
    # NOTE - Load the index now so that it does not have to be reloaded when processing a different tile.
    # FIXME - Move the index loader out of the download module.
    index = download.ahn34.load_index()
    download.ahn34.download(id_, index)


if __name__ == "__main__":
    main()
