import config
import downloaders


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
    # FIXME: Move the 3DBAG data parser to `downloaders.utils`.
    downloaders.bag3d.DataParser(obj_id).parse()

    # Download the corresponding AHN and BM data.
    # NOTE - Load the index now so that it does not have to be reloaded when processing
    #        a different tile.
    # FIXME - Aggregate all index loaders and move them to a separate file inside  a
    #         data module.
    index1 = downloaders.ahn34.load_index()
    downloaders.ahn34.download(obj_id, index1)

    index2 = downloaders.ortho.load_index()
    downloaders.ortho.download(obj_id, index2)


if __name__ == "__main__":
    main()
