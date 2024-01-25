import config
import downloaders
import parsers


def main():
    # Configure the program runtime.
    config.config()

    # Fake a random, valid user input.
    obj_id = "9-284-556"

    # Download the corresponding 3DBAG data.
    downloaders.bag3d.BAG3DDataDownloader().download(obj_id)

    # Parse the tile.
    # TODO: Do not parse previously processed tiles.
    # TODO: Rewrite the data parsers so that they can be reused across multiple object.
    parsers.bag3d.BAG3DDataParser(obj_id).parse()

    # Download the corresponding AHN and BM data.
    downloaders.lidar.LiDARDataDownloader().download(obj_id)
    downloaders.image.ImageDataDownloader().download(obj_id)


if __name__ == "__main__":
    main()
