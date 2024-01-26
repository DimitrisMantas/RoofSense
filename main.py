import config
import downloaders
import parsers


def main():
    # Configure the program runtime.
    config.config()

    # Fake a random, valid user input.
    obj_id = "9-284-556"

    # Download the corresponding 3DBAG data.
    downloaders.BAG3DDataDownloader().download(obj_id)

    # Parse the tile.
    parsers.bag3d.BAG3DDataParser().parse(obj_id)

    # Download the corresponding AHN and BM data.
    downloaders.AssetDataDownloader().download(obj_id)

    parsers.image.ImageDataParser().parse(obj_id)


if __name__ == "__main__":
    main()
