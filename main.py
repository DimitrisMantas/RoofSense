import config
import downloaders
import parsers


def main():
    # Configure the program runtime.
    config.config()

    # Fake a random, valid user input.
    obj_id = "NL.IMBAG.Pand.0503100000032914"

    # Download the corresponding 3DBAG data.
    downloaders.BAG3DDataDownloader().download(obj_id)

    # Parse the tile.
    parsers.BAG3DDataParser().parse(obj_id)

    # Download the corresponding AHN and BM data.
    downloaders.AssetDataDownloader().download(obj_id)

    parsers.ImageDataParser().parse(obj_id)
    parsers.LiDARDataParser().parse(obj_id)


if __name__ == "__main__":
    main()
