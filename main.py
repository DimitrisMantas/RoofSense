import config
import downloaders
import parsers


def main():
    # Configure the program runtime.
    config.config()

    # Fake a valid user input.
    obj_id = "9-284-556"

    # Download the corresponding 3DBAG data.
    downloaders.BAG3DDataDownloader().download(obj_id)

    # Parse the data.
    parsers.BAG3DDataParser().parse(obj_id)

    # Download the corresponding assets.
    downloaders.AssetDataDownloader().download(obj_id)

    # Parse the assets.
    parsers.ImageDataParser().parse(obj_id)
    parsers.LiDARDataParser().parse(obj_id)


if __name__ == "__main__":
    main()
