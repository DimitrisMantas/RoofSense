import config
import downloaders
import parsers


def main():
    # Configure the program runtime.
    config.config()

    # Fake a valid user input.
    obj_id = "9-284-556"

    # Download the corresponding 3DBAG data.
    downloaders.BAG3DDownloader().download(obj_id)

    # Parse the data.
    parsers.BAG3DParser().parse(obj_id)

    # Download the corresponding assets.
    downloaders.AssetDownloader().download(obj_id)

    # Parse the assets.
    parsers.ImageParser().parse(obj_id)
    parsers.LiDARParser().parse(obj_id)


if __name__ == "__main__":
    main()
