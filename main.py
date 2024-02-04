import config
import preprocessing


def main():
    # Configure the program runtime.
    config.config()

    # Fake a valid user input.
    obj_id = "9-284-556"

    # Download the corresponding 3DBAG data.
    preprocessing.downloaders.BAG3DDownloader().download(obj_id)

    # Parse the data.
    preprocessing.parsers.BAG3DParser().parse(obj_id)

    # Download the corresponding assets.
    preprocessing.downloaders.AssetDownloader().download(obj_id)

    # Parse the assets.
    preprocessing.parsers.ImageParser().parse(obj_id)
    preprocessing.parsers.LiDARParser().parse(obj_id)


if __name__ == "__main__":
    main()
