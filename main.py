import config
import preprocessing
import utils.iris


def gen_training_data(size: int = 10) -> None:
    """
    Generate training data of a particular size by repeatedly sampling random 3DBAG
    tiles, and pass it off to the user for annotation.

    This is the entry point of the program for inputs of the form:
    >> roofsense train 10
    or
    >> roofsense train --sample-size=10
    or
    >> roofsense train --size=10
    or
    >> roofsense train -s=10

    :param size: The desired sample size.
    """
    config.config()
    # Fake a sample size of one and get a random 3DBAG tile ID.
    # NOTE:
    obj_id = "NL.IMBAG.Pand.0363100012094841"

    # Download the corresponding 3DBAG data.
    preprocessing.downloaders.BAG3DDownloader().download(obj_id)

    # Parse the data.
    preprocessing.parsers.BAG3DParser().parse(obj_id)

    # Download the corresponding assets.
    preprocessing.downloaders.AssetDownloader().download(obj_id)

    # Parse the assets.
    preprocessing.parsers.ImageParser().parse(obj_id)
    preprocessing.parsers.LiDARParser().parse(obj_id)

    # Create the raster stack.
    preprocessing.merger.RasterStackBuilder().merge(obj_id)

    # Create the corresponding IRIS configuration file.
    utils.iris.ConfigurationFile().create(obj_id)


if __name__ == "__main__":
    gen_training_data()
