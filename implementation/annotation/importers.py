from roofsense.annotation.importers import (RoboflowAnnotationImporter, )
from roofsense.bag3d import BAG3DTileStore
from roofsense.split import DatasetSplittingMethod


def main():
    importer = RoboflowAnnotationImporter(
        # The name of the source directory generally depends on the structure defined by the annotation provider.
        # This example simulates Roboflow.
        src_dirpath=r"./provider", tile_store=BAG3DTileStore(), )
    importer.import_(
        # The dataset directory must already exist, and it should contain the original tiles in an appropriately named subdirectory according to 'importer.dst_image_dirname'.
        dst_dirpath=r"./dataset",
        # The dataset is split randomly in this example because its size does not allow for stratification.
        splitting_method=DatasetSplittingMethod.RANDOM,)


if __name__ == "__main__":
    main()
