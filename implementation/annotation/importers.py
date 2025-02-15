from roofsense.annotation.importers import RoboflowAnnotationImporter
from roofsense.bag3d import BAG3DTileStore
from roofsense.utilities.splits import DatasetSplittingMethod


def main():
    importer = RoboflowAnnotationImporter(
        # The name of the source directory generally depends on the structure defined by the annotation provider.
        # Hence, the user is responsible for the setting up or creating their new importer.
        # In the latter case, see 'RoboflowAnnotationImporter' for guidance on how to extend the corresponding base class.
        # This example simulates Roboflow.
        src_dirpath=r"data/provider",
        tile_store=BAG3DTileStore(),
    )
    importer.import_(
        # The dataset directory must already exist, and it should contain the original tiles in an appropriately named subdirectory according to 'importer.dst_image_dirname'.
        dst_dirpath=r"data/dataset",
        # The example dataset is not suitable for stratified splitting due to its small size which would make discovering splits with full class support purely by chance unlikely.
        splitting_method=DatasetSplittingMethod.RANDOM,
    )


if __name__ == "__main__":
    main()
