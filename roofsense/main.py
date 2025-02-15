import config

from roofsense.preprocessing.chip_samplers import BAG3DSampler


def generate_pretraining_data(size: int = 300, background_cutoff: float = 0.8) -> None:
    # Initialize the program runtime.
    config.config()
    # Fake a random sample.
    BAG3DSampler().sample(size, background_cutoff)


if __name__ == "__main__":
    generate_pretraining_data()
