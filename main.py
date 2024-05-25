from __future__ import annotations

import config
import training


def generate_pretraining_data(size: int = 200, background_cutoff: float = 0.8) -> None:
    """Entry point for:
    roofsense --gen-pretrain-data <size> --bg-cutoff <pct>
    """
    # Initialize the program runtime.
    config.config(training=True)
    # Fake a random sample.
    samples = training.sampler.BAG3DSampler().sample(size,background_cutoff)


def train() -> None:
    """Entry point for:
    roofsense --train
    """
    ...


if __name__ == "__main__":
    generate_pretraining_data()
