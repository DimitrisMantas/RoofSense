from abc import ABC, abstractmethod

import numpy as np

import config


class DataMergerABC(ABC):
    def __init__(self) -> None:
        self.rand = np.random.default_rng(int(config.var("SEED")))

    @abstractmethod
    def merge(self):
        ...
