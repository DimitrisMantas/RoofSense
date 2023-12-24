from abc import ABC, abstractmethod


class DataParser(ABC):
    def __init__(self, id_: str) -> None:
        self.id = id_

    @abstractmethod
    def parse(self) -> None:
        ...
