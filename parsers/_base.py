from abc import ABC, abstractmethod


class DataParser(ABC):
    def __init__(self, obj_id: str) -> None:
        self.obj_id = obj_id

    @abstractmethod
    def parse(self) -> None:
        ...
