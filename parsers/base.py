from abc import ABC, abstractmethod


class DataParser(ABC):
    def __init__(self) -> None:
        self._data = None
        self._surfs = None

    @abstractmethod
    def parse(self, obj_id: str) -> None:
        ...

    @abstractmethod
    def _update_fields(self, obj_id: str) -> None:
        ...
