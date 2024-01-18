from ._base import DataParser


class LiDARDataParser(DataParser):
    def __init__(self, id_: str) -> None:
        super().__init__(id_)

    def parse(self) -> None:
        pass
