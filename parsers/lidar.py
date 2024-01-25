from ._base import DataParser


# TODO: Do not parse previously processed tiles.
# TODO: Rewrite the data parsers so that they can be reused across multiple object.
class LiDARDataParser(DataParser):
    def __init__(self, id_: str) -> None:
        super().__init__(id_)

    def parse(self) -> None:
        pass
