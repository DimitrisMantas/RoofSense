from abc import ABC, abstractmethod
from typing import Optional

import geopandas as gpd


class DataDownloader(ABC):
    def __init__(
        self,
        index: Optional[str] = None,
    ) -> None:
        if index is not None:
            self._index = gpd.read_file(index)
        else:
            self._index = None

    @abstractmethod
    def download(self, obj_id: str) -> None:
        ...
