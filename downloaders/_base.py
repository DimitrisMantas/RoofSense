from abc import ABC, abstractmethod
from typing import Optional

import geopandas as gpd


# TODO: Harmonize this class name.
class DataDownloader(ABC):
    def __init__(
        self,  # TODO: Harmonize this parameter name.
        sindex: Optional[str] = None,
    ) -> None:
        if sindex is not None:
            self._sindex = gpd.read_file(sindex)
        else:
            self._sindex = None

    @abstractmethod
    def download(self, obj_id: str) -> None:
        ...
