from __future__ import annotations

from abc import ABC, abstractmethod


class DataDownloader(ABC):
    @abstractmethod
    def download(self, tile_id: str) -> None:
        ...
