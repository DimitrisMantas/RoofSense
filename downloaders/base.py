from __future__ import annotations

from abc import ABC, abstractmethod

from utils.type import BoundingBoxLike


class DataDownloader(ABC):
    """
    The interface all data downloaders must provide.
    """

    @abstractmethod
    def download(self, obj_id: str | BoundingBoxLike) -> None:
        ...
