from __future__ import annotations

from abc import ABC, abstractmethod


class _Downloader(ABC):
    """Interface all downloaders must implement."""

    @abstractmethod
    def download(self, tile_id: str) -> None:
        """Download a single 3DBAG tile or its corresponding assets (i.e., from the
        latest BM5, 8 cm RGB orthoimagery and AHN point cloud collections).

        :param tile_id: The tile ID (e.g., 9-284-556).
        """
        ...
