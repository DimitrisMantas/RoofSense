from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Generator, Any

import torch
from torch import default_generator
from torch.utils.data import Dataset
from torchgeo.datasets import (GeoDataset,
                               random_bbox_assignment,
                               random_grid_cell_assignment, )


def random_file_split(
    dataset: Dataset[dict[str, Any]],
    lengths: Sequence[int | float],
    generator: Generator | None = default_generator,
) -> list[Dataset[dict[str, Any]]]:
    splits = torch.utils.data.random_split(
        dataset, lengths=lengths, generator=generator
    )

    datasets = []
    for split in splits:
        tmp = copy.deepcopy(dataset)
        # noinspection PyUnresolvedReferences
        tmp._files = [split.dataset.files[i] for i in split.indices]
        datasets.append(tmp)

    return datasets


def random_bbox_split(
    dataset: GeoDataset,
    lengths: Sequence[int | float],
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    return random_bbox_assignment(dataset, lengths=lengths, generator=generator)


def random_grid_split(
    dataset: GeoDataset,
    lengths: Sequence[int | float],
    size: int,
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    return random_grid_cell_assignment(
        dataset, fractions=lengths, grid_size=size, generator=generator
    )
