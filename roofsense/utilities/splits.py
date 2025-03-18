from collections.abc import Iterable
from copy import deepcopy
from enum import UNIQUE, Enum, auto, verify
from itertools import accumulate, combinations
from math import floor, isclose
from typing import Any, Sequence

import numpy as np
import scipy as sp

# @dataclass(frozen=True, slots=True)
# class Split:
#     tra: list[int]
#     val: list[int]
#     tst: list[int]
#
#     def __getitem__(self, item: int) -> list[int]:
#         return getattr(self, fields(self)[item].name)
#
#     def __iter__(self) -> Generator[list[int]]:
#         for field in fields(self):
#             yield getattr(self, field.name)
#
#     def __len__(self) -> int:
#         return len(fields(self))


@verify(UNIQUE)
class DatasetSplittingMethod(Enum):
    """Methods to split a given dataset into training, validation, and test subsets."""

    RANDOM = auto()
    """Split a dataset of a given size into random, non-overlapping subsets of given lengths.
    This function mimics the functionality of 'torch.utils.data.random_split'.
    """
    STRATIFIED = auto()
    """Split a dataset of a given size into random, non-overlapping, stratified subsets of given lengths.
    This function implements the algorithm described in: 'https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549'.
    """


@verify(UNIQUE)
class SubsetHistogramNormalizationMethod(Enum):
    """Methods to normalize the histogram of a given subset."""

    GLOBAL = auto()
    """Normalize a given subset histogram by its maximum possible area as computed by the class counts of the corresponding dataset."""
    LOCAL = auto()
    """Normalize a given subset histogram by its area."""


@verify(UNIQUE)
class SubsetPairwiseDistanceReductionMethod(Enum):
    """Methods to reduce the pairwise distances amongst multiple given subsets."""

    AVG = auto()
    """Reduce a given collection of pairwise distances across all corresponding subsets by considering their arithmetic mean."""
    MAX = auto()
    """Reduce a given collection of pairwise distances across all corresponding subsets by considering their maximum value."""
    SUM = auto()
    """Reduce a given collection of pairwise distances across all corresponding subsets by considering their sum."""


def random_split(
    dataset_length: int,
    lengths: Iterable[float],
    generator: np.random.Generator = np.random.default_rng(0),
) -> list[list[int]]:
    """Split a dataset of a given size into random, non-overlapping subsets of given lengths.

    This function mimics the functionality of 'torch.utils.data.random_split'.

    Args:
        dataset_length:
            The length of the input dataset.
        lengths:
            The lengths of the subsets to be produced.
            Each length may be defined either as an integer or a fraction of the input dataset length.
            In this case, the corresponding integer length is computed by multiplying the given length with that of the input dataset and rounding the result towards negative infinity.
        generator:
            The RNG used to randomly compute a random permutation of the input dataset before it is split.

    Returns:
        A list containing the required subsets in the order in which they are defined by their lengths.
        Each subset is defined by a list of integer indices into the original dataset.

    Notes:
        If there are any remainders after splitting the input dataset, a single item is distributed in a round-robin fashion to each subset produced until there are no remainders left.
    """
    if isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, length in enumerate(lengths):
            if not 0 < length < 1:
                raise ValueError
            abs_len = int(floor(dataset_length * length))
            subset_lengths.append(abs_len)

        remainder = dataset_length - sum(subset_lengths)
        for i in range(remainder):
            subset = i % len(subset_lengths)
            subset_lengths[subset] += 1

        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                raise RuntimeError

    if (tot_len := sum(lengths)) != dataset_length:
        msg = (
            f"Provided lengths sum to {tot_len} but dataset length is {dataset_length}."
        )
        raise ValueError(msg)

    all_indices = generator.permutation(dataset_length)
    split_indices: list[list[int]] = []
    for index, (offset, length) in enumerate(zip(accumulate(lengths), lengths)):
        split_indices.append([i for i in all_indices[offset - length : offset]])

    return split_indices


def stratified_split(
    class_counts: np.ndarray[tuple[Any, Any], np.dtype[np.int32]],
    lengths: Iterable[float],
    max_generation_trials: int = 100,
    max_optimization_trials: int = 1000,
    generator: np.random.Generator = np.random.default_rng(0),
) -> list[list[int]]:
    r"""Split a dataset of a given size into random, non-overlapping, stratified subsets of given lengths.

    This function implements the algorithm described in: 'https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549'.

    Args:
        class_counts:
            A 2D array of shape :math:`\left(\left|\mathcal{D}\right|, C\right)` containing the support of each class in the input dataset in each corresponding segmentation mask.
        lengths:
            The lengths of the subsets to be produced.
            Each length may be defined either as an integer or a fraction of the input dataset length.
            In this case, the corresponding integer length is computed by multiplying the given length with that of the input dataset and rounding the result towards negative infinity.
        max_generation_trials:
            The number of valid random splits to generate and optimize.
            A split is considered to be valid iff each corresponding subset contains at least one pixel of each class in the input dataset.
            In this case, the subset is said to have full class support.
        max_optimization_trials:
            The number of valid item swaps to perform between each subset pair.
        generator:
            The RNG used to randomly compute a random permutation of the input dataset before it is split.

    Returns:
        A list containing the required subsets in the order in which they are defined by their lengths.
        Each subset is defined by a list of integer indices into the original dataset.

    Notes:
        Each initial split is generated using 'roofsense.split.random_split'.
    """
    dataset_length, _ = class_counts.shape

    best_splits: list[list[int]] | None = None
    best_dist = np.inf

    num_trials = 0
    while num_trials < max_generation_trials:
        splits = random_split(dataset_length, lengths, generator)

        proceed = True
        for split in splits:
            if not _has_full_class_support(class_counts, split):
                proceed = False
                break

        if not proceed:
            num_trials -= 1
            continue

        temp_splits, temp_dist = _optimize_splits(
            class_counts, splits, max_optimization_trials, generator
        )
        if temp_dist < best_dist:
            print(
                f"Mean Wasserstein distance improved in trial {num_trials} from {best_dist} to {temp_dist}."
            )
            best_splits = temp_splits
            best_dist = temp_dist

        num_trials += 1

    return best_splits


def _histogram(
    class_counts: np.ndarray[tuple[Any, Any], np.dtype[np.int32]],
    split: Sequence[int],
    normalize: SubsetHistogramNormalizationMethod
    | None = SubsetHistogramNormalizationMethod.LOCAL,
) -> np.ndarray[tuple[Any,], np.dtype[np.int32]]:
    r"""Compute and optionally normalize the histogram of a given subset.

    This function mimics the functionality of 'numpy.histogram'.

    Args:
        class_counts:
            A 2D array of shape :math:`\left(\left|\mathcal{D}\right|, C\right)` containing the support of each class in the input dataset in each corresponding segmentation mask.
        split:
            The subset to process.
        normalize:
            The normalization method to use.

    Returns:
        The subset histogram.
    """
    hist: np.ndarray[tuple[Any,], np.dtype[np.int32]] = class_counts[split].sum(axis=0)

    if normalize == SubsetHistogramNormalizationMethod.GLOBAL:
        hist /= class_counts.sum(axis=0)
    elif normalize == SubsetHistogramNormalizationMethod.LOCAL:
        hist = hist.astype(np.float64)
        hist /= hist.sum()

    return hist


def _has_full_class_support(
    class_counts: np.ndarray[tuple[Any, Any], np.dtype[np.int32]], split: Sequence[int]
) -> np.bool_:
    r"""Check if a given subset has full class support.

    Args:
        class_counts:
            A 2D array of shape :math:`\left(\left|\mathcal{D}\right|, C\right)` containing the support of each class in the input dataset in each corresponding segmentation mask.
        split:
            The subset to process.

    Returns:
        'True' if the subset has full class support; 'False' otherwise.
    """
    return np.all(class_counts[split].sum(axis=0) > 0)


def _pairwise_distance(
    class_counts: np.ndarray[tuple[Any, Any], np.dtype[np.int32]],
    splits: Sequence[Sequence[int]],
    reduction: SubsetPairwiseDistanceReductionMethod
    | None = SubsetPairwiseDistanceReductionMethod.AVG,
) -> float:
    r"""Compute and optionally reduce the pairwise distances amongst multiple given subsets.

    This function mimics the functionality of 'numpy.histogram'.

    Args:
        class_counts:
            A 2D array of shape :math:`\left(\left|\mathcal{D}\right|, C\right)` containing the support of each class in the input dataset in each corresponding segmentation mask.
        splits:
            A list containing the subsets.
            Each subset is defined by a list of integer indices into the original dataset.
        reduction:
            The reduction method to use.

    Returns:
        The reduced distance.
    """
    hists = [_histogram(class_counts, split_indices) for split_indices in splits]

    dists: list[float] = []
    for this, that in combinations(range(len(splits)), 2):
        dists.append(
            sp.spatial.distance.jensenshannon(hists[this], hists[that],base=class_counts.shape[1])
        )

    if reduction == SubsetPairwiseDistanceReductionMethod.AVG:
        dist = np.mean(dists)
    elif reduction == SubsetPairwiseDistanceReductionMethod.MAX:
        dist = np.max(dists)
    elif reduction == SubsetPairwiseDistanceReductionMethod.SUM:
        dist = np.sum(dists)

    return dist


def _optimize_splits(
    class_counts: np.ndarray[tuple[Any, Any], np.dtype[np.int32]],
    splits: Sequence[Sequence[int]],
    max_trials: int,
    generator: np.random.Generator,
) -> float | tuple[Sequence[Sequence[int]] | None, float]:
    r"""Optimize multiple given splits in a stratified fashion.

    Args:
        class_counts:
            A 2D array of shape :math:`\left(\left|\mathcal{D}\right|, C\right)` containing the support of each class in the input dataset in each corresponding segmentation mask.
        splits:
            A list containing the subsets.
            Each subset is defined by a list of integer indices into the original dataset.
        max_trials:
            The number of valid item swaps to perform between each subset pair.
        generator:
            The RNG used to randomly compute a random permutation of the input dataset before it is split.

    Returns:
        The optimized splits.
    """
    best_splits: Sequence[Sequence[int]] = splits

    prev_dist: float | None = None
    curr_dist = _pairwise_distance(class_counts, splits)
    if isclose(curr_dist, 0):
        return curr_dist

    num_trials = 0
    max_trials = -1 if max_trials is None else max_trials
    while num_trials < max_trials:
        for this, that in combinations(range(len(splits)), 2):
            if isclose(curr_dist, 0):
                return curr_dist

            if prev_dist is not None and isclose(curr_dist, prev_dist, rel_tol=1e-6):
                return curr_dist

            this_index = generator.choice(len(splits[this]))
            that_index = generator.choice(len(splits[that]))

            temp_splits: Sequence[Sequence[int]] = deepcopy(best_splits)
            temp_splits[this][this_index], temp_splits[that][that_index] = (
                temp_splits[that][that_index],
                temp_splits[this][this_index],
            )

            if not _has_full_class_support(
                class_counts, temp_splits[this]
            ) or not _has_full_class_support(class_counts, temp_splits[that]):
                continue

            dist = _pairwise_distance(class_counts, temp_splits)
            # print(dist)
            if dist < curr_dist:
                best_splits = temp_splits
                prev_dist, curr_dist = curr_dist, dist

            num_trials += 1

    return best_splits, curr_dist
