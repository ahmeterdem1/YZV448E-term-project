"""Utility helpers for preparing datasets and dataloaders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

StratifyInput = Union[str, Sequence[int], Sequence[str], Callable[[Any], Any]]


@dataclass(frozen=True)
class DatasetSplits:
    """Container that keeps references to the partitioned datasets."""

    train: Dataset
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def as_tuple(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        return self.train, self.val, self.test


def _validate_splits(val_split: float, test_split: float) -> None:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in the range [0, 1).")
    if not 0.0 <= test_split < 1.0:
        raise ValueError("test_split must be in the range [0, 1).")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be smaller than 1.0.")


def _resolve_targets(
    dataset: Dataset, stratify_by: Optional[StratifyInput]
) -> Optional[Sequence[Any]]:
    if stratify_by is None:
        for candidate in ("targets", "labels", "y"):
            if hasattr(dataset, candidate):
                targets = getattr(dataset, candidate)
                if len(targets) == len(dataset):
                    return targets
        return None

    if isinstance(stratify_by, str):
        if not hasattr(dataset, stratify_by):
            raise ValueError(f"Dataset does not expose attribute '{stratify_by}'.")
        targets = getattr(dataset, stratify_by)
        if len(targets) != len(dataset):
            raise ValueError(
                f"Attribute '{stratify_by}' has length {len(targets)} "
                f"but dataset length is {len(dataset)}."
            )
        return targets

    if isinstance(stratify_by, Sequence):
        if len(stratify_by) != len(dataset):
            raise ValueError(
                f"Provided stratify sequence has length {len(stratify_by)} "
                f"but dataset length is {len(dataset)}."
            )
        return stratify_by

    if callable(stratify_by):
        return [stratify_by(dataset[idx]) for idx in range(len(dataset))]

    raise TypeError("Unsupported stratify_by argument.")


def _split_lengths(
    total_len: int, val_split: float, test_split: float
) -> Tuple[int, int, int]:
    test_len = int(round(total_len * test_split))
    val_len = int(round(total_len * val_split))

    if test_split > 0.0 and test_len == 0:
        test_len = 1
    if val_split > 0.0 and val_len == 0:
        val_len = 1

    train_len = total_len - val_len - test_len
    if train_len <= 0:
        raise ValueError(
            "Dataset is too small for the requested splits. "
            "Reduce val/test ratios or provide more samples."
        )
    return train_len, val_len, test_len


def _stratified_indices(
    targets: Sequence[Any],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    label_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        label_to_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label_indices in label_to_indices.values():
        rng.shuffle(label_indices)
        n = len(label_indices)
        _, val_len, test_len = _split_lengths(n, val_split, test_split)
        test_slice = label_indices[:test_len]
        val_slice = label_indices[test_len : test_len + val_len]
        train_slice = label_indices[test_len + val_len :]
        train_idx.extend(train_slice)
        val_idx.extend(val_slice)
        test_idx.extend(test_slice)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def get_partitioned_dataset(
    dataset: Dataset,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    stratify_by: Optional[StratifyInput] = None,
    return_dataclass: bool = False,
) -> Union[
    Tuple[Dataset, Optional[Dataset], Optional[Dataset]],
    DatasetSplits,
]:
    """
    Partition a dataset into train/validation/test subsets.

    Parameters
    ----------
    dataset:
        Any ``torch.utils.data.Dataset`` implementation.
    val_split:
        Fraction reserved for validation. Set to ``0`` to skip.
    test_split:
        Fraction reserved for testing. Set to ``0`` to skip.
    seed:
        Random seed used for shuffling/stratification.
    stratify_by:
        Optional hint controlling stratified sampling. Can be:
        - the name of an attribute on the dataset (e.g. ``"targets"``),
        - a sequence with one entry per sample,
        - a callable that receives a dataset item and returns the label.
    return_dataclass:
        When ``True`` returns a ``DatasetSplits`` instance, otherwise a tuple.

    Returns
    -------
    (train, val, test) tuple or ``DatasetSplits`` container.
    """

    _validate_splits(val_split, test_split)
    dataset_len = len(dataset)
    train_len, val_len, test_len = _split_lengths(dataset_len, val_split, test_split)

    if stratify_by is not None:
        targets = _resolve_targets(dataset, stratify_by)
        if targets is None:
            raise ValueError(
                "Stratification was requested but no targets could be resolved."
            )
        train_idx, val_idx, test_idx = _stratified_indices(
            targets=targets, val_split=val_split, test_split=test_split, seed=seed
        )
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(dataset_len)
        test_idx = indices[:test_len]
        val_idx = indices[test_len : test_len + val_len]
        train_idx = indices[test_len + val_len :]

    def _subset(idxs: np.ndarray) -> Optional[Dataset]:
        return Subset(dataset, idxs.tolist()) if len(idxs) > 0 else None

    splits = DatasetSplits(
        train=_subset(train_idx),
        val=_subset(val_idx),
        test=_subset(test_idx),
    )
    return splits if return_dataclass else splits.as_tuple()


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """
    Convenience wrapper that instantiates a DataLoader with sensible defaults.

    Parameters
    ----------
    dataset:
        Dataset or subset instance to iterate over.
    batch_size:
        Number of samples per batch.
    shuffle:
        Whether to shuffle samples. Defaults to ``True`` when the dataset is used
        for training and ``False`` otherwise.
    """

    if shuffle is None:
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )