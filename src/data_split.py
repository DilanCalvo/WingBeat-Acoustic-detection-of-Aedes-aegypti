# Grouped, folder-stratified train/val/test split, shared by both models.

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from . import config


def dataset_of(folders: np.ndarray) -> np.ndarray:
    # Dataset name for each segment, from its "<dataset>/<class folder>" string.
    return np.array([str(f).split("/", 1)[0] for f in folders])


def holdout_mask(folders: np.ndarray) -> np.ndarray:
    # Segments reserved as an independent test set (config.HOLDOUT_DATASETS).
    return np.isin(dataset_of(folders), config.HOLDOUT_DATASETS)


def grouped_split(
    groups: np.ndarray,
    folders: np.ndarray | None = None,
    test_fraction: float = config.TEST_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
):
    # (train_idx, val_idx, test_idx) over the non-holdout segments, split by group.
    indices = np.arange(len(groups))

    if folders is None or not config.STRATIFY_SPLIT_BY_FOLDER:
        pool = indices
        train_idx, val_idx, test_idx = _split_one(pool, groups, test_fraction, val_fraction, seed)
    else:
        pool = indices[~holdout_mask(folders)]
        train_parts, val_parts, test_parts = [], [], []
        # Sorted for determinism: dict/array iteration order must not change the split.
        for offset, folder in enumerate(sorted(set(folders[pool].tolist()))):
            members = pool[folders[pool] == folder]
            # Vary the seed per folder so they don't all cut at the same place.
            tr, va, te = _split_one(members, groups, test_fraction, val_fraction, seed + offset)
            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)
        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
        test_idx = np.concatenate(test_parts)

    _assert_disjoint_groups(groups, train_idx, val_idx, test_idx)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _split_one(members: np.ndarray, groups: np.ndarray,
               test_fraction: float, val_fraction: float, seed: int):
    # Grouped split of one folder's segments (indices into the originals).
    n_groups = len(set(groups[members].tolist()))
    # GroupShuffleSplit needs at least 2 groups to cut anything.
    if n_groups < 2:
        empty = np.empty(0, dtype=members.dtype)
        return members, empty, empty

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_rel, test_rel = next(gss_test.split(members, groups=groups[members]))
    train_val, test_idx = members[train_val_rel], members[test_rel]

    if len(set(groups[train_val].tolist())) < 2:
        empty = np.empty(0, dtype=members.dtype)
        return train_val, empty, test_idx

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_rel, val_rel = next(gss_val.split(train_val, groups=groups[train_val]))
    return train_val[train_rel], train_val[val_rel], test_idx


def _assert_disjoint_groups(groups, train_idx, val_idx, test_idx) -> None:
    # No source recording may appear in more than one split.
    g_train, g_val, g_test = (set(groups[i].tolist()) for i in (train_idx, val_idx, test_idx))
    assert g_train.isdisjoint(g_test), "leakage: recording in both train and test"
    assert g_train.isdisjoint(g_val), "leakage: recording in both train and val"
    assert g_val.isdisjoint(g_test), "leakage: recording in both val and test"
