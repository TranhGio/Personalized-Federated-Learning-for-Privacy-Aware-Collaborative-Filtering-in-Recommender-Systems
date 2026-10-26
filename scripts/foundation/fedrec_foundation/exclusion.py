"""Per-user exclusion set with flat items + indptr layout (FND-03 + IMP-3).

Each user's exclusion set equals ``train_positives_u union {test_item_u}``
per D-13. The on-disk layout is a flat ``int32 items`` array plus an
``int64 indptr`` offset table (CSR-style) -- strictly better than a
keyed-dict layout at 6040 users: smaller zip footprint, O(1) per-user
slice via ``items[indptr[u]:indptr[u+1]]``, single ``np.load`` call.

Loaders use ``np.load(..., allow_pickle=False)`` per D-05 -- no pickle
anywhere in the foundation layer.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def build_exclusion(
    train_df_canonical: pd.DataFrame,
    split_manifest,
) -> Dict[int, np.ndarray]:
    """Compute ``exclude_items[u] = train_pos_u union {test_item_u}``.

    Parameters
    ----------
    train_df_canonical : pandas.DataFrame
        Train rows in canonical space (must have ``user_idx`` and
        ``item_idx`` columns). The test item has already been removed.
    split_manifest : SplitManifest
        Manifest providing ``test_item_per_user`` for the union.

    Returns
    -------
    Dict[int, numpy.ndarray]
        ``user_idx -> sorted int32 array of item_idx to exclude``
        from training-negative sampling. D-13 locks this definition.
    """
    exclusion: Dict[int, np.ndarray] = {}
    grouped = train_df_canonical.groupby("user_idx")["item_idx"].apply(
        lambda s: set(int(x) for x in s)
    )
    # Start with all users observed in train; add test item for each.
    for u_idx, train_items in grouped.items():
        test_item = split_manifest.test_item_per_user.get(int(u_idx))
        all_excluded = set(int(i) for i in train_items)
        if test_item is not None:
            all_excluded.add(int(test_item))
        exclusion[int(u_idx)] = np.array(sorted(all_excluded), dtype=np.int32)
    # Users with no train rows but a held-out test item (edge case): include them.
    for u_idx, t_item in split_manifest.test_item_per_user.items():
        if int(u_idx) not in exclusion:
            exclusion[int(u_idx)] = np.array([int(t_item)], dtype=np.int32)
    return exclusion


def save_exclusion(per_user_items: Dict[int, np.ndarray], path) -> None:
    """Save as NPZ with two arrays: ``items`` (flat int32) + ``indptr`` (int64 offsets).

    IMP-3 flat layout: ``items`` is the concatenation of all per-user
    arrays; ``indptr[u]`` is the starting offset of user ``u``'s slice
    and ``indptr[u+1]`` is its end. Atomic write via tempfile +
    ``os.replace``.

    Parameters
    ----------
    per_user_items : Dict[int, numpy.ndarray]
        Mapping ``user_idx -> int32 array of item_idx``.
    path : pathlib.Path or str
        Destination NPZ path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not per_user_items:
        items = np.empty(0, dtype=np.int32)
        indptr = np.zeros(1, dtype=np.int64)
    else:
        n_users = max(per_user_items.keys()) + 1
        indptr = np.zeros(n_users + 1, dtype=np.int64)
        for u in range(n_users):
            indptr[u + 1] = indptr[u] + len(per_user_items.get(u, np.empty(0, dtype=np.int32)))
        items_list = [
            per_user_items.get(u, np.empty(0, dtype=np.int32)).astype(np.int32)
            for u in range(n_users)
        ]
        items = (
            np.concatenate(items_list).astype(np.int32)
            if items_list
            else np.empty(0, dtype=np.int32)
        )

    # Atomic write: np.savez to a temp path, then os.replace onto the destination.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, items=items, indptr=indptr)
        # np.savez may append .npz when the path lacks the suffix.
        candidate = tmp if Path(tmp).exists() else tmp + ".npz"
        os.replace(candidate, str(p))
    except Exception:
        for candidate in (tmp, tmp + ".npz"):
            try:
                os.unlink(candidate)
            except FileNotFoundError:
                pass
        raise


class ExclusionTable:
    """Wrapper around a loaded flat-layout NPZ with O(1) per-user slicing.

    Use as a context manager to ensure the underlying ``NpzFile`` is
    closed::

        with load_exclusion(path) as tab:
            items = tab.for_user(user_idx)
    """

    def __init__(self, npz: "np.lib.npyio.NpzFile") -> None:
        self._npz = npz
        self._items = npz["items"]
        self._indptr = npz["indptr"]

    def for_user(self, user_idx: int) -> np.ndarray:
        """Return this user's excluded item_idx array (O(1) slice)."""
        u = int(user_idx)
        if u < 0 or u + 1 >= len(self._indptr):
            return np.empty(0, dtype=np.int32)
        start = int(self._indptr[u])
        end = int(self._indptr[u + 1])
        return self._items[start:end]

    def close(self) -> None:
        try:
            self._npz.close()
        except Exception:
            pass

    def __enter__(self) -> "ExclusionTable":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_exclusion(path) -> ExclusionTable:
    """Load ``exclusion_items.npz`` with ``allow_pickle=False`` (D-05).

    Parameters
    ----------
    path : pathlib.Path or str
        Source NPZ path.

    Returns
    -------
    ExclusionTable
        O(1) per-user slice accessor.
    """
    npz = np.load(str(path), allow_pickle=False)
    return ExclusionTable(npz)


def exclusion_for(npz, user_idx: int) -> np.ndarray:
    """Module-level helper for callers holding a raw ``NpzFile`` (CR-3).

    Returns the same slice as ``ExclusionTable.for_user(user_idx)`` but
    without constructing an ``ExclusionTable`` instance. Useful for
    callers who prefer to keep a raw ``np.load(...)`` object around.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        Loaded NPZ from ``np.load(path, allow_pickle=False)``.
    user_idx : int
        Target user index.

    Returns
    -------
    numpy.ndarray
        ``int32`` array of excluded item indices.
    """
    indptr = npz["indptr"]
    u = int(user_idx)
    if u < 0 or u + 1 >= len(indptr):
        return np.empty(0, dtype=np.int32)
    start = int(indptr[u])
    end = int(indptr[u + 1])
    return npz["items"][start:end]
