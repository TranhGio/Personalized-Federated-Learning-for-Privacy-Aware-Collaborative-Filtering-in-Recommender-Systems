"""Tests for fedrec_foundation.exclusion (FND-03 + IMP-3 flat layout + CR-3 module-level helper)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedrec_foundation.mapping import build_mapping
from fedrec_foundation.split import build_split
from fedrec_foundation.exclusion import (
    build_exclusion,
    save_exclusion,
    load_exclusion,
    exclusion_for,
)


@pytest.fixture
def synthetic_movies_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (10, "A", "Action"),
            (20, "B", "Drama"),
            (30, "C", "Comedy"),
            (40, "D", "Action"),
        ],
        columns=["movie_id", "title", "genres"],
    )


def _vectorized_train_split(
    df_canonical: pd.DataFrame, test_item_per_user: dict
) -> pd.DataFrame:
    """Remove each user's LOO test item via a merge (vectorized; no .apply)."""
    test_pairs = pd.DataFrame(
        [(u, i) for u, i in test_item_per_user.items()],
        columns=["user_idx", "item_idx"],
    )
    test_pairs["is_test"] = True
    merged = df_canonical.merge(test_pairs, on=["user_idx", "item_idx"], how="left")
    train = merged[merged["is_test"].isna()].drop(columns=["is_test"]).copy()
    return train


def _setup(df: pd.DataFrame, movies_df: pd.DataFrame, tmp_path: Path):
    m = build_mapping(df)
    s = build_split(df, m, movies_df, mapping_sha256="a" * 64, raw_data_hash="b" * 64)
    df_c = df.copy()
    df_c["user_idx"] = df_c["user_id"].map(m.user2idx)
    df_c["item_idx"] = df_c["movie_id"].map(m.item2idx)
    train_c = _vectorized_train_split(df_c, s.test_item_per_user)
    excl = build_exclusion(train_c, s)
    p = tmp_path / "excl.npz"
    save_exclusion(excl, p)
    return m, s, p


def test_includes_test_item(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """FND-03-a: exclusion_for(u) includes LOO test item + all train positives."""
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    with load_exclusion(p) as tab:
        u1_idx = m.user2idx[1]
        excluded = tab.for_user(u1_idx)
        assert s.test_item_per_user[u1_idx] in excluded.tolist()
        # User 1's interactions: (10, t=1000), (20, t=1001), (30, t=1002).
        # LOO held-out = item 30 (last by timestamp). Train positives = {10, 20}.
        for train_item_raw in (10, 20):
            assert m.item2idx[train_item_raw] in excluded.tolist()


def test_safe_load(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """FND-03-b + D-05: np.load(allow_pickle=False) must succeed and files = items + indptr."""
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    data = np.load(p, allow_pickle=False)
    assert "items" in data.files
    assert "indptr" in data.files
    assert data["items"].dtype == np.int32
    assert data["indptr"].dtype == np.int64


def test_indptr_layout(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """IMP-3 + FND-03-c: every user-slice returns int32 array of size >= 1 (test item)."""
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    with load_exclusion(p) as tab:
        for u_raw in synthetic_ratings_df["user_id"].unique():
            u_idx = m.user2idx[int(u_raw)]
            arr = tab.for_user(u_idx)
            assert arr.dtype == np.int32
            assert len(arr) >= 1  # every user has at least their test item


def test_module_level_exclusion_for(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """CR-3: module-level exclusion_for() returns the same slice as ExclusionTable.for_user()."""
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    npz = np.load(p, allow_pickle=False)
    with load_exclusion(p) as tab:
        for u_raw in synthetic_ratings_df["user_id"].unique():
            u_idx = m.user2idx[int(u_raw)]
            from_class = tab.for_user(u_idx)
            from_helper = exclusion_for(npz, u_idx)
            np.testing.assert_array_equal(from_class, from_helper)
