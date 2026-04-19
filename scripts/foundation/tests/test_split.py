"""Tests for fedrec_foundation.split (FND-02 + CR-5 + D-04 + IMP-2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fedrec_foundation.mapping import build_mapping
from fedrec_foundation.split import (
    SplitManifest,
    build_split,
    compute_split_hash,
    save_split_or_verify,
    load_split_manifest,
)


@pytest.fixture
def synthetic_movies_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (10, "A", "Action"),
            (20, "B", "Comedy|Drama"),
            (30, "C", "Action|Drama"),
            (40, "D", "Comedy"),
        ],
        columns=["movie_id", "title", "genres"],
    )


def test_hash_deterministic(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
) -> None:
    """FND-02-a + IMP-2: split_hash stable; SplitManifest stores both fingerprints as fields."""
    m = build_mapping(synthetic_ratings_df)
    s1 = build_split(
        synthetic_ratings_df, m, synthetic_movies_df,
        mapping_sha256="a" * 64, raw_data_hash="b" * 64,
    )
    s2 = build_split(
        synthetic_ratings_df, m, synthetic_movies_df,
        mapping_sha256="a" * 64, raw_data_hash="b" * 64,
    )
    assert s1.split_hash == s2.split_hash
    assert len(s1.split_hash) == 64
    # SplitManifest stores both fingerprints as fields (IMP-2).
    assert s1.raw_data_hash == "b" * 64
    assert s1.mapping_sha256 == "a" * 64


def test_timestamp_tiebreak(synthetic_movies_df: pd.DataFrame) -> None:
    """FND-02-b: ties broken by (user_idx, timestamp, item_idx) stable sort."""
    # user 1 has: (item 10, t=1000), (item 20, t=1000), (item 30, t=2000).
    # Stable sort by (user_idx, timestamp, item_idx) mergesort; tail(1) is the last row.
    rows = [(1, 10, 5, 1000), (1, 20, 4, 1000), (1, 30, 3, 2000)]
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    m = build_mapping(df)
    s = build_split(
        df, m, synthetic_movies_df,
        mapping_sha256="a" * 64, raw_data_hash="b" * 64,
    )
    # user_idx 0 (user 1); held-out item should be item 30 (last row by timestamp=2000).
    held = s.test_item_per_user[0]
    assert held == m.item2idx[30]


def test_split_lock_refuses_overwrite(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """FND-02-c + D-04: lock-forever — refuses to overwrite on divergent hash."""
    m = build_mapping(synthetic_ratings_df)
    s = build_split(
        synthetic_ratings_df, m, synthetic_movies_df,
        mapping_sha256="a" * 64, raw_data_hash="b" * 64,
    )
    p = tmp_path / "split.json"
    save_split_or_verify(s, p)
    # Second call with same hash: no-op.
    save_split_or_verify(s, p)
    # Different mapping_sha => different split_hash (because mapping_sha is part of the hash): raise.
    s2 = build_split(
        synthetic_ratings_df, m, synthetic_movies_df,
        mapping_sha256="c" * 64, raw_data_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="invalidate all cached results"):
        save_split_or_verify(s2, p)


def test_train_only_user_stats(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
) -> None:
    """CR-5: train_user_stats computed AFTER the LOO test item is removed."""
    m = build_mapping(synthetic_ratings_df)
    s = build_split(
        synthetic_ratings_df, m, synthetic_movies_df,
        mapping_sha256="a" * 64, raw_data_hash="b" * 64,
    )
    # User 1 has 3 interactions total; after LOO, train has 2.
    u1_idx = m.user2idx[1]
    assert s.train_user_stats[u1_idx].n_interactions == 2
    # The test item is NOT among the unique train items.
    assert s.train_user_stats[u1_idx].n_unique_items == 2
