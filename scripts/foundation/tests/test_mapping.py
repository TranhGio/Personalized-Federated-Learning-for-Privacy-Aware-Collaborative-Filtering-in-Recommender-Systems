"""Tests for fedrec_foundation.mapping (FND-01 + Codex CR-1)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedrec_foundation.mapping import (
    CanonicalMapping,
    MAPPING_SCHEMA_VERSION,
    build_mapping,
    save_mapping,
    load_mapping,
)


def test_sort_order(synthetic_ratings_df: pd.DataFrame) -> None:
    """FND-01-b: user_idx / item_idx assigned in sorted raw-ID order (deterministic)."""
    m = build_mapping(synthetic_ratings_df)
    user_ids_sorted = sorted(int(u) for u in synthetic_ratings_df["user_id"].unique())
    item_ids_sorted = sorted(int(i) for i in synthetic_ratings_df["movie_id"].unique())
    assert list(m.user2idx.keys()) == list(user_ids_sorted)
    assert list(m.item2idx.keys()) == list(item_ids_sorted)
    # Enumerate from 0.
    assert m.user2idx[user_ids_sorted[0]] == 0
    assert m.item2idx[item_ids_sorted[0]] == 0


def test_item_mapping_from_ratings_only() -> None:
    """CR-1 anchor: a movie that never appears in ratings is NOT in item2idx."""
    ratings = pd.DataFrame(
        [(1, 10, 5, 1000), (1, 20, 4, 2000)],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    # Movie 99 exists in the catalog but never rated — must be absent.
    m = build_mapping(ratings)
    assert 99 not in m.item2idx
    assert m.num_items == 2
    assert list(m.item2idx.keys()) == [10, 20]


def test_roundtrip(synthetic_ratings_df: pd.DataFrame, tmp_path: Path) -> None:
    """FND-01-a: build -> save -> load round-trip preserves raw<->idx bijection."""
    m = build_mapping(synthetic_ratings_df)
    p = tmp_path / "mapping.json"
    save_mapping(m, str(p))
    m2 = load_mapping(str(p))
    assert m.user2idx == m2.user2idx  # int keys preserved
    assert m.item2idx == m2.item2idx
    assert m.num_users == m2.num_users
    assert m.num_items == m2.num_items
    assert m.schema_version == m2.schema_version == MAPPING_SCHEMA_VERSION
    # On disk the keys are strings; loader restored ints.
    raw = json.loads(p.read_text())
    assert all(isinstance(k, str) for k in raw["user2idx"].keys())
