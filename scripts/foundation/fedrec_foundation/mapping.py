"""Canonical ID mapping for ML-1M (FND-01).

Builds raw_user_id -> user_idx and raw_item_id -> item_idx from the
raw ratings DataFrame. CRITICAL: item2idx is built from
``ratings_df["movie_id"].unique()`` -- NOT from movies.dat. ML-1M has
3,883 movies but only 3,706 unique rated items; using movies.dat
silently expands the embedding table and invalidates every cached
embedding (Codex CR-1).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict

import pandas as pd

from fedrec_foundation.atomic import atomic_write_json

MAPPING_SCHEMA_VERSION: int = 1


@dataclass
class CanonicalMapping:
    """Immutable raw_id <-> idx mapping for ML-1M.

    Attributes
    ----------
    user2idx : Dict[int, int]
        Raw MovieLens user_id -> canonical user_idx (0..num_users-1).
    item2idx : Dict[int, int]
        Raw MovieLens movie_id -> canonical item_idx (0..num_items-1).
    num_users : int
        Number of unique users (expected 6040 on real ML-1M).
    num_items : int
        Number of unique rated items (expected 3706 on real ML-1M --
        NOT 3883, which would be the count from movies.dat; see CR-1).
    schema_version : int
        Schema version tag for forward-compatibility.
    """

    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    num_users: int
    num_items: int
    schema_version: int = MAPPING_SCHEMA_VERSION


def build_mapping(ratings_df: pd.DataFrame) -> CanonicalMapping:
    """Build mapping deterministically from raw ratings DataFrame.

    Sort-ascending over unique ids, then enumerate. Reproducible across
    pandas versions because ``sorted()`` on ints is total-ordered.

    CR-1: ``item2idx`` is built from ``ratings_df["movie_id"].unique()``
    -- the set of movies that were actually rated. Using ``movies.dat``
    (the full catalog) would silently add 177 never-rated items to the
    embedding table, changing ``num_items`` from 3706 to 3883 and
    invalidating every cached item embedding.

    Parameters
    ----------
    ratings_df : pandas.DataFrame
        Raw ratings with columns ``user_id``, ``movie_id`` (at minimum).

    Returns
    -------
    CanonicalMapping
        Deterministic mapping with user2idx + item2idx populated.
    """
    unique_users = sorted(int(u) for u in ratings_df["user_id"].unique())
    unique_items = sorted(int(i) for i in ratings_df["movie_id"].unique())
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
    return CanonicalMapping(
        user2idx=user2idx,
        item2idx=item2idx,
        num_users=len(user2idx),
        num_items=len(item2idx),
    )


def save_mapping(mapping: CanonicalMapping, path: str) -> None:
    """Atomic write to ``data/derived/mapping.json``.

    JSON serializes ``int`` keys as strings; the loader restores them.

    Parameters
    ----------
    mapping : CanonicalMapping
        Mapping to persist.
    path : str
        Destination path for the JSON artifact.
    """
    atomic_write_json(path, asdict(mapping))


def load_mapping(path: str) -> CanonicalMapping:
    """Load mapping; verifies schema version.

    Restores ``int`` keys for ``user2idx`` and ``item2idx`` (JSON round-trip
    converts ``int`` keys to ``str``).

    Parameters
    ----------
    path : str
        Source JSON path.

    Returns
    -------
    CanonicalMapping
        Loaded mapping with ``int``-keyed dicts.

    Raises
    ------
    ValueError
        If the on-disk schema_version does not match
        ``MAPPING_SCHEMA_VERSION``.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if data["schema_version"] != MAPPING_SCHEMA_VERSION:
        raise ValueError(
            f"mapping.json schema version {data['schema_version']} != "
            f"expected {MAPPING_SCHEMA_VERSION}"
        )
    # JSON serializes int keys as strings; restore.
    return CanonicalMapping(
        user2idx={int(k): v for k, v in data["user2idx"].items()},
        item2idx={int(k): v for k, v in data["item2idx"].items()},
        num_users=data["num_users"],
        num_items=data["num_items"],
        schema_version=data["schema_version"],
    )
