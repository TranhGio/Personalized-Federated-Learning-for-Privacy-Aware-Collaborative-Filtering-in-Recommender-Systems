"""Deterministic leave-one-out split manifest (FND-02).

Stable-sort by ``(user_idx, timestamp, item_idx)`` and take the last
interaction per user as the test item. Computes ``split_hash`` over
``(mapping_sha256, raw_data_hash, sorted train_keys, sorted test_keys)``
per Codex IMP-2. The manifest is lock-forever: re-running the builder
refuses to overwrite if the new hash diverges from the committed one
(D-04).

Per-user stats (``train_user_stats``) are computed on TRAIN-ONLY
interactions (the held-out test item is removed first) per Codex CR-5.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.user_groups import (
    BUCKET_SEMANTICS,
    USER_GROUP_BOUNDARIES,
    classify_user_group,
)

SPLIT_SCHEMA_VERSION: int = 1
BUILDER_VERSION: str = "1.0.0"


@dataclass
class PerUserStats:
    """Precomputed per-user stats for Phase 4 (adaptive alpha).

    Attributes
    ----------
    n_interactions : int
        Number of TRAIN interactions (test item excluded per CR-5).
    genre_entropy : float
        Shannon entropy base-2 over the user's TRAIN-item genre histogram.
    n_unique_items : int
        Number of unique TRAIN items rated by this user.
    rating_std : float
        Population std (``ddof=0``) of the user's TRAIN ratings;
        0.0 if only one training rating.
    user_group : str
        One of "sparse" | "medium" | "dense" via ``classify_user_group``.
    """

    n_interactions: int
    genre_entropy: float
    n_unique_items: int
    rating_std: float
    user_group: str


@dataclass
class SplitManifest:
    """LOO split manifest with hash + per-user metadata.

    The manifest stores ``raw_data_hash`` and ``mapping_sha256`` as
    top-level fields (IMP-2). Downstream consumers (``publish_bundle``,
    RunManifest in Plan 04) read these fingerprints directly from the
    manifest -- no side-channel access, no post-hoc assignment.

    Attributes
    ----------
    schema_version : int
        Schema version tag.
    builder_version : str
        Builder version tag (embedded in foundation_index.json too).
    created_at : str
        ISO-8601 UTC timestamp.
    raw_data_hash : str
        sha256 of ``ratings.dat || movies.dat || users.dat`` -- LOCKED
        concatenation order from Plan 01.
    mapping_sha256 : str
        sha256 of the mapping.json this split was built against.
    split_hash : str
        sha256 of ``(mapping_sha256, raw_data_hash, sorted train_keys,
        sorted test_keys)`` (IMP-2).
    num_train : int
        Number of rows in the training split.
    num_test_users : int
        Number of users with a held-out test item.
    test_item_per_user : Dict[int, int]
        ``user_idx -> item_idx`` -- one held-out item per user.
    train_user_stats : Dict[int, PerUserStats]
        Per-user stats computed on TRAIN-ONLY rows (CR-5).
    bucket_boundaries : list
        ``[30, 100]`` -- ``classify_user_group`` thresholds (IMP-4).
    bucket_semantics : str
        ``"half_open"`` (IMP-4).
    """

    schema_version: int
    builder_version: str
    created_at: str
    raw_data_hash: str
    mapping_sha256: str
    split_hash: str
    num_train: int
    num_test_users: int
    test_item_per_user: Dict[int, int]
    train_user_stats: Dict[int, PerUserStats]
    bucket_boundaries: List[int] = field(default_factory=lambda: list(USER_GROUP_BOUNDARIES))
    bucket_semantics: str = BUCKET_SEMANTICS


def compute_split_hash(
    train_keys: Iterable[Tuple[int, int]],
    test_keys: Iterable[Tuple[int, int]],
    mapping_sha256: str,
    raw_data_hash: str,
) -> str:
    """SHA-256 of the canonical split key sets + mapping + raw-data hashes.

    Hashing ``(user_idx, item_idx)`` tuples (NOT timestamps) keeps the
    hash stable under timestamp-only metadata edits while still
    invalidating on any held-out-item change. Both fingerprints
    ``mapping_sha256`` and ``raw_data_hash`` are folded in per IMP-2,
    so the split_hash changes if either underlying artifact changes.

    Parameters
    ----------
    train_keys : iterable of (user_idx, item_idx)
        Training keys; will be sorted inside.
    test_keys : iterable of (user_idx, item_idx)
        Test keys; will be sorted inside.
    mapping_sha256 : str
        sha256 of the mapping this split was built against.
    raw_data_hash : str
        sha256 of the raw ML-1M concatenation.

    Returns
    -------
    str
        Lowercase 64-character hex digest.
    """
    h = hashlib.sha256()
    h.update(b"mapping:" + mapping_sha256.encode("ascii") + b";")
    h.update(b"raw:" + raw_data_hash.encode("ascii") + b";")
    h.update(b"train:")
    for uidx, iidx in sorted(train_keys):
        h.update(f"{int(uidx)},{int(iidx)};".encode("ascii"))
    h.update(b"test:")
    for uidx, iidx in sorted(test_keys):
        h.update(f"{int(uidx)},{int(iidx)};".encode("ascii"))
    return h.hexdigest()


def _compute_genre_entropy(user_train_df: pd.DataFrame, movies_df: pd.DataFrame) -> float:
    """Shannon entropy base-2 of a user's TRAIN-item genre histogram.

    Genres in ``movies.dat`` are ``|``-separated strings. We explode
    per-genre, count, normalize, then compute ``-sum(p * log2(p))``.
    Returns ``0.0`` if the user has zero train rows or all items share
    one genre.
    """
    if len(user_train_df) == 0:
        return 0.0
    # Merge on item_idx -> genres string; explode to per-genre rows.
    if "item_idx" not in movies_df.columns:
        # movies_df is keyed by movie_id; build a lookup from user_train's movie_id.
        # We expect user_train_df to have either 'movie_id' OR have been merged upstream.
        pass
    # We join via movie_id (raw) -- upstream preserves that column.
    merged = user_train_df.merge(
        movies_df[["movie_id", "genres"]], on="movie_id", how="left"
    )
    if merged["genres"].isna().all():
        return 0.0
    genre_lists = merged["genres"].fillna("").str.split("|")
    # Flatten and count non-empty genres.
    from collections import Counter

    counts: Counter = Counter()
    for gl in genre_lists:
        for g in gl:
            if g:
                counts[g] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    if len(probs) <= 1:
        return 0.0
    # Shannon entropy, base-2.
    return float(-np.sum(probs * np.log2(probs)))


def build_split(
    ratings_df: pd.DataFrame,
    mapping,
    movies_df: pd.DataFrame,
    mapping_sha256: str,
    raw_data_hash: str,
) -> SplitManifest:
    """Build deterministic LOO split with stable tiebreak + train-only stats.

    Stable-sort by ``(user_idx, timestamp, item_idx)`` using mergesort;
    ``tail(1)`` per ``user_idx`` is the held-out test item. Users with
    only one interaction are skipped (kept entirely in train).

    Per-user stats are computed AFTER removing the test item per CR-5.
    The returned ``SplitManifest`` carries both ``mapping_sha256`` and
    ``raw_data_hash`` as top-level fields per IMP-2 -- no caller-side
    post-hoc mutation.

    Parameters
    ----------
    ratings_df : pandas.DataFrame
        Raw ratings with ``user_id``, ``movie_id``, ``rating``, ``timestamp``.
    mapping : CanonicalMapping
        Canonical mapping to translate raw ids -> ``user_idx`` / ``item_idx``.
    movies_df : pandas.DataFrame
        Movies catalog with ``movie_id`` and ``genres`` columns
        (``|``-separated genre strings).
    mapping_sha256 : str
        sha256 of the mapping this split is built against.
    raw_data_hash : str
        sha256 of the raw ML-1M concatenation.

    Returns
    -------
    SplitManifest
        Deterministic manifest with ``split_hash``, ``train_user_stats``,
        and both fingerprints embedded as fields.
    """
    df = ratings_df.copy()
    df["user_idx"] = df["user_id"].map(mapping.user2idx)
    df["item_idx"] = df["movie_id"].map(mapping.item2idx)

    # Stable-sort by (user_idx, timestamp, item_idx). mergesort is stable.
    sorted_df = df.sort_values(
        by=["user_idx", "timestamp", "item_idx"],
        kind="mergesort",
    )

    # Users with >1 interaction are eligible for LOO.
    counts = sorted_df.groupby("user_idx").size()
    eligible = set(counts[counts > 1].index)

    test_idx = (
        sorted_df[sorted_df["user_idx"].isin(eligible)]
        .groupby("user_idx")
        .tail(1)
        .index
    )
    test_df = sorted_df.loc[test_idx]
    train_df = sorted_df.drop(test_idx)

    test_item_per_user = {
        int(u): int(i)
        for u, i in zip(test_df["user_idx"], test_df["item_idx"])
    }
    train_keys = list(
        zip(
            train_df["user_idx"].astype(int).tolist(),
            train_df["item_idx"].astype(int).tolist(),
        )
    )
    test_keys = list(
        zip(
            test_df["user_idx"].astype(int).tolist(),
            test_df["item_idx"].astype(int).tolist(),
        )
    )
    split_hash = compute_split_hash(train_keys, test_keys, mapping_sha256, raw_data_hash)

    # CR-5: per-user stats from TRAIN rows only.
    train_user_stats: Dict[int, PerUserStats] = {}
    for u_idx, user_train_df in train_df.groupby("user_idx"):
        n_interactions = int(len(user_train_df))
        n_unique_items = int(user_train_df["item_idx"].nunique())
        rating_std_val = float(user_train_df["rating"].std(ddof=0))
        if np.isnan(rating_std_val):  # single-rating user
            rating_std_val = 0.0
        genre_entropy = _compute_genre_entropy(user_train_df, movies_df)
        train_user_stats[int(u_idx)] = PerUserStats(
            n_interactions=n_interactions,
            genre_entropy=genre_entropy,
            n_unique_items=n_unique_items,
            rating_std=rating_std_val,
            user_group=classify_user_group(n_interactions),
        )

    return SplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        builder_version=BUILDER_VERSION,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        raw_data_hash=raw_data_hash,
        mapping_sha256=mapping_sha256,
        split_hash=split_hash,
        num_train=int(len(train_df)),
        num_test_users=int(len(test_df)),
        test_item_per_user=test_item_per_user,
        train_user_stats=train_user_stats,
        bucket_boundaries=list(USER_GROUP_BOUNDARIES),
        bucket_semantics=BUCKET_SEMANTICS,
    )


def _manifest_to_dict(manifest: SplitManifest) -> dict:
    """Custom dict serialization with int-keyed dicts preserved at top level."""
    out = asdict(manifest)
    # asdict() already recurses; int keys serialize to JSON strings as usual.
    return out


def save_split_or_verify(manifest: SplitManifest, path) -> None:
    """Write if absent; if present, verify hash matches and refuse overwrite (D-04).

    On hash mismatch, raises ``ValueError`` with the exact sentinel
    substring ``"invalidate all cached results"`` so tests and callers
    can pattern-match.

    Parameters
    ----------
    manifest : SplitManifest
        Manifest to persist.
    path : pathlib.Path or str
        Destination path for ``split_manifest.json``.

    Raises
    ------
    ValueError
        If the on-disk split_hash differs from ``manifest.split_hash``.
    """
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            existing = json.load(f)
        if existing.get("split_hash") != manifest.split_hash:
            raise ValueError(
                f"split_hash mismatch: on-disk={existing.get('split_hash')} "
                f"new={manifest.split_hash}. A new split would invalidate "
                f"all cached results. Refusing to overwrite."
            )
        # Hash matches; no-op.
        return
    atomic_write_json(str(p), _manifest_to_dict(manifest))


def load_split_manifest(path) -> SplitManifest:
    """Load a ``split_manifest.json`` from disk and restore typed fields.

    JSON converts ``int`` dict keys to strings; we restore them and
    rebuild ``PerUserStats`` instances.

    Parameters
    ----------
    path : pathlib.Path or str
        Source JSON path.

    Returns
    -------
    SplitManifest
        Fully typed manifest.
    """
    with open(path, "r") as f:
        data = json.load(f)
    test_item_per_user = {int(k): int(v) for k, v in data["test_item_per_user"].items()}
    train_user_stats = {
        int(k): PerUserStats(**v) for k, v in data["train_user_stats"].items()
    }
    return SplitManifest(
        schema_version=data["schema_version"],
        builder_version=data["builder_version"],
        created_at=data["created_at"],
        raw_data_hash=data["raw_data_hash"],
        mapping_sha256=data["mapping_sha256"],
        split_hash=data["split_hash"],
        num_train=data["num_train"],
        num_test_users=data["num_test_users"],
        test_item_per_user=test_item_per_user,
        train_user_stats=train_user_stats,
        bucket_boundaries=data.get("bucket_boundaries", list(USER_GROUP_BOUNDARIES)),
        bucket_semantics=data.get("bucket_semantics", BUCKET_SEMANTICS),
    )
