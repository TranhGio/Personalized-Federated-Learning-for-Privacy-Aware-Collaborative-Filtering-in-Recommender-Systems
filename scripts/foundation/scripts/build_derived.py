"""CLI: python -m fedrec_foundation.build

Builds the 4-file ``data/derived/`` bundle from the real ML-1M in
``data/ml-1m/``. Idempotent: re-running with an existing locked split
is a no-op by D-04 lock semantics and exits 0 with the same hashes.

Uses a VECTORIZED merge-based train filter (no row-wise ``.apply``
lambda) so the 1M-row ML-1M DataFrame runs in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from fedrec_foundation.bundle import publish_bundle
from fedrec_foundation.exclusion import build_exclusion
from fedrec_foundation.hashing import compute_raw_data_hash, sha256_file
from fedrec_foundation.mapping import build_mapping, load_mapping, save_mapping
from fedrec_foundation.paths import data_derived, ml1m_dir
from fedrec_foundation.split import build_split


def _load_ml1m(ml1m: Path):
    """Load ratings + movies from the ML-1M .dat files."""
    ratings = pd.read_csv(
        ml1m / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        ml1m / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def _vectorized_train_split(
    df_canonical: pd.DataFrame, test_item_per_user: dict
) -> pd.DataFrame:
    """Filter TRAIN rows via a left-join merge (O(N log N), not per-row .apply).

    Builds a small DataFrame of ``(user_idx, item_idx)`` test pairs and
    left-merges; rows where the merge-marker is NaN are train rows.
    """
    test_pairs = pd.DataFrame(
        [(u, i) for u, i in test_item_per_user.items()],
        columns=["user_idx", "item_idx"],
    )
    test_pairs["is_test"] = True
    merged = df_canonical.merge(test_pairs, on=["user_idx", "item_idx"], how="left")
    return merged[merged["is_test"].isna()].drop(columns=["is_test"]).copy()


def main() -> int:
    """Entrypoint: build the ``data/derived/`` bundle from ``data/ml-1m/``."""
    derived = data_derived()
    ml1m = ml1m_dir()

    ratings_df, movies_df = _load_ml1m(ml1m)
    raw_data_hash = compute_raw_data_hash(ml1m)

    # 1. Mapping: build or load (idempotent; save_mapping is atomic).
    mapping_path = derived / "mapping.json"
    if mapping_path.exists():
        mapping = load_mapping(str(mapping_path))
    else:
        mapping = build_mapping(ratings_df)
        derived.mkdir(parents=True, exist_ok=True)
        save_mapping(mapping, str(mapping_path))
    mapping_sha = sha256_file(mapping_path)

    # 2. Split: build_split stores raw_data_hash + mapping_sha256 as fields
    # on the returned SplitManifest (IMP-2) -- no post-hoc mutation needed.
    split = build_split(
        ratings_df,
        mapping,
        movies_df,
        mapping_sha256=mapping_sha,
        raw_data_hash=raw_data_hash,
    )

    # 3. Exclusion: built from canonical train-only rows; vectorized filter.
    df_c = ratings_df.copy()
    df_c["user_idx"] = df_c["user_id"].map(mapping.user2idx)
    df_c["item_idx"] = df_c["movie_id"].map(mapping.item2idx)
    train_c = _vectorized_train_split(df_c, split.test_item_per_user)
    exclusion = build_exclusion(train_c, split)

    # 4. Atomic bundle publication (4-arg signature; raw_data_hash read from split).
    idx = publish_bundle(derived, mapping, split, exclusion)
    print(f"[build] mapping: {mapping.num_users} users, {mapping.num_items} items")
    print(f"[build] split_hash={idx.split_hash[:12]}...")
    print(f"[build] foundation_contract_sha256={idx.foundation_contract_sha256[:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
