---
phase: 01-foundation-contract
plan: 02
type: execute
wave: 2
depends_on: [01-foundation-contract-01]
files_modified:
  - scripts/foundation/fedrec_foundation/mapping.py
  - scripts/foundation/fedrec_foundation/split.py
  - scripts/foundation/fedrec_foundation/exclusion.py
  - scripts/foundation/fedrec_foundation/user_groups.py
  - scripts/foundation/fedrec_foundation/bundle.py
  - scripts/foundation/scripts/build_derived.py
  - scripts/foundation/tests/test_mapping.py
  - scripts/foundation/tests/test_split.py
  - scripts/foundation/tests/test_exclusion.py
  - scripts/foundation/tests/test_integration.py
  - data/derived/mapping.json
  - data/derived/split_manifest.json
  - data/derived/exclusion_items.npz
  - data/derived/foundation_index.json
autonomous: true
requirements: [FND-01, FND-02, FND-03]
must_haves:
  truths:
    - "mapping.json, split_manifest.json, exclusion_items.npz, and foundation_index.json exist under data/derived/ after running `python -m fedrec_foundation.build`."
    - "Running the builder twice produces the same split_hash and the same test_item_per_user; the second run is a no-op by D-04 lock semantics."
    - "For every user, exclusion_for(user) contains that user's held-out test item AND every training positive for that user."
    - "Built on this repo's ML-1M, the mapping reports exactly 6040 users and 3706 items (NOT 3883 from movies.dat — Codex CR-1 anchor)."
    - "train_user_stats is computed from TRAIN-ONLY interactions (test item removed first) per CR-5."
    - "Bundle publication is atomic: if any of mapping.json / split_manifest.json / exclusion_items.npz write fails, foundation_index.json is not published, and loaders reject the bundle."
    - "SplitManifest stores raw_data_hash as a dataclass field; publish_bundle reads it directly from split_manifest.raw_data_hash (no post-hoc assignment, no extra parameter)."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/mapping.py"
      provides: "build_mapping, save_mapping, load_mapping, CanonicalMapping dataclass"
      exports: ["build_mapping", "save_mapping", "load_mapping", "CanonicalMapping", "MAPPING_SCHEMA_VERSION"]
    - path: "scripts/foundation/fedrec_foundation/split.py"
      provides: "build_split, save_split_or_verify, load_split_manifest, SplitManifest (with raw_data_hash field), PerUserStats, compute_split_hash"
      exports: ["SplitManifest", "PerUserStats", "build_split", "save_split_or_verify", "load_split_manifest", "compute_split_hash", "SPLIT_SCHEMA_VERSION", "BUILDER_VERSION"]
    - path: "scripts/foundation/fedrec_foundation/exclusion.py"
      provides: "build_exclusion, save_exclusion (flat items+indptr), load_exclusion, ExclusionTable, exclusion_for module-level helper"
      exports: ["build_exclusion", "save_exclusion", "load_exclusion", "ExclusionTable", "exclusion_for"]
    - path: "scripts/foundation/fedrec_foundation/user_groups.py"
      provides: "classify_user_group with frozen half-open semantics [0,30), [30,100), [100, inf)"
      exports: ["classify_user_group", "USER_GROUP_BOUNDARIES", "BUCKET_SEMANTICS"]
    - path: "scripts/foundation/fedrec_foundation/bundle.py"
      provides: "publish_bundle(derived_dir, mapping, split_manifest, exclusion): 4-param signature; reads raw_data_hash from split_manifest.raw_data_hash. verify_bundle() on load."
      exports: ["publish_bundle", "verify_bundle", "FoundationIndex", "compute_foundation_contract_sha256"]
    - path: "scripts/foundation/scripts/build_derived.py"
      provides: "CLI entry point: python -m fedrec_foundation.build"
    - path: "data/derived/mapping.json"
      provides: "Canonical raw_user_id -> user_idx + raw_item_id -> item_idx"
      contains: "\"num_users\": 6040"
    - path: "data/derived/split_manifest.json"
      provides: "LOO split + split_hash + raw_data_hash + train_user_stats + user_group classification"
      contains: "\"bucket_semantics\": \"half_open\""
    - path: "data/derived/exclusion_items.npz"
      provides: "Flat items array + indptr (IMP-3); loaded with allow_pickle=False"
    - path: "data/derived/foundation_index.json"
      provides: "mapping_sha256 + split_hash + exclusion_sha256 + foundation_contract_sha256 + builder_version + created_at"
  key_links:
    - from: "scripts/foundation/scripts/build_derived.py"
      to: "publish_bundle(derived, mapping, split, exclusion)"
      via: "atomic multi-file publication; 4-arg signature"
      pattern: "publish_bundle\\(derived"
    - from: "scripts/foundation/fedrec_foundation/mapping.py::build_mapping"
      to: "sorted(ratings_df[\"movie_id\"].unique())"
      via: "CR-1 fix: ratings-only item set, not movies.dat"
      pattern: "ratings_df\\[.movie_id.\\]\\.unique"
    - from: "scripts/foundation/fedrec_foundation/split.py::build_split"
      to: "build_split(ratings_df, mapping, movies_df, mapping_sha256, raw_data_hash) -> SplitManifest"
      via: "IMP-2: fingerprints are explicit parameters; SplitManifest stores raw_data_hash as a field"
      pattern: "mapping_sha256.*raw_data_hash"
    - from: "scripts/foundation/fedrec_foundation/split.py::SplitManifest"
      to: "raw_data_hash: str (dataclass field)"
      via: "Consumed by Plan 04 RunManifest and by publish_bundle"
      pattern: "raw_data_hash: str"
    - from: "scripts/foundation/fedrec_foundation/split.py::build_split"
      to: "train_user_stats computed AFTER test-item removal"
      via: "CR-5 train-only user stats"
      pattern: "train_user_stats"
---

<objective>
Implement FND-01 (canonical ID mapping), FND-02 (deterministic LOO split manifest with lock-forever policy), and FND-03 (per-user exclusion set with flat items+indptr layout). Produce the three on-disk artifacts under `data/derived/` and publish them atomically via a `foundation_index.json` sentinel.

Purpose: These three artifacts are the hard-dependency every downstream module imports. FND-01 fixes the ID space (6040 users, 3706 items); FND-02 locks the LOO split forever; FND-03 provides the exclusion set that prevents training-negative test-leakage. The atomic publication sentinel (N-3) ensures loaders never read a partially-written bundle.

Output: Three data artifacts + one index sentinel in `data/derived/`, plus four implemented Python modules (`mapping`, `split`, `exclusion`, `user_groups`, `bundle`) and a CLI builder (`scripts/build_derived.py`). All Plan 01 skipped tests for these modules flip GREEN.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-foundation-contract/01-CONTEXT.md
@.planning/phases/01-foundation-contract/01-RESEARCH.md
@.planning/phases/01-foundation-contract/01-VALIDATION.md
@.planning/phases/01-foundation-contract/01-foundation-contract-01-SUMMARY.md
@CLAUDE.md
@.planning/codebase/CONVENTIONS.md

<interfaces>
<!-- These are the EXACT public APIs Plans 03-06 will import. Signatures are LOCKED; changing them requires a replan. -->

From scripts/foundation/fedrec_foundation/mapping.py:
```python
from dataclasses import dataclass
from typing import Dict

MAPPING_SCHEMA_VERSION: int = 1

@dataclass
class CanonicalMapping:
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    num_users: int
    num_items: int
    schema_version: int

def build_mapping(ratings_df) -> CanonicalMapping: ...    # CR-1: ratings-only item set
def save_mapping(mapping: CanonicalMapping, path: str) -> None: ...
def load_mapping(path: str) -> CanonicalMapping: ...      # restores int keys
```

From scripts/foundation/fedrec_foundation/split.py:
```python
from dataclasses import dataclass
from typing import Dict

SPLIT_SCHEMA_VERSION: int = 1
BUILDER_VERSION: str = "1.0.0"

@dataclass
class PerUserStats:
    n_interactions: int
    genre_entropy: float
    n_unique_items: int
    rating_std: float
    user_group: str   # "sparse" | "medium" | "dense"

@dataclass
class SplitManifest:
    schema_version: int
    builder_version: str
    created_at: str
    raw_data_hash: str                    # <-- explicit field; consumed by RunManifest (Plan 04) and publish_bundle
    mapping_sha256: str                   # fingerprint of the mapping this split was built against
    split_hash: str
    num_train: int
    num_test_users: int
    test_item_per_user: Dict[int, int]    # user_idx -> item_idx
    train_user_stats: Dict[int, PerUserStats]
    bucket_boundaries: list               # [30, 100]
    bucket_semantics: str                 # "half_open"

def compute_split_hash(train_keys, test_keys, mapping_sha256: str, raw_data_hash: str) -> str: ...

def build_split(
    ratings_df,
    mapping,
    movies_df,
    mapping_sha256: str,
    raw_data_hash: str,
) -> SplitManifest: ...
# ^^ 5-param signature. Fingerprints are required inputs; the returned
# SplitManifest stores them as fields so downstream consumers (publish_bundle,
# RunManifest) don't need side-channel access.

def save_split_or_verify(manifest: SplitManifest, path) -> None: ...   # D-04 lock
def load_split_manifest(path) -> SplitManifest: ...
```

From scripts/foundation/fedrec_foundation/exclusion.py (IMP-3 flat layout):
```python
import numpy as np
from typing import Dict

def build_exclusion(train_df_canonical, split_manifest) -> Dict[int, np.ndarray]: ...
def save_exclusion(per_user_items: Dict[int, np.ndarray], path) -> None: ...  # items + indptr
def load_exclusion(path) -> ExclusionTable: ...

# Module-level helper for callers who don't want the class instance.
def exclusion_for(npz, user_idx: int) -> np.ndarray: ...

class ExclusionTable:
    def for_user(self, user_idx: int) -> np.ndarray: ...    # O(1) via indptr slice
    def close(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *a): ...
```

From scripts/foundation/fedrec_foundation/user_groups.py:
```python
USER_GROUP_BOUNDARIES = (30, 100)     # frozen, half-open
BUCKET_SEMANTICS = "half_open"

def classify_user_group(n_interactions: int) -> str:
    # n_interactions < 30  -> "sparse"
    # 30 <= n_interactions < 100 -> "medium"
    # 100 <= n_interactions -> "dense"
```

From scripts/foundation/fedrec_foundation/bundle.py (N-3 atomic bundle):
```python
from dataclasses import dataclass

@dataclass
class FoundationIndex:
    schema_version: int
    builder_version: str
    created_at: str
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str     # IMP-2

def compute_foundation_contract_sha256(mapping_sha256: str, split_hash: str, exclusion_sha256: str) -> str: ...

def publish_bundle(
    derived_dir,
    mapping,
    split_manifest,
    exclusion,
) -> FoundationIndex: ...
# ^^ 4-param signature. raw_data_hash is READ from split_manifest.raw_data_hash
# (which build_split populated). No extra kwarg, no post-hoc assignment.

def verify_bundle(derived_dir) -> FoundationIndex: ...   # raises on index mismatch/missing
```
</interfaces>

<!-- Brownfield references the executor MUST read to lift logic correctly. -->
@federated-baseline-cf/federated_baseline_cf/dataset.py
@federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/user_groups.py
@federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement FND-01 canonical ID mapping + user_groups + flip test_mapping green</name>
  <files>
    scripts/foundation/fedrec_foundation/mapping.py
    scripts/foundation/fedrec_foundation/user_groups.py
    scripts/foundation/tests/test_mapping.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md §"Implementation Decisions" (D-01..D-05)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"CODEX PEER REVIEW" CR-1 (build item2idx from ratings.dat, NOT movies.dat — LOCKED DECISION)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 1: Canonical ID Mapping (FND-01)" (implementation sketch lines 335-411)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pitfall 2: JSON int keys silently become strings on load"
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §IMP-4 (user-group bucket_semantics half-open; LOCK to `[30, 100]`)
    - federated-baseline-cf/federated_baseline_cf/dataset.py (read the existing `create_global_mappings` for the established pattern — lift and adapt, don't reinvent)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/user_groups.py (existing classifier reference — but the FOUNDATION version uses frozen half-open semantics per IMP-4, not the existing `≤` semantics)
    - CLAUDE.md (Python 3.9+ typing: `typing.Dict` not `dict[int, int]`)
  </read_first>
  <behavior>
    - `build_mapping(ratings_df)` builds `user2idx` from `sorted(int(u) for u in ratings_df["user_id"].unique())` and `item2idx` from `sorted(int(i) for i in ratings_df["movie_id"].unique())`. For this repo's ML-1M: 6040 users and 3706 items. NEVER builds item2idx from movies.dat (Codex CR-1 — silent 177-item expansion).
    - `save_mapping(mapping, path)` writes via `atomic_write_json`; int keys serialize as JSON strings.
    - `load_mapping(path)` casts string keys back to int: `{int(k): v for k, v in data["user2idx"].items()}`; raises `ValueError` if schema_version mismatch.
    - `classify_user_group(n_interactions)` uses half-open buckets: `< 30 -> "sparse"`, `< 100 -> "medium"`, else `"dense"`. `USER_GROUP_BOUNDARIES = (30, 100)` and `BUCKET_SEMANTICS = "half_open"` frozen at module level.
    - `test_mapping.py` tests flip from SKIP to GREEN: `test_sort_order`, `test_item_mapping_from_ratings_only`, `test_roundtrip`.
  </behavior>
  <action>
Copy the `CanonicalMapping` dataclass + `build_mapping`/`save_mapping`/`load_mapping` functions verbatim from `01-RESEARCH.md` Pattern 1 (research lines 347-410) into `scripts/foundation/fedrec_foundation/mapping.py`. Use `typing.Dict` (NOT `dict[int, int]`). Add NumPy-style docstrings.

Add to top of `mapping.py`:
```python
"""Canonical ID mapping for ML-1M (FND-01).

Builds raw_user_id -> user_idx and raw_item_id -> item_idx from the
raw ratings DataFrame. CRITICAL: item2idx is built from
``ratings_df["movie_id"].unique()`` — NOT from movies.dat. ML-1M has
3,883 movies but only 3,706 unique rated items; using movies.dat
silently expands the embedding table and invalidates every cached
embedding (Codex CR-1).
"""
```

Create `scripts/foundation/fedrec_foundation/user_groups.py`:
```python
"""User-group bucket classifier with FROZEN half-open semantics.

Half-open: sparse = [0, 30), medium = [30, 100), dense = [100, inf).
Boundary value 30 lands in "medium", not "sparse" — the half-open
decision is recorded in split_manifest.json as bucket_semantics so
future readers never have to infer.
"""
from __future__ import annotations

USER_GROUP_BOUNDARIES = (30, 100)
BUCKET_SEMANTICS = "half_open"


def classify_user_group(n_interactions: int) -> str:
    """Return "sparse" | "medium" | "dense" using half-open semantics.

    Parameters
    ----------
    n_interactions : int
        Number of training interactions (TRAIN-ONLY per Codex CR-5).

    Returns
    -------
    str
        "sparse" if n < 30, "medium" if 30 <= n < 100, "dense" if n >= 100.
    """
    sparse_hi, medium_hi = USER_GROUP_BOUNDARIES
    if n_interactions < sparse_hi:
        return "sparse"
    if n_interactions < medium_hi:
        return "medium"
    return "dense"
```

Flip `tests/test_mapping.py` to GREEN — remove `pytestmark = pytest.mark.skip(...)`, replace `raise NotImplementedError(...)` stubs with real test bodies:

```python
"""Tests for fedrec_foundation.mapping (FND-01 + Codex CR-1)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedrec_foundation.mapping import (
    CanonicalMapping, MAPPING_SCHEMA_VERSION,
    build_mapping, save_mapping, load_mapping,
)


def test_sort_order(synthetic_ratings_df: pd.DataFrame) -> None:
    m = build_mapping(synthetic_ratings_df)
    user_ids_sorted = sorted(synthetic_ratings_df["user_id"].unique())
    item_ids_sorted = sorted(synthetic_ratings_df["movie_id"].unique())
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
    m = build_mapping(synthetic_ratings_df)
    p = tmp_path / "mapping.json"
    save_mapping(m, str(p))
    m2 = load_mapping(str(p))
    assert m.user2idx == m2.user2idx  # int keys preserved
    assert m.item2idx == m2.item2idx
    assert m.num_users == m2.num_users
    assert m.num_items == m2.num_items
    # On disk the keys are strings; loader restored ints.
    raw = json.loads(p.read_text())
    assert all(isinstance(k, str) for k in raw["user2idx"].keys())
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_mapping.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/mapping.py` exists and defines `build_mapping`, `save_mapping`, `load_mapping`, `CanonicalMapping`, `MAPPING_SCHEMA_VERSION`.
    - `grep "ratings_df\[.movie_id.\]\.unique" scripts/foundation/fedrec_foundation/mapping.py` matches (CR-1 compliance).
    - File `scripts/foundation/fedrec_foundation/user_groups.py` exists and defines `classify_user_group`, `USER_GROUP_BOUNDARIES = (30, 100)`, `BUCKET_SEMANTICS = "half_open"`.
    - `cd scripts/foundation && pytest tests/test_mapping.py -v` reports 3 passed (sort_order, item_mapping_from_ratings_only, roundtrip).
    - `cd scripts/foundation && python -c "from fedrec_foundation.user_groups import classify_user_group; assert classify_user_group(29) == 'sparse'; assert classify_user_group(30) == 'medium'; assert classify_user_group(99) == 'medium'; assert classify_user_group(100) == 'dense'"` succeeds (boundary values).
  </acceptance_criteria>
  <done>FND-01 implemented; mapping tests green; user_groups classifier has frozen half-open semantics.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement FND-02 deterministic LOO split + FND-03 exclusion set + flip tests green</name>
  <files>
    scripts/foundation/fedrec_foundation/split.py
    scripts/foundation/fedrec_foundation/exclusion.py
    scripts/foundation/tests/test_split.py
    scripts/foundation/tests/test_exclusion.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md §"Implementation Decisions" (D-03 manifest content, D-04 lock-forever, D-13 exclusion = train_pos ∪ test_pos)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 2: Deterministic LOO Split Builder" (research lines 415-575)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §CR-5 (train-only user stats — LOCKED)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §IMP-2 (split_hash inputs include mapping_sha256 + raw_data_hash)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §IMP-3 (exclusion NPZ: flat items + indptr, NOT keyed-dict)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pitfall 4: split_hash computed over train/test row order"
    - federated-baseline-cf/federated_baseline_cf/dataset.py lines 358-428 (existing `create_leave_one_out_split` — lift and extend with stable sort on `(user_idx, timestamp, item_idx)`)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py (existing `compute_user_genre_distribution` and user-stats computation — lift for the per-user stats block; BUT apply on TRAIN-ONLY rows per CR-5)
    - CLAUDE.md (typing, docstrings)
  </read_first>
  <behavior>
    - `SplitManifest` dataclass INCLUDES `raw_data_hash: str` and `mapping_sha256: str` as top-level fields (populated by `build_split` from its input params). This allows consumers (publish_bundle, RunManifest) to read these fingerprints directly from the manifest.
    - `build_split(ratings_df, mapping, movies_df, mapping_sha256, raw_data_hash)` — 5-param signature. Maps raw IDs to canonical idx, stable-sorts by `(user_idx, timestamp, item_idx)` with `kind="mergesort"`, takes `.tail(1)` per `user_idx` as the held-out item. Users with ≤1 interaction are skipped. Returns `SplitManifest` with `train_user_stats` computed on TRAIN rows (test item removed) AND `raw_data_hash`/`mapping_sha256` populated from the input args.
    - `compute_split_hash(train_keys, test_keys, mapping_sha256, raw_data_hash)` sorts both key lists then SHA-256 of `b"mapping:" + mapping_sha256 + b";raw:" + raw_data_hash + b";train:" + joined + b";test:" + joined`. The mapping + raw-data hashes are INSIDE the split_hash per IMP-2.
    - `save_split_or_verify(manifest, path)`: if path exists, load and compare `split_hash` — if mismatch raise `ValueError("split_hash mismatch: on-disk=X new=Y. A new split would invalidate all cached results. Refusing to overwrite.")`. If match, no-op. If absent, `atomic_write_json`.
    - Per-user genre entropy: for each user, histogram over the set of genres (from movies.dat pipe-delimited genre column) of their TRAIN items, normalize to probabilities, compute `-sum(p * log2(p))`. Rating std: `np.std(user_train_ratings)`. n_interactions: len(user_train_rows). n_unique_items: len(set of user's train items). user_group: `classify_user_group(n_interactions)`.
    - `build_exclusion(train_df_canonical, split_manifest)` groups train rows by `user_idx`, adds `split_manifest.test_item_per_user[u]`, returns `Dict[int, np.ndarray[int32]]` sorted per user.
    - `save_exclusion(per_user_items, path)` writes NPZ with TWO arrays: `items` (flat int32 concat) and `indptr` (int64 offsets, shape `(num_users + 1,)`). IMP-3 layout. Atomic via tempfile + os.replace.
    - `load_exclusion(path)` returns `ExclusionTable(np.load(path, allow_pickle=False))` with `for_user(u) -> items[indptr[u]:indptr[u+1]]`.
    - Module-level `exclusion_for(npz, user_idx)` returns the same slice directly from a loaded NPZ object (CR-3 helper — for callers that don't want the class wrapper).
    - `test_split.py` tests flip green: `test_hash_deterministic`, `test_timestamp_tiebreak`, `test_split_lock_refuses_overwrite`, `test_train_only_user_stats`.
    - `test_exclusion.py` tests flip green: `test_includes_test_item`, `test_safe_load`, `test_indptr_layout`, `test_module_level_exclusion_for`.
  </behavior>
  <action>
Start from `01-RESEARCH.md` Pattern 2 (lines 421-573) and Pattern 3 (lines 589-660), then apply Codex overrides:

1. **`scripts/foundation/fedrec_foundation/split.py`:**
   - Use the dataclasses `PerUserStats` and `SplitManifest` from research Pattern 2 BUT:
     - Rename `per_user_stats` field to `train_user_stats` (CR-5).
     - Add fields `raw_data_hash: str` and `mapping_sha256: str` to `SplitManifest` (so consumers can read both fingerprints directly from the manifest — no side-channel, no post-hoc assignment).
     - Add fields `bucket_boundaries: list` and `bucket_semantics: str`.
   - `compute_split_hash` signature: `compute_split_hash(train_keys, test_keys, mapping_sha256: str, raw_data_hash: str) -> str`. Hash payload explicitly prefixes both fingerprints (IMP-2).
   - `build_split(ratings_df, mapping, movies_df, mapping_sha256: str, raw_data_hash: str) -> SplitManifest`:
     - Map `user_id -> user_idx`, `movie_id -> item_idx`.
     - Stable sort by `(user_idx, timestamp, item_idx)` with `kind="mergesort"`.
     - LOO: `test_idx = sorted_df[sorted_df["user_idx"].isin(eligible)].groupby("user_idx").tail(1).index` — eligible = users with >1 interaction.
     - Build `train_df = sorted_df.drop(test_idx)`.
     - Compute `train_user_stats` over `train_df` ONLY (CR-5): for each user_idx, `n_interactions=len(user_train_df)`, `n_unique_items=user_train_df["item_idx"].nunique()`, `rating_std=float(user_train_df["rating"].std(ddof=0))` (0 if only one interaction), `genre_entropy=_compute_genre_entropy(user_train_df, movies_df)`, `user_group=classify_user_group(n_interactions)`.
     - `_compute_genre_entropy(user_train_df, movies_df)`: merge on item_idx → genre pipe-separated string; explode to per-genre rows; normalize; shannon entropy base-2; return 0.0 if only one genre.
     - `test_item_per_user = {int(u): int(i) for u, i in zip(test_df["user_idx"], test_df["item_idx"])}`.
     - `split_hash = compute_split_hash(train_keys, test_keys, mapping_sha256, raw_data_hash)`.
     - **Construct the returned `SplitManifest` with `raw_data_hash=raw_data_hash, mapping_sha256=mapping_sha256` fields populated explicitly** — no caller-side post-hoc mutation.

2. **`scripts/foundation/fedrec_foundation/exclusion.py`** — replace the keyed-dict research version with the flat items+indptr layout from CODEX PEER REVIEW §IMP-3 (copy that code block verbatim, lines 112-132 of research CODEX section). The `ExclusionTable` class holds the loaded NPZ and offers `for_user(user_idx: int) -> np.ndarray[int32]` that returns `items[indptr[u]:indptr[u+1]]`. Export a module-level `exclusion_for(npz, user_idx)` helper matching the CODEX CR-3 pseudocode — callable directly on a loaded `np.load(...)` object without constructing an `ExclusionTable`.

3. **Flip `tests/test_split.py` to GREEN** with these four tests (note the 5-arg `build_split` call):
```python
"""Tests for fedrec_foundation.split (FND-02 + CR-5 + D-04)."""
from __future__ import annotations

from pathlib import Path
import pytest

from fedrec_foundation.mapping import build_mapping
from fedrec_foundation.split import (
    SplitManifest, build_split, compute_split_hash,
    save_split_or_verify, load_split_manifest,
)


@pytest.fixture
def synthetic_movies_df():
    import pandas as pd
    return pd.DataFrame(
        [(10, "A", "Action"),
         (20, "B", "Comedy|Drama"),
         (30, "C", "Action|Drama"),
         (40, "D", "Comedy")],
        columns=["movie_id", "title", "genres"],
    )


def test_hash_deterministic(synthetic_ratings_df, synthetic_movies_df):
    m = build_mapping(synthetic_ratings_df)
    s1 = build_split(synthetic_ratings_df, m, synthetic_movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    s2 = build_split(synthetic_ratings_df, m, synthetic_movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    assert s1.split_hash == s2.split_hash
    assert len(s1.split_hash) == 64
    # SplitManifest stores both fingerprints as fields.
    assert s1.raw_data_hash == "b" * 64
    assert s1.mapping_sha256 == "a" * 64


def test_timestamp_tiebreak(synthetic_movies_df):
    import pandas as pd
    # Two interactions for user 1 with SAME timestamp -> item with larger item_idx wins via tail(1).
    rows = [(1, 10, 5, 1000), (1, 20, 4, 1000), (1, 30, 3, 2000)]
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    m = build_mapping(df)
    s = build_split(df, m, synthetic_movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    # user_idx 0 (user 1); held-out should be the LAST interaction after stable sort.
    # Stable sort by (user_idx=0, timestamp, item_idx); last row is (0, 2000, item_idx(30)) => held-out item 30.
    held = s.test_item_per_user[0]
    # Item 30 has idx 2 in the sorted-item mapping (items 10,20,30 -> 0,1,2).
    assert held == m.item2idx[30]


def test_split_lock_refuses_overwrite(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    m = build_mapping(synthetic_ratings_df)
    s = build_split(synthetic_ratings_df, m, synthetic_movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    p = tmp_path / "split.json"
    save_split_or_verify(s, p)
    # Second call with same hash: no-op.
    save_split_or_verify(s, p)
    # Different hash: raise.
    s2 = build_split(synthetic_ratings_df, m, synthetic_movies_df, mapping_sha256="c"*64, raw_data_hash="b"*64)
    with pytest.raises(ValueError, match="invalidate all cached results"):
        save_split_or_verify(s2, p)


def test_train_only_user_stats(synthetic_ratings_df, synthetic_movies_df):
    """CR-5: per-user stats must be computed AFTER removing the LOO test item."""
    m = build_mapping(synthetic_ratings_df)
    s = build_split(synthetic_ratings_df, m, synthetic_movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    # User 1 has 3 interactions total; after LOO, train has 2.
    u1_idx = m.user2idx[1]
    assert s.train_user_stats[u1_idx].n_interactions == 2
    # The test item is NOT among the unique train items.
    assert s.train_user_stats[u1_idx].n_unique_items == 2
```

4. **Flip `tests/test_exclusion.py` to GREEN** (uses a VECTORIZED train/test filter — no row-wise `.apply` lambda):
```python
"""Tests for fedrec_foundation.exclusion (FND-03 + IMP-3 flat layout + CR-3 module-level helper)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedrec_foundation.mapping import build_mapping
from fedrec_foundation.split import build_split
from fedrec_foundation.exclusion import (
    build_exclusion, save_exclusion, load_exclusion, exclusion_for,
)


@pytest.fixture
def synthetic_movies_df():
    return pd.DataFrame(
        [(10, "A", "Action"), (20, "B", "Drama"), (30, "C", "Comedy"), (40, "D", "Action")],
        columns=["movie_id", "title", "genres"],
    )


def _vectorized_train_split(df_canonical: pd.DataFrame, test_item_per_user: dict) -> pd.DataFrame:
    """Remove each user's LOO test item via a merge (vectorized; no .apply)."""
    test_pairs = pd.DataFrame(
        [(u, i) for u, i in test_item_per_user.items()],
        columns=["user_idx", "item_idx"],
    )
    test_pairs["is_test"] = True
    merged = df_canonical.merge(test_pairs, on=["user_idx", "item_idx"], how="left")
    train = merged[merged["is_test"].isna()].drop(columns=["is_test"]).copy()
    return train


def _setup(df, movies_df, tmp_path):
    m = build_mapping(df)
    s = build_split(df, m, movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    df_c = df.copy()
    df_c["user_idx"] = df_c["user_id"].map(m.user2idx)
    df_c["item_idx"] = df_c["movie_id"].map(m.item2idx)
    train_c = _vectorized_train_split(df_c, s.test_item_per_user)
    excl = build_exclusion(train_c, s)
    p = tmp_path / "excl.npz"
    save_exclusion(excl, p)
    return m, s, p


def test_includes_test_item(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    with load_exclusion(p) as tab:
        u1_idx = m.user2idx[1]
        excluded = tab.for_user(u1_idx)
        assert s.test_item_per_user[u1_idx] in excluded.tolist()
        # User 1's interactions are (10, t=1000), (20, t=1001), (30, t=1002).
        # LOO held-out = item 30 (last by timestamp). Train positives = {10, 20}.
        for train_item_raw in (10, 20):
            assert m.item2idx[train_item_raw] in excluded.tolist()


def test_safe_load(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    # np.load with allow_pickle=False must succeed (no pickled objects).
    data = np.load(p, allow_pickle=False)
    assert "items" in data.files
    assert "indptr" in data.files
    assert data["items"].dtype == np.int32
    assert data["indptr"].dtype == np.int64


def test_indptr_layout(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    with load_exclusion(p) as tab:
        for u_raw in synthetic_ratings_df["user_id"].unique():
            u_idx = m.user2idx[int(u_raw)]
            arr = tab.for_user(u_idx)
            assert arr.dtype == np.int32
            assert len(arr) >= 1  # every user has at least their test item


def test_module_level_exclusion_for(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    """CR-3: module-level exclusion_for() returns the same slice as ExclusionTable.for_user()."""
    m, s, p = _setup(synthetic_ratings_df, synthetic_movies_df, tmp_path)
    npz = np.load(p, allow_pickle=False)
    with load_exclusion(p) as tab:
        for u_raw in synthetic_ratings_df["user_id"].unique():
            u_idx = m.user2idx[int(u_raw)]
            from_class = tab.for_user(u_idx)
            from_helper = exclusion_for(npz, u_idx)
            np.testing.assert_array_equal(from_class, from_helper)
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_split.py tests/test_exclusion.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/split.py` defines `SplitManifest` (with `raw_data_hash: str` and `mapping_sha256: str` fields), `PerUserStats`, `build_split` (5-param signature), `save_split_or_verify`, `load_split_manifest`, `compute_split_hash`, `SPLIT_SCHEMA_VERSION`, `BUILDER_VERSION`.
    - `grep "raw_data_hash: str" scripts/foundation/fedrec_foundation/split.py` matches (SplitManifest field).
    - `grep "train_user_stats" scripts/foundation/fedrec_foundation/split.py` matches (CR-5 field name).
    - `grep "invalidate all cached results" scripts/foundation/fedrec_foundation/split.py` matches (D-04 error message).
    - `grep "mapping_sha256.*raw_data_hash" scripts/foundation/fedrec_foundation/split.py` matches (IMP-2 hash inputs as build_split params).
    - File `scripts/foundation/fedrec_foundation/exclusion.py` defines `build_exclusion`, `save_exclusion`, `load_exclusion`, `ExclusionTable`, and module-level `exclusion_for`.
    - `grep "indptr" scripts/foundation/fedrec_foundation/exclusion.py` matches (IMP-3 flat layout).
    - `grep "allow_pickle=False" scripts/foundation/fedrec_foundation/exclusion.py` matches (D-05 no-pickle).
    - `grep "^def exclusion_for" scripts/foundation/fedrec_foundation/exclusion.py` matches (module-level helper exported).
    - `cd scripts/foundation && pytest tests/test_split.py -v` prints 4 passed.
    - `cd scripts/foundation && pytest tests/test_exclusion.py -v` prints 4 passed.
  </acceptance_criteria>
  <done>FND-02 + FND-03 implemented; split is deterministic + lock-forever; SplitManifest stores both fingerprints as fields; exclusion uses flat layout + no-pickle loading + module-level helper.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Implement atomic bundle publication + CLI builder + flip integration tests green + GENERATE data/derived/*</name>
  <files>
    scripts/foundation/fedrec_foundation/bundle.py
    scripts/foundation/scripts/build_derived.py
    scripts/foundation/fedrec_foundation/__init__.py
    scripts/foundation/tests/test_integration.py
    data/derived/mapping.json
    data/derived/split_manifest.json
    data/derived/exclusion_items.npz
    data/derived/foundation_index.json
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §N-3 (atomic bundle publication)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §IMP-2 (composite foundation_contract_sha256)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Code Examples — Building the canonical mapping (one-off)" (CLI skeleton, lines 1314-1375)
    - .planning/phases/01-foundation-contract/01-VALIDATION.md (integration test row list: `test_build_idempotent`, `test_bundle_atomic_publication`, `test_build_creates_all_artifacts`, `test_ml1m_counts_6040_3706`)
  </read_first>
  <behavior>
    - `FoundationIndex` dataclass + `publish_bundle(derived_dir, mapping, split_manifest, exclusion) -> FoundationIndex` — 4-param signature. Reads `raw_data_hash` from `split_manifest.raw_data_hash` internally.
    - `publish_bundle` order:
      1. Writes `mapping.json` via `save_mapping` to `derived_dir`.
      2. Writes `split_manifest.json` via `save_split_or_verify`.
      3. Writes `exclusion_items.npz` via `save_exclusion`.
      4. Computes `mapping_sha256 = sha256_file(mapping.json)`, `exclusion_sha256 = sha256_file(exclusion_items.npz)`, `foundation_contract_sha256 = sha256(mapping_sha256 + split_hash + exclusion_sha256)`.
      5. Writes `foundation_index.json` (LAST) via `atomic_write_json` with all four fingerprints + `builder_version` + `created_at`.
    - `verify_bundle(derived_dir) -> FoundationIndex`: loads index; recomputes `mapping_sha256`/`exclusion_sha256` and `foundation_contract_sha256`; raises `RuntimeError("Bundle incomplete or corrupted...")` if any fingerprint mismatches. Loaders (future code) call `verify_bundle` before reading any payload.
    - CLI `python -m fedrec_foundation.build` wires Tasks 1-2 together on the real ML-1M files in `data/ml-1m/`. Produces `data/derived/mapping.json` with exactly 6040 users and 3706 items (empirical anchor). The CLI calls `build_split(..., mapping_sha256=mapping_sha, raw_data_hash=raw_data_hash)` — fingerprints are now SplitManifest fields, so the CLI does NOT mutate `split.raw_data_hash` after the fact.
    - CLI uses a VECTORIZED train-set filter (merge-based) — not a row-wise `.apply` lambda — for the 1M-row ML-1M DataFrame.
    - Integration tests flip green: `test_build_idempotent`, `test_bundle_atomic_publication`, `test_build_creates_all_artifacts`, `test_ml1m_counts_6040_3706`.
  </behavior>
  <action>
1. Create `scripts/foundation/fedrec_foundation/bundle.py`:
```python
"""Atomic bundle publication for foundation artifacts (N-3)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.hashing import sha256_file
from fedrec_foundation.mapping import CanonicalMapping, save_mapping
from fedrec_foundation.split import SplitManifest, save_split_or_verify
from fedrec_foundation.exclusion import save_exclusion

BUNDLE_SCHEMA_VERSION = 1


@dataclass
class FoundationIndex:
    schema_version: int
    builder_version: str
    created_at: str
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str


def compute_foundation_contract_sha256(mapping_sha256: str, split_hash: str, exclusion_sha256: str) -> str:
    """Composite fingerprint — changes if ANY of the three inputs change (IMP-2)."""
    h = hashlib.sha256()
    h.update(b"mapping:" + mapping_sha256.encode("ascii") + b";")
    h.update(b"split:" + split_hash.encode("ascii") + b";")
    h.update(b"exclusion:" + exclusion_sha256.encode("ascii"))
    return h.hexdigest()


def publish_bundle(
    derived_dir: Path,
    mapping: CanonicalMapping,
    split_manifest: SplitManifest,
    exclusion: Dict[int, "np.ndarray"],
) -> FoundationIndex:
    """Atomically publish the 4-file bundle. Index file is written LAST.

    Reads ``raw_data_hash`` from ``split_manifest.raw_data_hash`` — no
    extra parameter needed (the manifest owns that fingerprint after
    build_split populates it). 4-param signature is the LOCKED contract.
    """
    derived_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = derived_dir / "mapping.json"
    split_path = derived_dir / "split_manifest.json"
    excl_path = derived_dir / "exclusion_items.npz"
    index_path = derived_dir / "foundation_index.json"

    # raw_data_hash is a SplitManifest field (populated by build_split).
    # We don't take it as a parameter — single source of truth.
    _ = split_manifest.raw_data_hash  # sanity-touch; used by RunManifest later

    # Step 1-3: payload files (each atomic individually).
    save_mapping(mapping, str(mapping_path))
    save_split_or_verify(split_manifest, split_path)
    save_exclusion(exclusion, excl_path)

    # Step 4: fingerprints, then index.
    mapping_sha = sha256_file(mapping_path)
    excl_sha = sha256_file(excl_path)
    contract = compute_foundation_contract_sha256(
        mapping_sha, split_manifest.split_hash, excl_sha,
    )
    idx = FoundationIndex(
        schema_version=BUNDLE_SCHEMA_VERSION,
        builder_version=split_manifest.builder_version,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mapping_sha256=mapping_sha,
        split_hash=split_manifest.split_hash,
        exclusion_sha256=excl_sha,
        foundation_contract_sha256=contract,
    )
    atomic_write_json(str(index_path), asdict(idx))
    return idx


def verify_bundle(derived_dir: Path) -> FoundationIndex:
    """Load foundation_index.json and verify every fingerprint matches the payload.

    Raises
    ------
    RuntimeError
        If the index is missing, any payload is missing, or any fingerprint mismatches.
    """
    derived_dir = Path(derived_dir)
    index_path = derived_dir / "foundation_index.json"
    if not index_path.exists():
        raise RuntimeError(f"Bundle incomplete: {index_path} missing. Run `python -m fedrec_foundation.build`.")
    with open(index_path) as f:
        data = json.load(f)
    idx = FoundationIndex(**data)

    for name in ("mapping.json", "split_manifest.json", "exclusion_items.npz"):
        if not (derived_dir / name).exists():
            raise RuntimeError(f"Bundle incomplete: {name} missing but index present.")

    mapping_sha = sha256_file(derived_dir / "mapping.json")
    excl_sha = sha256_file(derived_dir / "exclusion_items.npz")
    contract = compute_foundation_contract_sha256(mapping_sha, idx.split_hash, excl_sha)
    if mapping_sha != idx.mapping_sha256:
        raise RuntimeError(f"mapping.json sha mismatch: index={idx.mapping_sha256} actual={mapping_sha}")
    if excl_sha != idx.exclusion_sha256:
        raise RuntimeError(f"exclusion_items.npz sha mismatch: index={idx.exclusion_sha256} actual={excl_sha}")
    if contract != idx.foundation_contract_sha256:
        raise RuntimeError(f"foundation_contract_sha256 mismatch (index corruption)")
    return idx
```

2. Create `scripts/foundation/scripts/build_derived.py` AND a thin `scripts/foundation/fedrec_foundation/__main__.py` that invokes it, so `python -m fedrec_foundation.build` works. Wire Tasks 1-2 together. Use the VECTORIZED merge filter (no `.apply`):
```python
# scripts/foundation/scripts/build_derived.py
"""CLI: python -m fedrec_foundation.build

Builds the 4-file data/derived/ bundle from the real ML-1M in data/ml-1m/.
Idempotent: re-running with an existing locked split is a no-op and exits 0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedrec_foundation.paths import data_derived, ml1m_dir
from fedrec_foundation.hashing import compute_raw_data_hash, sha256_file
from fedrec_foundation.mapping import build_mapping, save_mapping, load_mapping
from fedrec_foundation.split import build_split
from fedrec_foundation.exclusion import build_exclusion
from fedrec_foundation.bundle import publish_bundle


def _load_ml1m(ml1m: Path):
    ratings = pd.read_csv(
        ml1m / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        ml1m / "movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def _vectorized_train_split(df_canonical: pd.DataFrame, test_item_per_user: dict) -> pd.DataFrame:
    """Filter TRAIN rows via a merge (vectorized; O(N log N), not O(N) per-row)."""
    test_pairs = pd.DataFrame(
        [(u, i) for u, i in test_item_per_user.items()],
        columns=["user_idx", "item_idx"],
    )
    test_pairs["is_test"] = True
    merged = df_canonical.merge(test_pairs, on=["user_idx", "item_idx"], how="left")
    return merged[merged["is_test"].isna()].drop(columns=["is_test"]).copy()


def main() -> int:
    derived = data_derived()
    ml1m = ml1m_dir()

    ratings_df, movies_df = _load_ml1m(ml1m)
    raw_data_hash = compute_raw_data_hash(ml1m)

    # 1. Mapping (idempotent via load).
    mapping_path = derived / "mapping.json"
    if mapping_path.exists():
        mapping = load_mapping(str(mapping_path))
    else:
        mapping = build_mapping(ratings_df)
        derived.mkdir(parents=True, exist_ok=True)
        save_mapping(mapping, str(mapping_path))
    mapping_sha = sha256_file(mapping_path)

    # 2. Split (LOCK-FOREVER via save_split_or_verify).
    # build_split returns a SplitManifest with raw_data_hash + mapping_sha256
    # stored as fields — no post-hoc mutation.
    split = build_split(
        ratings_df, mapping, movies_df,
        mapping_sha256=mapping_sha, raw_data_hash=raw_data_hash,
    )

    # 3. Exclusion (built from train-only canonical rows; vectorized filter).
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
```

And `scripts/foundation/fedrec_foundation/build.py`:
```python
"""Enable `python -m fedrec_foundation.build`.

Thin shim re-exporting scripts/build_derived.py::main so the CLI is
discoverable as a module entry point. We add the scripts/ directory
to sys.path at import time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_derived import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
```

Update `scripts/foundation/pyproject.toml` `[tool.hatch.build.targets.wheel]` if needed so `fedrec_foundation.build` is included.

3. Flip `tests/test_integration.py` to GREEN (using the vectorized filter):
```python
"""Integration tests for foundation bundle publication + ML-1M anchors."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedrec_foundation.paths import ml1m_dir
from fedrec_foundation.bundle import publish_bundle, verify_bundle
from fedrec_foundation.mapping import build_mapping
from fedrec_foundation.split import build_split
from fedrec_foundation.exclusion import build_exclusion


@pytest.fixture
def synthetic_movies_df():
    return pd.DataFrame(
        [(10, "A", "Action"), (20, "B", "Drama"), (30, "C", "Comedy"), (40, "D", "Action")],
        columns=["movie_id", "title", "genres"],
    )


def _vectorized_train_split(df_canonical: pd.DataFrame, test_item_per_user: dict) -> pd.DataFrame:
    test_pairs = pd.DataFrame(
        [(u, i) for u, i in test_item_per_user.items()],
        columns=["user_idx", "item_idx"],
    )
    test_pairs["is_test"] = True
    merged = df_canonical.merge(test_pairs, on=["user_idx", "item_idx"], how="left")
    return merged[merged["is_test"].isna()].drop(columns=["is_test"]).copy()


def _build_small_bundle(df, movies_df, derived):
    m = build_mapping(df)
    s = build_split(df, m, movies_df, mapping_sha256="a"*64, raw_data_hash="b"*64)
    df_c = df.copy()
    df_c["user_idx"] = df_c["user_id"].map(m.user2idx)
    df_c["item_idx"] = df_c["movie_id"].map(m.item2idx)
    train_c = _vectorized_train_split(df_c, s.test_item_per_user)
    excl = build_exclusion(train_c, s)
    return publish_bundle(derived, m, s, excl)  # 4-arg signature


def test_build_idempotent(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    derived = tmp_path / "derived"
    idx1 = _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    idx2 = _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    assert idx1.split_hash == idx2.split_hash
    assert idx1.foundation_contract_sha256 == idx2.foundation_contract_sha256


def test_bundle_atomic_publication(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    derived = tmp_path / "derived"
    _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    verify_bundle(derived)  # happy path OK
    # Delete one payload; verify_bundle must fail.
    (derived / "exclusion_items.npz").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_bundle(derived)


def test_build_creates_all_artifacts(synthetic_ratings_df, synthetic_movies_df, tmp_path):
    derived = tmp_path / "derived"
    _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    for name in ("mapping.json", "split_manifest.json", "exclusion_items.npz", "foundation_index.json"):
        assert (derived / name).exists(), f"{name} missing"


def test_ml1m_counts_6040_3706(tmp_path, monkeypatch):
    """Empirical anchor (Codex): real ML-1M produces 6040 users + 3706 items."""
    ml1m = ml1m_dir()
    if not (ml1m / "ratings.dat").exists():
        pytest.skip("real ML-1M not present in data/ml-1m/")
    ratings = pd.read_csv(
        ml1m / "ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    m = build_mapping(ratings)
    assert m.num_users == 6040
    assert m.num_items == 3706
```

4. After all tasks pass, RUN `python -m fedrec_foundation.build` from the repo root to actually produce `data/derived/mapping.json`, `data/derived/split_manifest.json`, `data/derived/exclusion_items.npz`, `data/derived/foundation_index.json`. Commit these four files (they are locked-forever D-04 artifacts; the commit is the lock).
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_integration.py -v &amp;&amp; cd .. &amp;&amp; python -m fedrec_foundation.build &amp;&amp; ls -la data/derived/</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/bundle.py` defines `publish_bundle` (4-param signature: `derived_dir, mapping, split_manifest, exclusion`), `verify_bundle`, `FoundationIndex`, `compute_foundation_contract_sha256`.
    - `grep -E "^def publish_bundle\(" scripts/foundation/fedrec_foundation/bundle.py` shows the 4-param signature (no `raw_data_hash` parameter).
    - `grep "split_manifest.raw_data_hash" scripts/foundation/fedrec_foundation/bundle.py` matches (raw_data_hash is read from the manifest field).
    - `grep "split.raw_data_hash =" scripts/foundation/scripts/build_derived.py` returns no matches (no post-hoc assignment — the field is populated by build_split).
    - File `scripts/foundation/fedrec_foundation/build.py` exists such that `python -m fedrec_foundation.build` from repo root succeeds.
    - Running `python -m fedrec_foundation.build` creates 4 files under `data/derived/`: `mapping.json`, `split_manifest.json`, `exclusion_items.npz`, `foundation_index.json`.
    - `python -c "import json; d=json.load(open('data/derived/mapping.json')); assert d['num_users']==6040 and d['num_items']==3706"` succeeds.
    - `python -c "import json; d=json.load(open('data/derived/split_manifest.json')); assert 'raw_data_hash' in d and len(d['raw_data_hash'])==64"` succeeds (manifest stores raw_data_hash as a top-level field).
    - `python -c "from fedrec_foundation.bundle import verify_bundle; from pathlib import Path; verify_bundle(Path('data/derived'))"` succeeds.
    - `cd scripts/foundation && pytest tests/test_integration.py -v` prints at least 4 passed.
    - Running `python -m fedrec_foundation.build` a SECOND time is a no-op (idempotent, D-04 lock).
    - Grep the builder and test fixtures for `df.*\.apply\(lambda` in this plan's files — ZERO matches allowed (we use vectorized merge, not row-wise apply).
  </acceptance_criteria>
  <done>FND-01, FND-02, FND-03 artifacts exist on disk under data/derived/ and are loaded via verify_bundle() gate. publish_bundle uses the 4-param signature. No `.apply(lambda)` row-wise filters remain.</done>
</task>

</tasks>

<verification>
- `cd scripts/foundation && pytest tests/test_mapping.py tests/test_split.py tests/test_exclusion.py tests/test_integration.py -v` — all pass (no skips for these modules).
- `data/derived/mapping.json` exists and has `"num_users": 6040` and `"num_items": 3706`.
- `data/derived/split_manifest.json` has `"bucket_semantics": "half_open"`, `"bucket_boundaries": [30, 100]`, `"train_user_stats"` block, AND top-level `"raw_data_hash"` + `"mapping_sha256"` fields.
- `data/derived/foundation_index.json` has `mapping_sha256`, `split_hash`, `exclusion_sha256`, `foundation_contract_sha256`.
- Running `python -m fedrec_foundation.build` a second time prints success without rewriting (idempotent, D-04 lock).
- `grep -n "publish_bundle(" scripts/foundation/fedrec_foundation/bundle.py scripts/foundation/scripts/build_derived.py` shows ONLY 4-arg call sites (no 5-arg, no `raw_data_hash=` kwarg).
</verification>

<success_criteria>
- FND-01: `data/derived/mapping.json` exists and any module can load via `load_mapping()` to get `num_users=6040, num_items=3706`.
- FND-02: `data/derived/split_manifest.json` exists, `split_hash` is deterministic, re-building refuses to overwrite on divergence, and the manifest stores `raw_data_hash` + `mapping_sha256` as dataclass fields (not side-channel).
- FND-03: `data/derived/exclusion_items.npz` exists, `load_exclusion(...).for_user(u)` includes that user's test item AND all their training positives, and loads with `allow_pickle=False`. Module-level `exclusion_for(npz, u)` returns the same slice.
- Atomic bundle: `foundation_index.json` is published last; `verify_bundle()` hard-fails if any payload drifts.
- Signature consistency: `publish_bundle` is 4-param everywhere (interface spec, bundle.py action code, CLI in build_derived.py); `build_split` is 5-param with `mapping_sha256` + `raw_data_hash` as explicit args; `SplitManifest` stores both fingerprints as fields.
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation-contract/01-foundation-contract-02-SUMMARY.md` — enumerate the three artifacts' hashes (captured from foundation_index.json), the key file-level decisions (CR-1 ratings-only, CR-5 train-only stats, IMP-2 composite hash, IMP-3 flat NPZ, N-3 atomic index, CR-3 module-level exclusion_for helper), the locked signatures (`build_split` 5-param, `publish_bundle` 4-param, `SplitManifest.raw_data_hash` field), and confirm the empirical 6040/3706 anchor.
</output>
</content>
</invoke>