# Phase 1: Foundation Contract - Research

**Researched:** 2026-04-19
**Domain:** Shared cross-device FedRec protocol artifacts — canonical ID mapping, deterministic LOO split, exclusion set, evaluator selector, weight-policy abstraction, four-tier RNG, mode resolver, run manifest.
**Confidence:** HIGH (code-level patterns are directly extractable from the existing brownfield repo; external best-practice already captured in `.planning/research/*`)

## Summary

Phase 1 is a **foundation plumbing phase**: it does NOT touch any of the four federated modules. It produces six on-disk artifacts plus one shared Python package (`scripts/foundation/`) that every downstream phase will import. The phase is high-leverage because every Phase 2–7 task inherits from its interfaces; getting field names, file layouts, and import paths right here avoids a rename cascade later.

The decisions locked in `01-CONTEXT.md` already close the large design questions (committed `data/derived/`, JSON+NPZ without pickle, locked-forever split, single `mode` selector, primary evaluator `sampled_loo_99`, four-tier RNG, embedded+sibling manifest). Research focuses on the **mechanics**: how to build each artifact deterministically, what exact schema to use, how to wire imports from `scripts/foundation/` into four installable Python packages, and how to validate correctness.

Three cross-cutting risks the planner must design against:

1. **Python's `hash()` is process-salted for tuples containing strings** (`PYTHONHASHSEED` defaults to random). The four-tier RNG `(run_seed, user_id, round, purpose)` cannot use `hash(...)` — it MUST use `hashlib.sha256`. This is a correctness bug if gotten wrong.
2. **`scripts/foundation/` is not inside any of the four installable packages** — it must either be made importable via `sys.path` manipulation inside each module's `dataset.py`, or installed as its own `pip install -e .` package. The latter is cleaner and is this research's recommendation.
3. **Hashing ml-1m raw files vs hashing the canonical split** are two different fingerprints that must both live in `split_manifest.json`. Mixing them silently lets upstream data changes sneak past the split lock.

**Primary recommendation:** Make `scripts/foundation/` a real installable Python package (`pip install -e scripts/foundation/`) named `fedrec_foundation`, with a dataclass-first API (`SplitManifest`, `RunManifest`, `ModeProfile`, `WeightPolicy`) and `hashlib.sha256`-based four-tier RNG. Use NPZ keyed-dict for exclusion sets (one key per user), not flat-array + offset. Write every JSON atomically via `tempfile.NamedTemporaryFile` + `os.replace`. Validate with a small pytest suite (framework install is a Wave 0 gap — not currently present).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Artifact storage & format**
- **D-01:** Foundation artifacts live in a committed `data/derived/` folder at the project root. Ground-truth, bit-exact reproducibility. ~1–5 MB footprint is acceptable for a thesis repo.
- **D-02:** Format is a **mix** — `mapping.json` (tiny, human-readable), `split_manifest.json` (tiny, human-readable), `exclusion_items.npz` (binary int32 arrays per user, fast load for the hot path).
- **D-03:** `split_manifest.json` carries ALL of: `split_hash` (mandatory), builder metadata (builder version, creation timestamp, raw-data hash of `ratings.dat`), per-user stats (`n_interactions`, `genre_entropy`, `n_unique_items`, `rating_std` — precomputed once here so Phase 4 doesn't recompute), and user-group classification (sparse/medium/dense bucket per user — Phase 6 reporting reads it directly).
- **D-04:** Canonical split is **locked forever** after first generation. Re-running the builder recomputes the hash; if it diverges from the committed manifest, the builder errors with "a new split would invalidate all cached results" and refuses to overwrite. Immutable by policy.
- **D-05:** NPZ loads use `numpy.load(..., allow_pickle=False)` and JSON loads are plain `json.load`. No pickle anywhere in the foundation layer.

**Mode-selector interface**
- **D-06:** Each `pyproject.toml` exposes a **single top-level `mode` selector** under `[tool.flwr.app.config]`: `mode = "benchmark_cross_device"` / `"paper_compat_pfedrec"` / `"cross_silo_legacy"` (PFedRec-compat mode only defined where relevant; baseline / personalized / adaptive expose `benchmark_cross_device` and `cross_silo_legacy` only).
- **D-07:** The mode value fully locks the downstream experiment: `num-supernodes`, `partition-mode`, `weight-policy`, `eval-protocol` (primary evaluator), AND the training hyperparameters (`embedding-dim`, optimizer, lr, local epochs, training negatives, num-server-rounds). A mode IS a complete experiment profile, not just a cross-silo/cross-device toggle.
- **D-08:** The mode-to-defaults mapping lives in a shared Python module (not scattered in pyproject.toml). Each client/server reads `mode` from the Flower config, then calls a `resolve_mode_defaults(mode: str) -> dict` helper that returns the locked values. Per-module overrides allowed where a module's paper-compat setting legitimately differs (e.g., PFedRec's `weight-policy`).
- **D-09:** Cross-silo (`num-supernodes=5`) is **kept reachable** as `mode = "cross_silo_legacy"`, not deleted.
- **D-10:** Mode-locked settings **can be overridden** at the CLI (`flwr run . --run-config "num-supernodes=3"`), but every override is captured in the run manifest's `overrides` field AND prints a loud warning at run start.
- **D-11:** Benchmark-mode startup assertion: when `mode = "benchmark_cross_device"` and no overrides are in play, the client asserts `num_users_in_client == 1` and fails loudly otherwise. Assertion stays on for `paper_compat_pfedrec` too; skipped only for `cross_silo_legacy`.

**Canonical evaluator and weight-policy**
- **D-12 (locked):** Primary evaluator is `sampled_loo_99` (leave-one-out + 99 negatives, NCF protocol). `allrank_*` is kept as a namespaced secondary and explicitly excluded from the thesis comparison table.
- **D-13 (locked):** Per-user exclusion set equals `train_pos ∪ test_pos`; no val set at the foundation layer.
- **D-14 (locked):** Four-tier RNG derivation: `run_seed` → `server_rng = Random(run_seed)` → `per_user_rng(user_id, round, purpose) = Random(hash((run_seed, user_id, round, purpose)))`. Purposes include `train_neg`, `eval_neg`, `model_init`. No module-level or evaluator-level `random.seed(...)` / `np.random.seed(...)` calls are permitted.

**Run manifest / protocol fingerprint**
- **D-15:** The run manifest is **embedded in every result JSON** under a top-level `_manifest` key AND written as a sibling `<run_id>-manifest.json` next to the result file.
- **D-16:** Manifest fields (minimum): `mode`, `num-supernodes`, `partition-mode`, `fraction-train`, `fraction-eval`, `weight-policy`, `primary-evaluator`, `num-train-negatives`, `num-eval-negatives` (99), `run-seed`, `checkpoint-rule`, `split_hash`, `raw-data-hash`, `builder-version`, `overrides`, `module`, `flwr-version`, `torch-version`, `git-commit`.

### Claude's Discretion

- **Shared code placement** — Recommendation (planner may override): place as `scripts/foundation/` at project root; each module imports it.
- **Weight-policy defaults per module** — baseline/personalized/adaptive default to `num_positives`; PFedRec depends on Phase 5 audit; PFedRec `benchmark_cross_device` uses `num_positives`.
- **Directory layout inside `data/derived/`** — flat vs subdirs; planner picks.
- **Atomic write pattern** — tempfile + `os.replace()`.
- **Per-user-group bucket boundaries** — sparse ≤ 30, 30 < medium ≤ 100, dense > 100 (unchanged from codebase).
- **`run_id` generation** — ULID or short UUID or timestamp-slug.
- **Validation split** — not introduced at foundation layer.

### Deferred Ideas (OUT OF SCOPE)

- Shared code refactor into `fedrec_common/` (revisit post-thesis).
- Validation split (Phase 5/6 decision if needed).
- DP / privacy accounting (v2).
- ML-10M / ML-20M generalization (v2).
- Profile-based config mechanism (single `mode` selector is sufficient).
- Cross-silo deletion (kept reachable as `cross_silo_legacy`).
- Atomic multi-writer semantics for `data/derived/` (single-writer assumed).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-01 | Canonical `raw_user_id → user_idx` / `raw_item_id → item_idx` mapping artifact under `data/derived/`, imported by every module. | §"Canonical ID Mapping" — sort-ascending, single build, JSON schema. Reuses `create_global_mappings()` logic. |
| FND-02 | Deterministic LOO split manifest with `split_hash`, stable-sort by `(user_id, timestamp, movie_id)`. | §"Deterministic LOO Split Builder" — exact sort keys, SHA-256 input bytes, immutable lock. Reuses `create_leave_one_out_split()` logic + tiebreak fix. |
| FND-03 | Per-user exclusion set `exclude_items[user] = train_pos ∪ test_pos` consumed by every train-negative sampler. | §"Exclusion Set NPZ Layout" — per-user keyed dict pattern, `allow_pickle=False`, O(1) load for `user_idx` key. |
| FND-04 | `sampled_loo_99` declared primary evaluator; selector in foundation layer. | §"Primary Evaluator Selector" — `EvalProtocol` enum + `get_primary_evaluator(mode) -> str`. Does NOT replace evaluator code in each `task.py`; only provides the config constant. |
| FND-05 | Explicit `weight-policy` config (`uniform` / `num_positives` / `num_training_examples`) per module, logged in manifest. | §"Weight-Policy Abstraction" — `WeightPolicy` enum + `compute_aggregation_weight(result, policy) -> float`. Wiring goes in each module's `strategy.py` / `server_app.py` (Phase 2–5). |
| FND-06 | Four-tier RNG derivation `(run_seed, user_id, round, purpose)`; no global reseeding in evaluators. | §"Four-Tier RNG Derivation" — `hashlib.sha256` based (NOT Python `hash()`), exposed as `derive_rng(...)` factory. |
| FND-07 | Run manifest embedded under `_manifest` in result JSON AND sibling `<run_id>-manifest.json`. | §"Run Manifest Schema" — dataclass with all D-16 fields, `to_dict()` method, atomic write helpers. |
</phase_requirements>

## Standard Stack

No stack changes. Phase 1 uses the existing Flower / PyTorch / Python 3.9 baseline and the standard library heavily.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `hashlib` | built-in | `sha256` for split_hash, raw-data-hash, RNG derivation | Deterministic across processes; `hash()` is salted and UNSAFE for this use |
| Python stdlib `json` | built-in | `mapping.json`, `split_manifest.json`, `<run_id>-manifest.json` | Human-readable, no pickle, portable across tooling |
| Python stdlib `dataclasses` | built-in (3.7+) | `SplitManifest`, `RunManifest`, `ModeProfile`, etc. | Matches codebase convention (`AlphaConfig`, `UserGroupConfig`, `EarlyStoppingState`) |
| Python stdlib `random.Random` | built-in | Per-user RNG instances (NOT the global `random` module) | Already the codebase idiom (`random.Random(seed)` appears in `federated-pfedrec/.../task.py:134`) |
| Python stdlib `tempfile` | built-in | Atomic writes (`NamedTemporaryFile` → `os.replace`) | Codebase already uses this for embedding cache |
| Python stdlib `os.replace` | built-in | Atomic rename (POSIX + Windows) | Same pattern as existing cache save in client apps |
| Python stdlib `uuid` | built-in | Short UUID for `run_id` (recommended over ULID) | No new dep; `uuid.uuid4().hex[:12]` is enough collision-resistance for research runs |
| `numpy` | ≥ 1.24.0 (already pinned) | `np.savez`, `np.load(allow_pickle=False)`, per-user int32 arrays | Already required by every module; exclusion set format matches numpy idioms |
| `pandas` | ≥ 2.0.0 (already pinned) | Read ml-1m, stable-sort split | Already used in every `dataset.py::load_movielens_1m()` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest` | latest (NOT currently installed) | Unit tests for builder determinism, RNG reproducibility, mode-resolver dispatch | Wave 0 gap — install in the foundation's own venv or repo-wide |
| `pytest-forked` or subprocess isolation | latest | Cross-process determinism test (hash-based RNG must be stable across processes with different `PYTHONHASHSEED`) | Critical: verifies `hashlib.sha256` usage replaces `hash()` correctly |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `hashlib.sha256` for RNG seeds | Python `hash()` | REJECTED — `hash()` of tuples with strings is salted per-process; breaks determinism across runs |
| `hashlib.sha256` for RNG seeds | `numpy.random.SeedSequence` | Also valid; produces deterministic BitGenerator child streams. `hashlib.sha256 → int → random.Random` is simpler and matches codebase RNG idioms (downstream code wants `random.Random`, not `np.random.Generator`). Prefer sha256. |
| NPZ keyed-dict for exclusion set | Single flat int32 array + offset index | Keyed-dict is O(1) by `user_idx` with `np.load(...)["user_123"]`; flat+offset requires two lookups. Keyed-dict memory is minor (6040 tiny arrays). PREFER keyed-dict. |
| NPZ keyed-dict | JSON lists | JSON is ~5x larger on disk and slower to parse; keyed-dict NPZ is the hot-path load. |
| `ulid-py` for `run_id` | `uuid.uuid4().hex[:12]` or timestamp-slug | ULID requires a new dep; UUID4-short gives 48 bits of entropy which is plenty for a research thesis. Timestamp-slug (`20260419-142301-abcd`) is human-readable and sortable. RECOMMEND timestamp-slug format: `{YYYYMMDD}-{HHMMSS}-{uuid4hex[:6]}`. |
| `pip install -e scripts/foundation/` | `sys.path.insert(0, "../scripts/foundation")` in each `dataset.py` | Editable install is cleaner (real import, real IDE support, real pytest discovery). `sys.path` hack is fragile. PREFER editable install. |
| Separate installable package | Duplicate foundation source into each module | Duplication is the existing sin (see CONCERNS.md "Code duplicated across four modules"); foundation should NOT propagate it. |

**Installation (Wave 0):**
```bash
# Install pytest for testing (not currently available)
pip install pytest

# After creating scripts/foundation/, install as editable package
pip install -e scripts/foundation/
```

**Version verification:** pandas 2.x and numpy 1.24+ are already pinned in every `federated-*/pyproject.toml`. No new pins needed. Confirmed Python 3.9+ target from `CLAUDE.md`.

## Architecture Patterns

### Recommended Project Structure

```
movie-recommendation-system/
├── data/
│   ├── ml-1m/                    # raw data (already exists)
│   └── derived/                  # NEW — committed foundation artifacts (D-01)
│       ├── mapping.json          # FND-01 — canonical raw_id <-> idx
│       ├── split_manifest.json   # FND-02 — LOO split + hash + per-user stats + groups
│       └── exclusion_items.npz   # FND-03 — per-user int32 arrays
├── scripts/
│   └── foundation/               # NEW — installable `fedrec_foundation` package
│       ├── pyproject.toml        # hatchling build, editable install
│       ├── fedrec_foundation/
│       │   ├── __init__.py
│       │   ├── paths.py          # locate data/derived/, data/ml-1m/
│       │   ├── hashing.py        # sha256 helpers (raw_data_hash, split_hash, seed derivation)
│       │   ├── mapping.py        # build + load canonical ID mapping
│       │   ├── split.py          # build + load LOO split manifest; DETERMINISTIC
│       │   ├── exclusion.py      # build + load exclusion NPZ; API: exclusion_for(user_idx)
│       │   ├── evaluator.py      # EvalProtocol enum + get_primary_evaluator()
│       │   ├── weight_policy.py  # WeightPolicy enum + compute_aggregation_weight()
│       │   ├── rng.py            # derive_rng(run_seed, user_id, round, purpose)
│       │   ├── mode.py           # ModeProfile dataclass + resolve_mode_defaults()
│       │   ├── manifest.py       # RunManifest dataclass + write helpers
│       │   └── atomic.py         # atomic_write_json() helper
│       ├── scripts/
│       │   └── build_derived.py  # CLI: python -m fedrec_foundation.build (builds all 3 artifacts)
│       └── tests/
│           ├── test_hashing.py
│           ├── test_mapping.py
│           ├── test_split.py
│           ├── test_exclusion.py
│           ├── test_rng.py
│           ├── test_mode.py
│           └── test_manifest.py
├── federated-baseline-cf/        # (existing)
├── federated-pfedrec/            # (existing)
├── federated-personalized-cf/    # (existing)
└── federated-adaptive-personalized-cf/  # (existing)
```

The four federated modules add ONE line to their `pyproject.toml` dependencies: `"fedrec-foundation @ file://${PROJECT_ROOT}/scripts/foundation"` (or `pip install -e` it manually as a setup step documented in a README).

### Pattern 1: Canonical ID Mapping (FND-01)

**What:** Build `raw_user_id → user_idx` and `raw_item_id → item_idx` once, persist to `mapping.json`, refuse to rebuild if the committed file exists and differs.

**When to use:** At repo setup (one-off), then every module loads it at startup.

**Implementation sketch:**

```python
# fedrec_foundation/mapping.py
# Source: adapted from existing federated-baseline-cf/.../dataset.py:408-428
# (create_global_mappings) + CONTEXT.md D-01/D-04 locked semantics.

from dataclasses import dataclass, asdict
from typing import Dict
import json
import pandas as pd

from fedrec_foundation.atomic import atomic_write_json

MAPPING_SCHEMA_VERSION = 1


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
    num_items : int
    schema_version : int
    """
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    num_users: int
    num_items: int
    schema_version: int = MAPPING_SCHEMA_VERSION


def build_mapping(ratings_df: pd.DataFrame) -> CanonicalMapping:
    """Build mapping deterministically from raw ratings DataFrame.

    Sort-ascending over unique ids, then enumerate. Reproducible across
    pandas versions because `sorted()` on ints is total-ordered.
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
    """Atomic write to `data/derived/mapping.json`."""
    atomic_write_json(path, asdict(mapping))


def load_mapping(path: str) -> CanonicalMapping:
    """Load mapping; verifies schema version."""
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
```

**Gotcha:** JSON keys are always strings. `user2idx[1]` in Python is written as `"1": 0` on disk. The loader MUST restore `int(k)` or every downstream `user2idx[user_id]` lookup silently breaks.

### Pattern 2: Deterministic LOO Split Builder (FND-02)

**What:** Stable-sort by `(user_id, timestamp, movie_id)`, take the last interaction per user as test, compute a `split_hash` fingerprint, refuse to overwrite a committed manifest if the hash diverges (D-04 lock).

**Implementation sketch:**

```python
# fedrec_foundation/split.py
# Source: adapted from existing federated-baseline-cf/.../dataset.py:358-405
# (create_leave_one_out_split), extended with deterministic tiebreak
# on (user_id, timestamp, movie_id) per PITFALLS.md §23.

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List
import pandas as pd

from fedrec_foundation.atomic import atomic_write_json

SPLIT_SCHEMA_VERSION = 1
BUILDER_VERSION = "1.0.0"


@dataclass
class PerUserStats:
    """Precomputed per-user stats for Phase 4 (adaptive alpha)."""
    n_interactions: int
    genre_entropy: float
    n_unique_items: int
    rating_std: float
    user_group: str  # "sparse" | "medium" | "dense"


@dataclass
class SplitManifest:
    """LOO split manifest with hash + per-user metadata.

    One train row == (user_id, item_id, rating, timestamp).
    One test row per user == that user's most-recent interaction by timestamp.
    """
    schema_version: int
    builder_version: str
    created_at: str  # ISO-8601 UTC
    raw_data_hash: str  # sha256 of ratings.dat + movies.dat + users.dat
    split_hash: str  # sha256 of serialized (train_keys, test_keys)
    num_train: int
    num_test_users: int
    test_item_per_user: Dict[int, int]  # user_idx -> item_idx held out
    per_user_stats: Dict[int, PerUserStats]  # user_idx -> stats


def compute_raw_data_hash(ml1m_dir: Path) -> str:
    """SHA-256 of ratings.dat || movies.dat || users.dat (byte-concat, in that order).

    Hashing all three files locks the entire ML-1M fingerprint. Users who
    upgrade to a different ML-1M revision will see the hash change and
    can then either bless the new hash or pin to the old one.
    """
    h = hashlib.sha256()
    for fname in ("ratings.dat", "movies.dat", "users.dat"):
        with open(ml1m_dir / fname, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def compute_split_hash(
    train_keys: List[tuple],  # list of (user_idx, item_idx) sorted
    test_keys: List[tuple],   # list of (user_idx, item_idx) sorted
) -> str:
    """SHA-256 of the canonical split key sets.

    Hashing (user_idx, item_idx) tuples — NOT timestamps — keeps the hash
    stable under timestamp-only metadata edits while still invalidating
    on any held-out-item change. Sorted before hashing for determinism.
    """
    h = hashlib.sha256()
    h.update(b"train:")
    for uidx, iidx in sorted(train_keys):
        h.update(f"{uidx},{iidx};".encode("ascii"))
    h.update(b"test:")
    for uidx, iidx in sorted(test_keys):
        h.update(f"{uidx},{iidx};".encode("ascii"))
    return h.hexdigest()


def build_split(ratings_df: pd.DataFrame, mapping, user_stats: dict) -> SplitManifest:
    """Build deterministic LOO split with stable tiebreak.

    Stable-sort by (user_id, timestamp, movie_id); tail(1) per user_id
    is the held-out test item. This deterministically resolves
    same-timestamp ties (PITFALLS.md §23).
    """
    # Map raw ids to canonical idx; work in canonical space from here.
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

    test_item_per_user = dict(
        zip(test_df["user_idx"].astype(int), test_df["item_idx"].astype(int))
    )
    train_keys = list(zip(train_df["user_idx"].astype(int), train_df["item_idx"].astype(int)))
    test_keys = list(zip(test_df["user_idx"].astype(int), test_df["item_idx"].astype(int)))
    split_hash = compute_split_hash(train_keys, test_keys)

    return SplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        builder_version=BUILDER_VERSION,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        raw_data_hash="",  # set by caller after hashing raw files
        split_hash=split_hash,
        num_train=len(train_df),
        num_test_users=len(test_df),
        test_item_per_user=test_item_per_user,
        per_user_stats=user_stats,
    )


def save_split_or_verify(manifest: SplitManifest, path: Path) -> None:
    """Write if absent; if present, verify hash matches and refuse overwrite (D-04).

    Raises ValueError on hash mismatch with message 'a new split would
    invalidate all cached results'.
    """
    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
        if existing["split_hash"] != manifest.split_hash:
            raise ValueError(
                f"split_hash mismatch: on-disk={existing['split_hash']} "
                f"new={manifest.split_hash}. A new split would invalidate "
                f"all cached results. Refusing to overwrite."
            )
        # Hash matches; manifest already correct, no-op.
        return
    atomic_write_json(str(path), asdict(manifest))
```

**Source:** `federated-baseline-cf/federated_baseline_cf/dataset.py:358-405` (existing) for the LOO core, extended with `(user_id, timestamp, movie_id)` stable sort per PITFALLS.md §23 and D-04's lock semantics.

### Pattern 3: Exclusion Set NPZ Layout (FND-03)

**What:** One NPZ archive `exclusion_items.npz` with one keyed entry per user: `user_{uidx}` → `np.int32` array of excluded `item_idx`. `np.load(..., allow_pickle=False)` returns an `NpzFile` that is O(1) by key.

**Why keyed-dict over flat+offset:**
- Load time: `npz[f"user_{uidx}"]` is O(1) on demand; flat+offset needs two lookups (offset table + slice).
- Memory: NPZ stores each array zipped; 6040 small arrays total ~1-3 MB.
- Code simplicity: caller does `excluded = npz[f"user_{uidx}"]` — no index arithmetic.
- Safety: `allow_pickle=False` works with keyed arrays.

**Implementation sketch:**

```python
# fedrec_foundation/exclusion.py
import numpy as np
from typing import Dict, Set
from pathlib import Path

from fedrec_foundation.split import SplitManifest


def build_exclusion(
    ratings_df_in_canonical_space: "pd.DataFrame",
    split: SplitManifest,
) -> Dict[int, np.ndarray]:
    """For each user, compute exclude_items[u] = train_pos_u ∪ {test_item_u}.

    Returns dict user_idx -> sorted int32 array of item_idx to exclude
    from training-negative sampling. D-13 locks this definition.
    """
    exclusion: Dict[int, np.ndarray] = {}
    grouped = ratings_df_in_canonical_space.groupby("user_idx")["item_idx"].apply(set)
    for uidx, train_items in grouped.items():
        test_item = split.test_item_per_user.get(int(uidx))
        all_excluded: Set[int] = set(int(i) for i in train_items)
        if test_item is not None:
            all_excluded.add(int(test_item))
        exclusion[int(uidx)] = np.array(sorted(all_excluded), dtype=np.int32)
    return exclusion


def save_exclusion(exclusion: Dict[int, np.ndarray], path: Path) -> None:
    """Save as NPZ with keys 'user_{uidx}'. allow_pickle=False safe."""
    kwargs = {f"user_{uidx}": arr for uidx, arr in exclusion.items()}
    # np.savez uses zipfile; atomic via tempfile.
    tmp = path.with_suffix(".npz.tmp")
    np.savez(str(tmp), **kwargs)
    import os
    os.replace(str(tmp), str(path))


def load_exclusion(path: Path):
    """Return an object with O(1) .for_user(uidx) -> np.ndarray[int32].

    IMPORTANT: np.load(..., allow_pickle=False) per D-05. Consumer holds
    the NpzFile open; caller should use inside `with` or close explicitly.
    """
    npz = np.load(str(path), allow_pickle=False)
    return ExclusionTable(npz)


class ExclusionTable:
    def __init__(self, npz: "np.lib.npyio.NpzFile"):
        self._npz = npz

    def for_user(self, user_idx: int) -> np.ndarray:
        key = f"user_{int(user_idx)}"
        if key not in self._npz.files:
            # Returning empty array is safer than KeyError for users with
            # no train interactions (edge case: n=1 user with only the LOO
            # test item -- which we skip in LOO eligibility, so shouldn't
            # happen, but defensive).
            return np.empty(0, dtype=np.int32)
        return self._npz[key]

    def close(self) -> None:
        self._npz.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### Pattern 4: Primary Evaluator Selector (FND-04)

**Important:** Phase 1 does NOT replace evaluator implementations (`evaluate_ranking_sampled` already exists in every module's `task.py`). Phase 1 provides a **config-level constant** and a **resolver function**. The per-module surgical RNG refactor is Phase 2–5 work.

```python
# fedrec_foundation/evaluator.py
from enum import Enum


class EvalProtocol(str, Enum):
    SAMPLED_LOO_99 = "sampled_loo_99"   # D-12 primary
    ALLRANK = "allrank"                  # D-12 secondary, namespaced


def get_primary_evaluator(mode: str) -> str:
    """Return the primary evaluator string for a given mode.

    For ALL three modes the primary evaluator is sampled_loo_99; this
    function exists so that future modes (e.g., paper-compat with a
    different protocol) have a clean extension point.
    """
    return EvalProtocol.SAMPLED_LOO_99.value
```

Downstream phases use `EvalProtocol.SAMPLED_LOO_99.value` as the manifest field and metric namespace prefix (e.g., `sampled_ndcg@10`).

### Pattern 5: Weight-Policy Abstraction (FND-05)

```python
# fedrec_foundation/weight_policy.py
from enum import Enum
from typing import Dict, Union


class WeightPolicy(str, Enum):
    UNIFORM = "uniform"
    NUM_POSITIVES = "num_positives"
    NUM_TRAINING_EXAMPLES = "num_training_examples"


def compute_aggregation_weight(
    client_metrics: Dict[str, Union[int, float]],
    policy: str,
) -> float:
    """Compute this client's aggregation weight from its returned metrics.

    Expected metric keys (produced by each client's @app.train()):
    - "num_positives": int, count of positive train samples (for NUM_POSITIVES)
    - "num_training_examples": int, total train sample count (for NUM_TRAINING_EXAMPLES)

    UNIFORM returns 1.0 unconditionally.

    Raises ValueError on unknown policy or missing required metric.
    """
    p = WeightPolicy(policy)
    if p is WeightPolicy.UNIFORM:
        return 1.0
    if p is WeightPolicy.NUM_POSITIVES:
        if "num_positives" not in client_metrics:
            raise ValueError("weight-policy=num_positives requires 'num_positives' metric")
        return float(client_metrics["num_positives"])
    if p is WeightPolicy.NUM_TRAINING_EXAMPLES:
        if "num_training_examples" not in client_metrics:
            raise ValueError(
                "weight-policy=num_training_examples requires "
                "'num_training_examples' metric"
            )
        return float(client_metrics["num_training_examples"])
    raise ValueError(f"Unknown weight policy: {policy}")
```

**Per-module wiring (Phase 2–5 work, not Phase 1):**
- Each module's `strategy.py` (where split learning applies) or `server_app.py` reads `weight-policy` from `context.run_config`, then during `aggregate_fit()` calls `compute_aggregation_weight(fit_res.metrics, policy)` to populate `num_examples` for `flwr.server.strategy.FedAvg`.
- Baseline/personalized/adaptive default: `num_positives`.
- PFedRec `benchmark_cross_device` default: `num_positives`.
- PFedRec `paper_compat_pfedrec` default: deferred to Phase 5 audit (likely `uniform` or `num_positives`; the reference `IJCAI-23-PFedRec/engine.py:117-119` needs verification).

### Pattern 6: Four-Tier RNG Derivation (FND-06) — CRITICAL CORRECTNESS

**The trap:** Python's built-in `hash()` is **salted per-process** when `PYTHONHASHSEED` is not fixed to 0. Hashing strings (or tuples containing strings like `"train_neg"`) gives different values in different processes. CONTEXT.md's D-14 phrasing `Random(hash((run_seed, user_id, round, purpose)))` is a **shorthand, not a literal recipe** — if implemented with `hash()` it silently becomes non-deterministic across runs.

**Correct implementation uses `hashlib.sha256`:**

```python
# fedrec_foundation/rng.py
import hashlib
import random
from typing import Literal

# The full set of legal purposes. Keeping a closed enum prevents typos
# silently producing new RNG streams (a classic reproducibility footgun).
_ALLOWED_PURPOSES = frozenset({"train_neg", "eval_neg", "model_init"})


def _derive_seed(run_seed: int, user_id: int, round_num: int, purpose: str) -> int:
    """SHA-256 -> 64-bit int seed. Fully deterministic across processes.

    Python's built-in hash() is process-salted; we MUST use hashlib here.
    """
    if purpose not in _ALLOWED_PURPOSES:
        raise ValueError(
            f"Unknown RNG purpose {purpose!r}. Allowed: {sorted(_ALLOWED_PURPOSES)}"
        )
    payload = f"{run_seed}|{user_id}|{round_num}|{purpose}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    # Take first 8 bytes as unsigned 64-bit int.
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def derive_rng(
    run_seed: int,
    user_id: int,
    round_num: int,
    purpose: Literal["train_neg", "eval_neg", "model_init"],
) -> random.Random:
    """Derive an independent random.Random stream for (user, round, purpose).

    D-14 four-tier derivation. Call once per (user, round, purpose) and
    thread the resulting Random instance through whatever samples negatives
    or inits weights. Never call random.seed(...) or np.random.seed(...)
    in the caller.
    """
    return random.Random(_derive_seed(run_seed, user_id, round_num, purpose))


def server_rng(run_seed: int) -> random.Random:
    """Top-level server RNG for per-round client selection.

    Server calls this once at app.main() startup and passes the instance
    to grid.get_node_ids() sampler each round.
    """
    return random.Random(run_seed)
```

**Verification test (MUST exist in `tests/test_rng.py`):**

```python
# tests/test_rng.py
import subprocess
import sys
import textwrap

def test_derive_rng_stable_across_processes():
    """Seeds must match across fresh Python processes with different PYTHONHASHSEED."""
    script = textwrap.dedent('''
        from fedrec_foundation.rng import _derive_seed
        print(_derive_seed(42, 123, 7, "train_neg"))
    ''')
    outputs = []
    for hashseed in ("0", "1", "random"):
        env = {"PYTHONHASHSEED": hashseed} if hashseed != "random" else {}
        r = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        outputs.append(r.stdout.strip())
    assert len(set(outputs)) == 1, (
        f"RNG seed must be stable across PYTHONHASHSEED; got {outputs}"
    )
```

### Pattern 7: Mode Resolver (D-06 to D-11)

```python
# fedrec_foundation/mode.py
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from fedrec_foundation.weight_policy import WeightPolicy
from fedrec_foundation.evaluator import EvalProtocol

ModeName = Literal["benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"]


@dataclass(frozen=True)
class ModeProfile:
    """Complete experiment profile for one mode. D-07: a mode IS an experiment."""
    mode: str
    num_supernodes: int
    partition_mode: str  # "natural" | "dirichlet"
    weight_policy: str   # from WeightPolicy
    primary_evaluator: str  # from EvalProtocol
    fraction_train: float
    fraction_eval: float
    num_train_negatives: int
    num_eval_negatives: int
    # Training hyperparams (D-07 says mode locks these too).
    embedding_dim: int
    optimizer: str       # "adam" | "sgd"
    lr: float
    local_epochs: int    # K
    num_server_rounds: int  # R
    checkpoint_rule: str  # "best_round" | "last_round"
    # Benchmark-mode assertion (D-11).
    assert_one_user_per_client: bool


# Per-module overrides layer: each module MAY override a single field if its
# paper-compat setting legitimately differs. Enforced by resolve_mode_defaults
# which takes a `module_overrides` dict.

_BENCHMARK_CROSS_DEVICE = ModeProfile(
    mode="benchmark_cross_device",
    num_supernodes=6040,
    partition_mode="natural",
    weight_policy=WeightPolicy.NUM_POSITIVES.value,
    primary_evaluator=EvalProtocol.SAMPLED_LOO_99.value,
    fraction_train=0.1,   # sweep-tunable default
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=64,
    optimizer="adam",
    lr=0.001,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",
    assert_one_user_per_client=True,
)

_PAPER_COMPAT_PFEDREC = ModeProfile(
    mode="paper_compat_pfedrec",
    num_supernodes=6040,
    partition_mode="natural",
    weight_policy=WeightPolicy.NUM_POSITIVES.value,  # deferred confirmation to PFR-02
    primary_evaluator=EvalProtocol.SAMPLED_LOO_99.value,
    fraction_train=1.0,  # paper uses full participation
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=32,
    optimizer="sgd",
    lr=0.1,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",
    assert_one_user_per_client=True,
)

_CROSS_SILO_LEGACY = ModeProfile(
    mode="cross_silo_legacy",
    num_supernodes=5,
    partition_mode="dirichlet",
    weight_policy=WeightPolicy.NUM_TRAINING_EXAMPLES.value,
    primary_evaluator=EvalProtocol.SAMPLED_LOO_99.value,
    fraction_train=1.0,
    fraction_eval=1.0,
    num_train_negatives=1,
    num_eval_negatives=99,
    embedding_dim=128,
    optimizer="adam",
    lr=0.001,
    local_epochs=5,
    num_server_rounds=10,
    checkpoint_rule="last_round",
    assert_one_user_per_client=False,  # D-11: disabled in legacy
)

_REGISTRY: Dict[str, ModeProfile] = {
    "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
    "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
    "cross_silo_legacy": _CROSS_SILO_LEGACY,
}


def resolve_mode_defaults(mode: str, module_overrides: Optional[Dict[str, object]] = None) -> ModeProfile:
    """Return the ModeProfile for a mode name, with optional per-module field overrides.

    D-08: per-module overrides allowed where paper-compat setting differs.
    module_overrides example: {"weight_policy": "uniform"} for PFedRec's paper mode.
    """
    if mode not in _REGISTRY:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of {sorted(_REGISTRY)}."
        )
    profile = _REGISTRY[mode]
    if not module_overrides:
        return profile
    # Use dataclass replace so we return a fresh (still frozen) instance.
    from dataclasses import replace
    return replace(profile, **module_overrides)
```

**Circular-import safety:** `mode.py` imports from `weight_policy.py` and `evaluator.py`, which do not import back. No circular risk.

**Import path from each module:**
```python
# In e.g. federated-baseline-cf/federated_baseline_cf/server_app.py:
from fedrec_foundation.mode import resolve_mode_defaults
from fedrec_foundation.rng import server_rng, derive_rng
from fedrec_foundation.mapping import load_mapping
from fedrec_foundation.split import SplitManifest, load_split_manifest
from fedrec_foundation.exclusion import load_exclusion
from fedrec_foundation.manifest import RunManifest, write_manifest
```

**Override-logging at run start (server_app.py integration — done in Phases 2–5, but pattern lives in foundation):**

```python
# fedrec_foundation/mode.py (additional helper)
def log_mode_and_overrides(
    mode: str,
    profile: ModeProfile,
    run_config: Dict[str, object],
) -> Dict[str, object]:
    """Print loud warning for any run_config key that overrides a mode field.

    Returns the subset of run_config that actually overrode a profile field —
    this dict goes into RunManifest.overrides (D-10).
    """
    overrides: Dict[str, object] = {}
    for key, val in run_config.items():
        # kebab-case run_config key -> snake_case dataclass field
        snake = key.replace("-", "_")
        if hasattr(profile, snake):
            profile_val = getattr(profile, snake)
            if profile_val != val:
                overrides[snake] = val
                print(
                    f"[MODE OVERRIDE] {key}: mode={mode} default={profile_val!r} "
                    f"user-override={val!r}"
                )
    if overrides:
        print(f"[MODE OVERRIDE] {len(overrides)} override(s) active; "
              f"captured in manifest.overrides")
    return overrides
```

### Pattern 8: Run Manifest (FND-07)

```python
# fedrec_foundation/manifest.py
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import json
import uuid

from fedrec_foundation.atomic import atomic_write_json


RUN_MANIFEST_SCHEMA_VERSION = 1


def generate_run_id() -> str:
    """Timestamp-slug + short uuid. Human-readable and sortable.

    Format: '20260419-142301-a1b2c3'
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


@dataclass
class RunManifest:
    """Run fingerprint. D-16 field set.

    Written twice (D-15):
    - Embedded under top-level '_manifest' key in the result JSON.
    - Sibling <run_id>-manifest.json next to the result file.
    """
    schema_version: int
    run_id: str
    # Mode + locked config
    mode: str
    num_supernodes: int
    partition_mode: str
    fraction_train: float
    fraction_eval: float
    weight_policy: str
    primary_evaluator: str
    num_train_negatives: int
    num_eval_negatives: int
    run_seed: int
    checkpoint_rule: str
    # Data fingerprint
    split_hash: str
    raw_data_hash: str
    builder_version: str
    # Overrides + module metadata
    overrides: Dict[str, object]  # from log_mode_and_overrides()
    module: str  # "baseline" | "personalized" | "adaptive" | "pfedrec"
    # Environment
    flwr_version: str
    torch_version: str
    git_commit: str  # from `git rev-parse HEAD`; "unknown" if not a git dir


def _git_commit() -> str:
    """Best-effort; 'unknown' on failure (e.g., not a git checkout)."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    run_id: str,
    mode_profile,
    run_seed: int,
    split_hash: str,
    raw_data_hash: str,
    builder_version: str,
    overrides: Dict[str, object],
    module: str,
) -> RunManifest:
    import flwr
    import torch
    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        mode=mode_profile.mode,
        num_supernodes=mode_profile.num_supernodes,
        partition_mode=mode_profile.partition_mode,
        fraction_train=mode_profile.fraction_train,
        fraction_eval=mode_profile.fraction_eval,
        weight_policy=mode_profile.weight_policy,
        primary_evaluator=mode_profile.primary_evaluator,
        num_train_negatives=mode_profile.num_train_negatives,
        num_eval_negatives=mode_profile.num_eval_negatives,
        run_seed=run_seed,
        checkpoint_rule=mode_profile.checkpoint_rule,
        split_hash=split_hash,
        raw_data_hash=raw_data_hash,
        builder_version=builder_version,
        overrides=dict(overrides),
        module=module,
        flwr_version=getattr(flwr, "__version__", "unknown"),
        torch_version=torch.__version__,
        git_commit=_git_commit(),
    )


def write_manifest_sibling(manifest: RunManifest, result_json_path: Path) -> Path:
    """D-15 sibling: write <run_id>-manifest.json next to the result JSON.

    Returns the manifest path.
    """
    sibling = result_json_path.parent / f"{manifest.run_id}-manifest.json"
    atomic_write_json(str(sibling), asdict(manifest))
    return sibling


def embed_manifest_in_result(manifest: RunManifest, result_dict: dict) -> dict:
    """D-15 embedded: inject '_manifest' key into an existing result dict.

    Caller is responsible for json.dump(result_dict, ...). Returns
    result_dict for fluent chaining.
    """
    result_dict["_manifest"] = asdict(manifest)
    return result_dict
```

**W&B integration (wandb.config duplication avoidance):**

```python
# In server_app.py (Phase 2–5 wiring):
wandb.init(project=..., config={"_manifest": asdict(manifest), **run_config})
```

The `_manifest` key appears as a nested dict in `wandb.config`. Individual fields are auto-flattened by W&B (e.g., `config._manifest.weight_policy` queryable), so there is no duplication. Alternatively, `wandb.run.summary["_manifest"] = asdict(manifest)` avoids even the surface duplication.

### Pattern 9: Atomic Write Helper

```python
# fedrec_foundation/atomic.py
import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path: str, data: object) -> None:
    """Write JSON atomically via tempfile + os.replace().

    Works on POSIX and Windows. On crash mid-write, the destination
    file is either untouched or fully-new — never partial.
    """
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem required for atomic replace.
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _json_default(obj):
    """Handle numpy scalars and Path objects that json.dumps rejects."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
```

### Anti-Patterns to Avoid

- **Don't use Python `hash()` for seed derivation** — process-salted, silent non-determinism. Use `hashlib.sha256` (Pattern 6).
- **Don't use `pickle` anywhere** (D-05). JSON + NPZ only. Never `torch.save` for foundation artifacts; never `np.load(allow_pickle=True)`.
- **Don't let `mapping.json` int keys become strings silently.** Loader must cast back to int (Pattern 1 gotcha).
- **Don't hash timestamps into `split_hash`.** Hash the `(user_idx, item_idx)` key sets only — makes the hash stable under metadata edits.
- **Don't re-derive RNG streams at module import.** Derive on-demand inside the function that samples.
- **Don't emit `warnings.warn()` for mode overrides — use `print()`** with a distinctive prefix so they show up in every run's stdout log (CONVENTIONS.md: "This codebase does NOT use the logging module for application logs").
- **Don't use `dict()` type hints** — codebase uses `typing.Dict` / `typing.List` / `typing.Optional` / `typing.Union` per CONVENTIONS.md "Python Target". Phase 1 must match.
- **Don't put `scripts/foundation/` inside one of the four module packages** — it has to be shared.
- **Don't put the foundation module at the repo top-level as `foundation/`** — clashes with the user's request phrasing that put it under `scripts/`. Keep `scripts/foundation/`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deterministic hashing | Python `hash()` or custom mixing | `hashlib.sha256` | `hash()` is process-salted; see correctness discussion in Pattern 6 |
| Atomic file writes | Naive `open("w")` | `tempfile.mkstemp` + `os.replace` | Existing codebase pattern; POSIX-atomic |
| Binary array archives | Custom flat-buffer file | `np.savez` + `np.load(allow_pickle=False)` | Keyed dict access, zipped, safe |
| Human-readable config | TOML, YAML, custom parser | `json` stdlib | No new deps; already used everywhere in codebase |
| Per-thread RNG | Global `random.seed()` + subroutine | `random.Random(seed)` instances | Codebase already uses `random.Random(seed)` pattern (see `federated-pfedrec/.../task.py:134`) |
| Run-id generator | External ULID lib | `datetime.strftime` + `uuid.uuid4().hex[:6]` | No new deps; human-readable + sortable |
| Git-commit capture | Parse `.git/HEAD` by hand | `subprocess.run(["git", "rev-parse", "HEAD"])` | Portable, handles detached HEAD |
| Enum dispatch | `if/elif/else` string ladder | `enum.Enum` + factory (`weight_policy.py`, `mode.py`) | Codebase convention (see `create_alpha_computer` factory in `adaptive_alpha.py:604`) |

**Key insight:** Foundation code is 80% standard-library plumbing. Every external dependency added here multiplies by four downstream packages. Stay stdlib + existing pins.

## Runtime State Inventory

Phase 1 is a **greenfield-on-brownfield** phase: it adds new artifacts and a new package, but does NOT rename or migrate existing state. Runtime-state inventory is mostly "nothing — verified by scope". Included for completeness:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified by scope. Phase 1 creates `data/derived/` from scratch; no existing database or data-store is renamed. Existing `.embedding_cache/` directories are untouched by Phase 1 (they get run-namespaced in Phase 2–4 per PSN-05 / ADP-06). | None. |
| Live service config | None — verified by scope. No live services; this is a simulation-only thesis codebase. No n8n / Datadog / Cloudflare. W&B project switch (`thesis-crossdevice-*`) is EVL-05, not Phase 1. | None. |
| OS-registered state | None — verified by scope. No systemd units, no Windows tasks, no launchd plists. | None. |
| Secrets / env vars | `WANDB_API_KEY` (existing; managed outside repo via `~/.netrc` per STACK.md). Phase 1 does NOT change any env var names. | None. |
| Build artifacts / installed packages | **NEW:** `scripts/foundation/` will be installed as `fedrec-foundation` via `pip install -e scripts/foundation/`. Each of the four federated modules' `pyproject.toml` gets a new dependency line. No existing egg-info becomes stale; the four federated packages' package names are unchanged. | Run `pip install -e scripts/foundation/` then `pip install -e federated-<name>/` per module once Phase 1 lands. Document in README / commit message. |

**Canonical question:** *After every file in the repo is updated, what runtime systems still have old state cached?* Answer: **nothing.** Phase 1 adds artifacts; it does not rename or invalidate prior artifacts. The `.embedding_cache/` directories created under the pre-migration cross-silo setup are still valid for `cross_silo_legacy` mode runs (D-09) and only become stale under `benchmark_cross_device` mode — but that staleness surfaces via the run-namespaced cache keying added in Phase 2–4, not Phase 1.

## Common Pitfalls

### Pitfall 1: `hash()` is process-salted — silently breaks reproducibility

**What goes wrong:** Implementing D-14 literally as `Random(hash((run_seed, user_id, round, purpose)))` produces different RNG streams in two fresh Python processes with the same inputs.

**Why it happens:** CPython randomizes `hash()` of `str` (and tuples containing `str`) between processes unless `PYTHONHASHSEED=0`. Thesis reviewers re-running from scratch won't have `PYTHONHASHSEED=0` set, and their numbers will drift from the committed results.

**How to avoid:** Use `hashlib.sha256` (Pattern 6). Add the cross-process test in Pattern 6 as a mandatory unit test.

**Warning signs:** `test_derive_rng_stable_across_processes` flakes. Different wandb runs of the "same" config produce different selected-client sequences.

### Pitfall 2: JSON int keys silently become strings on load

**What goes wrong:** `json.dump({1: 0, 2: 1}, f)` writes `{"1": 0, "2": 1}`. `json.load(f)` returns `{"1": 0, "2": 1}`. `user2idx[1]` then raises `KeyError`.

**Why it happens:** JSON has no int-key type; everything is a string. Python dict doesn't complain on type mismatch in lookups — it just misses.

**How to avoid:** Loader must cast: `{int(k): v for k, v in data.items()}` (Pattern 1).

**Warning signs:** `KeyError: 1` on lookup. Silent fallback to defaults hiding the problem.

### Pitfall 3: NPZ loaded with `allow_pickle=True` opens RCE window

**What goes wrong:** `np.load(..., allow_pickle=True)` deserializes arbitrary Python objects. If anyone ever commits a malicious NPZ, `import`-time code execution follows.

**Why it happens:** NumPy defaults to `allow_pickle=True` pre-1.17 (now False) but passing it explicitly is safer and documents intent.

**How to avoid:** D-05 is locked. Always `np.load(..., allow_pickle=False)` — exclusion.py Pattern 3 enforces this.

**Warning signs:** `ValueError: Object arrays cannot be loaded when allow_pickle=False` — this is SUCCESS; means you tried to load pickled data and refused. Keep it and emit a clear error message.

### Pitfall 4: `split_hash` computed over train/test row order

**What goes wrong:** Hashing `(user_idx, item_idx, timestamp)` or hashing the DataFrame row-wise gives a hash that changes when pandas version updates change row order for identical data.

**Why it happens:** `sort_values` stability guarantees depend on `kind=` parameter. Default changed between pandas 1.x and 2.x.

**How to avoid:** Hash `(user_idx, item_idx)` pairs AFTER sorting them explicitly in the hashing function (Pattern 2 `compute_split_hash`). Never depend on DataFrame row order for hash input.

**Warning signs:** Re-running the builder yields a different `split_hash` with zero code changes.

### Pitfall 5: Foundation module is not on `sys.path` in each Flower module's process

**What goes wrong:** `from fedrec_foundation.rng import derive_rng` raises `ModuleNotFoundError` when each Flower client process starts.

**Why it happens:** Each of the four modules is a separate installable package with its own `pyproject.toml`. If `fedrec_foundation` is not a declared dependency, `pip install -e federated-baseline-cf/` does not pull it in.

**How to avoid:** (a) Make `scripts/foundation/` a real installable package. (b) Add it to each module's `pyproject.toml` `[project].dependencies` as `"fedrec-foundation"`. (c) Install order is `pip install -e scripts/foundation/` first, THEN `pip install -e federated-<name>/`. Document this order in the README and in a `scripts/setup_dev_env.sh` (tiny script, optional).

**Warning signs:** First `flwr run .` after Phase 1 lands fails at import. Diagnostic: `python -c "import fedrec_foundation"` from inside a module directory.

### Pitfall 6: `mode` override-logging misses snake_case vs kebab-case mismatch

**What goes wrong:** User runs `flwr run . --run-config "weight-policy=uniform"`. Run config key is `"weight-policy"` (kebab). `ModeProfile.weight_policy` is snake. Naive `hasattr(profile, key)` returns False; override is silently dropped from the manifest.

**Why it happens:** Flower config uses kebab-case (CONVENTIONS.md); Python fields use snake_case.

**How to avoid:** `log_mode_and_overrides` in Pattern 7 explicitly does `snake = key.replace("-", "_")` before the hasattr check.

**Warning signs:** Manifest's `overrides` field is `{}` despite CLI override. Run behaves as mode-default instead of as override.

### Pitfall 7: Split builder runs under different ML-1M revision

**What goes wrong:** Someone re-downloads ml-1m (or the GroupLens server updates the tar). `raw_data_hash` changes. Builder detects this but `split_hash` may or may not change (depends on whether the user/item set changed).

**Why it happens:** Immutable artifacts + mutable upstream data.

**How to avoid:** `build_derived.py` CLI checks existing `split_manifest.json.raw_data_hash`. If the current ML-1M hash differs, it errors: "Raw ML-1M changed (hash X → Y). Pin the old ML-1M, or bless the new hash by deleting and regenerating data/derived/. Regeneration invalidates all cached results."

**Warning signs:** `raw_data_hash` mismatch on CI or a fresh checkout.

### Pitfall 8: Benchmark-mode assertion triggers in `cross_silo_legacy` by accident

**What goes wrong:** `client_app.py` asserts `num_users_in_client == 1` unconditionally. `cross_silo_legacy` runs (5 users per partition) raise at round 1.

**Why it happens:** Shared assertion wired before checking mode.

**How to avoid:** `ModeProfile.assert_one_user_per_client` bool (Pattern 7) gates the assert. `cross_silo_legacy` sets it False. Benchmark + paper-compat set it True.

**Warning signs:** `AssertionError: num_users_in_client=1200 != 1` from a `cross_silo_legacy` run.

### Pitfall 9: Test that `evaluate_ranking_sampled` no longer reseeds globals is phase-2 work, not phase-1

**What goes wrong:** Foundation builds the RNG factory but each module's `task.py` still has `random.seed(seed)` at top of `evaluate_ranking_sampled`. Foundation's RNG never actually gets used.

**Why it happens:** Phase 1 is non-invasive; the per-module surgical refactor is Phases 2–5.

**How to avoid:** Make it explicit in the plan that Phase 1 **provides** `derive_rng` but Phases 2–5 **consume** it. Foundation's unit tests cover the factory only; per-module integration tests (Phases 2–5) cover the consumption.

**Warning signs:** Scope creep into Phase 2 territory during Phase 1 execution.

## Code Examples

### Building the canonical mapping (one-off)

```python
# scripts/foundation/scripts/build_derived.py
# CLI: python -m fedrec_foundation.build
# Source: synthesized from patterns 1, 2, 3 above.

from pathlib import Path
import pandas as pd

from fedrec_foundation.mapping import build_mapping, save_mapping, load_mapping
from fedrec_foundation.split import (
    build_split, save_split_or_verify, compute_raw_data_hash,
)
from fedrec_foundation.exclusion import build_exclusion, save_exclusion
from fedrec_foundation.paths import DATA_DERIVED, ML1M_DIR


def main():
    # 1. Load raw ml-1m.
    ratings_df = pd.read_csv(
        ML1M_DIR / "ratings.dat",
        sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    # 2. Build mapping (idempotent).
    mapping_path = DATA_DERIVED / "mapping.json"
    if mapping_path.exists():
        mapping = load_mapping(str(mapping_path))
        print(f"mapping.json already exists with {mapping.num_users} users. OK.")
    else:
        mapping = build_mapping(ratings_df)
        save_mapping(mapping, str(mapping_path))
        print(f"Wrote mapping.json: {mapping.num_users} users, {mapping.num_items} items.")

    # 3. Build split (locks on disk per D-04).
    user_stats = _compute_user_stats(ratings_df, mapping)  # per-user dict; Phase-4-ready
    manifest = build_split(ratings_df, mapping, user_stats)
    manifest.raw_data_hash = compute_raw_data_hash(ML1M_DIR)
    split_path = DATA_DERIVED / "split_manifest.json"
    save_split_or_verify(manifest, split_path)  # errors if committed hash differs
    print(f"Wrote split_manifest.json. split_hash={manifest.split_hash[:12]}...")

    # 4. Build exclusion NPZ.
    # Work in canonical-idx space.
    df_idx = ratings_df.copy()
    df_idx["user_idx"] = df_idx["user_id"].map(mapping.user2idx)
    df_idx["item_idx"] = df_idx["movie_id"].map(mapping.item2idx)
    # NOTE: build_exclusion expects TRAIN rows only -- remove test rows first.
    # For LOO split these are exactly the rows whose (user_idx, item_idx)
    # match manifest.test_item_per_user -- removed for symmetry with training.
    train_df = df_idx.drop(
        df_idx[df_idx.apply(
            lambda r: manifest.test_item_per_user.get(r["user_idx"]) == r["item_idx"],
            axis=1,
        )].index
    )
    exclusion = build_exclusion(train_df, manifest)
    excl_path = DATA_DERIVED / "exclusion_items.npz"
    save_exclusion(exclusion, excl_path)
    print(f"Wrote exclusion_items.npz: {len(exclusion)} users.")


if __name__ == "__main__":
    main()
```

### Using the RNG factory from a client

```python
# In Phase-2 federated-baseline-cf/federated_baseline_cf/client_app.py (future work)
# Source: pattern 6 applied to existing train-negative sampling.

from fedrec_foundation.rng import derive_rng

def sample_train_negatives(user_idx, round_num, run_seed, catalog, exclude_items, n_neg):
    """Sample n_neg training negatives for one user. Per-user, per-round, per-purpose RNG."""
    rng = derive_rng(run_seed, user_idx, round_num, purpose="train_neg")
    eligible = [i for i in catalog if i not in exclude_items]
    return rng.sample(eligible, n_neg)  # uses random.Random, fully deterministic
```

### Writing the run manifest at end of training

```python
# In Phase-2 federated-baseline-cf/federated_baseline_cf/server_app.py (future work)
# Source: patterns 7, 8 applied.

from fedrec_foundation.manifest import (
    generate_run_id, build_run_manifest,
    write_manifest_sibling, embed_manifest_in_result,
)
from fedrec_foundation.mode import resolve_mode_defaults, log_mode_and_overrides

def save_final_results(context, run_config, final_metrics, results_dir):
    mode = context.run_config.get("mode", "benchmark_cross_device")
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))
    run_id = generate_run_id()

    # Load split fingerprints.
    from fedrec_foundation.split import load_split_manifest
    split = load_split_manifest(Path("data/derived/split_manifest.json"))

    manifest = build_run_manifest(
        run_id=run_id, mode_profile=profile,
        run_seed=run_config.get("run-seed", 42),
        split_hash=split.split_hash, raw_data_hash=split.raw_data_hash,
        builder_version=split.builder_version,
        overrides=overrides, module="baseline",
    )

    result_path = results_dir / f"{run_id}-results.json"
    result_dict = {"final_metrics": final_metrics}
    embed_manifest_in_result(manifest, result_dict)
    with open(result_path, "w") as f:
        import json; json.dump(result_dict, f, indent=2, sort_keys=True)
    write_manifest_sibling(manifest, result_path)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-module `create_global_mappings()` (4 copies) | One canonical mapping under `data/derived/` | Phase 1 | Single source of truth; cache keys stable across modules |
| `np.random.seed(seed)` / `random.seed(seed)` at process start | `random.Random(hashlib-derived seed)` per (user, round, purpose) | Phase 1 provides; Phases 2-5 consume | Reproducibility across processes; no global RNG reseeding |
| Split recomputed per client | Split computed once, persisted with `split_hash`, verified on subsequent runs | Phase 1 | Bit-exact reproducibility; detects raw-data drift |
| `.embedding_cache/partition_{id}/` | Run-namespaced `.embedding_cache/{run_id}/...` with `split_hash` in key | Phases 2-4 (after Phase 1 provides `split_hash`) | Prevents cross-experiment contamination |
| Implicit "all params global" vs ad-hoc `_LOCAL_PARAMS` | Explicit `weight-policy` enum, logged in manifest | Phase 1 provides; Phases 2-5 wire | Apples-to-apples comparisons across methods |
| Python `hash()` | `hashlib.sha256` | Phase 1 | Cross-process determinism (the Python `hash()` pattern NEVER actually worked across fresh processes; this is a correctness fix, not a style upgrade) |

**Deprecated / outdated:**
- `partition-mode = "dirichlet"` as primary — downgraded to `cross_silo_legacy` mode (D-09). Still reachable, never default.
- `num-supernodes = 5` default — replaced by 6040 in benchmark mode.
- `random.seed(42)` inside evaluators — banned by D-14.

## Open Questions

1. **Should the foundation module's `pyproject.toml` pin `numpy`, `pandas`, `flwr`, `torch`?**
   - What we know: each federated module pins them already.
   - What's unclear: pinning in foundation risks version mismatch if a module upgrades its pin.
   - Recommendation: foundation declares only minimum-version floors matching current modules (`numpy>=1.24.0`, `pandas>=2.0.0`, `flwr>=1.22.0`, `torch>=2.7.1`). No upper bounds. Planner confirms.

2. **Per-user-group classification boundaries — confirm sparse ≤ 30, 30 < medium ≤ 100, dense > 100?**
   - What we know: existing `UserGroupConfig` uses `(0, 30)`, `(30, 100)`, `(100, 10000)` per `federated-adaptive-personalized-cf/.../evaluation/user_groups.py:26-28`.
   - What's unclear: whether the thesis comparison prefers different boundaries.
   - Recommendation: reuse existing config unchanged (CONTEXT.md "Claude's Discretion" confirms this is fine). Boundaries are stored in `split_manifest.json` so future changes are one-edit operations — but lock `split_hash` recomputation to a boundary change via D-04.

3. **Where do the four federated modules' `pyproject.toml` dependency declarations gain `fedrec-foundation`?**
   - What we know: each module currently has its own `[project].dependencies` list.
   - What's unclear: whether we pin the foundation dep as `fedrec-foundation @ file://../scripts/foundation` (PEP 440 direct ref) or rely on `pip install -e` being run manually.
   - Recommendation: use direct-file reference so `pip install -e federated-baseline-cf/` pulls the foundation in automatically. Pattern: `fedrec-foundation @ file:///${PROJECT_ROOT}/scripts/foundation` — requires env-var expansion; alternative is relative path `file://../scripts/foundation` which works in practice but is non-canonical. Planner picks; both need tested.

4. **Will `flwr run .` change `cwd` such that `data/derived/` relative paths break?**
   - What we know: Flower local simulation spawns subprocess supernodes; `cwd` may change.
   - What's unclear: whether to use absolute paths or a `paths.py` helper that walks up to find `data/derived/`.
   - Recommendation: `paths.py` helper that uses `pathlib.Path(__file__).parent` to locate the repo root, then resolves `data/derived/`. This makes foundation work regardless of where `flwr run .` starts supernodes from. Add a `FEDREC_FOUNDATION_DATA_DIR` env-var override for CI / remote runs.

5. **Does hashing `ratings.dat + movies.dat + users.dat` capture everything?**
   - What we know: these are the three ML-1M files in `data/ml-1m/`.
   - What's unclear: whether `README` (present in `data/ml-1m/`) should also be hashed.
   - Recommendation: hash only the three data files; README is non-normative and would needlessly invalidate on harmless edits.

## Validation Architecture

`nyquist_validation: true` in `.planning/config.json` — this section is mandatory.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest` (NOT currently installed — Wave 0 gap) |
| Config file | `scripts/foundation/pyproject.toml` declares `[tool.pytest.ini_options] testpaths = ["tests"]` |
| Quick run command | `cd scripts/foundation && pytest -x tests/` |
| Full suite command | `cd scripts/foundation && pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-01 | `build_mapping()` is deterministic; `load_mapping(save_mapping(x)) == x`; int keys round-trip | unit | `pytest tests/test_mapping.py -x` | ❌ Wave 0 |
| FND-01 | `mapping.json` sorts raw IDs ascending; rebuild is idempotent | unit | `pytest tests/test_mapping.py::test_sort_order -x` | ❌ Wave 0 |
| FND-02 | `build_split()` produces same `split_hash` on two fresh calls over same input | unit | `pytest tests/test_split.py::test_hash_deterministic -x` | ❌ Wave 0 |
| FND-02 | Stable tiebreak on `(user_id, timestamp, movie_id)` — two users with identical timestamps produce same held-out item across runs | unit | `pytest tests/test_split.py::test_timestamp_tiebreak -x` | ❌ Wave 0 |
| FND-02 | `save_split_or_verify` refuses overwrite on hash mismatch (D-04 lock) | unit | `pytest tests/test_split.py::test_split_lock_refuses_overwrite -x` | ❌ Wave 0 |
| FND-03 | `exclusion_for(user)` includes both train positives AND held-out test item | unit | `pytest tests/test_exclusion.py::test_includes_test_item -x` | ❌ Wave 0 |
| FND-03 | `np.load(..., allow_pickle=False)` succeeds on generated NPZ | unit | `pytest tests/test_exclusion.py::test_safe_load -x` | ❌ Wave 0 |
| FND-04 | `get_primary_evaluator("benchmark_cross_device")` returns `"sampled_loo_99"` for all three modes | unit | `pytest tests/test_evaluator.py -x` | ❌ Wave 0 |
| FND-05 | `compute_aggregation_weight({"num_positives": 10}, "num_positives") == 10.0` | unit | `pytest tests/test_weight_policy.py -x` | ❌ Wave 0 |
| FND-05 | Unknown policy raises `ValueError` | unit | `pytest tests/test_weight_policy.py::test_unknown_policy_raises -x` | ❌ Wave 0 |
| FND-06 | `derive_rng(...)` produces same seed across `PYTHONHASHSEED={0, 1, "random"}` in subprocess | integration | `pytest tests/test_rng.py::test_derive_rng_stable_across_processes -x` | ❌ Wave 0 |
| FND-06 | Different `(user_id, round, purpose)` produce different seeds | unit | `pytest tests/test_rng.py::test_tuple_uniqueness -x` | ❌ Wave 0 |
| FND-06 | `random.Random(seed).sample(...)` is reproducible | unit | `pytest tests/test_rng.py::test_sample_reproducible -x` | ❌ Wave 0 |
| FND-07 | `build_run_manifest` populates all D-16 fields | unit | `pytest tests/test_manifest.py::test_all_fields_populated -x` | ❌ Wave 0 |
| FND-07 | `write_manifest_sibling` produces both embedded and sibling JSON | unit | `pytest tests/test_manifest.py::test_both_writes -x` | ❌ Wave 0 |
| FND-07 | `log_mode_and_overrides` handles kebab↔snake conversion | unit | `pytest tests/test_mode.py::test_override_logging -x` | ❌ Wave 0 |
| D-04 lock | Running `build_derived.py` twice is idempotent (no hash change) | integration | `pytest tests/test_integration.py::test_build_idempotent -x` | ❌ Wave 0 |
| D-11 assert | `ModeProfile.assert_one_user_per_client` is True in benchmark/paper-compat, False in legacy | unit | `pytest tests/test_mode.py::test_assertion_flags -x` | ❌ Wave 0 |
| Cross-module import | `import fedrec_foundation` succeeds from any of the four federated module directories | smoke | `for mod in federated-*-cf; do (cd $mod && python -c "import fedrec_foundation"); done` | ❌ Wave 0 |
| End-to-end artifact | `python -m fedrec_foundation.build` creates all three `data/derived/*` files | integration | `pytest tests/test_integration.py::test_build_creates_all_artifacts -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cd scripts/foundation && pytest -x tests/` — should finish in < 10 seconds.
- **Per wave merge:** `cd scripts/foundation && pytest -v tests/` full suite including subprocess-isolation RNG test (< 30 seconds).
- **Phase gate:** Full suite green AND `python -m fedrec_foundation.build` runs clean AND all four federated modules successfully `python -c "import fedrec_foundation"` after `pip install -e` — before `/gsd:verify-work`.

### Wave 0 Gaps

- [ ] `scripts/foundation/pyproject.toml` — hatchling build, `[project.name] = "fedrec-foundation"`, `[tool.pytest.ini_options]` block
- [ ] `scripts/foundation/fedrec_foundation/__init__.py` — version string only
- [ ] `scripts/foundation/tests/conftest.py` — shared fixtures (tiny synthetic ratings_df for fast tests; temp-dir fixtures for file-IO tests)
- [ ] `scripts/foundation/tests/test_mapping.py` — FND-01 behaviors
- [ ] `scripts/foundation/tests/test_split.py` — FND-02 behaviors + D-04 lock
- [ ] `scripts/foundation/tests/test_exclusion.py` — FND-03 behaviors
- [ ] `scripts/foundation/tests/test_evaluator.py` — FND-04 behaviors
- [ ] `scripts/foundation/tests/test_weight_policy.py` — FND-05 behaviors
- [ ] `scripts/foundation/tests/test_rng.py` — FND-06 behaviors incl. cross-process
- [ ] `scripts/foundation/tests/test_mode.py` — D-06 to D-11 behaviors
- [ ] `scripts/foundation/tests/test_manifest.py` — FND-07 behaviors
- [ ] `scripts/foundation/tests/test_integration.py` — end-to-end build + idempotence
- [ ] Framework install: `pip install pytest` (not currently pinned anywhere)
- [ ] `scripts/setup_dev_env.sh` (optional convenience) — documents the `pip install -e scripts/foundation && pip install -e federated-*-cf` install order

## Sources

### Primary (HIGH confidence)

- `.planning/phases/01-foundation-contract/01-CONTEXT.md` — all locked decisions D-01..D-16, Claude's Discretion list, deferred items
- `.planning/REQUIREMENTS.md` §Foundation FND-01..07 — verbatim requirements
- `.planning/ROADMAP.md` §Phase 1 — four success criteria phrased observably
- `.planning/research/PITFALLS.md` §§1, 2, 3, 12, 13, 14, 21, 23, 24 — tested against the exact failure modes Phase 1 is immunizing against
- `.planning/research/FEATURES.md` §P0, §P1 tables — table-stakes features cross-referenced against the FND-01..07 set
- `.planning/research/SUMMARY.md` — build-order implications
- `.planning/research/ARCHITECTURE.md` §Data Layer, §Orchestration Layer — migration deltas consumed by Phase 1
- `.planning/codebase/ARCHITECTURE.md` §Personalization Boundary Matrix, §Data Flow — brownfield existing structure
- `.planning/codebase/CONCERNS.md` — already-catalogued bugs Phase 1 avoids reintroducing
- `.planning/codebase/CONVENTIONS.md` — typing, naming, dataclass patterns, import style
- `.planning/codebase/STACK.md` — dep pins confirmed (Python 3.9+, Flower 1.22+, PyTorch 2.7+)
- `CLAUDE.md` — root project instructions, notation convention `w / theta_i / D_i / K / R / N / C`
- `federated-baseline-cf/federated_baseline_cf/dataset.py:358-428` — direct source for `create_leave_one_out_split` and `create_global_mappings`, lifted with determinism additions
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/user_groups.py:17-62` — direct source for `UserGroupConfig` and `classify_user_group`

### Secondary (MEDIUM confidence)

- Python 3.9 `random.Random` docs — confirmed `random.Random(seed).sample()` is deterministic given seed (Python stdlib behavior is stable)
- NumPy docs for `np.savez` + `np.load(allow_pickle=False)` — confirmed safe-load + keyed-dict access pattern
- `docs/superpowers/plans/2026-04-04-cross-device-migration.md` — prior draft cited only for the `partition-mode` config pattern; explicitly superseded by D-06/D-07

### Tertiary (LOW confidence)

- None — all recommendations are grounded in either locked decisions, existing repo code, or standard-library behavior with cited docs. The only LOW-confidence area is Open Question #3 (direct-file vs PEP 660 editable reference) which the planner will confirm by testing.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — almost entirely stdlib + existing pins
- Architecture: HIGH — decisions locked in CONTEXT.md; sketch code is straight-line
- Pitfalls: HIGH — Pitfalls 1-5 are verified against codebase evidence; Pitfall 6 is a conversion bug that will surface in tests
- Runtime state inventory: HIGH — phase is additive, no pre-existing state to invalidate
- Mode resolver: MEDIUM — locked values for `benchmark_cross_device` are research-recommended defaults; planner should confirm `fraction_train=0.1` and `embedding_dim=64` against THS-01's eventual standardized config
- Validation architecture: HIGH — concrete test commands for each FND-0X requirement, framework Wave 0 gap acknowledged

**Research date:** 2026-04-19
**Valid until:** 2026-05-19 (30 days for stable foundation patterns; revisit if Flower 1.23+ or NumPy 2.x behavior changes)
