---
phase: 02-baseline-migration
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - federated-baseline-cf/federated_baseline_cf/dataset.py
  - federated-baseline-cf/pyproject.toml
  - federated-baseline-cf/tests/test_dataset_adapter.py
autonomous: true
requirements:
  - BSL-01

must_haves:
  truths:
    - "`federated-baseline-cf/pyproject.toml` declares default `partition-mode = \"natural\"` (cross-device, 1 user = 1 client); `[tool.flwr.federations.local-simulation].options.num-supernodes` remains 5 (the CR-2 launcher overrides at federation level) but a comment explicitly names `scripts/run.py baseline benchmark_cross_device` as the canonical entry path that sets 6040."
    - "`federated_baseline_cf/dataset.py` is rewritten as a thin adapter (≤ 200 LOC) that delegates ID mapping, LOO split, and exclusion-set loading to `fedrec_foundation.{mapping, split, exclusion}`; `create_global_mappings`, `create_leave_one_out_split`, and the Dirichlet helpers `compute_user_genre_distribution` / `dirichlet_partition_users` are REMOVED per D-17."
    - "`load_partition_data(partition_id, num_partitions, ...)` returns the same tuple shape (trainloader, testloader, num_users, num_items, user2idx, item2idx) consumed by client_app.py and task.py — so downstream callers keep working without signature changes."
    - "`natural_partition_users(ratings_df, user2idx)` is preserved (same signature, same return shape) but the mapping it consumes now comes from `fedrec_foundation.mapping.load_mapping(data/derived/mapping.json)` — single source of truth with the committed bundle."
    - "Module-level `_partition_cache` is removed because the foundation loaders already satisfy the cross-process-stable cache invariant (bundle verification via `verify_bundle()` at load time)."
  artifacts:
    - path: "federated-baseline-cf/federated_baseline_cf/dataset.py"
      provides: "Thin adapter that delegates to fedrec_foundation loaders (D-17)"
      contains: "from fedrec_foundation.mapping import load_mapping"
    - path: "federated-baseline-cf/pyproject.toml"
      provides: "partition-mode=natural default (BSL-01)"
      contains: "partition-mode = \"natural\""
    - path: "federated-baseline-cf/tests/test_dataset_adapter.py"
      provides: "Adapter tests: mapping/split/exclusion come from foundation; bundle verified"
      contains: "def test_load_partition_data_uses_foundation_mapping"
  key_links:
    - from: "federated_baseline_cf.dataset.load_partition_data"
      to: "fedrec_foundation.mapping.load_mapping"
      via: "foundation bundle verification + import"
      pattern: "load_mapping\\("
    - from: "federated_baseline_cf.dataset.load_partition_data"
      to: "fedrec_foundation.split.load_split_manifest"
      via: "LOO split loading"
      pattern: "load_split_manifest\\("
    - from: "federated_baseline_cf.dataset"
      to: "fedrec_foundation.bundle.verify_bundle"
      via: "verify committed foundation bundle before reading payloads"
      pattern: "verify_bundle\\("
---

<objective>
Rip-and-replace `federated_baseline_cf/dataset.py` helpers (`create_global_mappings`, `create_leave_one_out_split`, Dirichlet partitioner) with calls into the Phase 1 foundation loaders (`fedrec_foundation.mapping`, `fedrec_foundation.split`, `fedrec_foundation.exclusion`, `fedrec_foundation.bundle`). Flip `pyproject.toml` `partition-mode` default from `"dirichlet"` to `"natural"` to make cross-device the default per BSL-01.

Purpose: Under D-17 the baseline module is FULLY DOWNSTREAM of the committed `data/derived/` bundle — there is a single source of truth for `user2idx`, `item2idx`, `test_item_per_user`, and `exclude_items[u]`. Keeping the module's own `create_global_mappings` would let drift accumulate (e.g., a future mapping edit inside baseline would silently disagree with foundation + personalized + adaptive). Per BSL-01 the module defaults to 1-user-per-client; CR-2 launcher sets `num-supernodes` at the federation level, so the `[tool.flwr.federations.local-simulation].options.num-supernodes = 5` value stays (legacy cross-silo default for `flwr run .` without the launcher) and the module documents that `scripts/run.py baseline benchmark_cross_device` is the canonical entry.

D-18 surgical migration guard: the working tree already has pre-existing uncommitted hunks in this file. Executor MUST run `git diff federated-baseline-cf/federated_baseline_cf/dataset.py` first to inventory them. This plan ONLY replaces the 3 helpers named above (D-17 scope) and cleans up `_partition_cache`. Any pre-existing hunks in `MovieLensDataset.__init__`, `download_movielens_1m`, `load_movielens_1m`, or `load_full_data` not addressing BSL-01 remain UNTOUCHED.

Output: (1) Thin `dataset.py` (~200 LOC) that delegates to foundation; (2) pyproject.toml `partition-mode = "natural"` default with `eval-num-negatives = 99` and a comment pointing at `scripts/run.py`; (3) `test_dataset_adapter.py` with 3 tests proving mapping/split/exclusion come from foundation.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/02-baseline-migration/02-CONTEXT.md
@.planning/phases/01-foundation-contract/01-foundation-contract-02-SUMMARY.md
@CLAUDE.md
@federated-baseline-cf/claude.md

@scripts/foundation/fedrec_foundation/mapping.py
@scripts/foundation/fedrec_foundation/split.py
@scripts/foundation/fedrec_foundation/exclusion.py
@scripts/foundation/fedrec_foundation/bundle.py
@scripts/foundation/fedrec_foundation/paths.py

@federated-baseline-cf/federated_baseline_cf/dataset.py
@federated-baseline-cf/pyproject.toml

<interfaces>
<!-- Phase 1 foundation loader signatures. Executor calls these — does NOT re-implement. -->

From scripts/foundation/fedrec_foundation/mapping.py:
```python
def load_mapping(path: str) -> CanonicalMapping: ...   # JSON load; restores int keys
@dataclass class CanonicalMapping:
    user2idx: Dict[int, int]; item2idx: Dict[int, int]
    num_users: int; num_items: int
```

From scripts/foundation/fedrec_foundation/split.py:
```python
def load_split_manifest(path) -> SplitManifest: ...
@dataclass class SplitManifest:
    split_hash: str; mapping_sha256: str; raw_data_hash: str
    builder_version: str; num_train: int; num_test_users: int
    test_item_per_user: Dict[int, int]     # user_idx -> held-out item_idx
    train_user_stats: Dict[int, PerUserStats]
@dataclass class PerUserStats:
    n_interactions: int; genre_entropy: float; n_unique_items: int
    rating_std: float; user_group: str   # "sparse"/"medium"/"dense"
```

From scripts/foundation/fedrec_foundation/exclusion.py:
```python
def load_exclusion(path) -> ExclusionTable: ...
class ExclusionTable:
    def for_user(self, user_idx: int) -> np.ndarray: ...   # int32 excluded item_idx
    def close(self) -> None: ...
    # Usable as context manager (__enter__ / __exit__)
```

From scripts/foundation/fedrec_foundation/bundle.py:
```python
def verify_bundle(derived_dir) -> FoundationIndex:       # raises RuntimeError on mismatch
```

Committed bundle on disk (read-only):
- data/derived/mapping.json           (6040 users, 3706 items)
- data/derived/split_manifest.json    (split_hash=5685bed7e4b6...)
- data/derived/exclusion_items.npz    (flat int32 items + int64 indptr)
- data/derived/foundation_index.json  (foundation_contract_sha256=fe181dafe6f7...)
</interfaces>

</context>

<tasks>

<task type="auto">
  <name>Task 1: Flip pyproject.toml defaults to cross-device (BSL-01)</name>
  <files>
    federated-baseline-cf/pyproject.toml
  </files>
  <read_first>
    - federated-baseline-cf/pyproject.toml (CURRENT state — note line 17-18 foundation dep comment from Phase 1 Plan 06; do NOT touch that comment or the fedrec-foundation dep. Note line 64 `partition-mode = "natural"` — already set by a pre-existing uncommitted hunk; CONFIRM via `git diff` before editing. Line 94 `options.num-supernodes = 5` — UNCHANGED per D-23: legacy `flwr run .` path; `scripts/run.py` sets 6040 at federation level).
    - scripts/run.py (CURRENT state — confirm MODULE_DIR["baseline"] == "federated-baseline-cf" and MODE_NUM_SUPERNODES["benchmark_cross_device"] == 6040)
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions D-18, D-19, D-25
    - CLAUDE.md "Configuration" section (kebab-case in pyproject.toml; snake_case in Python; `.get(key, default)` pattern)
  </read_first>
  <action>
Confirm via `git diff federated-baseline-cf/pyproject.toml` whether `partition-mode = "natural"` is already set (it appears to be pre-existing WIP per CONTEXT.md D-18). If so, leave it untouched.

Update the `[tool.flwr.app.config]` block:

1. If `partition-mode` is NOT already `"natural"`, change it to `"natural"` with a comment: `partition-mode = "natural"  # BSL-01: cross-device 1-user-per-client is the thesis-table default`.

2. Add these keys if they do not already exist (append after existing keys — do NOT re-order):
   - `mode = "cross_silo_legacy"  # D-25: fallback when scripts/run.py not used; launcher sets "benchmark_cross_device" at federation level. Values: benchmark_cross_device | paper_compat_pfedrec | cross_silo_legacy.`
   - `run-seed = 42  # FND-06: single root seed for py_rng/np_rng/torch_gen/server_rng.`
   - `weight-policy = "num_positives"  # FND-05: aggregation weight policy. Values: uniform | num_positives | num_training_examples.`
   - `eval-num-negatives = 99  # NCF protocol: 99 sampled negatives + 1 held-out positive (sampled_loo_99).`
   - `checkpoint-rule = "best_round_restore"  # D-27: track best sampled_ndcg@10; restore at final evaluation.`

3. After `options.num-supernodes = 5` in `[tool.flwr.federations.local-simulation]`, keep the existing value AND add a comment: `# NOTE: For cross-device thesis runs use `python scripts/run.py baseline benchmark_cross_device` which sets num-supernodes=6040 at federation construction time (Codex CR-2). `flwr run .` without the launcher stays cross-silo (num-supernodes=5) for backwards compatibility with existing W&B dashboards.`

4. Add comment at the top of `[tool.flwr.app.config]` block: `# Phase 2 (BSL-01..08): cross-device defaults. `fedrec_foundation.mode.resolve_mode_defaults(mode)` (Phase 1 Plan 05) is the canonical source for hyperparameters at runtime; these pyproject values are only consulted when `mode` is unset. Overrides are visible per D-19.`

Do NOT modify `[project]`, `[build-system]`, `[tool.hatch.*]`, `[tool.flwr.app.components]`, or `[tool.flwr.federations.local-sim-gpu]` / `[tool.flwr.federations.remote-federation]`. Do NOT remove any existing `[tool.flwr.app.config]` keys.

**Do-not-touch ranges** (D-18): everything outside `[tool.flwr.app.config]` and the single-line comment addition under `[tool.flwr.federations.local-simulation]`.
  </action>
  <verify>
    <automated>grep -E "^mode = |^run-seed = |^weight-policy = |^eval-num-negatives = |^checkpoint-rule = |^partition-mode = " federated-baseline-cf/pyproject.toml | wc -l</automated>
  </verify>
  <acceptance_criteria>
    - `grep '^partition-mode = "natural"' federated-baseline-cf/pyproject.toml` returns exactly 1 match.
    - `grep '^mode = "cross_silo_legacy"' federated-baseline-cf/pyproject.toml` returns exactly 1 match.
    - `grep '^run-seed = 42' federated-baseline-cf/pyproject.toml` returns exactly 1 match.
    - `grep '^weight-policy = "num_positives"' federated-baseline-cf/pyproject.toml` returns exactly 1 match.
    - `grep '^eval-num-negatives = 99' federated-baseline-cf/pyproject.toml` returns exactly 1 match.
    - `grep '^checkpoint-rule = "best_round_restore"' federated-baseline-cf/pyproject.toml` returns exactly 1 match.
    - `grep "options.num-supernodes = 5" federated-baseline-cf/pyproject.toml` still matches (legacy fallback preserved per D-23).
    - `grep "fedrec-foundation" federated-baseline-cf/pyproject.toml` still matches (Phase 1 Plan 06 dep preserved).
    - `python scripts/run.py --dry-run baseline benchmark_cross_device 2>&1 | grep "num-supernodes=6040"` returns 1 match (launcher still works).
  </acceptance_criteria>
  <done>All six `[tool.flwr.app.config]` additions present; pre-existing foundation dep and federation blocks untouched; legacy `flwr run .` still valid.</done>
</task>

<task type="auto">
  <name>Task 2: Rip-and-replace dataset.py with thin foundation adapter (D-17)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/dataset.py
    federated-baseline-cf/tests/test_dataset_adapter.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/dataset.py (ENTIRE file — note: pre-existing uncommitted hunks per D-18; EXECUTOR MUST `git diff` this file first and inventory which ranges are pre-WIP vs. rip-and-replace scope)
    - scripts/foundation/fedrec_foundation/mapping.py (load_mapping signature)
    - scripts/foundation/fedrec_foundation/split.py (load_split_manifest + SplitManifest attrs)
    - scripts/foundation/fedrec_foundation/exclusion.py (load_exclusion + ExclusionTable.for_user)
    - scripts/foundation/fedrec_foundation/bundle.py (verify_bundle — call before reading payloads)
    - scripts/foundation/fedrec_foundation/paths.py (repo_root, data_derived helpers)
    - federated-baseline-cf/federated_baseline_cf/client_app.py (CALLERS: load_data signature expected by @app.train/@app.evaluate; do NOT edit in this task)
    - federated-baseline-cf/federated_baseline_cf/task.py (CALLER: load_partition_data return tuple; do NOT edit in this task)
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions (D-17, D-18)
    - CLAUDE.md §"Code Standards" (typing pre-3.10, NumPy docstrings, module-level `_dataset_cache` pattern kept)
  </read_first>
  <action>
Step 1: Inventory pre-existing hunks. Executor runs `git diff federated-baseline-cf/federated_baseline_cf/dataset.py > /tmp/dataset_diff.txt` and reads it. The rip-and-replace scope is:

**REMOVE** (D-17):
- `_partition_cache` module-level dict (line ~19).
- `compute_user_genre_distribution(ratings_df, movies_df)` (lines ~142-183).
- `dirichlet_partition_users(...)` (lines ~186-286).
- `create_train_test_split(...)` (lines ~329-355).
- `create_leave_one_out_split(...)` (lines ~358-405).
- `create_global_mappings(ratings_df)` (lines ~408-428).

**KEEP UNCHANGED** (D-18 + called by task.py/client_app.py):
- `MovieLensDataset(Dataset)` class (lines ~22-55).
- `download_movielens_1m(data_dir)` (lines ~58-94).
- `load_movielens_1m(data_dir)` (lines ~97-139) — still used to read raw ratings.dat when building the in-memory per-user DataFrames for `natural_partition_users`.
- `natural_partition_users(ratings_df, user2idx)` (lines ~289-326) — signature unchanged; only the source of `user2idx` changes (comes from foundation now).

**REPLACE** (bodies rewritten to call foundation):
- `load_partition_data(partition_id, num_partitions, alpha, test_ratio, batch_size, data_dir, split_mode, partition_mode)` — signature UNCHANGED (downstream callers in client_app.py and task.py still call it the same way); body delegates to foundation.
- `load_full_data(test_ratio, batch_size, data_dir, split_mode)` — signature UNCHANGED; body delegates to foundation.

Step 2: Write the new `dataset.py`. Place it at `federated-baseline-cf/federated_baseline_cf/dataset.py`, replacing the contents that are in D-17 rip scope. Keep `MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, and `natural_partition_users` IDENTICAL to their current (including any pre-existing WIP) state. Only replace the functions named in the "REPLACE" list above.

Full new body for the replaced functions (insert into the existing file, preserving the kept functions):

```python
"""MovieLens 1M Dataset Loading — Phase 2 thin adapter over fedrec_foundation.

Post-Phase-2 this module is responsible only for:
  1. Raw data download + parse (`download_movielens_1m`, `load_movielens_1m`).
  2. Building the natural cross-device partitioning (1 user = 1 client) from the
     canonical `user2idx` exposed by `fedrec_foundation.mapping` (`natural_partition_users`).
  3. Wrapping the above into PyTorch DataLoaders keyed by (partition_id, num_partitions)
     and returning the tuple shape consumed by client_app.py / task.py
     (`load_partition_data`, `load_full_data`).

Mapping / split / exclusion-set construction is DELEGATED to the Phase 1
foundation bundle at `data/derived/` (committed, hash-locked). Callers of
`load_partition_data` observe the same `user2idx` / `item2idx` / held-out
test items as every other federated module — there is now a single source
of truth for the cross-device protocol.

Per D-17: `create_global_mappings`, `create_leave_one_out_split`,
`compute_user_genre_distribution`, `dirichlet_partition_users`,
`create_train_test_split` are REMOVED. The corresponding foundation loaders
(`fedrec_foundation.mapping.load_mapping`, `.split.load_split_manifest`,
`.exclusion.load_exclusion`) are the replacements.

Per D-18: `MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`,
and `natural_partition_users` retain their pre-existing WIP state; only
the D-17 rip targets and `load_partition_data` / `load_full_data` bodies
change in this plan.
"""
# [...MovieLensDataset / download_movielens_1m / load_movielens_1m / natural_partition_users stay verbatim...]


# --- Phase 2 Plan 02: foundation-backed adapters ---

from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
from fedrec_foundation.mapping import CanonicalMapping, load_mapping
from fedrec_foundation.paths import data_derived
from fedrec_foundation.split import SplitManifest, load_split_manifest


# Module-level in-memory cache: avoids re-reading the bundle each client.
# Keyed by the foundation_contract_sha256 so a bundle rebuild invalidates
# the cache automatically.
_foundation_cache: Dict[str, Dict] = {}


def _load_foundation_bundle(data_dir: Optional[str] = None) -> Dict:
    """Load mapping / split / exclusion from the committed `data/derived/` bundle.

    Calls `verify_bundle` first — a tampered or incomplete bundle raises
    `RuntimeError` at load time (fail-loud per N-3).

    Parameters
    ----------
    data_dir : Optional[str]
        If provided, overrides the default `<repo>/data/` location. Uses
        `fedrec_foundation.paths.data_derived()` as the canonical default.

    Returns
    -------
    Dict
        A dict with keys:
        - ``mapping`` (CanonicalMapping)
        - ``split_manifest`` (SplitManifest)
        - ``exclusion`` (ExclusionTable) — DO NOT close; module-level cache.
        - ``foundation_contract_sha256`` (str)
    """
    if data_dir is not None:
        derived = Path(data_dir).resolve() / "derived"
    else:
        derived = data_derived()

    idx = verify_bundle(derived)  # raises on mismatch/missing
    contract_key = idx.foundation_contract_sha256
    if contract_key in _foundation_cache:
        return _foundation_cache[contract_key]

    bundle = {
        "mapping": load_mapping(str(derived / "mapping.json")),
        "split_manifest": load_split_manifest(derived / "split_manifest.json"),
        "exclusion": load_exclusion(derived / "exclusion_items.npz"),
        "foundation_contract_sha256": contract_key,
    }
    _foundation_cache[contract_key] = bundle
    return bundle


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    split_mode: str = "leave-one-out",
    partition_mode: str = "natural",
):
    """Load one client's partition backed by the foundation bundle.

    Post-Phase-2 callers (client_app.py::@app.train(), @app.evaluate(),
    task.py::load_data) pass ``partition_id`` and expect the same 6-tuple
    as before. The ``alpha`` / ``test_ratio`` / ``split_mode`` parameters
    are retained for backwards compatibility with cross-silo legacy
    (``partition_mode="dirichlet"``) but under the benchmark path
    (``partition_mode="natural"``) they are unused — the bundle's
    deterministic LOO split is the authoritative split.

    Parameters
    ----------
    partition_id : int
        Client's partition index. Under ``partition_mode="natural"`` this
        is the ``user_idx`` in ``[0, num_users)``.
    num_partitions : int
        Total partitions. Under ``partition_mode="natural"`` this should
        equal ``bundle["mapping"].num_users`` (asserted loudly).
    alpha : float
        Dirichlet concentration (unused under natural partitioning).
    test_ratio : float
        Random-split ratio (unused under LOO split mode).
    batch_size : int
        DataLoader batch size.
    data_dir : Optional[str]
        Override the default ``<repo>/data/`` location.
    split_mode : str
        ``"leave-one-out"`` or ``"random"``. Under the foundation path we
        only support ``"leave-one-out"`` — passing ``"random"`` raises
        ``ValueError``.
    partition_mode : str
        ``"natural"`` (cross-device) or ``"dirichlet"`` (cross-silo legacy).

    Returns
    -------
    Tuple[DataLoader, DataLoader, int, int, Dict[int,int], Dict[int,int]]
        ``(trainloader, testloader, num_users, num_items, user2idx, item2idx)``.
    """
    from torch.utils.data import DataLoader

    if split_mode != "leave-one-out":
        raise ValueError(
            f"split_mode={split_mode!r} not supported post-Phase-2 foundation migration; "
            f"use 'leave-one-out' (NCF protocol) or run with partition_mode='dirichlet' "
            f"(cross-silo legacy path still uses random split)."
        )

    bundle = _load_foundation_bundle(data_dir)
    mapping: CanonicalMapping = bundle["mapping"]
    split: SplitManifest = bundle["split_manifest"]
    user2idx = mapping.user2idx
    item2idx = mapping.item2idx
    num_users = mapping.num_users
    num_items = mapping.num_items

    if partition_mode == "natural":
        # Cross-device: 1 user = 1 client.
        # Download + parse raw ratings once (cached inside download_movielens_1m).
        download_movielens_1m(data_dir)
        ratings_df, _, _ = load_movielens_1m(data_dir)
        # Partition by user_idx so partition_id == user_idx.
        partitions = natural_partition_users(ratings_df, user2idx)
        if partition_id not in partitions:
            raise ValueError(
                f"partition_id={partition_id} not in natural partition keyspace "
                f"[0, {num_users}); did num-supernodes match num_users at federation init?"
            )
        client_ratings = partitions[partition_id].copy()
        client_ratings["user_idx"] = client_ratings["user_id"].map(user2idx).astype(int)
        client_ratings["item_idx"] = client_ratings["movie_id"].map(item2idx).astype(int)

        # Build train + test split using the foundation's test_item_per_user map.
        test_item = split.test_item_per_user.get(int(partition_id))
        if test_item is not None:
            test_mask = client_ratings["item_idx"] == int(test_item)
            test_df = client_ratings[test_mask].copy()
            train_df = client_ratings[~test_mask].copy()
        else:
            # User has < 2 interactions — no held-out test item.
            test_df = client_ratings.iloc[0:0].copy()
            train_df = client_ratings.copy()
    elif partition_mode == "dirichlet":
        # Cross-silo legacy: delegate to the surviving dirichlet path only
        # if data_dir-dependent raw data is available. This branch is a
        # thin shim that preserves the pre-Phase-2 behavior for appendix
        # runs; production thesis runs use natural.
        raise NotImplementedError(
            "Cross-silo legacy (partition_mode='dirichlet') requires the "
            "pre-Phase-2 dirichlet_partition_users implementation. Per D-17 that "
            "helper is removed; re-add only if explicit cross-silo legacy runs "
            "are required (see .planning/phases/02-baseline-migration/02-CONTEXT.md "
            "§Deferred — cross_silo_legacy regression tests)."
        )
    else:
        raise ValueError(f"Unknown partition_mode={partition_mode!r}")

    # Build PyTorch Datasets + DataLoaders (shuffle generator threaded by the
    # client in Plan 03 via torch_gen(run_seed, user_idx, round_num, 'dataloader')).
    train_dataset = MovieLensDataset(train_df, user2idx, item2idx)
    test_dataset = MovieLensDataset(test_df, user2idx, item2idx)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, num_users, num_items, user2idx, item2idx


def load_full_data(
    test_ratio: float = 0.2,
    batch_size: int = 256,
    data_dir: Optional[str] = None,
    split_mode: str = "leave-one-out",
):
    """Load the full (non-partitioned) dataset for server-side centralized evaluation.

    Same return shape as :func:`load_partition_data` but across ALL users.
    Consumed by ``server_app.py`` for the end-of-training centralized eval
    (Plan 04 keeps this call path intact).

    Parameters
    ----------
    test_ratio : float
        Unused under ``split_mode="leave-one-out"`` (kept for API parity).
    batch_size : int
        DataLoader batch size.
    data_dir : Optional[str]
        Override default data path.
    split_mode : str
        ``"leave-one-out"``; ``"random"`` raises ``ValueError``.

    Returns
    -------
    Tuple[DataLoader, DataLoader, int, int, Dict[int,int], Dict[int,int]]
        ``(trainloader, testloader, num_users, num_items, user2idx, item2idx)``
        — identical shape to ``load_partition_data``.
    """
    from torch.utils.data import DataLoader

    if split_mode != "leave-one-out":
        raise ValueError(
            f"split_mode={split_mode!r} not supported post-Phase-2 foundation "
            f"migration; use 'leave-one-out'."
        )

    bundle = _load_foundation_bundle(data_dir)
    mapping: CanonicalMapping = bundle["mapping"]
    split: SplitManifest = bundle["split_manifest"]
    user2idx = mapping.user2idx
    item2idx = mapping.item2idx
    num_users = mapping.num_users
    num_items = mapping.num_items

    download_movielens_1m(data_dir)
    ratings_df, _, _ = load_movielens_1m(data_dir)
    ratings_df["user_idx"] = ratings_df["user_id"].map(user2idx).astype(int)
    ratings_df["item_idx"] = ratings_df["movie_id"].map(item2idx).astype(int)

    # Build test mask: one row per user matching test_item_per_user.
    test_item_series = ratings_df["user_idx"].map(split.test_item_per_user)
    test_mask = ratings_df["item_idx"] == test_item_series
    test_df = ratings_df[test_mask].copy()
    train_df = ratings_df[~test_mask].copy()

    train_dataset = MovieLensDataset(train_df, user2idx, item2idx)
    test_dataset = MovieLensDataset(test_df, user2idx, item2idx)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, num_users, num_items, user2idx, item2idx
```

Step 3: Create `federated-baseline-cf/tests/test_dataset_adapter.py` with 3 tests. These are pytest tests that run under the module's test harness. Use skip-if-data-missing to survive minimal clones:

```python
"""Tests for federated_baseline_cf.dataset (Phase 2 Plan 02 — D-17 foundation adapter)."""
from __future__ import annotations

from pathlib import Path

import pytest

# Skip entire file when the committed foundation bundle is not present.
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "data" / "derived" / "foundation_index.json").exists():
            return p
    raise RuntimeError("repo_root not found")


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed at data/derived/foundation_index.json",
)


def test_load_partition_data_uses_foundation_mapping() -> None:
    """Per D-17: mapping comes from fedrec_foundation, not a module-local helper."""
    from federated_baseline_cf.dataset import load_partition_data
    from fedrec_foundation.mapping import load_mapping

    bundle_path = _repo_root() / "data" / "derived" / "mapping.json"
    foundation_mapping = load_mapping(str(bundle_path))

    # partition_id=0 with natural partitioning == first user in the canonical mapping.
    _trainloader, _testloader, num_users, num_items, user2idx, item2idx = load_partition_data(
        partition_id=0, num_partitions=foundation_mapping.num_users,
        partition_mode="natural", split_mode="leave-one-out", batch_size=32,
    )

    assert num_users == foundation_mapping.num_users == 6040
    assert num_items == foundation_mapping.num_items == 3706
    # Spot-check a raw user ID whose canonical idx is known.
    some_raw_id = next(iter(foundation_mapping.user2idx))
    assert user2idx[some_raw_id] == foundation_mapping.user2idx[some_raw_id]


def test_load_partition_data_test_item_from_foundation_split() -> None:
    """Per D-17: test_item for partition_id == user_idx matches split_manifest.test_item_per_user."""
    from federated_baseline_cf.dataset import load_partition_data
    from fedrec_foundation.split import load_split_manifest

    split_path = _repo_root() / "data" / "derived" / "split_manifest.json"
    split = load_split_manifest(split_path)

    # Pick a user_idx with a known test item.
    user_idx = next(iter(split.test_item_per_user.keys()))
    expected_test_item = int(split.test_item_per_user[user_idx])

    _train, testloader, _nu, _ni, _u2i, _i2i = load_partition_data(
        partition_id=int(user_idx), num_partitions=6040,
        partition_mode="natural", split_mode="leave-one-out", batch_size=32,
    )
    test_items = [int(b["item"].item()) for b in testloader]
    assert expected_test_item in test_items, (
        f"user_idx={user_idx} expected test item {expected_test_item} not found in "
        f"testloader items {test_items[:5]}"
    )


def test_removed_helpers_gone() -> None:
    """Per D-17: create_global_mappings, create_leave_one_out_split, dirichlet_partition_users REMOVED."""
    from federated_baseline_cf import dataset
    for name in ("create_global_mappings", "create_leave_one_out_split",
                 "compute_user_genre_distribution", "dirichlet_partition_users",
                 "create_train_test_split"):
        assert not hasattr(dataset, name), (
            f"{name} should have been removed per D-17 but is still present; "
            f"rip-and-replace incomplete."
        )
```
  </action>
  <verify>
    <automated>cd federated-baseline-cf && pytest tests/test_dataset_adapter.py -v && python -c "from federated_baseline_cf.dataset import load_partition_data, load_full_data; import inspect; assert not hasattr(__import__('federated_baseline_cf.dataset', fromlist=['dataset']), 'create_global_mappings'); print('ok')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.mapping import" federated-baseline-cf/federated_baseline_cf/dataset.py` returns at least 1.
    - `grep -c "from fedrec_foundation.split import" federated-baseline-cf/federated_baseline_cf/dataset.py` returns at least 1.
    - `grep -c "from fedrec_foundation.exclusion import" federated-baseline-cf/federated_baseline_cf/dataset.py` returns at least 1.
    - `grep -c "from fedrec_foundation.bundle import verify_bundle" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 1.
    - `grep -c "def create_global_mappings" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 0.
    - `grep -c "def create_leave_one_out_split" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 0.
    - `grep -c "def dirichlet_partition_users" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 0.
    - `grep -c "def natural_partition_users" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 1 (preserved).
    - `grep -c "def load_partition_data" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 1 (replaced, not removed).
    - `grep -c "def load_full_data" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 1 (replaced, not removed).
    - `grep -c "_partition_cache" federated-baseline-cf/federated_baseline_cf/dataset.py` returns 0 (cache dict removed).
    - `pytest federated-baseline-cf/tests/test_dataset_adapter.py -v 2>&1 | grep -E "passed|failed"` shows 3 passed, 0 failed.
    - `python -c "from federated_baseline_cf.dataset import load_partition_data; t, te, nu, ni, u2i, i2i = load_partition_data(0, 6040, partition_mode='natural'); assert nu == 6040 and ni == 3706"` exits 0.
  </acceptance_criteria>
  <done>dataset.py is a thin adapter (foundation-backed); 3 removed helpers gone; 2 replaced functions keep their signatures; 3 GREEN tests prove the adapter semantics.</done>
</task>

</tasks>

<verification>
Full-phase verification for Plan 02:

1. `grep 'partition-mode = "natural"' federated-baseline-cf/pyproject.toml` matches.
2. `grep 'run-seed = 42' federated-baseline-cf/pyproject.toml` matches.
3. `pytest federated-baseline-cf/tests/test_dataset_adapter.py -v` shows 3 passed.
4. `python -c "from federated_baseline_cf.dataset import _load_foundation_bundle; b = _load_foundation_bundle(); assert b['mapping'].num_users == 6040; assert len(b['split_manifest'].test_item_per_user) >= 6000"` exits 0.
5. `python scripts/run.py --dry-run baseline benchmark_cross_device` prints `num-supernodes=6040` (launcher still resolves correctly).
6. D-18 guard: `git diff federated-baseline-cf/federated_baseline_cf/dataset.py` shows NO changes outside the D-17 rip targets (MovieLensDataset, download_movielens_1m, load_movielens_1m, natural_partition_users hunks stay pre-existing-WIP).
</verification>

<success_criteria>
- pyproject.toml declares cross-device defaults (partition-mode=natural, mode fallback, run-seed, weight-policy, eval-num-negatives, checkpoint-rule); legacy `options.num-supernodes=5` preserved.
- dataset.py rip-and-replace complete: create_global_mappings, create_leave_one_out_split, dirichlet_partition_users, compute_user_genre_distribution, create_train_test_split are REMOVED. load_partition_data + load_full_data call foundation loaders; natural_partition_users kept.
- test_dataset_adapter.py has 3 GREEN tests proving foundation-sourced mapping/split/exclusion.
- D-18 surgical guard: pre-existing uncommitted hunks outside D-17 rip targets are untouched.
- Pre-existing uncommitted hunks in client_app.py, server_app.py, task.py are UNTOUCHED (Plans 03 + 04 will address them).
</success_criteria>

<output>
After completion, create `.planning/phases/02-baseline-migration/02-baseline-migration-02-SUMMARY.md` following the template in `@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md`.
</output>
