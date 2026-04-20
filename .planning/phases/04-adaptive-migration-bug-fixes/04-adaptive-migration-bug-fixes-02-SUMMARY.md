---
phase: 04-adaptive-migration-bug-fixes
plan: 02
subsystem: infra
tags: [pyproject-toml, dataset-adapter, cross-device, num-supernodes, partition-mode, fedrec-foundation, rip-and-replace, pytest-dev-dep, schema-version-2-keys, adp-01, d-02, d-09, d-10, d-11, d-12, d-17, d-18, wave-1]
requirements: [ADP-01]
dependency-graph:
  requires:
    - phase-01 foundation bundle (mapping.json, split_manifest.json, exclusion_items.npz) committed at data/derived/
    - phase-03 plan-02 pyproject.toml + dataset.py template (pattern cloned verbatim with adaptive-specific extensions)
  provides:
    - ADP-01 closed in-file: flwr run . inside federated-adaptive-personalized-cf/ defaults to cross-device (6040 supernodes) under the full thesis benchmark config
    - Schema_version=2 signature-driving defaults (5 keys) as pyproject defaults so Plan 03's cache manifest round-trips the benchmark config without relying on scripts/run.py
    - Thin foundation-backed dataset.py adapter that Plan 03 (client_app.py + task.py) consumes without signature changes; 7-tuple return preserved including user_stats sourced from SplitManifest.train_user_stats
    - 9 GREEN pytest regression tests (5 pyproject shape + 4 dataset adapter) anchoring the Phase-4 contract
  affects:
    - federated-adaptive-personalized-cf/pyproject.toml: 6 Phase-3 keys + 5 Phase-4 defaults flipped + [dev] pytest extra + both federation blocks flipped to 6040
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py: rip-and-replace — 7 module-local helpers removed + _partition_cache removed + foundation adapter in place; 4 D-18 WIP functions preserved verbatim
tech-stack:
  added:
    - pytest>=7.0 (as dev optional-dependency)
  patterns:
    - Clone of Phase 3 Plan 02 foundation-adapter pattern with adaptive-specific 5 schema-v2 signature keys and 7-tuple return preservation
    - D-02 tightening (Phase-3 pattern): NotImplementedError raised at BOTH load_partition_data AND load_full_data
    - PerUserStats -> user_stats dict translation (field-for-field, lossless; CR-5 train-only semantics preserved)
key-files:
  created:
    - federated-adaptive-personalized-cf/tests/test_pyproject_shape.py (5 tests)
    - federated-adaptive-personalized-cf/tests/test_dataset_adapter.py (4 tests)
  modified:
    - federated-adaptive-personalized-cf/pyproject.toml (~30 LOC: 6 Phase-3 carry-forward + 5 Phase-4 defaults + [dev] extra + 2 federation flips)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py (rip-and-replace: 693 -> 473 LOC; 7 D-17 helpers + _partition_cache removed; foundation-backed adapter implemented; 4 D-18 functions preserved verbatim)
decisions:
  - Cloned Phase 3 Plan 02 foundation-adapter pattern verbatim with adaptive-specific extensions (7-tuple return, user_stats dict sourced from SplitManifest.train_user_stats)
  - Preserved the 7-tuple return shape of load_partition_data / load_full_data (the adaptive module uses compute_stats=True to receive user_stats; 3-tuple load_data wrapper in task.py unchanged)
  - user_stats dict keys (n_interactions, genre_entropy, n_unique_items, rating_std, user_group) sourced from PerUserStats via a 4-line field-rename helper _per_user_stats_to_dict; compute_client_alpha / compute_per_user_alpha in task.py receive the same field names they expected pre-migration
  - Added 8th rip target (compute_user_stats, compute_partition_user_stats) beyond the 5-6 Phase-3 Plan 02 list; the adaptive module had 2 extra helpers that are now replaced by foundation CR-5 train-only stats
  - Added defensive fallback for users absent from SplitManifest.train_user_stats (single-interaction users); emits zero-entropy sparse stats so compute_client_alpha degrades gracefully
metrics:
  duration: "7m 23s"
  completed-date: "2026-04-20"
  tasks-completed: 2
  tests-green: 9
  files-touched: 4
commits:
  - "9b9d1f8 — feat(04-02): adaptive pyproject cross-device defaults + schema-v2 keys + dev dep (ADP-01)"
  - "eafcafc — refactor(04-02): rip-and-replace dataset.py as foundation adapter (D-17, D-02 both entry points, ADP-01)"
---

# Phase 04 Plan 02: Adaptive pyproject cross-device defaults + dataset.py foundation adapter Summary

**One-liner:** Flipped federated-adaptive-personalized-cf/pyproject.toml defaults to cross-device (6040 supernodes in both federation blocks) with 6 Phase-3 carry-forward contract keys + 5 Phase-4 schema-v2 signature-driving thesis defaults + pytest dev dep, AND rip-and-replaced dataset.py as a thin fedrec_foundation adapter delegating mapping/split/exclusion/user-stats to the committed foundation bundle — closing ADP-01 in-file while preserving the adaptive-module's 7-tuple return contract.

## What Shipped

### Task 1: pyproject.toml cross-device defaults (commit `9b9d1f8`)

Four edits to `federated-adaptive-personalized-cf/pyproject.toml`:

1. **Both federation blocks flipped** `options.num-supernodes = 5` → `6040` (local-simulation + local-sim-gpu). Added cross-silo-opt-in comment block referencing pre-Phase-4 commits and the fact that `partition-mode=dirichlet` raises NotImplementedError at the dataset layer (D-02).

2. **Added 6 Phase-3 carry-forward contract keys** as a new block above the Early Stopping section:
   - `mode = "cross_silo_legacy"` (launcher flips to `benchmark_cross_device`)
   - `run-seed = 42`
   - `weight-policy = "num_positives"`
   - `eval-num-negatives = 99` (consolidated — removed duplicate declaration from Evaluation protocol section to avoid TOML duplicate-key error)
   - `checkpoint-rule = "best_round_restore"`
   - `reuse-cache = false` (D-09)

3. **Flipped 5 Phase-4 signature-driving keys** to the locked CONTEXT D-09..D-12 benchmark defaults:
   - `alpha-method`: `"multi_factor"` → `"hierarchical_conditional"` (D-10)
   - `enable-per-user-alpha`: `false` → `true` (D-03 + D-12)
   - `enable-item-perturbation`: `false` → `true` (D-03 + D-12)
   - `contrastive-lambda`: `0.0` → `0.1` (D-12)
   - `fusion-type`, `model-type` already matched (`concat`, `dual`)

4. **Added `[project.optional-dependencies]` block** with `dev = ["pytest>=7.0"]`.

5. Created 5 GREEN regression tests in `tests/test_pyproject_shape.py`.

### Task 2: dataset.py rip-and-replace (commit `eafcafc`)

**D-17 removed** (8 targets — 1 more than Phase-3's 5-6 because adaptive had extra stats helpers):
- `create_global_mappings`
- `create_leave_one_out_split`
- `compute_user_genre_distribution`
- `compute_user_stats` (adaptive-specific — now replaced by SplitManifest.train_user_stats CR-5)
- `compute_partition_user_stats` (adaptive-specific — now replaced by dict comprehension over split.train_user_stats)
- `dirichlet_partition_users`
- `create_train_test_split`
- `_partition_cache` module dict

**D-18 preserved verbatim:**
- `MovieLensDataset` torch.utils.data.Dataset subclass
- `download_movielens_1m` (ML-1M URL)
- `load_movielens_1m` parser
- `natural_partition_users` (1 user = 1 client)

**Foundation-backed adapters** implemented:
- `_load_foundation_bundle(data_dir)` — calls `verify_bundle` and caches by `foundation_contract_sha256`.
- `_per_user_stats_to_dict(stats: PerUserStats)` — 4-line field-rename translator so `compute_client_alpha` / `compute_per_user_alpha` in task.py receive the `n_interactions` / `genre_entropy` / `n_unique_items` / `rating_std` keys they expect. `user_group` also forwarded for downstream per-group metrics.
- `load_partition_data(partition_id, num_partitions, alpha, test_ratio, batch_size, data_dir, compute_stats, split_mode, partition_mode)` — 7-tuple return preserved (the adaptive-specific shape that task.py::load_data destructures). Delegates mapping/item/user2idx/item2idx/test_item to foundation. Raises NotImplementedError on `partition_mode="dirichlet"` per D-02.
- `load_full_data(test_ratio, batch_size, data_dir, compute_stats, split_mode, partition_mode)` — same 7-tuple shape; also raises NotImplementedError on `partition_mode="dirichlet"` per Phase-3 D-02 tightening (BOTH entry points).

Edge case handled: users absent from `SplitManifest.train_user_stats` (single-interaction users that LOO split excludes from test) get a defensive zero-entropy sparse-stats row so `compute_client_alpha` degrades gracefully to the sparse-user penalty path.

Created 4 GREEN tests in `tests/test_dataset_adapter.py`.

## Test Counts

| File | GREEN | Contents |
| --- | --- | --- |
| tests/test_pyproject_shape.py | 5 | num-supernodes flipped in both federations; Phase-3 foundation-contract keys present; Phase-4 signature keys at thesis defaults; dev pytest extra declared; fedrec-foundation dep preserved |
| tests/test_dataset_adapter.py | 4 | load_partition_data uses foundation mapping; test_item matches foundation split; D-17 rip + D-18 preservation; D-02 raises at BOTH load_partition_data AND load_full_data |
| **Total** | **9** | |

Phase 3 test suite regression-checked: 34/34 still GREEN.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Consolidated duplicate `eval-num-negatives` declaration**
- **Found during:** Task 1 Step 3
- **Issue:** The pre-existing pyproject.toml had `eval-num-negatives = 99` under the "Evaluation protocol" section (line ~185) AND the plan required adding it again in the Phase-3 block — TOML rejects duplicate top-level keys within a table.
- **Fix:** Removed the standalone `eval-num-negatives` declaration under "Evaluation protocol" and left only the Phase-3 block declaration. Matches Phase 3 Plan 02's same-problem fix (STATE.md entry "Duplicate eval-num-negatives declaration auto-fixed to avoid TOML duplicate-key error").
- **Files modified:** federated-adaptive-personalized-cf/pyproject.toml
- **Commit:** 9b9d1f8

**2. [Rule 2 - Critical correctness] Added 8th D-17 rip target (adaptive-specific stats helpers)**
- **Found during:** Task 2 Step 1 (dataset.py inventory)
- **Issue:** The plan listed 5-6 D-17 targets from Phase 3. The adaptive module had 2 extra module-local helpers (`compute_user_stats`, `compute_partition_user_stats`) that duplicated the foundation's CR-5 train-only stats. Leaving them would have left two competing sources of user_stats — a silent contamination vector for the adaptive alpha heuristic.
- **Fix:** Removed both helpers and sourced user_stats exclusively from `SplitManifest.train_user_stats` via a 4-line `_per_user_stats_to_dict` translator. The 4 field names (n_interactions, genre_entropy, n_unique_items, rating_std) are preserved verbatim so task.py::compute_client_alpha needs zero changes.
- **Files modified:** federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py
- **Commit:** eafcafc

**3. [Rule 1 - Bug safety] Defensive fallback for users missing from train_user_stats**
- **Found during:** Task 2 Step 4 (load_partition_data implementation)
- **Issue:** `SplitManifest.train_user_stats` only contains users with ≥1 training interaction (after LOO test-item removal). Single-interaction users would produce a KeyError when `compute_stats=True`.
- **Fix:** Defensive fallback emits a zero-entropy sparse-stats row so compute_client_alpha degrades to the sparse-user penalty path (CONTEXT D-13 cold-start branch consumes this without panic).
- **Files modified:** federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py
- **Commit:** eafcafc

**None of the Rule 4 architectural checkpoints were triggered.**

### Wave-1 write-race safety

Plan 01 committed first (commits `05d8ee3` and `9269477`, touching strategy.py + test_strategy.py + test_dual_model.py + test_dataset_adapter_dual.py + conftest.py + __init__.py). Plan 02 touched strictly disjoint files:
- pyproject.toml
- federated_adaptive_personalized_cf/dataset.py
- tests/test_pyproject_shape.py
- tests/test_dataset_adapter.py

Zero file overlap; `--no-verify` flag used on both commits per parallel execution rule.

### D-18 surgical-edit discipline (confirmed)

No edits to client_app.py, server_app.py, task.py, strategy.py, early_stopping.py, models/, or evaluation/. `git show --stat eafcafc 9b9d1f8` confirms only the 4 in-scope files were touched.

## ADP-01 Closure Note

**ADP-01 ("Cross-device defaults in-file; `flwr run .` spawns 6040 supernodes by default") is now closed.** Observable via:

1. `python -c "import tomllib; d = tomllib.load(open('federated-adaptive-personalized-cf/pyproject.toml', 'rb')); print(d['tool']['flwr']['federations']['local-simulation']['options']['num-supernodes'])"` → `6040`
2. `python -c "import tomllib; d = tomllib.load(open('federated-adaptive-personalized-cf/pyproject.toml', 'rb')); print(d['tool']['flwr']['federations']['local-sim-gpu']['options']['num-supernodes'])"` → `6040`
3. `cd federated-adaptive-personalized-cf && python -m pytest tests/test_pyproject_shape.py -v` → 5 passed

The full thesis benchmark config (dual + hierarchical_conditional + concat + per-user-alpha ON + item-perturbation ON + contrastive-lambda=0.1) is now the in-file default — no dependence on scripts/run.py launcher overrides.

## Plan 03 Readiness Confirmation

Plan 03 (client_app.py + task.py contract wire) can proceed with the following guarantees:

- `dataset.py::load_partition_data` returns a 7-tuple `(trainloader, testloader, num_users, num_items, user2idx, item2idx, user_stats)` — identical to pre-migration.
- `task.py::load_data` continues to receive the 7-tuple, unpacks it, and returns the 3-tuple `(trainloader, testloader, user_stats)` to client_app.py — zero signature churn.
- `user_stats` dict keys (`n_interactions`, `genre_entropy`, `n_unique_items`, `rating_std`) unchanged; `compute_client_alpha` and `compute_per_user_alpha` continue to work without modification.
- The 5 schema_version=2 signature-driving keys (alpha-method, fusion-type, enable-per-user-alpha, enable-item-perturbation, contrastive-lambda) are now pyproject defaults; Plan 03's cache manifest can read them via `context.run_config.get(...)` and drop them into the `schema_version=2` signature without further coordination.

## Self-Check: PASSED

- Task 1 commit `9b9d1f8` exists: FOUND in git log
- Task 2 commit `eafcafc` exists: FOUND in git log
- pyproject.toml exists: FOUND
- dataset.py exists: FOUND
- test_pyproject_shape.py exists: FOUND
- test_dataset_adapter.py exists: FOUND
- 9 tests GREEN under `cd federated-adaptive-personalized-cf && pytest tests/test_pyproject_shape.py tests/test_dataset_adapter.py`
- Phase 3 regression: 34/34 tests GREEN (no contamination)
- D-18 scope: Plan 01 source files (strategy.py) untouched by Plan 02 commits; other pre-existing WIP (client_app.py, server_app.py, task.py) untouched

## Known Stubs

None — all returned values are backed by real data from the foundation bundle or the raw ML-1M download. No placeholder text, no hardcoded empty values flowing to UI.
