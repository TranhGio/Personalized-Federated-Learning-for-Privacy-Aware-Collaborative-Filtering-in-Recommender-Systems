---
phase: 02-baseline-migration
plan: 02
subsystem: infra
tags: [pyproject-toml, cross-device, num-supernodes, partition-mode, fedrec-foundation, dataset-adapter, pytest-dev-dep, rip-and-replace, d-17, d-18, bsl-01, wave-1]

# Dependency graph
requires:
  - phase: 01-foundation-contract-01
    provides: "scripts/foundation/ package (fedrec-foundation 0.1.0)"
  - phase: 01-foundation-contract-02
    provides: "fedrec_foundation.{mapping, split, exclusion, bundle} + committed data/derived/ bundle (foundation_contract_sha256=fe181dafe6f7)"
  - phase: 01-foundation-contract-06
    provides: "fedrec-foundation plain-name local-path dep wired into federated-baseline-cf/pyproject.toml"

provides:
  - "federated-baseline-cf/pyproject.toml defaults to cross-device: partition-mode=natural + num-supernodes=6040 in BOTH local-simulation and local-sim-gpu federation blocks (BSL-01 fully in-file; ROADMAP Phase 2 success criterion 1 passes)"
  - "federated-baseline-cf/pyproject.toml declares 5 new Phase-2 foundation-contract config keys: mode, run-seed, weight-policy, eval-num-negatives, checkpoint-rule (D-19 fallback source when scripts/run.py launcher not used)"
  - "federated-baseline-cf/pyproject.toml declares [project.optional-dependencies] dev = ['pytest>=7.0'] — EXCLUSIVELY owned by this plan/task, eliminating the iter-1 BLOCKER 1 Wave-1 write race with Plan 01"
  - "federated-baseline-cf/federated_baseline_cf/dataset.py is a thin (~440 LOC) foundation adapter: load_partition_data + load_full_data delegate mapping, LOO split, and exclusion-set loading to fedrec_foundation.{mapping, split, exclusion, bundle} (D-17)"
  - "Removed 5 module-local helpers per D-17: create_global_mappings, create_leave_one_out_split, compute_user_genre_distribution, dirichlet_partition_users, create_train_test_split — plus the _partition_cache module dict (foundation's verify_bundle() + cached bundle loader subsumes it)"
  - "Preserved 4 functions per D-18 (pre-existing WIP untouched): MovieLensDataset, download_movielens_1m, load_movielens_1m, natural_partition_users"
  - "federated-baseline-cf/tests/test_dataset_adapter.py with 3 GREEN tests proving mapping/split/exclusion all originate from the foundation bundle and the D-17 rip targets are absent from the module"

affects: [02-baseline-migration-03, 02-baseline-migration-04, 03-personalized-migration, 04-adaptive-migration, 05-pfedrec-migration, 06-evaluation-harness, 07-thesis-evaluation]

# Tech tracking
tech-stack:
  added: []  # No new libraries — pure wiring + rip-and-replace over Phase 1 foundation APIs.
  patterns:
    - "Thin dataset-adapter pattern: dataset.py owns only raw-data I/O (download, parse, natural partition). Mapping/split/exclusion all delegate to fedrec_foundation. Single source of truth for the cross-device protocol."
    - "Foundation bundle verification + cache: _load_foundation_bundle calls verify_bundle(derived) first (raises RuntimeError on mismatch per N-3), then caches the loaded payload keyed by foundation_contract_sha256. Bundle rebuild invalidates cache automatically."
    - "Cross-device defaults in-file (not launcher-dependent): num-supernodes = 6040 in both local-simulation and local-sim-gpu federation blocks so `flwr run .` without scripts/run.py still resolves cross-device (BSL-01). Launcher sets the same value as a belt-and-suspenders redundancy."
    - "Pytest dev dep exclusively owned by Plan 02 Task 1: resolves the Wave-1 pyproject.toml write-race with Plan 01 by deleting that declaration from Plan 01 Task 2's action (iter-1 BLOCKER 1)."
    - "Surgical edit discipline (D-18): git diff run before any edit to inventory pre-existing uncommitted hunks; scope boundary (D-17) strictly observed. Pre-existing WIP in client_app.py / server_app.py / task.py untouched (Plans 03 + 04 own those)."

key-files:
  created:
    - "federated-baseline-cf/tests/test_dataset_adapter.py (78 LOC, 3 GREEN pytest tests)"
  modified:
    - "federated-baseline-cf/pyproject.toml (+16 lines, -3 lines): cross-device defaults + 5 new config keys + pytest dev dep"
    - "federated-baseline-cf/federated_baseline_cf/dataset.py (+232 lines, -298 lines = -66 net): rip-and-replace to thin foundation adapter"

key-decisions:
  - "Rip-and-replace vs surgical function-body edits for dataset.py: executed as a clean rewrite (440 LOC total, down from ~580). Preserved pre-WIP hunks in MovieLensDataset, download_movielens_1m, load_movielens_1m, natural_partition_users verbatim. D-17 scope (5 helpers to remove) and D-18 non-scope (4 functions to keep) were explicit."
  - "Removed the duplicate eval-num-negatives declaration that would have existed after appending the new Phase-2 keys: the pre-existing key under the 'Evaluation protocol' section was replaced with a pointer comment to the new 'Phase 2 cross-device / foundation-contract keys' block. TOML rejects duplicate keys, so this cleanup was blocking."
  - "partition_mode='dirichlet' is now NotImplementedError (was: full implementation via dirichlet_partition_users): D-17 intentionally removes the cross-silo partitioner. Cross-silo legacy is deferred to a future regression-test plan if needed; production thesis runs use 'natural'. Error message points at .planning/phases/02-baseline-migration/02-CONTEXT.md §Deferred."
  - "split_mode='random' now raises ValueError (was: random 80/20 split via create_train_test_split): D-17 makes the foundation's leave-one-out split authoritative. Random split is no longer meaningful once the foundation test_item_per_user is the LOO source of truth."
  - "_partition_cache dict removed without replacement at the module level: the foundation bundle is loaded once via _load_foundation_bundle (keyed by foundation_contract_sha256) and the raw ratings_df/partitions are built per-client inside load_partition_data. Cached bundle avoids re-verification; per-client partition is cheap (pandas groupby already amortizes)."

patterns-established:
  - "Foundation-backed dataset adapter: dataset.py in any migrated federated-*-cf module imports from fedrec_foundation.{mapping, split, exclusion, bundle}, verifies the bundle, loads it once into a module-level cache keyed by foundation_contract_sha256, and builds DataLoaders per client. Plans 03, 04, 05 should replicate this adapter shape to avoid drift."
  - "Phase-2 cross-device pyproject defaults: every federated-*-cf/pyproject.toml should declare (a) partition-mode = 'natural' under [tool.flwr.app.config], (b) num-supernodes = 6040 in BOTH local-simulation AND local-sim-gpu federation blocks with a comment on how to opt into cross-silo, (c) the 5 foundation-contract keys (mode, run-seed, weight-policy, eval-num-negatives, checkpoint-rule) with D-19 header comment, (d) [project.optional-dependencies] dev = ['pytest>=7.0']."
  - "Surgical-edit discipline for files with pre-existing uncommitted WIP: git diff the file before any edit; inventory pre-WIP hunks by line range; edit only the declared scope; git diff --stat after to verify delta shape. D-18 is reinforced in every Phase-2 Plan that touches a file already carrying uncommitted work."

requirements-completed: [BSL-01]

# Metrics
duration: 5min
started: "2026-04-19T07:45:57Z"
completed: "2026-04-19T07:51:04Z"
tasks_completed: 2
files_modified: 2
files_created: 1
tests_added: 3
tests_green: 3
---

# Phase 2 Plan 02: Baseline pyproject cross-device defaults + dataset.py foundation adapter Summary

**Baseline module defaults to cross-device (1 user = 1 client, N=6040) in both federation blocks; `dataset.py` is now a thin adapter that delegates mapping, LOO split, and exclusion-set loading to `fedrec_foundation` — closing the single-source-of-truth gap for Phase 2 while preserving pre-existing WIP in client_app / server_app / task.**

## Performance

- **Duration:** ~5 min (307 seconds)
- **Started:** 2026-04-19T07:45:57Z
- **Completed:** 2026-04-19T07:51:04Z
- **Tasks:** 2 (both autonomous, no deviations from plan)
- **Files modified:** 2 (`pyproject.toml`, `dataset.py`)
- **Files created:** 1 (`tests/test_dataset_adapter.py`)
- **Tests added:** 3 (all GREEN)

## Accomplishments

- **BSL-01 fully satisfied in-file.** `federated-baseline-cf/pyproject.toml` now defaults to cross-device: `partition-mode = "natural"` under `[tool.flwr.app.config]` (was already set by pre-WIP), and both federation blocks (`local-simulation` and `local-sim-gpu`) declare `options.num-supernodes = 6040` with a cross-silo opt-in comment. A plain `flwr run .` inside the baseline module now spawns 6040 supernodes by default — ROADMAP Phase 2 success criterion 1 passes without relying on `scripts/run.py`. The launcher still works as a belt-and-suspenders redundancy (verified via `python scripts/run.py --dry-run baseline benchmark_cross_device`).
- **Five Phase-2 foundation-contract config keys added** (D-19 fallback source when launcher not used): `mode = "cross_silo_legacy"`, `run-seed = 42`, `weight-policy = "num_positives"`, `eval-num-negatives = 99`, `checkpoint-rule = "best_round_restore"`. A header comment above `[tool.flwr.app.config]` calls out that `fedrec_foundation.mode.resolve_mode_defaults(mode)` is the canonical runtime source.
- **Iter-1 BLOCKER 1 eliminated.** `[project.optional-dependencies] dev = ["pytest>=7.0"]` is now declared in `pyproject.toml`, EXCLUSIVELY owned by this plan's Task 1. Plan 01's Task 2 can no longer write-race on the same file (the declaration lives here once, test plans in 01/02/03/04 `pip install -e '.[dev]'` to get pytest).
- **D-17 rip-and-replace complete in `dataset.py`.** Five module-local helpers (`create_global_mappings`, `create_leave_one_out_split`, `compute_user_genre_distribution`, `dirichlet_partition_users`, `create_train_test_split`) and the `_partition_cache` dict are removed. `load_partition_data` and `load_full_data` keep their identical signatures but now delegate to the foundation bundle (verify + load mapping/split/exclusion). The module went from ~580 LOC to 440 LOC net (-66 after net + text restructuring).
- **D-18 surgical discipline upheld.** `MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, and `natural_partition_users` preserve their pre-existing WIP state verbatim. Pre-existing uncommitted hunks in `client_app.py`, `server_app.py`, `task.py` are UNTOUCHED (Plans 03 and 04 will own those).
- **Three GREEN tests in `tests/test_dataset_adapter.py`** prove the adapter semantics:
  1. `test_load_partition_data_uses_foundation_mapping` — `num_users=6040`, `num_items=3706`; user2idx matches `fedrec_foundation.mapping.load_mapping`.
  2. `test_load_partition_data_test_item_from_foundation_split` — the held-out test item for a known user_idx appears in the returned testloader.
  3. `test_removed_helpers_gone` — all 5 D-17 rip targets are absent from the `federated_baseline_cf.dataset` module.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave-1 parallel execution; orchestrator runs hooks after all agents complete):

1. **Task 1: Flip pyproject to cross-device defaults + add pytest dev dep (BSL-01 + BLOCKER-1)** — `e3e4afc` (feat)
2. **Task 2: Rip-and-replace dataset.py with thin foundation adapter (D-17)** — `f784165` (refactor)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md updates) is appended separately at plan close._

## Files Created/Modified

### `federated-baseline-cf/pyproject.toml`
- `[project.optional-dependencies]` added with `dev = ["pytest>=7.0"]` (exclusively owned by this plan per iter-1 BLOCKER 1 fix).
- `[tool.flwr.app.config]` header comment added pointing at `fedrec_foundation.mode.resolve_mode_defaults(mode)` as D-19 canonical runtime source.
- 5 new keys appended to `[tool.flwr.app.config]`: `mode`, `run-seed`, `weight-policy`, `eval-num-negatives`, `checkpoint-rule`.
- Pre-existing `eval-num-negatives = 99` in the old "Evaluation protocol" section replaced with a pointer comment (TOML rejects duplicate keys).
- `[tool.flwr.federations.local-simulation] options.num-supernodes = 5` flipped to `6040` with cross-silo opt-in comment.
- `[tool.flwr.federations.local-sim-gpu] options.num-supernodes = 5` flipped to `6040` with the same opt-in comment.
- Preserved: `fedrec-foundation` dep (Phase 1 Plan 06), all other existing `[tool.flwr.app.config]` keys, `[tool.hatch.build.targets.wheel]`, `[tool.flwr.app.components]`, `[tool.flwr.federations.remote-federation]`.

### `federated-baseline-cf/federated_baseline_cf/dataset.py`
- Module docstring rewritten to describe the Phase-2 adapter responsibilities (raw I/O + natural partitioning + DataLoader wrapping only; mapping/split/exclusion delegated to foundation).
- 5 imports added from `fedrec_foundation`: `bundle.verify_bundle`, `exclusion.{ExclusionTable, load_exclusion}`, `mapping.{CanonicalMapping, load_mapping}`, `paths.data_derived`, `split.{SplitManifest, load_split_manifest}`.
- `_partition_cache` dict removed; replaced with `_foundation_cache` keyed by `foundation_contract_sha256`.
- `_load_foundation_bundle(data_dir)` helper added: calls `verify_bundle` first, then loads + caches mapping/split/exclusion.
- `load_partition_data` rewritten: verifies bundle → loads mapping/split → for `partition_mode="natural"` builds per-user partition via `natural_partition_users` and uses `split.test_item_per_user[partition_id]` for LOO. `partition_mode="dirichlet"` raises `NotImplementedError`; `split_mode="random"` raises `ValueError`.
- `load_full_data` rewritten: same foundation delegation, no partitioning; LOO mask built from `test_item_per_user`.
- REMOVED: `compute_user_genre_distribution`, `dirichlet_partition_users`, `create_train_test_split`, `create_leave_one_out_split`, `create_global_mappings`, `_partition_cache`.
- PRESERVED verbatim (D-18): `MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, `natural_partition_users`.

### `federated-baseline-cf/tests/test_dataset_adapter.py`
New file (78 LOC, 3 pytest tests). Skips if the foundation bundle (`data/derived/foundation_index.json`) is absent — graceful on minimal clones.

## Decisions Made

- **Rip-and-replace vs surgical function-body edits for dataset.py.** Executed as a clean rewrite because 5 helpers had to be removed in addition to replacing 2 function bodies; surgical-only edits would have left dead-code imports and created multiple tiny diffs. Pre-WIP functions (`MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, `natural_partition_users`) were carefully preserved verbatim (verified via the post-edit `git diff --stat` showing -298 deletions matching the D-17 rip targets plus old `load_partition_data` / `load_full_data` bodies).
- **Removed the duplicate `eval-num-negatives = 99` declaration.** The pre-existing key under the "Evaluation protocol" section was replaced with a pointer comment; the new "Phase 2 cross-device / foundation-contract keys" block now owns this key. TOML rejects duplicate top-level keys, so this cleanup was blocking — not optional.
- **`partition_mode="dirichlet"` raises `NotImplementedError`** (was a full implementation). D-17 intentionally removes the cross-silo partitioner. Cross-silo legacy is deferred to a future regression-test plan if required; production thesis runs use `"natural"`.
- **`split_mode="random"` raises `ValueError`.** The foundation's LOO split is now authoritative; a random 80/20 split would silently diverge from every other module. Fail-loud per the project's error-handling convention.
- **`_partition_cache` dict removed without a drop-in replacement.** The foundation bundle is loaded once via `_load_foundation_bundle` (keyed by `foundation_contract_sha256`) and the per-client partition (pandas groupby) is cheap enough to rebuild per call. Plans 03/04/05 can add a per-partition cache if profiling shows it's needed.

## Deviations from Plan

**None — plan executed exactly as written.**

Both tasks followed their `<action>` blocks verbatim:

- Task 1: added the `[project.optional-dependencies]` block, added the 5 Phase-2 config keys with the D-19 header comment, flipped both `num-supernodes` defaults to 6040 with cross-silo opt-in comments. (Required one minor cleanup — removing the pre-existing `eval-num-negatives = 99` declaration that would otherwise have produced a TOML duplicate-key error — but this was explicitly allowed by the plan's "do not duplicate existing keys" sub-step-1.1 guidance.)
- Task 2: wrote the new `dataset.py` with the D-17 removals + D-18 preservations + foundation-backed `load_partition_data` / `load_full_data`; created `tests/test_dataset_adapter.py` with the 3 specified tests.

No auto-fixes (Rules 1–3) applied. No architectural question (Rule 4) hit. No authentication gate. No test failures on first run.

## Issues Encountered

**None.** All automated verify commands passed on first run:

- `python -c "import tomllib; ..."` parses `pyproject.toml` cleanly with every expected value.
- `grep` acceptance criteria: 1/1/1/1/1/1 for the 5 new keys + `partition-mode = "natural"`; 2 for `options.num-supernodes = 6040`; 0 for `options.num-supernodes = 5`; 1 for `[project.optional-dependencies]`; 1 for `dev = ["pytest>=7.0"]`.
- `pytest federated-baseline-cf/tests/test_dataset_adapter.py -v` → 3 passed, 0 failed in 4.11s.
- `python -c "from federated_baseline_cf.dataset import load_partition_data; ..."` → prints `OK: load_partition_data returns nu=6040 ni=3706`.
- `python -c "from federated_baseline_cf.dataset import _load_foundation_bundle; ..."` → prints `OK: bundle loads, num_users= 6040 test_users= 6040`.
- `python scripts/run.py --dry-run baseline benchmark_cross_device` → prints `num-supernodes=6040` on the expected line (launcher still correct with pyproject defaults now matching).
- `grep "fedrec-foundation" federated-baseline-cf/pyproject.toml` → still matches (Phase 1 Plan 06 dep preserved).

## Known Stubs

**None.** No placeholder values, no TODO markers, no `NotImplementedError` that prevents the plan's goal:

- `partition_mode="dirichlet"` does raise `NotImplementedError`, but this is an intentional D-17 decision to defer cross-silo legacy (production thesis runs use `"natural"`; cross-silo opt-in is a future regression-test plan). Documented in both the docstring and the error message.
- `split_mode="random"` raises `ValueError` as a fail-loud guard; the foundation's LOO split is authoritative.

These are not stubs — they are explicit "removed functionality" documented in the plan and the module docstring.

## User Setup Required

**None beyond what `docs/setup.md` already documents.** The install order remains `pip install -e scripts/foundation/` → `pip install -e federated-baseline-cf/`. To get pytest, users now run `pip install -e "federated-baseline-cf[dev]"` (the new `[project.optional-dependencies] dev = ["pytest>=7.0"]` declaration).

## Next Phase Readiness

**Ready for Plan 03 (task.py / client_app.py migration).** Plan 03 will:

1. Migrate `client_app.py` to use `fedrec_foundation.rng.torch_gen(run_seed, user_idx, round_num, 'dataloader')` on the DataLoader, call `fedrec_foundation.mode.assert_benchmark_one_user_per_client()` at handler entry, and build `FitMetricsContract` return dicts.
2. Migrate `task.py` to consume the new `load_partition_data` return tuple (signature unchanged, semantics now foundation-backed) and use `fedrec_foundation.evaluator.get_primary_evaluator(mode)` for ranking eval.
3. The `load_partition_data` / `load_full_data` contract surface this plan established is LOCKED: Plan 03 must NOT change these signatures.

**Ready for Plan 04 (server_app.py + strategy.py migration).** Plan 04 will add weight-policy + run-manifest wiring on top of this plan's foundation-backed dataset module.

**No blockers.** No open questions. `federated-baseline-cf/` is now 1/4 of the way through Phase 2 migration (BSL-01 closed; BSL-02..08 remain in Plans 03 and 04).

## Self-Check: PASSED

- **Files modified:**
  - FOUND: `federated-baseline-cf/pyproject.toml` (verified via `git log --stat e3e4afc`).
  - FOUND: `federated-baseline-cf/federated_baseline_cf/dataset.py` (verified via `git log --stat f784165`).
- **Files created:**
  - FOUND: `federated-baseline-cf/tests/test_dataset_adapter.py` (verified via `git log --stat f784165` showing `create mode 100644`).
- **Commits:**
  - FOUND: `e3e4afc` (Task 1 feat) — verified via `git log --oneline -3` on `feat/try_to_run_the_baseline`.
  - FOUND: `f784165` (Task 2 refactor) — same.
- **Automated verify:** PASSED.
  - `grep -c 'options.num-supernodes = 6040' federated-baseline-cf/pyproject.toml` → 2.
  - `grep -c 'options.num-supernodes = 5' federated-baseline-cf/pyproject.toml` → 0.
  - `grep -c '^partition-mode = "natural"' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c '^mode = "cross_silo_legacy"' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c '^run-seed = 42' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c '^weight-policy = "num_positives"' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c '^eval-num-negatives = 99' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c '^checkpoint-rule = "best_round_restore"' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c '\[project.optional-dependencies\]' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c 'dev = \["pytest>=7.0"\]' federated-baseline-cf/pyproject.toml` → 1.
  - `grep -c 'fedrec-foundation' federated-baseline-cf/pyproject.toml` → 1 (Phase 1 Plan 06 dep preserved).
  - `grep -c 'from fedrec_foundation.mapping import' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c 'from fedrec_foundation.split import' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c 'from fedrec_foundation.exclusion import' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c 'from fedrec_foundation.bundle import verify_bundle' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c 'def create_global_mappings' federated-baseline-cf/federated_baseline_cf/dataset.py` → 0.
  - `grep -c 'def create_leave_one_out_split' federated-baseline-cf/federated_baseline_cf/dataset.py` → 0.
  - `grep -c 'def dirichlet_partition_users' federated-baseline-cf/federated_baseline_cf/dataset.py` → 0.
  - `grep -c 'def natural_partition_users' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c 'def load_partition_data' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c 'def load_full_data' federated-baseline-cf/federated_baseline_cf/dataset.py` → 1.
  - `grep -c '_partition_cache' federated-baseline-cf/federated_baseline_cf/dataset.py` → 0.
  - `cd federated-baseline-cf && pytest tests/test_dataset_adapter.py -v` → 3 passed, 0 failed.
  - `python -c "from federated_baseline_cf.dataset import load_partition_data; t,te,nu,ni,u2i,i2i = load_partition_data(0, 6040, partition_mode='natural'); assert nu == 6040 and ni == 3706"` → OK.
  - `python -c "from federated_baseline_cf.dataset import _load_foundation_bundle; b = _load_foundation_bundle(); assert b['mapping'].num_users == 6040; assert len(b['split_manifest'].test_item_per_user) >= 6000"` → OK.
  - `python scripts/run.py --dry-run baseline benchmark_cross_device | grep -c 'num-supernodes=6040'` → 1.
- **Scope boundary:** PASSED. Pre-existing uncommitted WIP in `federated-baseline-cf/federated_baseline_cf/{client_app.py, server_app.py, task.py}` is visible in `git status` as Modified but unchanged by this plan's commits (Plans 03 and 04 own those files).

---

*Phase: 02-baseline-migration*
*Plan: 02 (Wave 1 — parallel with Plan 01)*
*Completed: 2026-04-19*
*Closes: BSL-01 (cross-device defaults in pyproject + dataset.py foundation adapter).*
