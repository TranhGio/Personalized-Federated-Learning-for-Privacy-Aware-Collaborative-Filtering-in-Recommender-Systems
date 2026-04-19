---
phase: 03-personalized-migration
plan: 02
subsystem: infra
tags: [pyproject-toml, cross-device, num-supernodes, partition-mode, fedrec-foundation, dataset-adapter, pytest-dev-dep, rip-and-replace, d-17, d-18, d-02, d-09, psn-01, wave-1]

# Dependency graph
requires:
  - phase: 01-foundation-contract-01
    provides: "scripts/foundation/ package (fedrec-foundation 0.1.0)"
  - phase: 01-foundation-contract-02
    provides: "fedrec_foundation.{mapping, split, exclusion, bundle} + committed data/derived/ bundle (foundation_contract_sha256=fe181dafe6f7)"
  - phase: 01-foundation-contract-06
    provides: "fedrec-foundation plain-name local-path dep wired into federated-personalized-cf/pyproject.toml"
  - phase: 02-baseline-migration-02
    provides: "Canonical template for pyproject + dataset.py rip-and-replace; foundation-backed adapter shape that this plan mirrors for the personalized module."

provides:
  - "federated-personalized-cf/pyproject.toml defaults to cross-device: partition-mode=natural + num-supernodes=6040 in BOTH local-simulation and local-sim-gpu federation blocks (PSN-01 fully in-file; ROADMAP Phase 3 success criterion for in-file defaults passes)"
  - "federated-personalized-cf/pyproject.toml declares 6 Phase-3 foundation-contract config keys: mode, run-seed=42, weight-policy=num_positives, eval-num-negatives=99, checkpoint-rule=best_round_restore, reuse-cache=false (D-09 new key)"
  - "federated-personalized-cf/pyproject.toml declares [project.optional-dependencies] dev = ['pytest>=7.0'] — EXCLUSIVELY owned by this plan, eliminating the Wave-1 write-race with Plan 01"
  - "federated-personalized-cf/federated_personalized_cf/dataset.py is a thin (~410 LOC) foundation adapter: load_partition_data + load_full_data delegate mapping, LOO split, and exclusion-set loading to fedrec_foundation.{mapping, split, exclusion, bundle} (D-17)"
  - "Removed 5 module-local helpers per D-17: create_global_mappings, create_leave_one_out_split, compute_user_genre_distribution, dirichlet_partition_users, create_train_test_split — plus the _partition_cache module dict"
  - "Preserved 4 functions per D-18 (pre-existing WIP untouched): MovieLensDataset, download_movielens_1m, load_movielens_1m, natural_partition_users"
  - "D-02 NotImplementedError enforced at the dataset layer for partition_mode='dirichlet' in BOTH load_partition_data and load_full_data, with message referencing pre-Phase-3 commits + CONTEXT.md §Deferred"
  - "federated-personalized-cf/tests/test_dataset_adapter.py with 3 GREEN tests proving mapping/split delegation + D-17 rip completeness + D-02 NotImplementedError"

affects: [03-personalized-migration-03, 03-personalized-migration-04, 03-personalized-migration-05, 04-adaptive-migration, 05-pfedrec-migration]

# Tech tracking
tech-stack:
  added: []  # No new libraries — pure wiring + rip-and-replace over Phase 1 foundation APIs.
  patterns:
    - "Thin dataset-adapter pattern replicated from Phase 2 Plan 02: dataset.py owns raw-data I/O (download, parse, natural partition). Mapping/split/exclusion all delegate to fedrec_foundation. Single source of truth for the cross-device protocol."
    - "D-02 NotImplementedError at the dataset layer (not just the config layer) — any caller that explicitly passes partition_mode='dirichlet' is told unambiguously to check out a pre-Phase-3 commit. Fail-loud per CONVENTIONS.md."
    - "Cross-device defaults in-file (not launcher-dependent): num-supernodes = 6040 in both local-simulation and local-sim-gpu federation blocks so `flwr run .` without scripts/run.py still resolves cross-device (PSN-01). Launcher sets the same value as a belt-and-suspenders redundancy."
    - "6 Phase-3 keys (includes new reuse-cache=false per D-09) in [tool.flwr.app.config] — pyproject values are the fallback when scripts/run.py launcher is not used. D-19 override surface preserved."
    - "Surgical edit discipline (D-18): pre-existing uncommitted WIP in client_app.py, server_app.py, task.py is UNTOUCHED by this plan; git diff --stat verified those 3 files carry the same pre-WIP delta both before and after this plan's commits."

key-files:
  created:
    - "federated-personalized-cf/tests/test_dataset_adapter.py (91 LOC, 3 GREEN pytest tests)"
  modified:
    - "federated-personalized-cf/pyproject.toml (+23 lines, -5 lines): cross-device defaults + 6 Phase-3 config keys + pytest dev dep"
    - "federated-personalized-cf/federated_personalized_cf/dataset.py (+250 lines, -296 lines = -46 net): rip-and-replace to thin foundation adapter"

key-decisions:
  - "Mirror Phase 2 Plan 02 template exactly (canonical reference). The baseline module's dataset.py adapter shape is the contract; differences are only (a) 6 keys instead of 5 (new reuse-cache=false per D-09), (b) D-02 NotImplementedError enforced at dataset.py instead of deferred — this module is cross-device-only by design per Phase 3 CONTEXT §D-02."
  - "reuse-cache defaulted to false in pyproject.toml per D-09: cache-staleness bugs silently corrupt thesis numbers; default-off protects the thesis. Opt-in reuse is Plan 04's cache-manifest work to consume."
  - "_partition_cache dict REMOVED from dataset.py (not just preserved as in baseline Plan 02): the personalized module's pre-existing uncommitted WIP re-added it; rip-and-replace wipes it because the foundation bundle cache (_foundation_cache keyed by foundation_contract_sha256) subsumes it. Pandas groupby per-client partition is cheap enough to rebuild."
  - "D-02 enforced in BOTH load_partition_data AND load_full_data: the baseline equivalent only raised in load_partition_data. Because Plan 04's server_app.py centralized-eval path calls load_full_data, we want the guard at every entry into dirichlet territory — unambiguous fail-loud."
  - "split_mode='random' raises ValueError (not NotImplementedError): random 80/20 split would silently diverge from every other module; the foundation's LOO split is authoritative. Same fail-loud pattern as baseline Plan 02."

patterns-established:
  - "Phase-3 cross-device pyproject defaults: every federated-*-cf/pyproject.toml in Phases 3, 4, 5 should declare (a) partition-mode = 'natural' under [tool.flwr.app.config], (b) num-supernodes = 6040 in BOTH local-simulation AND local-sim-gpu federation blocks with a comment on how to opt into cross-silo legacy, (c) the 6 foundation-contract keys (mode, run-seed, weight-policy, eval-num-negatives, checkpoint-rule, reuse-cache), (d) [project.optional-dependencies] dev = ['pytest>=7.0']."
  - "D-02 NotImplementedError at dataset layer: for modules whose cross-silo path is permanently frozen (personalized, pfedrec, adaptive — per the respective phase CONTEXT.md §Deferred), raise at every entry function (load_partition_data + load_full_data) with a message including 'D-02', 'cross-device', and a pointer to pre-Phase-3 commits. The Phase 2 baseline deferred this decision (non-implementation message only in load_partition_data); Phase 3+ tightens."

requirements-completed: [PSN-01]

# Metrics
duration: 3min
started: "2026-04-19T16:08:55Z"
completed: "2026-04-19T16:11:30Z"
tasks_completed: 2
files_modified: 2
files_created: 1
tests_added: 3
tests_green: 3
---

# Phase 3 Plan 02: Personalized pyproject cross-device defaults + dataset.py foundation adapter Summary

**`federated-personalized-cf` module defaults to cross-device (1 user = 1 client, N=6040) in both federation blocks; `dataset.py` is now a thin foundation adapter that delegates mapping / LOO split / exclusion-set loading to `fedrec_foundation` and enforces D-02 NotImplementedError for `partition_mode='dirichlet'`. Mirror of Phase 2 Plan 02 with the D-09 reuse-cache key added and D-02 tightened at the dataset boundary.**

## Performance

- **Duration:** ~3 min
- **Tasks:** 2 (both autonomous, no deviations from plan)
- **Files modified:** 2 (`pyproject.toml`, `dataset.py`)
- **Files created:** 1 (`tests/test_dataset_adapter.py`)
- **Tests added:** 3 (all GREEN)

## Accomplishments

- **PSN-01 fully satisfied in-file.** `federated-personalized-cf/pyproject.toml` now defaults to cross-device: `partition-mode = "natural"` under `[tool.flwr.app.config]` (flipped from the pre-existing `"dirichlet"`-comment default), and both federation blocks (`local-simulation` and `local-sim-gpu`) declare `options.num-supernodes = 6040` with a cross-silo opt-in comment. A plain `flwr run .` inside the personalized module now spawns 6040 supernodes by default — ROADMAP Phase 3 success criterion for in-file defaults passes without relying on `scripts/run.py`.
- **Six Phase-3 foundation-contract config keys added** (D-19 fallback source when launcher not used): `mode = "cross_silo_legacy"`, `run-seed = 42`, `weight-policy = "num_positives"`, `eval-num-negatives = 99`, `checkpoint-rule = "best_round_restore"`, `reuse-cache = false` (new D-09). A header comment calls out that `fedrec_foundation.mode.resolve_mode_defaults(mode)` is the canonical runtime source.
- **Wave-1 write-race eliminated.** `[project.optional-dependencies] dev = ["pytest>=7.0"]` is declared in `pyproject.toml`, exclusively owned by this plan. Plan 01 can no longer write-race on the same file.
- **D-17 rip-and-replace complete in `dataset.py`.** Five module-local helpers (`create_global_mappings`, `create_leave_one_out_split`, `compute_user_genre_distribution`, `dirichlet_partition_users`, `create_train_test_split`) and the `_partition_cache` dict are removed. `load_partition_data` and `load_full_data` keep their identical signatures but now delegate to the foundation bundle (verify + load mapping/split/exclusion).
- **D-02 NotImplementedError enforced at BOTH entry points.** `load_partition_data` and `load_full_data` each raise `NotImplementedError("Personalized cross-device migration removed multi-user support per D-02. Check out a pre-Phase-3 commit ... to reproduce legacy cross-silo numbers.")` when called with `partition_mode="dirichlet"`. Tightens the baseline Plan 02 pattern (which only raised in `load_partition_data`).
- **D-18 surgical discipline upheld.** `MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, and `natural_partition_users` preserve their pre-existing WIP state verbatim. Pre-existing uncommitted hunks in `client_app.py`, `server_app.py`, `task.py` are UNTOUCHED (Plans 03 and 04 will own those).
- **Three GREEN tests in `tests/test_dataset_adapter.py`** prove the adapter semantics:
  1. `test_load_partition_data_uses_foundation_mapping` — `num_users=6040`, `num_items=3706`; user2idx matches `fedrec_foundation.mapping.load_mapping`.
  2. `test_load_partition_data_test_item_from_foundation_split` — the held-out test item for a known user_idx appears in the returned testloader.
  3. `test_removed_helpers_gone_and_d02_raises` — all 5 D-17 rip targets + `_partition_cache` are absent from the module, AND `partition_mode="dirichlet"` raises `NotImplementedError` with a message referencing `D-02` / `cross-device` / `pre-Phase-3`.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave-1 parallel execution; orchestrator runs hooks after all agents complete):

1. **Task 1: Flip pyproject to cross-device defaults + 6 Phase-3 keys + pytest dev dep (PSN-01)** — `a1c2845` (feat)
2. **Task 2: Rip-and-replace dataset.py with thin foundation adapter + 3 GREEN tests (D-17, D-02)** — `9acc97d` (refactor)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md updates) is appended separately at plan close._

## Files Created/Modified

### `federated-personalized-cf/pyproject.toml`
- `[project.optional-dependencies]` added with `dev = ["pytest>=7.0"]` (exclusively owned by this plan).
- `[tool.flwr.app.config]` header comment added pointing at `fedrec_foundation.mode.resolve_mode_defaults(mode)` as D-19 canonical runtime source.
- 6 new keys appended to `[tool.flwr.app.config]`: `mode`, `run-seed`, `weight-policy`, `eval-num-negatives`, `checkpoint-rule`, `reuse-cache` (the last one is new to Phase 3 per D-09).
- Pre-existing comment documenting `partition-mode = "natural"` updated to note D-02 NotImplementedError for the `dirichlet` branch.
- `[tool.flwr.federations.local-simulation] options.num-supernodes = 5` flipped to `6040` with cross-silo opt-in comment.
- `[tool.flwr.federations.local-sim-gpu] options.num-supernodes = 5` flipped to `6040` with the same opt-in comment. `options.backend.client-resources.num-cpus` and `num-gpus` preserved verbatim.
- Preserved: `fedrec-foundation` dep (Phase 1 Plan 06), all other existing `[tool.flwr.app.config]` keys, `[tool.hatch.build.targets.wheel]`, `[tool.flwr.app.components]`, `[tool.flwr.federations.remote-federation]`.

### `federated-personalized-cf/federated_personalized_cf/dataset.py`
- Module docstring rewritten to describe the Phase-3 adapter responsibilities (raw I/O + natural partitioning + DataLoader wrapping only; mapping/split/exclusion delegated to foundation; D-02 enforced).
- 5 imports added from `fedrec_foundation`: `bundle.verify_bundle`, `exclusion.{ExclusionTable, load_exclusion}`, `mapping.{CanonicalMapping, load_mapping}`, `paths.data_derived`, `split.{SplitManifest, load_split_manifest}`.
- `_partition_cache` dict removed; replaced with `_foundation_cache` keyed by `foundation_contract_sha256`.
- `_load_foundation_bundle(data_dir)` helper added: calls `verify_bundle` first, then loads + caches mapping/split/exclusion.
- `load_partition_data` rewritten: verifies bundle → loads mapping/split → for `partition_mode="natural"` builds per-user partition via `natural_partition_users` and uses `split.test_item_per_user[partition_id]` for LOO. `partition_mode="dirichlet"` raises `NotImplementedError` with D-02 message; `split_mode="random"` raises `ValueError`.
- `load_full_data` rewritten: same foundation delegation; `partition_mode="dirichlet"` also raises `NotImplementedError` here (tightens baseline Plan 02 pattern).
- REMOVED: `compute_user_genre_distribution`, `dirichlet_partition_users`, `create_train_test_split`, `create_leave_one_out_split`, `create_global_mappings`, `_partition_cache`.
- PRESERVED verbatim (D-18): `MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, `natural_partition_users`.

### `federated-personalized-cf/tests/test_dataset_adapter.py`
New file (91 LOC, 3 pytest tests). Skips if the foundation bundle (`data/derived/foundation_index.json`) is absent — graceful on minimal clones.

## Decisions Made

- **Rip-and-replace vs surgical function-body edits for dataset.py.** Executed as a clean rewrite because 5 helpers had to be removed + `_partition_cache` wiped + `load_partition_data` / `load_full_data` re-authored. Pre-WIP functions (`MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, `natural_partition_users`) carefully preserved verbatim.
- **`partition_mode="dirichlet"` raises `NotImplementedError` in BOTH `load_partition_data` AND `load_full_data`.** Tightens baseline Plan 02 (which only raised in `load_partition_data`). Plan 04's centralized-eval path calls `load_full_data`, so the guard must be on every entry into dirichlet territory — no silent fallthrough. Message includes `D-02`, `cross-device`, and `pre-Phase-3` tokens matching the test assertion.
- **`split_mode="random"` raises `ValueError`.** The foundation's LOO split is authoritative; random 80/20 split would silently diverge from every other module. Fail-loud per CONVENTIONS.md.
- **`_partition_cache` dict removed without a drop-in replacement.** The foundation bundle is loaded once via `_load_foundation_bundle` (keyed by `foundation_contract_sha256`) and the per-client partition (pandas groupby) is cheap enough to rebuild per call. Plan 03 may add a per-partition cache later if profiling motivates it.
- **`reuse-cache = false` set in pyproject (D-09 default).** Matches Phase 3 CONTEXT decision: cache-staleness bugs silently corrupt thesis numbers; default-off protects the thesis. Opt-in reuse is Plan 04's cache-manifest work to consume via `--run-config "reuse-cache=true"`.

## Deviations from Plan

**None — plan executed exactly as written.**

Both tasks followed their `<action>` blocks verbatim:
- Task 1: added `[project.optional-dependencies]`, added 6 Phase-3 config keys with D-19 header comment, flipped both `num-supernodes` defaults to 6040 with cross-silo opt-in comments. No duplicate-key conflicts (pre-existing `eval-num-negatives` was only declared once in this module; no cleanup needed).
- Task 2: rewrote `dataset.py` with D-17 removals + D-18 preservations + foundation-backed `load_partition_data` / `load_full_data`; created `tests/test_dataset_adapter.py` with the 3 specified tests.

No auto-fixes (Rules 1–3) applied. No architectural question (Rule 4) hit. No authentication gate. No test failures on first run.

## Issues Encountered

**None.** All automated verify commands passed on first run:
- `python -c "import tomllib; ..."` parses `pyproject.toml` cleanly with every expected value (`num-supernodes=6040` in both federations; `reuse-cache=false`).
- `grep` acceptance criteria all match expected counts (2 for `options.num-supernodes = 6040`; 0 for `options.num-supernodes = 5`; 1 each for the 6 new keys + `[project.optional-dependencies]` + `dev = ["pytest>=7.0"]`; 1 for `partition-mode = "natural"`).
- `pytest federated-personalized-cf/tests/test_dataset_adapter.py -v` → 3 passed, 0 failed in 4.14s.
- `python -c "from federated_personalized_cf.dataset import load_partition_data, _load_foundation_bundle; ..."` → prints `bundle ok` then `D-02 raises ok` then `ok`.
- D-18 scope check: `git diff --stat` on client_app.py / server_app.py / task.py shows the same pre-WIP delta before and after this plan's commits.

## Known Stubs

**None.** No placeholder values, no TODO markers, no `NotImplementedError` that prevents the plan's goal:
- `partition_mode="dirichlet"` does raise `NotImplementedError`, but this is the intentional D-02 decision that removes multi-user support from the personalized cross-device module. Documented in both the docstring and the error message. Future cross-silo runs (if ever needed) check out pre-Phase-3 commits.
- `split_mode="random"` raises `ValueError` as a fail-loud guard; the foundation's LOO split is authoritative.

These are not stubs — they are explicit "removed functionality" documented in the plan, the module docstring, and the Phase 3 CONTEXT.md.

## User Setup Required

**None beyond what `docs/setup.md` already documents.** The install order remains `pip install -e scripts/foundation/` → `pip install -e federated-personalized-cf/`. To get pytest, users now run `pip install -e "federated-personalized-cf[dev]"` (the new `[project.optional-dependencies] dev = ["pytest>=7.0"]` declaration).

## Next Phase Readiness

**Ready for Plan 03 (client_app.py + task.py migration).** Plan 03 will:
1. Migrate `client_app.py` to use `fedrec_foundation.rng.torch_gen(run_seed, user_idx, round_num, 'dataloader')` on the DataLoader, call `fedrec_foundation.mode.assert_benchmark_one_user_per_client()` at handler entry, and build `FitMetricsContract` return dicts with single-row local user state + sufficient-stat metrics.
2. Migrate `task.py` to consume the new `load_partition_data` return tuple (signature unchanged, semantics now foundation-backed) and use `fedrec_foundation.evaluator.get_primary_evaluator(mode)` for ranking eval.
3. The `load_partition_data` / `load_full_data` contract surface this plan established is LOCKED: Plan 03 must NOT change these signatures.

**Ready for Plan 04 (server_app.py + strategy.py migration).** Plan 04 will add weight-policy + run-manifest + in-memory best-round restore + cache-signature manifest on top of this plan's foundation-backed dataset module, reading `reuse-cache` from `context.run_config`.

**No blockers.** No open questions. `federated-personalized-cf/` is now the second Wave-1 plan closing (Plans 01 + 02 running in parallel); PSN-01 fully satisfied in-file, PSN-02..07 remain for Plans 03, 04, 05.

## Self-Check: PASSED

- **Files modified:**
  - FOUND: `federated-personalized-cf/pyproject.toml` (verified via `git log --stat a1c2845`).
  - FOUND: `federated-personalized-cf/federated_personalized_cf/dataset.py` (verified via `git log --stat 9acc97d`).
- **Files created:**
  - FOUND: `federated-personalized-cf/tests/test_dataset_adapter.py` (verified via `git log --stat 9acc97d` showing `create mode 100644`).
- **Commits:**
  - FOUND: `a1c2845` (Task 1 feat) — verified via `git log --oneline -3`.
  - FOUND: `9acc97d` (Task 2 refactor) — same.
- **Automated verify:** PASSED.
  - `grep -c 'options.num-supernodes = 6040' federated-personalized-cf/pyproject.toml` → 2.
  - `grep -c 'options.num-supernodes = 5' federated-personalized-cf/pyproject.toml` → 0.
  - `grep -c '^partition-mode = "natural"' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '^mode = "cross_silo_legacy"' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '^run-seed = 42' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '^weight-policy = "num_positives"' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '^eval-num-negatives = 99' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '^checkpoint-rule = "best_round_restore"' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '^reuse-cache = false' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c '\[project.optional-dependencies\]' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c 'dev = \["pytest>=7.0"\]' federated-personalized-cf/pyproject.toml` → 1.
  - `grep -c 'fedrec-foundation' federated-personalized-cf/pyproject.toml` → 1 (Phase 1 Plan 06 dep preserved).
  - `grep -c 'from fedrec_foundation.mapping import' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c 'from fedrec_foundation.split import' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c 'from fedrec_foundation.exclusion import' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c 'from fedrec_foundation.bundle import verify_bundle' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c 'def create_global_mappings' federated-personalized-cf/federated_personalized_cf/dataset.py` → 0.
  - `grep -c 'def create_leave_one_out_split' federated-personalized-cf/federated_personalized_cf/dataset.py` → 0.
  - `grep -c 'def dirichlet_partition_users' federated-personalized-cf/federated_personalized_cf/dataset.py` → 0.
  - `grep -c 'def natural_partition_users' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c 'def load_partition_data' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c 'def load_full_data' federated-personalized-cf/federated_personalized_cf/dataset.py` → 1.
  - `grep -c '_partition_cache' federated-personalized-cf/federated_personalized_cf/dataset.py` → 0.
  - `grep -cE 'raise NotImplementedError' federated-personalized-cf/federated_personalized_cf/dataset.py` → 2 (both load_partition_data AND load_full_data guard the dirichlet branch).
  - `cd federated-personalized-cf && pytest tests/test_dataset_adapter.py -v` → 3 passed, 0 failed in 4.14s.
  - `python -c "..."` smoke test → `bundle ok`, `D-02 raises ok`, `ok`.
- **Scope boundary:** PASSED. Pre-existing uncommitted WIP in `federated-personalized-cf/federated_personalized_cf/{client_app.py, server_app.py, task.py}` is visible in `git status` as Modified but unchanged by this plan's commits (Plans 03 and 04 own those files). `git diff HEAD~1 --stat` on those 3 files shows the exact same delta as before this plan ran.

---

*Phase: 03-personalized-migration*
*Plan: 02 (Wave 1 — parallel with Plan 01)*
*Completed: 2026-04-19*
*Closes: PSN-01 (cross-device defaults in pyproject + dataset.py foundation adapter + D-02 dirichlet guard).*
