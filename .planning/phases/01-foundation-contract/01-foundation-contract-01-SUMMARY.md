---
phase: 01-foundation-contract
plan: 01
subsystem: infra
tags: [hatchling, pytest, python-packaging, editable-install, fedrec-foundation, sha256, atomic-write, pathlib, wave-0]

requires: []
provides:
  - "Installable fedrec-foundation package (v0.1.0) at scripts/foundation/"
  - "fedrec_foundation.paths: repo_root(), data_derived(), ml1m_dir() + FEDREC_FOUNDATION_DATA_DIR env override"
  - "fedrec_foundation.atomic: atomic_write_json() with tempfile + os.replace + numpy/Path-aware _json_default"
  - "fedrec_foundation.hashing: sha256_file(), compute_raw_data_hash() with LOCKED concat order ratings.dat || movies.dat || users.dat"
  - "Wave-0 test harness: pytest config, conftest.py fixtures (synthetic_ratings_df, tmp_derived_dir, pythonhashseed_random)"
  - "13 test files with 33 collected tests — 2 GREEN (hashing) + 31 SKIPPED stubs enumerated by pytest --collect-only"
  - "docs/setup.md documenting Plan 01 -> 06 install order (pytest, foundation, four federated-*-cf modules)"
affects: [02-foundation-contract, 03-foundation-contract, 04-foundation-contract, 05-foundation-contract, 06-foundation-contract]

tech-stack:
  added:
    - "hatchling (build backend for fedrec-foundation package)"
    - "pytest >= 7.0 (test discovery via [tool.pytest.ini_options] testpaths)"
  patterns:
    - "Skip-stub TDD handoff: Plans 02-05 flip pytestmark.skip off and replace NotImplementedError bodies"
    - "Atomic JSON writes: tempfile.mkstemp + os.replace + _json_default for numpy/Path coercion"
    - "LOCKED raw-data hash order: ratings.dat || movies.dat || users.dat — NEVER change"
    - "Env-var override FEDREC_FOUNDATION_DATA_DIR for CI/remote paths without touching ml1m_dir()"

key-files:
  created:
    - "scripts/foundation/pyproject.toml"
    - "scripts/foundation/fedrec_foundation/__init__.py"
    - "scripts/foundation/fedrec_foundation/paths.py"
    - "scripts/foundation/fedrec_foundation/atomic.py"
    - "scripts/foundation/fedrec_foundation/hashing.py"
    - "scripts/foundation/tests/__init__.py"
    - "scripts/foundation/tests/conftest.py"
    - "scripts/foundation/tests/test_hashing.py"
    - "scripts/foundation/tests/test_mapping.py"
    - "scripts/foundation/tests/test_split.py"
    - "scripts/foundation/tests/test_exclusion.py"
    - "scripts/foundation/tests/test_evaluator.py"
    - "scripts/foundation/tests/test_weight_policy.py"
    - "scripts/foundation/tests/test_rng.py"
    - "scripts/foundation/tests/test_mode.py"
    - "scripts/foundation/tests/test_manifest.py"
    - "scripts/foundation/tests/test_launcher.py"
    - "scripts/foundation/tests/test_integration.py"
    - "docs/setup.md"
  modified: []

key-decisions:
  - "Package location: scripts/foundation/ (D-Discretion option 1 from 01-CONTEXT.md) — shared, not duplicated; not at repo top-level to avoid namespace collision."
  - "Python typing: used typing.Optional / typing.Any (NOT PEP 604 X | Y) to match CONVENTIONS.md Python 3.9+ target."
  - "Skip stubs raise NotImplementedError so Plans 02-05 only have to delete the pytestmark line and fill in one body — no need to also rewrite scaffolding."
  - "Kept test_rng.py::test_derive_rng_stable_across_processes real subprocess body (PYTHONHASHSEED varied over '0'/'1'/'random') so Plan 04 un-skip is one-line change."
  - "pyproject.toml addopts='-ra' so SKIPPED tests are visible in every run's short summary, surfacing the outstanding work to downstream plans."

patterns-established:
  - "Wave-0 test handoff: module-level pytestmark=pytest.mark.skip(reason='Plan NN implements <module>') + test bodies raising NotImplementedError. Plans 02-05 delete the marker and write real assertions."
  - "Atomic JSON helper: tempfile.mkstemp(dir=parent, prefix='.tmp-', suffix='.json') + os.replace; _json_default coerces numpy scalars and pathlib.Path. Reuse across all manifest writers in Plans 02-04."
  - "LOCKED file-order hashing: compute_raw_data_hash concatenates ratings.dat || movies.dat || users.dat in that exact sequence — every FND-02/FND-07 field downstream depends on this order."

requirements-completed: []

duration: "5 min"
completed: "2026-04-19"
---

# Phase 01 Plan 01: Foundation Package Scaffold + Wave-0 Test Harness Summary

**Installable fedrec-foundation 0.1.0 package with paths/atomic/hashing modules plus a 33-test pytest harness (2 GREEN hashing tests + 31 SKIPPED stubs enumerating FND-01..07 behaviors for Plans 02-05 to fill in).**

## Performance

- **Duration:** ~5 min (277 seconds)
- **Started:** 2026-04-19T03:00:39Z
- **Completed:** 2026-04-19T03:05:16Z
- **Tasks:** 2 (both completed autonomously)
- **Files created:** 19 (6 package/doc + 13 test files)
- **Files modified:** 0 (greenfield in scripts/foundation/ and docs/)

## Accomplishments

- `fedrec-foundation` is now `pip install -e scripts/foundation/` — installs cleanly in the existing conda env (Python 3.13.5, numpy 2.2.6, pandas 2.3.3).
- Real implementations for three foundation modules: `paths` (repo-root walk + env override), `atomic` (tempfile+os.replace JSON writer), `hashing` (sha256 over files and over the LOCKED ml-1m .dat concatenation).
- `pytest tests/ --collect-only -q` enumerates 33 tests across 11 test files — satisfies VALIDATION.md's "≥25 tests collected" Nyquist criterion and maps 1:1 onto the FND-01..07 / mode / bundle / import / empirical verification matrix.
- `pytest tests/ -v` exits 0 with 2 PASSED + 31 SKIPPED + 0 FAILED; no pollution of the pre-existing working tree (staged only the 19 new files).
- `docs/setup.md` documents the install order: pytest -> foundation -> four modules -> smoke test — and flags that Plan 06 will wire foundation as a local-path dep in each module.

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold fedrec-foundation package + install-order doc** - `c05858c` (feat)
2. **Task 2: Create Wave-0 test stubs (13 test files + conftest)** - `73d9e47` (test)

_Note: Plan metadata commit (SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md) is appended separately at the end of execution._

## Files Created/Modified

### Package scaffold (Task 1)
- `scripts/foundation/pyproject.toml` - Hatchling build, `fedrec-foundation 0.1.0`, numpy/pandas deps, optional dev=pytest, `[tool.pytest.ini_options] testpaths=["tests"] addopts="-ra"`.
- `scripts/foundation/fedrec_foundation/__init__.py` - Single line `__version__ = "0.1.0"`.
- `scripts/foundation/fedrec_foundation/paths.py` - `repo_root()` walks up from `__file__` looking for `data/ml-1m`, `data_derived()` honors `FEDREC_FOUNDATION_DATA_DIR` override, `ml1m_dir()` is non-overridable.
- `scripts/foundation/fedrec_foundation/atomic.py` - `atomic_write_json(path, data)` via tempfile+os.replace with sort_keys=True; `_json_default` coerces numpy scalars/arrays + `pathlib.Path`.
- `scripts/foundation/fedrec_foundation/hashing.py` - `sha256_file(path)` reads in 65536-byte chunks; `compute_raw_data_hash(ml1m_dir)` hashes `ratings.dat || movies.dat || users.dat` in LOCKED order.
- `docs/setup.md` - Install order + smoke-test commands + note on Plans 01->06 readiness.

### Test harness (Task 2)
- `scripts/foundation/tests/__init__.py` - Empty package marker.
- `scripts/foundation/tests/conftest.py` - `synthetic_ratings_df` (5 users/4 items/12 rows), `tmp_derived_dir` (points env override at tmp), `pythonhashseed_random` (for subprocess tests).
- `scripts/foundation/tests/test_hashing.py` - **GREEN**: `test_sha256_file_deterministic`, `test_compute_raw_data_hash_order_matters` (2 passing).
- `scripts/foundation/tests/test_mapping.py` - **SKIPPED (Plan 02)**: `test_sort_order`, `test_item_mapping_from_ratings_only`, `test_roundtrip`.
- `scripts/foundation/tests/test_split.py` - **SKIPPED (Plan 02)**: `test_hash_deterministic`, `test_timestamp_tiebreak`, `test_split_lock_refuses_overwrite`, `test_train_only_user_stats`.
- `scripts/foundation/tests/test_exclusion.py` - **SKIPPED (Plan 02)**: `test_includes_test_item`, `test_safe_load`, `test_indptr_layout`, `test_module_level_exclusion_for`.
- `scripts/foundation/tests/test_evaluator.py` - **SKIPPED (Plan 03)**: `test_primary_evaluator_all_modes`.
- `scripts/foundation/tests/test_weight_policy.py` - **SKIPPED (Plan 03)**: `test_num_positives`, `test_unknown_policy_raises`, `test_fit_metrics_contract`, `test_from_dict_missing_required_raises`.
- `scripts/foundation/tests/test_rng.py` - **SKIPPED (Plan 04)**: `test_derive_rng_stable_across_processes` (real subprocess body preserved), `test_tuple_uniqueness`, `test_all_three_rng_factories`, `test_torch_generator_reproducible`, `test_sample_reproducible`.
- `scripts/foundation/tests/test_mode.py` - **SKIPPED (Plan 05)**: `test_override_logging`, `test_assertion_flags`.
- `scripts/foundation/tests/test_launcher.py` - **SKIPPED (Plan 05)**: `test_launcher_sets_num_supernodes`.
- `scripts/foundation/tests/test_manifest.py` - **SKIPPED (Plan 04)**: `test_all_fields_populated`, `test_both_writes`, `test_composite_foundation_hash`.
- `scripts/foundation/tests/test_integration.py` - **SKIPPED (Plans 02 + 06)**: `test_build_idempotent`, `test_bundle_atomic_publication`, `test_build_creates_all_artifacts`, `test_ml1m_counts_6040_3706`.

## Decisions Made

- **Package location = `scripts/foundation/`** (D-Discretion option 1). PROJECT.md rules out `fedrec_common/` in module packages this cycle; keeping the foundation at a neutral shared location avoids four-file duplication and namespace collisions with `federated-*-cf/`.
- **Typing uses `typing.Optional`, `typing.Any`** (not PEP 604 `X | Y`). CONVENTIONS.md pins the codebase to Python 3.9+ with `typing.*` module syntax; new foundation code matches.
- **Stub tests raise `NotImplementedError`**: with the module-level skip, the bodies never execute; Plans 02-05 simply drop the `pytestmark` line and fill in assertions — minimizing diff size and cognitive load.
- **`addopts = "-ra"`** in pyproject: SKIPPED reasons (with their "Plan NN implements …" text) appear in the short summary after every test run, so outstanding work stays visible.
- **`test_rng.py::test_derive_rng_stable_across_processes` has the real subprocess body** already (even though skipped) — Plan 04 un-skip is a one-line change, not a re-author.

## Deviations from Plan

None - plan executed exactly as written.

Both tasks followed the `<action>` blocks verbatim. Typing pattern (`typing.Any` for `_json_default(obj: Any)`) is a direct application of the CONVENTIONS.md rule referenced by the `<read_first>` list; no scope deviation.

## Issues Encountered

None. All automated verify commands passed on first run:
- `pip install -e scripts/foundation/` succeeded cleanly.
- Import smoke test printed `0.1.0`.
- `pytest tests/ --collect-only -q` reports 33 tests (>= 25 Nyquist floor).
- `pytest tests/ -v` exits 0 with 2 PASSED + 31 SKIPPED.

## Known Stubs

The 31 SKIPPED tests are **intentional handoff stubs**, not gaps in Plan 01's goal. Plan 01's goal is to produce the scaffold + test enumeration; filling the bodies is the defined responsibility of Plans 02-05 per this phase's `files_modified` allocation:

| Stub file | Un-skipping plan | Stub count | Covered requirement IDs |
|-----------|------------------|-----------:|-------------------------|
| `test_mapping.py` | Plan 02 | 3 | FND-01-a/b/c + CR-1 |
| `test_split.py` | Plan 02 | 4 | FND-02-a/b/c/d + CR-5 + D-04 |
| `test_exclusion.py` | Plan 02 | 4 | FND-03-a/b/c + CR-3 + IMP-3 |
| `test_evaluator.py` | Plan 03 | 1 | FND-04-a |
| `test_weight_policy.py` | Plan 03 | 4 | FND-05-a/b/c + CR-4 |
| `test_rng.py` | Plan 04 | 5 | FND-06-a/b/c/d/e + CR-3 |
| `test_manifest.py` | Plan 04 | 3 | FND-07-a/b/c + IMP-2 |
| `test_mode.py` | Plan 05 | 2 | D-06..D-11 + CR-2 |
| `test_launcher.py` | Plan 05 | 1 | mode-c + D-06 + CR-2 |
| `test_integration.py` | Plan 02 + Plan 06 | 4 | bundle-a/b, build-e2e, empirical-a |
| **TOTAL** | — | **31** | — |

Each plan un-skips its files by deleting the module-level `pytestmark = pytest.mark.skip(...)` and replacing `raise NotImplementedError("Plan NN fills this in")` with real assertions.

## User Setup Required

None - no external service configuration required. Dev setup steps are documented in `docs/setup.md` and will be referenced by Plan 06 when it wires foundation as a local-path dep into each `federated-*-cf/pyproject.toml`.

## Next Phase Readiness

**Ready for Plan 02 (foundation-contract-02)** — the mapping, split, and exclusion builders. Plan 02 agents:

1. Delete `pytestmark = pytest.mark.skip(...)` at the top of `test_mapping.py`, `test_split.py`, `test_exclusion.py`, and the Plan-02 subset of `test_integration.py`.
2. Replace each `raise NotImplementedError("Plan 02 fills this in")` body with real assertions matching the docstring's intent.
3. Create `fedrec_foundation/mapping.py`, `fedrec_foundation/split.py`, `fedrec_foundation/exclusion.py`, and the `python -m fedrec_foundation.build` entrypoint.

**Ready for Plans 03, 04, 05, 06** — same pattern per the "Known Stubs" table. Each plan imports from `fedrec_foundation`, un-skips its subset of test files, and ships artifacts under `data/derived/` using `atomic_write_json` + `compute_raw_data_hash`.

**No blockers.** No architectural decisions deferred. Install order documented. Scaffolding is frozen (should not change in Plans 02-06).

## Self-Check: PASSED

- **Files created:** FOUND: scripts/foundation/pyproject.toml, scripts/foundation/fedrec_foundation/{__init__,paths,atomic,hashing}.py, scripts/foundation/tests/{__init__,conftest,test_hashing,test_mapping,test_split,test_exclusion,test_evaluator,test_weight_policy,test_rng,test_mode,test_manifest,test_launcher,test_integration}.py, docs/setup.md (verified via git log + grep).
- **Commits:** FOUND: c05858c (Task 1), 73d9e47 (Task 2). `git log --oneline -5` confirms both on `feat/try_to_run_the_baseline`.
- **Automated verify:** PASSED. `pip install -e scripts/foundation/` -> 0. Import smoke test -> `0.1.0`. `pytest tests/ -v` -> 2 passed, 31 skipped, exit 0. `pytest tests/ --collect-only -q` -> `33 tests collected`.

---

*Phase: 01-foundation-contract*
*Plan: 01*
*Completed: 2026-04-19*
