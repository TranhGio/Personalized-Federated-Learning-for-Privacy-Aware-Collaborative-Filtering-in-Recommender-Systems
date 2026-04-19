---
phase: 01-foundation-contract
verified: 2026-04-19T04:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 01: Foundation Contract Verification Report

**Phase Goal:** A single shared cross-device protocol contract — canonical ID mapping, deterministic LOO split, exclusion set, primary evaluator choice, weight policy, seeding discipline, and run manifest — exists on disk and is ready to be imported by every downstream module.

**Verified:** 2026-04-19
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A single canonical `raw_user_id → user_idx` / `raw_item_id → item_idx` artifact exists and any module that imports it observes the same indices for the same raw IDs. | VERIFIED | `data/derived/mapping.json` exists with `num_users=6040`, `num_items=3706`, `schema_version=1`. Loaded twice in the same process: user_id `1` maps to idx `0` both times. `fedrec_foundation.mapping.load_mapping()` is importable from all 4 `federated-*-cf/` module directories via subprocess. |
| 2 | Running the split builder twice produces the same `split_hash` and the same held-out test item per user (deterministic tiebreaking), and `exclude_items[user]` always contains that held-out item. | VERIFIED | `split_hash=5685bed7e4b650807e58a49e25ea611cdef82444a034bdf105e46aa3755284d6` is committed to `data/derived/split_manifest.json`. D-04 lock (`save_split_or_verify`) refuses overwrite on divergent hash. 5/5 sampled users have their `test_item_per_user` entry present in `exclusion_items.npz`. `test_split_lock_refuses_overwrite` passes. |
| 3 | A config flag picks ONE primary evaluator (`sampled_loo_99`) and ONE aggregation `weight-policy` per module, both of which appear in the run manifest alongside partition mode, num-supernodes, fractions, seeds, and negative counts. | VERIFIED | `EvalProtocol.SAMPLED_LOO_99.value == "sampled_loo_99"` confirmed. `get_primary_evaluator()` returns `"sampled_loo_99"` for all three modes. `ModeProfile.weight_policy = "num_positives"` for `benchmark_cross_device`. `RunManifest` carries all required FND-07 fields: `partition_mode`, `num_supernodes`, `fraction_train`, `fraction_eval`, `weight_policy`, `primary_evaluator`, `num_train_negatives`, `num_eval_negatives`, `run_seed`, `checkpoint_rule`. |
| 4 | Setting a single run seed produces identical server-selected client sequences and identical evaluator negative samples across two back-to-back runs of the same config; no evaluator internally calls `random.seed(...)` or `np.random.seed(...)`. | VERIFIED | `server_rng(42).sample(range(6040), 300)` produces identical results on two separate calls. `np_rng(42, 100, 5, 'eval_neg')` produces identical 99-negative sequences on two calls. SHA-256 seed derivation is PYTHONHASHSEED-immune: `PYTHONHASHSEED={0, 1, random}` all produce `0.06309475747172966` from `py_rng(42, 1, 0, 'train_neg')`. Zero `random.seed` or `np.random.seed` calls in any `fedrec_foundation/` module. `test_derive_rng_stable_across_processes` passes. |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `scripts/foundation/pyproject.toml` | Installable `fedrec-foundation` package | VERIFIED | `name="fedrec-foundation"`, hatchling build backend, `packages=["fedrec_foundation"]`, `testpaths=["tests"]` |
| `scripts/foundation/fedrec_foundation/__init__.py` | Package marker | VERIFIED | `__version__ = "0.1.0"` |
| `scripts/foundation/fedrec_foundation/paths.py` | Repo-root + data path resolution | VERIFIED | Exports `repo_root()`, `data_derived()`, `ml1m_dir()` |
| `scripts/foundation/fedrec_foundation/atomic.py` | Atomic JSON write | VERIFIED | Exports `atomic_write_json` |
| `scripts/foundation/fedrec_foundation/hashing.py` | SHA-256 file hashing | VERIFIED | Exports `sha256_file`, `compute_raw_data_hash` |
| `scripts/foundation/fedrec_foundation/mapping.py` | FND-01 canonical mapping | VERIFIED | `build_mapping`, `load_mapping`, `CanonicalMapping`; CR-1 ratings-only items (3706 not 3883) |
| `scripts/foundation/fedrec_foundation/user_groups.py` | Frozen bucket semantics | VERIFIED | `classify_user_group`, `USER_GROUP_BOUNDARIES=(30,100)`, half-open |
| `scripts/foundation/fedrec_foundation/split.py` | FND-02 LOO split + D-04 lock | VERIFIED | `build_split`, `save_split_or_verify`, `load_split_manifest`, `SplitManifest` with both fingerprints |
| `scripts/foundation/fedrec_foundation/exclusion.py` | FND-03 exclusion set | VERIFIED | IMP-3 flat NPZ layout, `ExclusionTable`, `exclusion_for`, `allow_pickle=False` |
| `scripts/foundation/fedrec_foundation/bundle.py` | Atomic bundle publication (N-3) | VERIFIED | `publish_bundle`, `verify_bundle`, `compute_foundation_contract_sha256` |
| `scripts/foundation/fedrec_foundation/evaluator.py` | FND-04 primary evaluator | VERIFIED | `EvalProtocol.SAMPLED_LOO_99`, `get_primary_evaluator()` whitelists 3 modes |
| `scripts/foundation/fedrec_foundation/weight_policy.py` | FND-05 weight policy | VERIFIED | `WeightPolicy`, `compute_aggregation_weight()` |
| `scripts/foundation/fedrec_foundation/fit_metrics.py` | CR-4 FitMetricsContract | VERIFIED | `FitMetricsContract`, `from_dict` wraps TypeError as ValueError |
| `scripts/foundation/fedrec_foundation/rng.py` | FND-06 four-tier RNG | VERIFIED | `py_rng`, `np_rng`, `torch_gen`, `server_rng`, `_ALLOWED_PURPOSES`, `_derive_seed` with full 256-bit SHA-256 |
| `scripts/foundation/fedrec_foundation/manifest.py` | FND-07 run manifest | VERIFIED | `RunManifest` (24 fields), `build_run_manifest`, `embed_manifest_in_result`, `write_manifest_sibling` |
| `scripts/foundation/fedrec_foundation/mode.py` | Mode resolver + 3 profiles | VERIFIED | `ModeProfile`, `resolve_mode_defaults`, `assert_benchmark_one_user_per_client`, `MODE_NAMES` |
| `scripts/foundation/fedrec_foundation/build.py` | CLI entrypoint shim | VERIFIED | `python -m fedrec_foundation.build` resolves |
| `data/derived/mapping.json` | FND-01 on-disk artifact | VERIFIED | 6040 users, 3706 items (CR-1 confirmed), `mapping_sha256=0cffcdde...` |
| `data/derived/split_manifest.json` | FND-02 on-disk artifact | VERIFIED | `split_hash=5685bed7...`, `raw_data_hash`, `mapping_sha256`, `train_user_stats`, `test_item_per_user` |
| `data/derived/exclusion_items.npz` | FND-03 on-disk artifact | VERIFIED | flat `int32` items (1000209,) + `int64` indptr (6041,), `allow_pickle=False` safe |
| `data/derived/foundation_index.json` | Bundle sentinel (N-3) | VERIFIED | All 4 fingerprints + `builder_version=1.0.0` + `created_at` |
| `scripts/run.py` | CR-2 launcher | VERIFIED | `--dry-run` flag; emits `num-supernodes=6040 mode=benchmark_cross_device` for baseline benchmark mode |
| `scripts/foundation/tests/` (13 test files) | Test coverage | VERIFIED | 70 passed, 0 failed, 0 skipped |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fedrec_foundation.mapping` | `data/derived/mapping.json` | `build_mapping` + `save_mapping` | WIRED | Artifact on disk matches the committed `mapping_sha256` in `foundation_index.json` |
| `fedrec_foundation.split` | `data/derived/split_manifest.json` | `build_split` + `save_split_or_verify` | WIRED | D-04 lock active; `split_hash` matches `foundation_index.json.split_hash` |
| `fedrec_foundation.exclusion` | `data/derived/exclusion_items.npz` | `build_exclusion` + `save_exclusion` | WIRED | `exclusion_sha256` matches index; `exclusion_for(npz, user_idx)` returns correct exclusion arrays |
| `fedrec_foundation.bundle` | `data/derived/foundation_index.json` | `publish_bundle` (sentinel-last) | WIRED | `verify_bundle()` passes without exception |
| `scripts/run.py` | `federated-*-cf/` modules | argparse `MODULE_DIR` map + `num-supernodes` | WIRED | `--dry-run` confirms `num-supernodes=6040` for benchmark, `num-supernodes=5` for cross_silo_legacy |
| `fedrec_foundation.rng` | SHA-256 seed derivation | `_derive_seed` with namespace prefix | WIRED | Cross-process PYTHONHASHSEED test passes (0, 1, random all produce identical value) |
| `fedrec_foundation.evaluator` | `sampled_loo_99` string | `get_primary_evaluator(mode)` | WIRED | Returns `"sampled_loo_99"` for all 3 whitelisted mode names |
| `fedrec_foundation` dep | all 4 `federated-*-cf/` modules | `pyproject.toml` `[project] dependencies` | WIRED | `fedrec-foundation` present in dependencies list of all 4 modules; subprocess import test passes for all 4 |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| FND-01 | Plan 02 | Canonical `raw_user_id → user_idx` / `raw_item_id → item_idx` artifact, imported by every module | SATISFIED | `mapping.json` exists, 6040/3706 empirical counts confirmed, CR-1 ratings-only item set enforced |
| FND-02 | Plan 02 | Deterministic LOO split manifest with `split_hash`, stable sort by `(user_id, timestamp, movie_id)` | SATISFIED | `split_manifest.json` committed, D-04 lock active, `test_hash_deterministic` and `test_timestamp_tiebreak` pass |
| FND-03 | Plan 02 | Per-user exclusion set `exclude_items[user] = train_pos ∪ test_pos` | SATISFIED | `exclusion_items.npz` on disk, test items confirmed present for sampled users, `test_includes_test_item` passes |
| FND-04 | Plan 03 | ONE primary evaluation protocol (`sampled_loo_99`); `allrank_*` namespaced secondary | SATISFIED | `EvalProtocol.SAMPLED_LOO_99`, `get_primary_evaluator()` returns it for all 3 modes |
| FND-05 | Plan 03 | Explicit `weight-policy` config; module picks one default; logged | SATISFIED | `WeightPolicy` enum with 3 values; `compute_aggregation_weight()` with clear ValueError; `ModeProfile.weight_policy` field |
| FND-06 | Plan 04 | Run-scoped seeding; server RNG derived for client sampling; per-user RNG from `(run_seed, user_id, round, purpose)`; no global `random.seed` inside evaluators | SATISFIED | SHA-256 namespaced `py_rng`/`np_rng`/`torch_gen`, `server_rng`, `_ALLOWED_PURPOSES` frozenset; zero `random.seed`/`np.random.seed` in any foundation module |
| FND-07 | Plan 04 | Run manifest with partition mode, num-supernodes, fractions, weight policy, eval protocol, negative counts, seeds, checkpoint rule | SATISFIED | `RunManifest` (24 fields) covers all required fields; D-15 double-write via `embed_manifest_in_result` + `write_manifest_sibling` |

All 7 FND requirements marked `[x]` (complete) in `REQUIREMENTS.md`. All requirements accounted for across plans — no orphaned requirements found.

---

### Anti-Patterns Found

No anti-patterns detected. Zero `TODO`, `FIXME`, `HACK`, `PLACEHOLDER`, `NotImplementedError`, or stub return values in any `fedrec_foundation/` module source file.

---

### Human Verification Required

None. All success criteria are fully verifiable programmatically:

- Canonical mapping determinism confirmed by loading the same file twice and checking equality.
- Split hash determinism confirmed by the committed `split_hash` value and D-04 lock test.
- Exclusion set correctness confirmed by checking `test_item_per_user` entries appear in `exclusion_items.npz`.
- Evaluator and weight policy contracts confirmed by import + function calls.
- RNG reproducibility confirmed by cross-process subprocess test with varying `PYTHONHASHSEED`.
- Launcher correctness confirmed by `--dry-run` flag outputting expected `num-supernodes` values.

---

## Final Summary

Phase 01 (foundation-contract) goal is fully achieved.

The shared cross-device protocol contract exists as an installable `fedrec-foundation` package at `scripts/foundation/` with 14 submodules, 4 committed on-disk artifacts under `data/derived/`, and a `scripts/run.py` CR-2 launcher. All 7 FND requirements are satisfied and marked complete in `REQUIREMENTS.md`.

The full test suite runs 70 tests, all passing with 0 failures and 0 skipped. Every downstream module (`federated-baseline-cf`, `federated-pfedrec`, `federated-personalized-cf`, `federated-adaptive-personalized-cf`) declares `fedrec-foundation` as a dependency in its `pyproject.toml`, and cross-module subprocess import tests confirm each can load all 10 foundation submodules from its own working directory.

Phases 2-5 can now freely import from `fedrec_foundation` without any `sys.path` manipulation.

---

_Verified: 2026-04-19_
_Verifier: Claude (gsd-verifier)_
