---
phase: 05-pfedrec-migration-reproduction
plan: 03
subsystem: federated-learning
tags: [pfedrec, client-app, task, cache, manifest-sidecar, schema-v3, fnd-06, fnd-03, exclusion, rng, single-user-collapse, discover-only, eval-bce, dual-lr-preserved, cross-device, wave-2]
requirements: [PFR-02, PFR-03, PFR-04, PFR-05, PFR-06, PFR-07]

# Dependency graph
dependency-graph:
  requires:
    - phase-01 foundation contract (np_rng, ExclusionTable, atomic_write_json, mode resolver, FitMetricsContract / EvaluateMetricsContract, classify_user_group)
    - phase-05 plan-01 (PFedRecMLP._GLOBAL_PARAMS / _LOCAL_PARAMS tuples, PFedRecSplitFedAvg, GLOBAL_PARAM_KEYS / LOCAL_PARAM_KEYS frozensets, PFR-02-AUDIT.md)
    - phase-05 plan-02 (cross-device pyproject defaults, dataset.py foundation-adapter scaffolding with D-09 guards + Plan-03-deferred placeholders, mode.py D-25 weight_policy='uniform')
  provides:
    - federated-pfedrec/federated_pfedrec/task.py: FND-06 RNG factories + FND-03 exclusion threading + D-04 eval BCE over (positive + 99 negatives); stdlib random eradicated; dual-LR preserved (Pitfall 3).
    - federated-pfedrec/federated_pfedrec/client_app.py: PFR-05 single-user collapse; D-22 cold-round probe-then-load; D-16 / D-17 manifest-sidecar cache schema_v3; G-03-01 discover_only short-circuit; D-21 strict-contract wire payloads.
    - federated-pfedrec/federated_pfedrec/dataset.py: natural-path body fills (replaces Plan-02 NotImplementedError placeholders for both load_partition_data and load_full_data).
    - federated-pfedrec/tests/conftest.py: shared fixtures (foundation-bundle skip marker, run_seed=42).
    - federated-pfedrec/tests/test_task.py: 4 GREEN tests covering PFR-04 / PFR-07 / PFR-06 / D-04.
    - federated-pfedrec/tests/test_cache.py: 7 GREEN tests covering D-16 / D-17 / D-18 / D-21 / Pitfall 6.
    - federated-pfedrec/tests/test_client_app.py: 3 GREEN tests covering PFR-05 / D-22 / G-03-01.
  affects:
    - phase-05 plan-04 (server_app.py): consumes EvaluateMetricsContract surface, the discover_only short-circuit, the .embedding_cache/{run_id}/partition_{pid}.pt cache path, and the schema_v3 manifest sidecar.
    - phase-05 plan-05 (subprocess determinism guard): asserts byte-identity on partition_{pid}.pt cache files (single-key affine_output.weight payload after D-01).

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Phase-3/4 manifest-sidecar cache pattern cloned with PFedRec-specific schema_v3 (10 fields including bias_classification='global' D-01 sentinel + loss='bce' + num_train_negatives=4 paper-default).
    - FND-06 RNG factory threading per (run_seed, user_idx, round_num, purpose) tuple — same pattern Phase 2/3/4 use; closes BSL-05 (zero stdlib random) cross-file regression in BOTH task.py and client_app.py.
    - Per-user single-key disk payload (\`affine_output.weight\` only) — bias channel aggregated atomically server-side per engine.py:143. SC-2 / D-01 reconciliation surfaced explicitly in the \`_signature_fields\` docstring + cache-layout helpers.
    - D-22 probe-then-load idiom: cache-miss returns None (cold round), cache-load failure raises (D-17 / D-21 hard-fail). Cleanly separates "first round" from "stale cache".
    - G-03-01 discover_only short-circuit FIRST in @app.evaluate before any bundle/model/data load — keeps the discovery-round handshake O(N=6040) lightweight.

key-files:
  created:
    - federated-pfedrec/federated_pfedrec/task.py (untracked WIP at start; canonical Phase-5 version 661 LOC)
    - federated-pfedrec/federated_pfedrec/client_app.py (untracked WIP at start; canonical Phase-5 version 767 LOC)
    - federated-pfedrec/tests/test_task.py (4 GREEN tests, ~120 LOC)
    - federated-pfedrec/tests/test_cache.py (7 GREEN tests, ~210 LOC)
    - federated-pfedrec/tests/test_client_app.py (3 GREEN tests, ~135 LOC)
  modified:
    - federated-pfedrec/federated_pfedrec/dataset.py (filled both Plan-02 NotImplementedError placeholders with the natural-path body)
    - federated-pfedrec/tests/conftest.py (added foundation-bundle skip marker + run_seed fixture; was placeholder pre-Plan-03)

key-decisions:
  - "Plan-02 dataset.py placeholders filled in this plan: \`load_partition_data\` and \`load_full_data\` natural-path bodies cloned from Phase 3 Plan 02 dataset.py with foundation-bundle attribute access reshaped for the Plan-02 \`_FoundationBundle\` dataclass (mapping/split_manifest/exclusion attrs vs Phase 3 dict). D-09 guards retained verbatim above the bodies."
  - "Per-user single-key payload (\`affine_output.weight\` only) per D-01: bias channel persisted server-side via aggregate_clients_params (engine.py:143 deletes \`affine_output.weight\` before aggregation, so the bias travels and is averaged across users). SC-2 reconciliation note surfaced in \`_signature_fields\` docstring + \`_save_local_user_state\` docstring + this SUMMARY's Decisions section."
  - "Schema_v3 \`bias_classification='global'\` field is a sentinel: any future maintainer reverting D-01 (moving bias back to LOCAL) will trip the cache-load D-17 manifest-mismatch check immediately. Same defense-in-depth pattern as Phase 4 schema_v2 sentinels (mlp_hidden_dims, fusion_type, etc.)."
  - "BSL-05 cross-file regression chosen over per-call grep: \`test_eval_neg_rng_factory_used\` reads BOTH task.py and client_app.py source files and asserts (a) \`np_rng\` import present, (b) zero \`random.seed(\`, (c) zero \`random.sample(\`, (d) zero module-level \`import random\`. Catches the case where one file is cleaned but the other regresses — exactly the failure mode I avoided this iteration by re-wording task.py docstring tokens."
  - "PFR-05 single-user collapse mechanically pinned: \`for user_idx in user_train_data\` substring is forbidden in client_app.py source (test 1). Plus the legacy \`save_user_local_params\` / \`load_user_local_params\` per-user-subdir helpers are explicitly forbidden — caught by the same test."
  - "Dual-LR preserved with explicit comment + grep guard: 4 occurrences of \`lr * num_items * lr_eta\` in task.py (1 in \`train_pfedrec_single_user\` body + 3 in docstrings). Pitfall 3 — PFR-08 ±2 reproduction is sensitive to this boost; do NOT 'fix' it."
  - "D-22 probe-then-load implemented twice in client_app.py: once before \`_load_local_user_state\` in @app.train (cold-round flag controls the \`set_local_parameters\` call), once before in @app.evaluate (no-op when no cache file present). The probe uses \`pt_path.exists()\` directly so a missing manifest doesn't crash on first round."

patterns-established:
  - "Pattern 1: BSL-05 cross-file regression guard for stdlib RNG eradication — single test reads multiple sibling source files and asserts zero \`random.seed(\` / \`random.sample(\` / module-level \`import random\`. Generalizes to any future migration that needs to retire stdlib random across multiple files in one phase."
  - "Pattern 2: Schema-version sentinel field — schema_v3 manifest carries \`bias_classification='global'\` as a literal sentinel that catches future reverts of the parameter classification decision. Cache load hard-fails on D-17 mismatch immediately. Same precedent: Phase 4 schema_v2's \`mlp_hidden_dims\` / \`fusion_type\` / etc."
  - "Pattern 3: Plan-02 placeholder closure pattern — when Wave-1 ships D-09 guards + adapter scaffolding only, Wave-2 fills the natural-path body in the same plan that owns client_app.py + task.py wiring (so the integration is atomic). Two consecutive \`raise NotImplementedError\` calls (D-09 frozen + Plan-03-deferred) with distinct messages let the verifier diff the messages mechanically."

requirements-completed: [PFR-04, PFR-05, PFR-06, PFR-07]

# Metrics
duration: 12min
completed: 2026-04-29
---

# Phase 5 Plan 03: Cross-device PFedRec client + task wire Summary

**One-liner:** Cross-device PFedRec client_app.py + task.py rewritten with PFR-05 single-user collapse, FND-03 exclusion threading (PFR-04), FND-06 per-round RNG (PFR-07), D-04 eval-time BCE over (positive + 99 negs), D-16/D-17/D-21/D-22 manifest-sidecar cache schema_v3 with `bias_classification='global'` D-01 sentinel, G-03-01 discover_only short-circuit, dual-LR preserved (Pitfall 3); 14 new GREEN tests + 14 inherited = 28 GREEN; full PFR-04..07 closure in-file.

## Performance

- **Duration:** ~12 min (focused work; commits span 2026-04-29 00:52 -> 01:04 +07)
- **Tasks:** 2 (per the plan; 5 logical steps including dataset.py placeholder fill)
- **Files created:** 5 (task.py, client_app.py, test_task.py, test_cache.py, test_client_app.py)
- **Files modified:** 2 (dataset.py — fill Plan-02 placeholders; conftest.py — fixtures)
- **Tests added:** 14 GREEN (4 task + 7 cache + 3 client_app)
- **Cumulative module suite:** 28 GREEN (14 inherited + 14 new)
- **Foundation suite:** 82 passed + 2 skipped (no regression)

## What Shipped

### Task 1 — task.py rewrite + test_task.py + conftest.py

**task.py (commit `adae872`):**

- New helper `_sample_train_negatives_seeded(user_rated_items, num_items, num_negatives, rng)`: rejection-uniform sampler driven by an FND-06 `np.random.Generator`. Distribution-equivalent to the reference's stdlib uniform-without-replacement at `IJCAI-23-PFedRec/data.py:80`. Closes BSL-05 (no stdlib random in this module) + PFR-07 (per-round resampling).
- Rewritten `prepare_user_train_data(user_idx, user_train_items, *, num_items, num_negatives, run_seed, round_num, exclude_items, rng)`: builds the per-user (items, ratings) BCE batch. FND-03 `exclude_items` folded into the no-go set so the held-out test positive is never drawn (PFR-04). Per-round RNG via `np_rng(run_seed, user_idx, round_num, "train_neg")` (PFR-07).
- Updated `train_pfedrec_single_user` signature: accepts (run_seed, user_idx, round_num, exclude_items) kwargs. **Pitfall 3 preserved**: `optimizer_item.lr = lr * num_items * lr_eta` (matches engine.py:117-119). Source-level comment pin added.
- Rewritten `evaluate_pfedrec_sampled(model, test_items, train_items_set, num_items, device, k_values, num_negatives, *, run_seed, user_idx, round_num, exclude_items)`: eval-negative draws via `np_rng(run_seed, user_idx, round_num, "eval_neg").choice(...)` (PFR-06). FND-03 `exclude_items` folded into rated-items union (PFR-04). **D-04 BCE scope**: per-user `eval_loss` computed from `torch.cat((test_score, negative_score))` over 100 candidates with `ratings = [1, 0, 0, ..., 0]` — mirrors `engine.py:195-196` exactly.
- `test_pfedrec` retained for the legacy diagnostic path (RMSE/MAE on direct test rows).
- Module-top: `import numpy as np`, `from fedrec_foundation.rng import np_rng`. Module-top has NO `import random`.

**tests/test_task.py (commit `1c1286e` RED + `adae872` GREEN — 4 tests):**

- `test_train_negs_exclude_held_out_test_positive` (PFR-04 / FND-03): asserts the rejection sampler's output is disjoint from `(user_rated ∪ exclude)`.
- `test_train_negs_resampled_every_round` (PFR-07 / D-02): asserts `np_rng(run_seed, 0, round=1)` and `np_rng(run_seed, 0, round=2)` produce different `integers(...)` outputs; same key produces same output (FND-06).
- `test_eval_neg_rng_factory_used` (PFR-06 / BSL-05 cross-file): asserts `np_rng` referenced in eval source; zero stdlib random literals in BOTH task.py and client_app.py.
- `test_eval_bce_over_positives_plus_99_negs` (D-04): asserts the eval source contains the `torch.cat((test_score, negative_score))` idiom from engine.py:195-196.

**tests/conftest.py:** added foundation-bundle skip marker + `run_seed=42` fixture (was a placeholder pre-Plan 03).

### Task 1.5 — dataset.py natural-path body fill (commit `8d2c1d9`)

Filled the two Plan-02 `raise NotImplementedError("Plan 03 implements...")` placeholders:

- `load_partition_data(partition_id, ...)`: builds (trainloader, testloader, num_users, num_items, user2idx, item2idx) for the cross-device path. Loads the foundation bundle, partitions ratings by `user_idx`, splits by foundation `test_item_per_user`, wraps in `MovieLensDataset` + `DataLoader`. partition_id == user_idx under cross-device.
- `load_full_data(...)`: same shape across all users (centralized-eval data flow).
- D-09 frozen-cross-silo guards retained verbatim above the bodies. D-18 helpers (`MovieLensDataset`, `download_movielens_1m`, `load_movielens_1m`, `natural_partition_users`) untouched.
- The natural-path body is cloned from `federated-personalized-cf/federated_personalized_cf/dataset.py` with foundation-bundle attribute access reshaped for the Plan-02 `_FoundationBundle` dataclass (mapping/split_manifest/exclusion attrs vs Phase 3's dict-style bundle).

### Task 2 — client_app.py rewrite + test_cache.py + test_client_app.py

**client_app.py (commit `2a663e7`):**

- Module-top imports: `atomic_write_json`, `EvaluateMetricsContract`, `FitMetricsContract`, `validate_*_metrics`, `assert_benchmark_one_user_per_client`, `log_mode_and_overrides`, `resolve_mode_defaults`, `get_primary_evaluator`, `classify_user_group`, `_load_foundation_bundle`. NO `import random`.
- Module-level constants: `_CACHE_BASE_DIR` exposed for test-time monkeypatching, `_device_cache` for CUDA fallback.
- New cache helpers (clones of Phase 3 idiom with PFedRec-specific schema_v3):
  - `_signature_fields(...)`: 10-field schema_v3 with `bias_classification='global'` D-01 sentinel + SC-2 reconciliation note in the docstring.
  - `_cache_dir_for_run(...)`: D-18 reuse_cache=true switches to `sig_<sha256[:16]>/` (run_id-agnostic).
  - `_save_local_user_state(...)`: D-21 single-key shape guard fires BEFORE disk write; atomic .pt via `tempfile.mkstemp(prefix='partition_tmp_', ...)` + `os.replace` (Phase 3 Rule-1 fix); manifest sidecar via `atomic_write_json`.
  - `_load_local_user_state(...)`: D-22 cold-round probe-then-load returns `None` on cache miss; D-17 loud manifest-mismatch RuntimeError with per-field delta + literal `rm -rf` hint; `torch.load(weights_only=True)` (Pitfall 6); D-21 shape guard AFTER load.
  - `_classify_partition_user_group(...)`: reads `bundle.split_manifest.train_user_stats[pid].user_group` (Phase 3 idiom, dataclass-style attr access).
- `@app.train`:
  1. Resolve mode profile + log overrides.
  2. Identify partition_id == user_idx.
  3. Load foundation bundle + build schema_v3 signature.
  4. Construct `PFedRecMLP` (Kaiming default per D-19) + load GLOBAL params (`embedding_item.weight` + `affine_output.bias` per D-01).
  5. **D-22 probe-then-load**: `pt_path.exists()` first; on hit call `_load_local_user_state` + `model.set_local_parameters(local_state, strict=True, run_id=run_id)`; on miss keep cold-round Kaiming init.
  6. Load partition data + **PFR-05 single-user assertion** (`assert_benchmark_one_user_per_client`).
  7. Build BCE batch via `prepare_user_train_data(... exclude_items=bundle.exclusion.for_user(user_idx) ...)` (PFR-04 + PFR-07).
  8. Train via `train_pfedrec_single_user(...)` (Pitfall 3 dual-LR inside).
  9. **D-16 atomic save** of `{"affine_output.weight": ...}` (single-key payload per D-21).
  10. Build wire payload — return GLOBAL params + `FitMetricsContract.to_dict()` validated via `validate_fit_metrics`.
- `@app.evaluate`:
  1. **G-03-01 discover_only short-circuit**: FIRST check `config.get('discover_only', False)`; if True, return zero-suffstats `EvaluateMetricsContract.to_dict()` payload validated via `validate_evaluate_metrics`.
  2. Resolve mode profile; assert `get_primary_evaluator(mode) == 'sampled_loo_99'`.
  3. (Same identity / bundle / signature steps as @app.train.)
  4. Construct model + load GLOBAL params.
  5. D-22 probe-then-load (no cold-round flag — eval just runs against the loaded state).
  6. Load partition data + PFR-05 assertion.
  7. `evaluate_pfedrec_sampled(... exclude_items=bundle.exclusion.for_user(user_idx) ...)`.
  8. Build D-22 per-group sufficient-stat routing + D-21 strict-contract payload via `EvaluateMetricsContract.to_dict()` validated via `validate_evaluate_metrics`.

**tests/test_cache.py (commit `508c98d` RED + `2a663e7` GREEN — 7 tests):**

- `test_partition_pid_pt_layout` (D-16): cache file at `{base}/{run_id}/partition_{pid}.pt`, no per-user-subdir.
- `test_manifest_schema_v3_fields` (D-17): manifest has all 10 schema_v3 keys; `schema_version=3`, `method='pfedrec'`, `loss='bce'`.
- `test_bias_classification_sentinel_global` (D-17 sentinel): `manifest['bias_classification'] == 'global'`.
- `test_strict_load_shape_mismatch_raises` (D-21): RuntimeError on latent_dim mismatch with per-field delta + `rm -rf` hint + run_id substring.
- `test_reuse_cache_sig_path` (D-18): two run_ids with identical signatures collide on `sig_<16-hex>/` directory.
- `test_save_payload_shape_guard` (D-21): rejecting payload with extra key fires AssertionError BEFORE any disk write (`{tmp_path}/r1/partition_0.pt` does not exist post-AssertionError).
- `test_torch_load_weights_only_true` (Pitfall 6): `inspect.getsource(_load_local_user_state)` contains `weights_only=True`.

**tests/test_client_app.py (commit `508c98d` RED + `2a663e7` GREEN — 3 tests):**

- `test_benchmark_one_user_per_client_assert` (PFR-05): client_app source contains `assert_benchmark_one_user_per_client`; ZERO occurrences of legacy `for user_idx in user_train_data`; ZERO `def save_user_local_params` / `def load_user_local_params` (legacy per-user-subdir helpers retired).
- `test_cold_round_probe_then_load` (D-22): `_load_local_user_state` returns `None` on cache miss; round-trips correctly post-save.
- `test_discover_only_short_circuit` (G-03-01): `discover_only` referenced in evaluate source; `EvaluateMetricsContract` referenced; the `discover_only` check appears BEFORE the `_load_foundation_bundle` call (positional source-index check).

## Test Counts

| File | GREEN | Notes |
| --- | --- | --- |
| federated-pfedrec/tests/test_pyproject.py | 3 | inherited from Plan 02 |
| federated-pfedrec/tests/test_dataset.py | 3 | inherited from Plan 02 |
| federated-pfedrec/tests/test_strategy.py | 5 | inherited from Plan 01 |
| federated-pfedrec/tests/test_pfedrec_mlp.py | 3 | inherited from Plan 01 |
| federated-pfedrec/tests/test_task.py | 4 | NEW (Plan 03 Task 1) |
| federated-pfedrec/tests/test_cache.py | 7 | NEW (Plan 03 Task 2) |
| federated-pfedrec/tests/test_client_app.py | 3 | NEW (Plan 03 Task 2) |
| **Module total** | **28** | 14 inherited + 14 new |
| scripts/foundation/tests/ | 82 + 2 skipped | no regression |

## VALIDATION.md Per-Task Coverage Map

| Task ID | Test |
| --- | --- |
| 5-03-01 | `test_client_app.py::test_benchmark_one_user_per_client_assert` |
| 5-03-02 | `test_task.py::test_train_negs_exclude_held_out_test_positive` |
| 5-03-03 | `test_task.py::test_train_negs_resampled_every_round` |
| 5-03-04 | `test_task.py::test_eval_neg_rng_factory_used` |
| 5-03-05 | `test_task.py::test_eval_bce_over_positives_plus_99_negs` |
| 5-03-06 | `test_cache.py::test_partition_pid_pt_layout` |
| 5-03-07 | `test_cache.py::test_manifest_schema_v3_fields` |
| 5-03-08 | `test_cache.py::test_bias_classification_sentinel_global` |
| 5-03-09 | `test_client_app.py::test_cold_round_probe_then_load` |
| 5-03-10 | (D-19 — already pinned by Plan 01's `test_pfedrec_mlp.py::test_kaiming_default_init_paper_faithful`) |

Plus 4 supporting tests: `test_strict_load_shape_mismatch_raises` (D-21), `test_reuse_cache_sig_path` (D-18), `test_save_payload_shape_guard` (D-21 save-side), `test_torch_load_weights_only_true` (Pitfall 6).

## Acceptance Criteria Status

All Plan 03 acceptance criteria PASSED:

**Task 1 (task.py):**
- `from fedrec_foundation.rng import` -> 1 ✓
- `np_rng(` -> 6 ✓ (≥2; covers train_neg + eval_neg purposes + factory threading)
- stdlib `random` -> 0 ✓
- `exclude_items` -> 15 ✓ (≥3; threaded through helper + 2 entry points + signatures + comments)
- `_sample_train_negatives_seeded` -> 3 ✓ (≥2; def + call + docstring)
- `torch.cat((test_score, negative_score))` -> 3 ✓ (D-04 idiom)
- `lr * num_items * lr_eta` -> 4 ✓ (Pitfall 3 preserved across body + 3 docstring references)

**Task 2 (client_app.py):**
- `from fedrec_foundation.atomic import` -> 1 ✓
- `from fedrec_foundation.mode import` -> 1 ✓
- `from fedrec_foundation.fit_metrics import` -> 1 ✓
- `assert_benchmark_one_user_per_client` -> 4 ✓ (≥1; module-import + train-site + evaluate-site + docstring)
- `discover_only` -> 7 ✓ (≥1; G-03-01 short-circuit + ConfigRecord access + docstrings)
- `for user_idx in user_train_data` -> 0 ✓ (PFR-05 collapsed)
- `schema_version=3` -> 1 ✓ (D-17)
- `bias_classification` -> 5 ✓ (≥2; sentinel field + docstring + signature kwargs + reconciliation note)
- `weights_only=True` -> 4 ✓ (Pitfall 6)
- `partition_tmp_` -> 1 ✓ (Phase 3 Rule-1 tempfile prefix)
- stdlib `random` -> 0 ✓
- `EvaluateMetricsContract` -> 6 ✓ (≥2; discover_only + normal eval + import + docstring + per-group + validate)
- `validate_*_metrics` -> 6 ✓ (≥2; train + evaluate sides + import + discover_only + docstring)

**Verification automation:**
- `cd federated-pfedrec && pytest tests/test_task.py -x -v` -> 4 passed ✓
- `cd federated-pfedrec && pytest tests/test_cache.py tests/test_client_app.py -x -v` -> 10 passed ✓
- `cd federated-pfedrec && pytest tests/ -x` -> 28 passed ✓
- `pytest scripts/foundation/tests/` -> 82 passed + 2 skipped (no regression) ✓

## SC-2 / D-01 Reconciliation (surfaced in code + docstrings)

ROADMAP §Phase 5 SC-2 phrase *"each user's `(affine_output.weight, affine_output.bias)` is persisted/restored as one atomic per-user artifact keyed by stable `user_idx`"* is reconciled with CONTEXT.md D-01 as:

- **Weight channel:** Per-user disk cache payload — single key `affine_output.weight` of shape `(1, latent_dim)` written atomically to `{base}/{run_id}/partition_{pid}.pt` via `tempfile + os.replace`.
- **Bias channel:** Aggregated atomically server-side per `IJCAI-23-PFedRec/engine.py:143` — the reference deletes `affine_output.weight` from each user's dict before aggregation, so `affine_output.bias` is averaged across users.
- **Atomicity contract preserved:** Each round, each user gets one atomic `(weight-on-disk, bias-aggregated-server-side)` artifact pair; the channels just live on different sides of the wire.
- **Sentinel:** `manifest['bias_classification'] == 'global'` in schema_v3 mechanically catches any future regression that reverts D-01.
- **Reconciliation surface points:**
  1. `_signature_fields` docstring (client_app.py).
  2. `_save_local_user_state` docstring (client_app.py).
  3. Module-level docstring (client_app.py).
  4. This SUMMARY's frontmatter `key-decisions` + body.
  5. `PFR-02-AUDIT.md` (Plan 01 Task 3 — human-readable cross-walk).

## Wave-2 Single-Plan Ownership Held

Plan 03 touched STRICTLY this file set:

- `federated-pfedrec/federated_pfedrec/task.py` ✓ (created — was untracked WIP)
- `federated-pfedrec/federated_pfedrec/client_app.py` ✓ (created — was untracked WIP)
- `federated-pfedrec/federated_pfedrec/dataset.py` ✓ (filled Plan-02 placeholders only)
- `federated-pfedrec/tests/conftest.py` ✓ (extended placeholder)
- `federated-pfedrec/tests/test_task.py` ✓ (new)
- `federated-pfedrec/tests/test_cache.py` ✓ (new)
- `federated-pfedrec/tests/test_client_app.py` ✓ (new)

ZERO touch of: `pyproject.toml` (Plan 02), `strategy.py` (Plan 01), `models/` (Plan 01), `server_app.py` (Plan 04 ownership), Wave-1 files.

## Deviations from Plan

**None — plan executed exactly as written.**

Two non-deviation auto-fixes applied without expanding scope:

1. **Docstring rewording (task.py):** The plan's `<action>` block included docstring text containing the literal substring `random.sample(` (in the `_sample_train_negatives_seeded` docstring describing what the helper replaces). The BSL-05 cross-file regression test catches that literal substring anywhere in the file (deliberately, to mechanically prevent any regression — including in comments). I reworded "the reference's `random.sample(neg_pool, k)`" to "the reference's stdlib uniform-without-replacement sampler" in 3 task.py docstring/comment locations. This is a Rule-3 inline fix (eliminating a literal token that would trip the cross-file test); the underlying contract is unchanged.

2. **Docstring rewording (client_app.py):** Same pattern — the module docstring contained the literal `for user_idx in user_train_data.keys()` substring describing what the legacy loop looked like. I reworded it to "legacy per-user-key iteration" so the test_client_app.py PFR-05 forbidden-token check passes. Rule-3 inline fix.

These are not contract changes — they are cosmetic docstring eradication of forbidden-token literals so the cross-file regression guards remain mechanical (substring-based) rather than AST-aware (which would have been a substantially heavier test design). The underlying behavior (zero stdlib random; zero legacy multi-user loop) is the load-bearing thing being tested.

## Issues Encountered

- **Untracked source files at start:** `federated-pfedrec/federated_pfedrec/task.py` and `federated-pfedrec/federated_pfedrec/client_app.py` were both untracked at plan start (the original module had pre-existing WIP that was never `git add`-ed). The Phase-5-aligned versions committed by this plan are now the canonical tracked versions. No regression — pre-existing on-disk content (the per-user-subdir cache + multi-user-per-partition loop) was replaced wholesale per the plan's `<action>` block.
- **Plan-02 dataset.py placeholders:** Plan 02 explicitly left `raise NotImplementedError("Plan 03 implements...")` placeholders for both `load_partition_data` and `load_full_data` natural-path bodies. Task 1 of this plan filled them; this is documented in the `<read_first>` block of the PLAN.md and was an explicit inter-plan handoff, not a deviation.
- **No flow-control issues during execution:** RED tests committed once per task; GREEN landed on the first run for the 3 directly-task tests; the 4th task test (cross-file regression) required two iterations of docstring eradication described above. All commits with normal pre-commit hook flow (sequential plan, no `--no-verify` discipline needed).

## Plan 04 / Plan 05 Readiness

**Plan 04 (server_app.py)** can now consume:

- `EvaluateMetricsContract` wire payload surface (12 sufficient-stat fields + optional `eval_loss` + `partition_id`) — `PFedRecSplitFedAvg.aggregate_evaluate` already consumes these via Plan 01's strategy.
- `discover_only=True` short-circuit — Plan 04's G-03-01 discovery round can broadcast `discover_only=True` configs in the discovery round to build `partition_id -> node_id` before round 1.
- Cache path layout `.embedding_cache/{run_id}/partition_{pid}.pt` — Plan 04 D-13 cold-start counter probes this path (matches Phase 3 D-13 idiom).
- Schema_v3 manifest sidecar `.embedding_cache/{run_id}/manifest.json` — Plan 04 may surface `bias_classification` in result-JSON manifest for cross-walk auditability.
- D-21 strict-contract validation — server-side `validate_evaluate_metrics` enforced before strategy aggregation.
- D-04 eval BCE diagnostic surfaced as `EvaluateMetricsContract.eval_loss` — Plan 04 may aggregate per-round mean for W&B logging (already shown in `PFedRecSplitFedAvg.aggregate_evaluate` return value).

**Plan 05 (subprocess determinism guard)** can now check byte-identity on `partition_{pid}.pt` (single-key `affine_output.weight` payload after D-01) and on `selected_clients_per_round` JSON.

## Self-Check: PASSED

- `federated-pfedrec/federated_pfedrec/task.py` exists.
- `federated-pfedrec/federated_pfedrec/client_app.py` exists.
- `federated-pfedrec/federated_pfedrec/dataset.py` exists with bodies filled.
- `federated-pfedrec/tests/conftest.py` exists.
- `federated-pfedrec/tests/test_task.py` exists.
- `federated-pfedrec/tests/test_cache.py` exists.
- `federated-pfedrec/tests/test_client_app.py` exists.
- Commits `1c1286e`, `adae872`, `8d2c1d9`, `508c98d`, `2a663e7` exist on `feat/try_to_run_the_baseline`.
- Module test suite: `cd federated-pfedrec && pytest tests/ -x` -> 28 passed.
- Foundation regression: `pytest scripts/foundation/tests/` -> 82 passed + 2 skipped.
- All 14 acceptance-criteria grep checks pass with documented counts above.

## Known Stubs

None — all `NotImplementedError` paths in `dataset.py` are intentional D-09 frozen-cross-silo guards for `partition_mode != 'natural'`. There are no remaining Plan-02 placeholders. No mock-data flows to UI. No "coming soon" / "TODO" placeholders.

---

*Phase: 05-pfedrec-migration-reproduction*
*Plan: 03 (Wave 2 — depends on Plans 01 + 02)*
*Completed: 2026-04-29*
*Closes: PFR-04 (FND-03 exclusion in train + eval neg pools), PFR-05 (single-user collapse), PFR-06 (FND-06 RNG client half), PFR-07 (per-round resampling), D-04 (eval BCE over 99 negs), D-22 (probe-then-load), D-17 (schema_v3 + bias_classification sentinel), Pitfall 6 (weights_only=True). Plan-02 dataset.py natural-path placeholders filled.*
