---
phase: 03-personalized-migration
verified: 2026-04-20T07:30:00Z
status: passed
score: 4/4 success criteria verified
gaps: []
human_verification:
  - test: "Run `flwr run .` inside federated-personalized-cf/ end-to-end with benchmark_cross_device mode"
    expected: "6040 clients spawned, one round completes, result JSON written with _manifest.module='personalized', cold_starts reported"
    why_human: "The full Flower simulation loop requires a live ML-1M dataset download and GPU/CPU time not available in automated verification. All constituent parts verified in isolation via tests."
---

# Phase 03: Personalized Migration Verification Report

**Phase Goal:** `federated-personalized-cf` runs as a correct cross-device split-learning benchmark — 6040 clients, one local user per client, run-namespaced embedding cache, local user row collapsed to a single-user representation, sufficient-stat metrics, and protocol fingerprint logged.
**Verified:** 2026-04-20T07:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `flwr run .` spawns 6040 supernodes under natural partitioning by default; benchmark mode asserts exactly one local user | VERIFIED | `pyproject.toml` `num-supernodes=6040` in both federation blocks; `client_app.py` calls `assert_benchmark_one_user_per_client` before any training |
| 2 | Cache path scoped to `run_id/method/num_users/num_items/dim/split_hash`; mismatched signature hard-fails | VERIFIED | `_signature_fields()` and `_cache_dir_for_run()` in `client_app.py`; D-05 `RuntimeError` with per-field delta on any mismatch; D-10 shape guard on both save and load |
| 3 | Client footprint is a single-user row (not `num_users×d`); only GLOBAL params returned to server | VERIFIED | `BPRMF.local_user_row` is `nn.Parameter(shape=(d,))`; `get_global_parameters()` returns only `item_embeddings.weight`, `item_bias.weight`, `global_bias`; `get_local_parameters()` returns 2-key OrderedDict |
| 4 | Training negatives never include held-out test item; result artifact carries protocol fingerprint with sufficient-stat metrics | VERIFIED | `train_bpr_mf` merges `exclude_items` into `user_rated_items` before negative draw; `evaluate_ranking_sampled` unions `excluded_set` before candidate pool; `server_app.py` calls `build_run_manifest(..., module="personalized")` and double-writes |

**Score:** 4/4 truths verified

---

## Success Criterion 1: 6040 Supernodes + One-User Assertion

### pyproject.toml defaults

- `federated-personalized-cf/pyproject.toml` line 108: `options.num-supernodes = 6040` under `[tool.flwr.federations.local-simulation]`
- `federated-personalized-cf/pyproject.toml` line 115: `options.num-supernodes = 6040` under `[tool.flwr.federations.local-sim-gpu]`
- `federated-personalized-cf/pyproject.toml` line 68: `partition-mode = "natural"` under `[tool.flwr.app.config]`
- Comment on line 109 documents how to opt into cross-silo legacy (which raises `NotImplementedError` at runtime)

### One-user assertion in client_app.py

`@app.train()` handler (`client_app.py` line 501):
```python
assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)
```
Called BEFORE any training, after counting distinct user IDs in the trainloader.

`@app.evaluate()` handler (`client_app.py` line 710):
```python
assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)
```
Same assertion on the eval side, also called before evaluation.

### discover_only short-circuit

`@app.evaluate()` lines 616–625: returns zero-suffstats `EvaluateMetricsContract` with `partition_id` only when `discover_only=True`, without loading model, data, or running evaluation.

**VERDICT: PASS**

---

## Success Criterion 2: Run-Namespaced Cache with Hard-Fail Mismatch

### Cache path structure

`_signature_fields()` in `client_app.py` (lines 122–163) builds a dict with 6 fields + `schema_version=1`:
- `run_id`, `method`, `num_users`, `num_items`, `dim`, `split_hash`

`_cache_dir_for_run()` (lines 166–204):
- Default (`reuse_cache=False`): `.embedding_cache/{run_id}/`
- D-09 opt-in (`reuse_cache=True`): `.embedding_cache/sig_{sha256(non-run_id-fields)[:16]}/`

This scopes `partition_{pid}.pt` files under `{_CACHE_BASE_DIR}/{run_id}/partition_{pid}.pt`, satisfying the run-namespace requirement.

### Hard-fail on mismatch

`_load_local_user_state()` (lines 271–344) compares every field in `("schema_version", "run_id", "method", "num_users", "num_items", "dim", "split_hash")` against the on-disk `manifest.json`. On any divergence:
```python
raise RuntimeError(
    "Embedding-cache signature mismatch (D-05):\n  "
    + "\n  ".join(deltas)
    + f"\nRun: rm -rf {cache_dir}/ to reset, ..."
)
```

D-10 shape guard fires on BOTH save (line 244) and load (line 341):
```python
assert set(state_dict.keys()) == {"local_user_row", "local_user_bias"}
```

Test `test_manifest_mismatch_raises_runtime_error` (GREEN) confirms this path.

**VERDICT: PASS**

---

## Success Criterion 3: Single-User Row Footprint + GLOBAL-Only Server Return

### No ghost user table

`BPRMF.__init__` (lines 76–104 in `models/bpr_mf.py`):
- `self.local_user_row = nn.Parameter(torch.empty(embedding_dim))` — shape `(d,)`, not `(num_users, d)`
- `num_users` constructor arg retained for API compat but explicitly NOT stored as `self.num_users`
- No `nn.Embedding(num_users, ...)` for user representation

Tests `test_bpr_mf_no_ghost_table` and `test_basic_mf_no_ghost_table` structurally assert `'nn.Embedding(num_users' not in src` and `'self.user_embeddings' not in src` (GREEN).

### GLOBAL-only server return

`@app.train()` (lines 549–550 in `client_app.py`):
```python
global_params_out = model.get_global_parameters()
model_record = ArrayRecord(global_params_out)
```
Only GLOBAL parameters (`item_embeddings.weight`, `item_bias.weight`, `global_bias`) are placed in the wire `ArrayRecord`. LOCAL `local_user_row` / `local_user_bias` are saved to disk (line 540–546) and never sent.

`_GLOBAL_PARAM_KEYS` frozenset in `strategy.py` (lines 30–34):
```python
_GLOBAL_PARAM_KEYS = frozenset({
    "item_embeddings.weight",
    "item_bias.weight",
    "global_bias",
})
```

D-23 split-learning invariant: `PersonalizedSplitFedAvg.aggregate_fit` is NOT overridden — it is inherited unchanged from `BaseFedAvg`. Test `test_aggregate_fit_inherited_unchanged` asserts `PersonalizedSplitFedAvg.aggregate_fit is BaseFedAvg.aggregate_fit` (GREEN).

**VERDICT: PASS**

---

## Success Criterion 4: No Held-Out Test Item in Training Negatives + Protocol Fingerprint

### Training negatives exclude held-out item

`train_bpr_mf()` in `task.py` (lines 440–444):
```python
# PSN-03: merge the FND-03 exclusion set so the held-out test positive
# is NEVER sampled as a training negative.
if exclude_items is not None:
    excluded = [int(x) for x in np.asarray(list(exclude_items)).tolist()]
    user_rated_items |= set(excluded)
```

`evaluate_ranking_sampled()` in `task.py` (lines 1092–1130):
```python
excluded_set: Set[int] = set()
if exclude_items is not None:
    excluded_set = set(int(x) for x in ...)
...
all_user_items = train_items | set(test_items) | excluded_set
negative_candidates = list(all_items - all_user_items)
```

The `exclude_items` is sourced from `bundle["exclusion"].for_user(partition_id)` in `client_app.py` (FND-03 `ExclusionTable`), which carries the held-out test positive.

Test `test_train_negatives_exclude_test_positive` (GREEN) confirms the exclusion logic.

### Protocol fingerprint (PSN-07)

`server_app.py` (lines 876–890):
```python
manifest = build_run_manifest(
    run_id=run_id,
    mode_profile=profile,
    run_seed=run_seed,
    mapping_sha256=foundation_idx.mapping_sha256,
    split_hash=foundation_idx.split_hash,
    exclusion_sha256=foundation_idx.exclusion_sha256,
    foundation_contract_sha256=foundation_idx.foundation_contract_sha256,
    raw_data_hash=split_manifest.raw_data_hash,
    builder_version=split_manifest.builder_version,
    overrides=overrides,
    module="personalized",
)
```

D-15 double-write:
- `embed_manifest_in_result(manifest, results_data)` — embeds `_manifest` key in result JSON
- `write_manifest_sibling(manifest, results_filename)` — writes `{run_id}-manifest.json` beside result file

Sufficient-stat metrics: `PersonalizedSplitFedAvg.aggregate_evaluate()` sums `hit_count_*` and `evaluated_users_*` fields across all clients and computes `sampled_hr@10` and `sampled_ndcg@10` as ratios (sum(hit)/sum(users)).

Test `test_build_run_manifest_module_personalized` (GREEN) asserts `manifest.module == "personalized"`.
Test `test_personalized_split_fedavg_aggregate_evaluate_sum_not_average` (GREEN) confirms server-side ratio computation.

**VERDICT: PASS**

---

## Required Artifacts

| Artifact | Expected | Status | Evidence |
|----------|----------|--------|---------|
| `federated-personalized-cf/pyproject.toml` | num-supernodes=6040, partition-mode=natural | VERIFIED | Lines 108, 115 (both federation blocks), line 68 |
| `federated-personalized-cf/federated_personalized_cf/strategy.py` | PersonalizedSplitFedAvg/FedProx + sufficient-stat aggregate_evaluate | VERIFIED | 191-line file with both classes + D-23 identity |
| `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` | local_user_row nn.Parameter(d,), no ghost table | VERIFIED | Line 83 `nn.Parameter(torch.empty(embedding_dim))` |
| `federated-personalized-cf/federated_personalized_cf/models/basic_mf.py` | same single-row contract | VERIFIED | Confirmed by test_basic_mf_no_ghost_table (GREEN) |
| `federated-personalized-cf/federated_personalized_cf/client_app.py` | one-user assertion, discover_only, manifest-sidecar cache | VERIFIED | Lines 501, 616-625, 122-344 |
| `federated-personalized-cf/federated_personalized_cf/task.py` | FND-06 RNG, FND-03 exclusion, no stdlib random | VERIFIED | Lines 41, 440-444, 1092-1130; zero stdlib random grep |
| `federated-personalized-cf/federated_personalized_cf/dataset.py` | foundation adapter, D-02 NotImplementedError | VERIFIED | Lines 307-317, 404-415 |
| `federated-personalized-cf/federated_personalized_cf/server_app.py` | mode resolver, discovery round, seeded sampler, D-27 restore, D-15 double-write | VERIFIED | Lines 271-299 (mode+D-02), 417-450 (discovery), 486-487 (sampler), 770-775 (D-27), 876-908 (manifest) |
| `scripts/clean_cache.py` | standalone CLI, --keep N flag, sig_* preserved | VERIFIED | 155 LOC, executable (rwxrwxr-x), `--keep` arg with `>= 0` validation |
| `scripts/foundation/tests/test_personalized_determinism.py` | @pytest.mark.slow subprocess determinism guard | VERIFIED | File exists, dual-run structure confirmed |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `client_app.@app.train` | `model.get_global_parameters()` | Returns only GLOBAL ArrayRecord | WIRED | Lines 549-550 |
| `client_app.@app.train` | `_save_local_user_state(...)` | Saves 2-key local state with D-10 guard | WIRED | Lines 539-546 |
| `client_app.@app.evaluate` | `discover_only` short-circuit | Returns zero-suffstats without model load | WIRED | Lines 616-625 |
| `client_app.@app.evaluate` | `exclude_items` from ExclusionTable | FND-03 exclusion threaded to evaluate_ranking_sampled | WIRED | Lines 720, 736-747 |
| `server_app.main` | `PersonalizedSplitFedAvg` | Imported from strategy, used as aggregator | WIRED | Lines 63-66, 465-468 |
| `server_app.main` | `_server_sampler = server_rng(run_seed)` | Single-instance seeded sampler for all rounds | WIRED | Line 486 |
| `server_app.main` | discovery round -> `partition_to_node_id` | G-03-01 pre-loop mapping | WIRED | Lines 417-450 |
| `server_app.main` | `build_run_manifest(..., module="personalized")` | PSN-07 fingerprint | WIRED | Lines 876-889 |
| `strategy.aggregate_evaluate` | `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics` | Sum-then-ratio pattern | WIRED | Lines 152-156 |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| PSN-01 | Plan 02 | `pyproject.toml` defaults to num-supernodes=6040 + partition-mode=natural | SATISFIED | pyproject.toml lines 108, 115, 68 |
| PSN-02 | Plan 03 | Benchmark-mode one-user assertion in client_app.py | SATISFIED | client_app.py lines 501, 710; test_client_assertion.py GREEN |
| PSN-03 | Plan 03 | Training negatives exclude held-out test positive | SATISFIED | task.py lines 440-444, 1092-1130; test_task_rng.py GREEN |
| PSN-04 | Plans 03+04 (server half), Plan 05 | Server-side sampling seeded; evaluator RNG fixed; sufficient-stat metrics | SATISFIED | server_app.py line 486 (_server_sampler); strategy.py aggregate_evaluate; test_server_integration.py GREEN |
| PSN-05 | Plan 03 | .embedding_cache path includes run-id + method + num_users + num_items + dim + split_hash; hard-fail on mismatch | SATISFIED | client_app.py _signature_fields + _load_local_user_state D-05 RuntimeError; test_embedding_cache_manifest.py GREEN |
| PSN-06 | Plans 01+03 | Local user embedding row collapses to single-user row | SATISFIED | bpr_mf.py local_user_row nn.Parameter(d,); test_single_row_model.py GREEN |
| PSN-07 | Plan 04 | Module logs FND-07 protocol fingerprint | SATISFIED | server_app.py build_run_manifest(module="personalized") + D-15 double-write; test_build_run_manifest_module_personalized GREEN |

All 7 PSN requirements are satisfied. No orphaned requirements.

---

## Test Suite Summary

**Location:** `federated-personalized-cf/tests/`
**Result:** 34 PASSED in 4.66s (0 failures, 0 errors)

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_strategy.py` | 5 | ALL PASS |
| `test_single_row_model.py` | 7 | ALL PASS |
| `test_dataset_adapter.py` | 3 | ALL PASS |
| `test_client_assertion.py` | 5 | ALL PASS |
| `test_embedding_cache_manifest.py` | 4 | ALL PASS |
| `test_task_rng.py` | 4 | ALL PASS |
| `test_server_integration.py` | 6 | ALL PASS |

Subprocess-determinism test at `scripts/foundation/tests/test_personalized_determinism.py` (1 `@pytest.mark.slow` test) not run in this verification pass — requires a full Flower simulation launch. This is by design (FEDREC_SKIP_SLOW=1 escape hatch).

**Pre-existing known issue (NOT a Phase 3 regression):** `federated-baseline-cf/tests/test_server_integration.py::test_selected_partitions_byte_identical_across_subprocess_reruns` fails due to a Phase 2 path-mismatch bug (results written to `../results/federated/` not `repo_root/results/federated/`). Captured as todo at `.planning/todos/pending/phase2-baseline-determinism-path-bug.md`. Excluded from this verification scope.

---

## Anti-Patterns Scan

Scanned all 7 source files modified/created in Phase 3. No blocking anti-patterns found.

| File | Concern | Finding |
|------|---------|---------|
| `client_app.py` | `TODO`/placeholder | None |
| `client_app.py` | `return null` / stub handlers | None; discover_only returns valid zero-suffstats contract |
| `task.py` | module-level `random.seed` / `np.random.seed` | None (verified by grep + test_random_seed_calls_stripped) |
| `models/bpr_mf.py` | `sample_negatives` still uses `np.random.randint` | Present but flagged inline as legacy fallback; train_bpr_mf bypasses it when `rng is not None` (which client_app always provides). Not a blocker — the path is backward-compat only. |
| `server_app.py` | hardcoded empty data | None; discovery round, strategy wire, cold-start counter all substantive |
| `strategy.py` | `return {}` stub | None; aggregate_evaluate returns computed thesis metrics dict |
| `dataset.py` | dirichlet path | NotImplementedError raised — not a stub, it's the intended behavior |

One informational note: `BPRMF.sample_negatives()` (the instance method, lines 219-272) still uses `np.random.randint`. This is only called in `train_bpr_mf()` when `rng is None` — a legacy-caller fallback. In the Phase 3 cross-device client flow, `client_app.py` always constructs an FND-06 `rng` instance via `np_rng(...)` and passes it to `train_fn`, so the seeded path is always taken. This is ℹ️ Info severity.

---

## Human Verification Required

### 1. End-to-End Flower Simulation

**Test:** Run `cd federated-personalized-cf && flwr run . --run-config "mode=benchmark_cross_device run-seed=42 num-server-rounds=2 wandb-enabled=false"` (requires ML-1M download and simulation runtime)
**Expected:** 6040 supernodes instantiated, discovery round completes, 2 training rounds complete, result JSON written to `../results/federated/` with `_manifest.module == "personalized"` and `cold_starts` block present
**Why human:** Full Flower simulation requires dataset download (~700 MB), several minutes of compute, and live process orchestration. All individual building blocks are verified via unit tests but the integrated run cannot be confirmed programmatically.

---

## Gaps Summary

No gaps. All four success criteria are satisfied by direct code evidence and 34 GREEN tests.

The single human-verification item (end-to-end simulation) is a standard integration check that applies to any phase — it does not indicate a code deficiency.

---

## Commit Correspondence

Phase 3 shipped across 10 feature/test commits:

| Commit | Content |
|--------|---------|
| `a1c2845` | feat(03-02): personalized pyproject cross-device defaults + dev dep (PSN-01) |
| `9acc97d` | refactor(03-02): rip-and-replace dataset.py as foundation adapter (D-17, D-02, PSN-01) |
| `858915d` | feat(03-01): PersonalizedSplitFedAvg + PersonalizedSplitFedProx (D-20, PSN-06) |
| `fabc7eb` | feat(03-01): BPRMF + BasicMF single-row refactor (D-01, D-03, PSN-06) |
| `a563b76` | feat(03-03): task.py FND-06 RNG + FND-03 exclusion + _sample_negatives_seeded (PSN-03) |
| `a0b8bf8` | feat(03-03): client_app mode + assert + manifest-sidecar cache (PSN-02, PSN-04, PSN-05, PSN-06) |
| `969bc6d` | feat(03-04): server_app cross-device migration + D-13 cold-start + D-02 guard (PSN-04, PSN-07) |
| `52f56d6` | test(03-04): server integration tests — PSN-04 reproducibility + D-15 + D-13 + D-02 guard |
| `f906ac5` | feat(03-05): scripts/clean_cache.py — manual N-keep cache pruner (D-10) |
| `23ead96` | test(03-05): subprocess determinism regression guard (PSN-04 + PSN-05) |

---

_Verified: 2026-04-20T07:30:00Z_
_Verifier: Claude (gsd-verifier)_
