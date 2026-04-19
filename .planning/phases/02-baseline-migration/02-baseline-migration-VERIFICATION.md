---
phase: 02-baseline-migration
verified: 2026-04-19T15:30:00Z
status: passed
score: 4/4 success criteria verified + 8/8 requirements satisfied
re_verification:
  previous_status: none
  previous_score: n/a
  gaps_closed: []
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "End-to-end `flwr run .` from inside `federated-baseline-cf/` with no launcher and no run-config overrides"
    expected: "Flower spawns 6040 supernodes; each client reports exactly one raw user (partition-mode=natural + num-supernodes=6040 defaults); benchmark-mode assertion is a no-op because the pyproject default mode is `cross_silo_legacy` (per Plan 02 D-25: launcher is the canonical entry point); `num-supernodes=6040` is visible in Flower's startup log"
    why_human: "22 pytest tests green and launcher dry-run prints `num-supernodes=6040 mode=benchmark_cross_device`, but an actual 6040-supernode Flower simulation is too expensive to run inside verification; human confirms no silent regression when the whole round loop wires together (server_sampler -> client_app train/evaluate -> strategy.aggregate_evaluate -> manifest write)"
  - test: "End-to-end `python scripts/run.py baseline benchmark_cross_device` for at least 2-3 rounds"
    expected: "Both `results/federated/{run_id}_results.json` AND sibling `results/federated/{run_id}-manifest.json` appear; the result JSON has `_manifest.foundation_contract_sha256` (and all other fingerprint keys); `selected_clients_per_round` is a list-of-lists (one inner list per round); W&B dashboard shows `federated-cf-cross-device` project with per-round `round/selected_clients` logs; `sampled_hr@10` and `sampled_ndcg@10` equal `sum(hit_count)/sum(evaluated_users)` (not mean-of-per-client-ratios)"
    why_human: "The five test_server_integration.py tests isolate each aggregation / manifest path synthetically; a real Flower round loop is the only way to confirm they compose correctly when wired together"
  - test: "Re-run the same command with the same `run-seed=42` and compare the two result JSONs"
    expected: "`selected_clients_per_round` is byte-identical between the two runs; per-user sampled 99 negatives are identical (can be spot-checked by dumping the evaluator RNG sequence in one run and diffing against a second run)"
    why_human: "Deterministic client selection at the `server_rng` level is unit-tested in `test_server_rng_reproducible_per_round_selection`; the full-loop determinism (including eval-neg sampling per (user, round)) is an integration invariant that needs an actual run"
---

# Phase 02 Baseline Migration Verification Report

**Phase Goal:** `federated-baseline-cf` runs as a correct cross-device benchmark — 6040 clients, one user per client in benchmark mode, seeded sampling, sufficient-statistic metrics, test-positive excluded from training negatives, and protocol fingerprint logged.

**Verified:** 2026-04-19T15:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (derived from ROADMAP Success Criteria)

| #   | Truth                                                                                                                                                                                                                                                                                | Status     | Evidence                                                                                                                                                                                                                                                                                                                                                    |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `flwr run .` spawns 6040 supernodes under `partition-mode=natural` by default, AND the per-round client-loader for each selected node contains exactly one raw user (benchmark assertion passes)                                                                                        | ✓ VERIFIED | `pyproject.toml` has `options.num-supernodes = 6040` in BOTH `local-simulation` and `local-sim-gpu` (lines 105, 111); `partition-mode = "natural"` default (line 68); `natural_partition_users` creates 1-user-per-partition in `dataset.py:201`; `assert_benchmark_one_user_per_client` wired in `client_app.py:183,318` for BOTH `@app.train` and `@app.evaluate` |
| 2   | With a fixed run seed, two back-to-back runs select the same client IDs per round and log the same selected-client list, AND the sampled evaluator produces the same 99 negatives per (user, round) without reseeding globals                                                         | ✓ VERIFIED | `test_server_rng_reproducible_per_round_selection` PASS; `test_server_rng_different_seeds_different_selections` PASS; live check: `server_rng(42)` produces byte-identical 3x604 selections across two instances; `np_rng(42,5,1,"eval_neg")` produces byte-identical 99 draws; global `np.random.get_state()` unchanged after FND-06 draws                          |
| 3   | Running one round with a user whose held-out test item is known shows that test item never appears among the sampled training negatives for that user                                                                                                                                  | ✓ VERIFIED | `ExclusionTable.build_exclusion` constructs `train_positives[u] ∪ {test_item[u]}` in `exclusion.py:49-53`; `task.train_bpr_mf` merges `exclude_items` into `user_rated_items` before `_sample_negatives_seeded` (`task.py:485-493`); `evaluate_ranking_sampled` folds `excluded_set` into `all_user_items` before neg-candidate pool (`task.py:1225`); `test_train_negatives_exclude_test_positive` PASS; live: user 0's test item `47` IS in `exclusion.for_user(0)` |
| 4   | The result artifact contains a protocol fingerprint (partition mode, num-supernodes, fractions, weight policy, primary evaluator, seeds, checkpoint rule) AND reports headline NDCG@10 / HR@10 computed ONCE at the server from summed `hit_count@10`, `ndcg_sum@10`, `evaluated_users` | ✓ VERIFIED | `RunManifest` dataclass carries all 8 fingerprint fields + 4 IMP-2 hashes + overrides + env metadata; `server_app.py:710-751` calls `generate_run_id` → `verify_bundle` → `build_run_manifest` → `embed_manifest_in_result` → `write_manifest_sibling` (D-15 double-write); `BaselineFedAvg.aggregate_evaluate` at `strategy.py:117-149` sums per-client stats THEN divides once (NOT mean-of-ratios); `test_aggregate_evaluate_uses_sum_not_average` PASS proves 1/100=0.01 not 0.5; `test_embed_and_sibling_double_write_roundtrip` PASS |

**Score:** 4/4 truths verified.

### Required Artifacts

| Artifact                                                                       | Expected                                                                                                     | Status     | Details                                                                                                                                                                             |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `federated-baseline-cf/pyproject.toml`                                          | `partition-mode=natural` default + `num-supernodes=6040` in BOTH federations + foundation-contract keys      | ✓ VERIFIED | 6040 appears in both `local-simulation` and `local-sim-gpu`; `mode=cross_silo_legacy`, `run-seed=42`, `weight-policy=num_positives`, `eval-num-negatives=99`, `checkpoint-rule=best_round_restore` |
| `federated_baseline_cf/dataset.py`                                              | Thin foundation adapter; delegates mapping/split/exclusion to `fedrec_foundation`                             | ✓ VERIFIED | Imports `load_mapping`, `load_split_manifest`, `load_exclusion`, `verify_bundle`, `data_derived`; `_load_foundation_bundle()` caches by `foundation_contract_sha256`; D-17 rip targets GONE (verified by `test_removed_helpers_gone`) |
| `federated_baseline_cf/client_app.py`                                           | Mode resolution + benchmark assertion + strict-contract payloads + per-group routing                          | ✓ VERIFIED | `resolve_mode_defaults` (2 uses), `log_mode_and_overrides` (2), `assert_benchmark_one_user_per_client` (2), `FitMetricsContract` + `validate_fit_metrics` in `@app.train`; `EvaluateMetricsContract` + `validate_evaluate_metrics` in `@app.evaluate`; `_classify_partition_user_group` routes into one of sparse/medium/dense with zeros in the other two |
| `federated_baseline_cf/task.py`                                                 | FND-06 RNG threading, FND-03 exclusion merge, D-24 gradient masking                                           | ✓ VERIFIED | 5 `np_rng(run_seed,…)` calls; 3 D-24 helpers (`_apply_user_row_grad_mask`, `_snapshot_non_user_rows`, `_restore_non_user_rows`); `_sample_negatives_seeded` replaces `BPRMF.sample_negatives`; `exclude_items` merged in both `train_bpr_mf` (line 485-493) and `evaluate_ranking_sampled` (line 1196-1201); **zero** matches for `^import random$` / `random.seed(` / `random.sample(` |
| `federated_baseline_cf/strategy.py`                                             | `BaselineFedAvg(FedAvg)` + `BaselineFedProx(FedProx)` with sum-based `aggregate_evaluate`; `aggregate_fit` inherited | ✓ VERIFIED | `_sum_sufficient_stats` iterates 12 keys; `_sufficient_stats_to_thesis_metrics` divides once per group; `aggregate_fit` untouched (inherited from `FedAvg`/`FedProx`); `test_aggregate_fit_inherited_unchanged` identity-checks `BaselineFedAvg.aggregate_fit is FedAvg.aggregate_fit` |
| `federated_baseline_cf/server_app.py`                                           | Mode-first bootstrap, seeded sampler, strategy wire-up, D-27 best-round, D-15 manifest double-write           | ✓ VERIFIED | `server_rng(run_seed)` (1), `resolve_mode_defaults`/`log_mode_and_overrides` (1 each), `BaselineFedAvg`/`BaselineFedProx` (1 each), `strategy.aggregate_evaluate(…)` (1), `selected_clients_per_round` (4), `best_arrays`/`best_metric`/`best_round_num` (17), `build_run_manifest` + `embed_manifest_in_result` + `write_manifest_sibling` (1 each), **zero** `import random`/`random.sample` |
| `scripts/foundation/fedrec_foundation/fit_metrics.py`                           | Extended `FitMetricsContract` with 12 per-group/overall sufficient-stat fields + sibling `EvaluateMetricsContract` + `EVAL_METRICS_REQUIRED_KEYS` + `validate_evaluate_metrics` | ✓ VERIFIED | 32 grep hits for per-group keys; `EvaluateMetricsContract` dataclass with 3 required + 3 diagnostic + 9 per-group fields; `validate_evaluate_metrics` rejects free-form extras AND enforces required keys |
| `federated-baseline-cf/tests/` (5 files: strategy, dataset_adapter, task_rng, client_assertion, server_integration) | 22 GREEN pytest tests                                                                                         | ✓ VERIFIED | `pytest federated-baseline-cf/tests/` → 22 passed in 4.60s; `pytest scripts/foundation/tests/` → 77 passed in 7.92s                                                                  |
| Committed foundation bundle (`data/derived/{mapping.json, split_manifest.json, exclusion_items.npz, foundation_index.json}`) | Verifiable by `verify_bundle(data_derived())`                                                                | ✓ VERIFIED | `verify_bundle` prints `foundation_contract_sha256=fe181dafe6f7…`; `mapping.num_users=6040`, `mapping.num_items=3706`                                                                 |

### Key Link Verification

| From                                                              | To                                                                       | Via                                                     | Status | Details                                                                                                                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client_app.train`                                                | `fedrec_foundation.mode.assert_benchmark_one_user_per_client`            | benchmark-mode single-user lock                         | WIRED  | `client_app.py:183` — called BEFORE `train_fn`                                                                                                       |
| `client_app.evaluate`                                             | `fedrec_foundation.mode.assert_benchmark_one_user_per_client`            | same lock for evaluate                                  | WIRED  | `client_app.py:318`                                                                                                                                  |
| `client_app.train`                                                | `task.train_bpr_mf`                                                      | RNG + exclusion kwargs threaded                         | WIRED  | `client_app.py:212-217` — `run_seed`, `user_idx`, `round_num`, `exclude_items`, `rng` all passed                                                      |
| `client_app.evaluate`                                             | `task.evaluate_ranking_sampled`                                          | RNG + exclusion kwargs threaded                         | WIRED  | `client_app.py:344-355` — `run_seed`, `user_idx`, `round_num`, `exclude_items` all passed                                                              |
| `client_app.evaluate`                                             | `fedrec_foundation.fit_metrics.EvaluateMetricsContract`                  | strict-contract payload + `validate_evaluate_metrics`   | WIRED  | `client_app.py:402-420` builds full 15-field contract; `validate_evaluate_metrics` called BEFORE return (`client_app.py:420`)                          |
| `task.train_bpr_mf`                                               | FND-03 exclusion set                                                     | union into `user_rated_items` before neg sampling       | WIRED  | `task.py:485-493` — `excluded_set` unioned per user; also seeds current `user_idx` key if loader empty                                                |
| `task.evaluate_ranking_sampled`                                   | FND-03 exclusion set                                                     | union into `all_user_items` before neg-candidate pool   | WIRED  | `task.py:1225` — `all_user_items = train_items \| set(test_items) \| excluded_set`                                                                    |
| `task._sample_negatives_seeded`                                   | `fedrec_foundation.rng.np_rng`                                           | FND-06 deterministic negative sampling                  | WIRED  | `task.py:470` — `np_rng(run_seed, user_idx, round_num, "train_neg")`; `rng.integers` for rejection-sample draw                                          |
| `server_app.@app.main`                                            | `fedrec_foundation.mode.resolve_mode_defaults`                           | mode-first bootstrap                                    | WIRED  | `server_app.py:209` — profile canonical; all hyperparam reads use `int(run_config.get(key, profile.field))`                                              |
| `server_app.@app.main`                                            | `fedrec_foundation.rng.server_rng`                                       | seeded per-round client sampler (BSL-04)                | WIRED  | `server_app.py:350` — `_server_sampler = server_rng(run_seed)`; `_server_sampler.sample(node_ids, num_selected)` per round                               |
| `server_app.@app.main`                                            | `federated_baseline_cf.strategy.BaselineFedAvg`                          | strategy instantiation (BSL-06)                         | WIRED  | `server_app.py:323-329` — replaces raw `FedAvg/FedProx` instantiation; `strategy.aggregate_evaluate(round_num, eval_results, [])` at line 500         |
| `server_app.@app.main`                                            | `fedrec_foundation.manifest.build_run_manifest`                          | protocol fingerprint (BSL-08)                           | WIRED  | `server_app.py:715-727` — all 4 IMP-2 fingerprints + ModeProfile fields + overrides + module passed in                                                  |
| `server_app.@app.main`                                            | `fedrec_foundation.manifest.{embed_manifest_in_result, write_manifest_sibling}` | D-15 double-write                                 | WIRED  | `server_app.py:738, 751` — result JSON gains `_manifest`; sibling `{run_id}-manifest.json` written next to results                                      |
| `BaselineFedAvg.aggregate_evaluate`                               | `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics`          | sum-first divide-once aggregation                       | WIRED  | `strategy.py:144-145` — sums 12 sufficient-stat keys across clients, then divides per group with zero-div safety                                     |
| `dataset._load_foundation_bundle`                                 | `fedrec_foundation.bundle.verify_bundle`                                 | bundle integrity before cache                           | WIRED  | `dataset.py:248` — `idx = verify_bundle(derived)` raises on mismatch; cache keyed by `foundation_contract_sha256`                                      |
| `dataset.load_partition_data` (partition_mode="natural")          | `fedrec_foundation.split.load_split_manifest.test_item_per_user`         | foundation-backed LOO split                             | WIRED  | `dataset.py:340-350` — uses bundle's `test_item_per_user[partition_id]` for the held-out item                                                          |

All 16 critical links WIRED.

### Requirements Coverage

| Requirement | Source Plan      | Description                                                                                                                | Status        | Evidence                                                                                                                                                                 |
| ----------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BSL-01      | Plan 02          | `pyproject.toml` defaults to `num-supernodes=6040` + `partition-mode="natural"`; cross-silo remains explicit opt-in          | ✓ SATISFIED   | `pyproject.toml:105,111` (num-supernodes=6040 in both federations); `pyproject.toml:68` (partition-mode="natural"); comment at both federation blocks points at cross-silo opt-in |
| BSL-02      | Plan 03          | `client_app.py` asserts exactly one local user per client in benchmark mode                                                 | ✓ SATISFIED   | `client_app.py:183` (@app.train) and `client_app.py:318` (@app.evaluate); `test_benchmark_mode_asserts_one_user` + `test_benchmark_mode_skipped_with_override` PASS      |
| BSL-03      | Plan 03          | Training negative sampling uses FND-03 exclusion set so held-out test item is NEVER drawn as training negative              | ✓ SATISFIED   | `task.py:485-493` (train-side merge); `task.py:1225` (eval-side merge); `build_exclusion` constructs `train_positives ∪ {test_item}`; `test_train_negatives_exclude_test_positive` PASS; live check confirmed test_item is in exclusion set |
| BSL-04      | Plan 04          | Server-side `random.sample` replaced by seeded RNG from run seed; selected client IDs logged per round                      | ✓ SATISFIED   | `server_app.py:350` (`_server_sampler = server_rng(run_seed)`); `server_app.py:371` (`.sample(node_ids, …)`); `selected_clients_per_round` appended each round and written to result JSON + W&B; `test_server_rng_reproducible_per_round_selection` + negative guard PASS |
| BSL-05      | Plan 03          | Sampled evaluator no longer calls `random.seed(seed)`; accepts seeded RNG instance from FND-06                              | ✓ SATISFIED   | **ZERO** matches for `^import random$`/`random.seed(`/`random.sample(` across `task.py` + `client_app.py` + `server_app.py`; `np_rng(run_seed, user_idx, round_num, …)` threaded; `test_random_seed_calls_stripped` + `test_evaluate_ranking_sampled_accepts_rng_signature` PASS |
| BSL-06      | Plans 01 + 04    | Clients return sufficient stats; server computes final ratio ONCE                                                          | ✓ SATISFIED   | `FitMetricsContract` + `EvaluateMetricsContract` carry 12/15 per-group/overall sufficient-stat fields; `BaselineFedAvg.aggregate_evaluate` sums then divides once in `strategy.py:117-149`; `server_app.py:500` wires strategy into loop; `test_aggregate_evaluate_uses_sum_not_average` proves 1/100=0.01 (not 0.5) |
| BSL-07      | Plan 03          | Module-level evaluator path uses only FND-04 primary protocol; any secondary `allrank_*` stays explicitly namespaced        | ✓ SATISFIED   | `client_app.py:321` asserts `get_primary_evaluator(mode) == "sampled_loo_99"`; `evaluate_ranking` (all-items) runs only if `enable-ranking-eval=true` and return value is DROPPED so `allrank_*` keys never enter wire payload; `test_get_primary_evaluator_selects_sampled_loo_99` PASS |
| BSL-08      | Plan 04          | Module logs FND-07 protocol fingerprint alongside results                                                                  | ✓ SATISFIED   | `server_app.py:710-751` — `generate_run_id` + `verify_bundle` + `build_run_manifest` + `embed_manifest_in_result` (mutates result dict) + `write_manifest_sibling` (D-15 double-write); all 4 IMP-2 fingerprints + 8 criterion-4 fields + raw_data_hash + overrides + env metadata in `RunManifest`; `test_build_run_manifest_integrates_foundation_index` + `test_embed_and_sibling_double_write_roundtrip` PASS |

**Coverage:** 8/8 BSL requirements satisfied (100%). No orphaned requirements.

### Anti-Patterns Found

| File                                                | Line | Pattern                          | Severity | Impact                                                                                                                                                                         |
| --------------------------------------------------- | ---- | -------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `federated_baseline_cf/dataset.py`                  | 359  | `raise NotImplementedError(...)` | ℹ Info    | Intentional D-17 removal — cross-silo legacy `partition_mode="dirichlet"` is fail-loud, per plan design. Documented in the error message. Benchmark path (`natural`) unaffected. |
| `federated_baseline_cf/server_app.py`               | 90   | `return {}`                      | ℹ Info    | Edge-case guard inside legacy `weighted_average_metrics` helper for RMSE/MAE fallback when no eval examples. Preserved per D-18 surgical scope; not part of thesis-metric path.   |

No blocker or warning anti-patterns. No TODO / FIXME / placeholder markers anywhere in the migrated surface.

Stub / zero-value / empty-data scan of the five modified files (`client_app.py`, `server_app.py`, `task.py`, `dataset.py`, `strategy.py`) returned:

- 0 empty-handler patterns (`=> {}`, `() => {}`, `onClick={() => {}}`) — not applicable (Python, not React).
- 0 hardcoded empty returns that feed renders (the one `return {}` is an explicit empty-dict for rating fallback, not a stub).
- 0 `NotImplementedError` on the benchmark path (the one occurrence is on the deprecated `partition_mode="dirichlet"` branch, which isn't used in cross-device mode).
- 0 `PLACEHOLDER` / `coming soon` / `not yet implemented` strings.
- 0 `console.log`-only implementations (again, Python — no such thing here).

### Human Verification Required

See YAML frontmatter `human_verification:` block for three recommended manual checks. All three address integration-level invariants that automated checks cover synthetically but cannot exercise in a real 6040-client Flower loop.

1. **Full `flwr run .` with default pyproject** — confirm Flower actually spawns 6040 supernodes (pyproject parsing works end-to-end, not just in `tomllib.load`).
2. **Full `python scripts/run.py baseline benchmark_cross_device` for 2-3 rounds** — confirm server_app loop composes correctly (server_sampler + client handlers + strategy + manifest write) and produces both result JSON AND sibling manifest.
3. **Two re-runs with same seed** — confirm end-to-end determinism (including eval-neg draws per (user, round)).

### Gaps Summary

**No gaps.** All four success criteria are satisfied by a combination of:

1. In-tree pyproject defaults (`num-supernodes=6040`, `partition-mode=natural` in both federations).
2. Wired foundation delegates (`_load_foundation_bundle`, `ExclusionTable.for_user`, `verify_bundle`).
3. Client-side contract (`FitMetricsContract` / `EvaluateMetricsContract` + validators + per-group routing).
4. Server-side sum-then-divide aggregation (`BaselineFedAvg.aggregate_evaluate` + `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics`).
5. Seeded RNGs throughout (`server_rng` for client selection; `np_rng` for train/eval negatives; no stdlib `random` residue).
6. Protocol fingerprint manifest (`build_run_manifest` + D-15 double-write).

One item previously flagged in `deferred-items.md` (the D-24 `test_gradient_mask_zeros_non_user_rows` failure owned by Plan 03) is **resolved** — the test is now in the 22/22 GREEN baseline suite (verified with a targeted `pytest … ::test_gradient_mask_zeros_non_user_rows` invocation). `deferred-items.md` as a scope-boundary record of the Wave 2 cross-plan handoff remains accurate but no longer blocks Phase 2.

Requirement coverage is perfect (8/8 BSL-01..08 SATISFIED). REQUIREMENTS.md entries for Phase 2 are already marked `[x] Complete` for every BSL item and the phase-2 section of the status table shows all BSL IDs Complete.

---

_Verified: 2026-04-19T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
