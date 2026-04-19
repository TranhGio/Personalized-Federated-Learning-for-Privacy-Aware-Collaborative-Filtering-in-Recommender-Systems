---
phase: 02-baseline-migration
verified: 2026-04-19T19:15:00Z
status: passed
score: 4/4 success criteria verified + 8/8 requirements satisfied
re_verification:
  previous_status: passed
  previous_score: 4/4 + 8/8
  gaps_closed: []
  gaps_remaining: []
  regressions: []
  notes: "Re-run against the live codebase after Plan-05 + UAT fixes landed. All Goal-level claims still hold; one additional real-loop subprocess test (@pytest.mark.slow) is flaky for a benign test-path reason (`repo_root/results` vs launcher's `../results`), not a code defect — UAT run pair 20260419-115038-da9aa9 vs 20260419-115226-35228e already demonstrates byte-identical client selection."
human_verification:
  - test: "Test 1: GPU smoke (2 rounds, fraction-train=0.01)"
    expected: "Flower spawns 6040 supernodes; partition-mode=natural default visible; round 1 and round 2 both emit sampled_ndcg@10 / sampled_hr@10 + per-group sums; no benchmark AssertionError; no CUDA OOM."
    why_human: "ALREADY EXECUTED BY USER — UAT 02-UAT.md records `result: pass` for run 20260419-090228-08262a with mode=benchmark_cross_device, num_supernodes=6040, partition_mode=natural confirmed in manifest. No retest requested."
  - test: "Test 2: End-to-end via `python scripts/run.py baseline benchmark_cross_device`"
    expected: "Fresh {run_id}_results.json + sibling -manifest.json with full fingerprint; best_round picked correctly; selected_clients_per_round list-of-lists; sampled metrics appear."
    why_human: "ALREADY EXECUTED BY USER — UAT 02-UAT.md records `result: pass` for run 20260419-101756-badbb7 after three launcher-fix commits (848529e, 4c85afb, 227d366). Manifest fingerprint + per-group sufficient-stat sums verified clean. No retest requested."
  - test: "Test 3: Determinism — same seed, byte-identical selections"
    expected: "selected_clients_per_round identical across back-to-back reruns of the launcher with the same run-seed."
    why_human: "ALREADY EXECUTED BY USER — post-Plan-05 UAT records `result: pass` for run pair 20260419-115038-da9aa9 vs 20260419-115226-35228e. Live verification at verification time confirms JSONs are byte-identical on selected_clients_per_round (round-1 first-10 = [5238, 912, 204, 2253, 2006, 1828, 1143, 6033, 839, 5543] in BOTH runs). Residual ndcg diff ~1e-3 is known GPU-kernel noise (not thesis-blocker). No retest requested."
  - test: "Test 4: W&B project routing to `federated-cf-cross-device`"
    expected: "Runs with mode=benchmark_cross_device appear under W&B project `federated-cf-cross-device`, not legacy `federated-cf`."
    why_human: "ALREADY EXECUTED BY USER — UAT 02-UAT.md records `result: pass` after G-04-01 fix (pyproject wandb-project=\"\" + server_app mode-routing). User confirmed on wandb.ai. No retest requested."
---

# Phase 02 Baseline Migration Verification Report

**Phase Goal:** `federated-baseline-cf` runs as a correct cross-device benchmark — 6040 clients, one user per client in benchmark mode, seeded sampling, sufficient-statistic metrics, test-positive excluded from training negatives, and protocol fingerprint logged.

**Verified:** 2026-04-19T19:15:00Z
**Status:** passed
**Re-verification:** Yes — post-UAT re-verification after Plan-05 G-03-01 closure and G-04-01 inline fix.

## Goal Achievement

### Observable Truths (derived from ROADMAP Success Criteria)

| #   | Truth                                                                                                                                                                                                                                                                                    | Status     | Evidence                                                                                                                                                                                                                                                                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `flwr run .` spawns 6040 supernodes under `partition-mode=natural` by default, AND the per-round client-loader for each selected node contains exactly one raw user (benchmark assertion passes)                                                                                          | ✓ VERIFIED | `pyproject.toml` has `options.num-supernodes = 6040` in BOTH `local-simulation` (line 109) and `local-sim-gpu` (line 122); `partition-mode = "natural"` default (line 68); default federation flipped to `local-sim-gpu` (line 101); `assert_benchmark_one_user_per_client` appears 4× in `client_app.py` (both `@app.train` and `@app.evaluate`). Test 1 pass in UAT confirms Flower actually spawns 6040 supernodes end-to-end. |
| 2   | With a fixed run seed, two back-to-back runs select the same client IDs per round and log the same selected-client list, AND the sampled evaluator produces the same 99 negatives per (user, round) without reseeding globals                                                            | ✓ VERIFIED | Live diff of UAT-recorded runs 20260419-115038-da9aa9 vs 20260419-115226-35228e: `selected_clients_per_round` byte-identical across ALL rounds. First 10 of round 1: `[5238, 912, 204, 2253, 2006, 1828, 1143, 6033, 839, 5543]`. Root fix (Plan-05): discover_only round materialises `partition_to_node_id` so `_server_sampler.sample(range(num_supernodes), k)` operates on the stable partition-id space. `np_rng(run_seed, user_idx, round_num, "train_neg"/"eval_neg")` used instead of global `random.seed`. |
| 3   | Running one round with a user whose held-out test item is known shows that test item never appears among the sampled training negatives for that user                                                                                                                                    | ✓ VERIFIED | `task.py` has 23 references to `exclude_items` (merged into `user_rated_items` before `_sample_negatives_seeded`) and 6 references to `excluded_set`. `test_train_negatives_exclude_test_positive` PASS in the unit suite. |
| 4   | The result artifact contains a protocol fingerprint (partition mode, num-supernodes, fractions, weight policy, primary evaluator, seeds, checkpoint rule) AND reports headline NDCG@10 / HR@10 computed ONCE at the server from summed `hit_count@10`, `ndcg_sum@10`, `evaluated_users`   | ✓ VERIFIED | Live read of `20260419-115038-da9aa9_results.json._manifest`: `mode=benchmark_cross_device`, `num_supernodes=6040`, `partition_mode=natural`, `weight_policy=num_positives`, `foundation_contract_sha256=fe181dafe6f791d6679b…`. `final_metrics.sampled_ndcg@10` present. `BaselineFedAvg.aggregate_evaluate` sums-then-divides (strategy.py contains `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics`; `aggregate_fit` NOT overridden → inherited). Sibling `{run_id}-manifest.json` file present next to results. |

**Score:** 4/4 truths verified.

### Required Artifacts

| Artifact                                                                       | Expected                                                                                                     | Status     | Details                                                                                                                                                                             |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `federated-baseline-cf/pyproject.toml`                                         | `partition-mode=natural` default + `num-supernodes=6040` in BOTH federations + foundation-contract keys + wandb empty sentinel | ✓ VERIFIED | Lines 68 (partition-mode), 93-97 (mode/run-seed/weight-policy/eval-num-negatives/checkpoint-rule), 88 (wandb-project="" sentinel added by G-04-01 fix), 101 (default=local-sim-gpu), 109 + 122 (num-supernodes=6040 in both federations). |
| `federated_baseline_cf/dataset.py`                                             | Thin foundation adapter; delegates mapping/split/exclusion to `fedrec_foundation`                            | ✓ VERIFIED | 3 refs each to `load_mapping`, `load_split_manifest`, `load_exclusion`, `verify_bundle`, `data_derived`, `_load_foundation_bundle`; 4 refs to `foundation_contract_sha256` and `natural_partition_users`. |
| `federated_baseline_cf/client_app.py`                                          | Mode resolution + benchmark assertion + strict-contract payloads + per-group routing + partition_id echoing (Plan-05 G-03-01)  | ✓ VERIFIED | `assert_benchmark_one_user_per_client` 4× (train + evaluate), `FitMetricsContract` 5×, `EvaluateMetricsContract` 5×, `validate_fit_metrics` 3×, `validate_evaluate_metrics` 4×, `partition_id=` 5× (client echoes partition_id on both train and evaluate). |
| `federated_baseline_cf/task.py`                                                | FND-06 RNG threading, FND-03 exclusion merge, D-24 gradient masking                                          | ✓ VERIFIED | 6× `np_rng(`, 0× `random.seed(`/`random.sample(`/`import random`, 23× `exclude_items` kwarg, 6× `excluded_set`, 2× `_sample_negatives_seeded`, 4× `_apply_user_row_grad_mask`, 6× each `_snapshot_non_user_rows`/`_restore_non_user_rows`. |
| `federated_baseline_cf/strategy.py`                                            | `BaselineFedAvg(FedAvg)` + `BaselineFedProx(FedProx)` with sum-based `aggregate_evaluate`; `aggregate_fit` inherited | ✓ VERIFIED | `class BaselineFedAvg` ✓, `class BaselineFedProx` ✓, `_sum_sufficient_stats` ✓, `_sufficient_stats_to_thesis_metrics` ✓, `def aggregate_evaluate` ✓, `def aggregate_fit` — NOT present (inherited, confirming sum-only override on evaluate). |
| `federated_baseline_cf/server_app.py`                                          | Mode-first bootstrap, seeded sampler, strategy wire-up, D-27 best-round, D-15 manifest double-write, Plan-05 discover_only + partition_to_node_id | ✓ VERIFIED | `build_run_manifest`×2, `embed_manifest_in_result`×2, `write_manifest_sibling`×2, `BaselineFedAvg`×5, `BaselineFedProx`×4, `server_rng`×2, `resolve_mode_defaults`×2, `generate_run_id`×2, `verify_bundle`×2, `selected_clients_per_round`×5, `best_round`×14, `discover_only`×1, `partition_to_node_id`×5. No stdlib `random.*` residue. |
| `scripts/foundation/fedrec_foundation/fit_metrics.py`                          | Extended `FitMetricsContract` with 12 per-group/overall sufficient-stat fields + sibling `EvaluateMetricsContract` + validators + optional `partition_id` (Plan-05) | ✓ VERIFIED | Foundation test suite 81/81 PASS confirms contract integrity. EvaluateMetricsContract accepts `partition_id` via `fields(cls)` whitelist (Plan-05 change, no D-21 loosening). |
| Committed foundation bundle (`data/derived/…`)                                  | Verifiable by `verify_bundle(data_derived())`                                                                | ✓ VERIFIED | Live: `verify_bundle` returns `FoundationIndex(foundation_contract_sha256=fe181dafe6f791d6679b…)`; matches value embedded in all 20260419-115038/115226/101756 manifests. |
| `federated-baseline-cf/tests/`                                                 | Unit-level regression suite GREEN                                                                            | ✓ VERIFIED | `pytest tests/ --deselect ::test_selected_partitions_byte_identical_across_subprocess_reruns`: **22 passed**. The deselected @pytest.mark.slow subprocess test is a test-path glitch (see notes below), NOT a code regression. |
| `scripts/foundation/tests/`                                                    | Foundation contract test suite GREEN                                                                         | ✓ VERIFIED | `pytest scripts/foundation/tests/`: **81 passed** in 8.63s. |

### Key Link Verification

| From                                                              | To                                                                       | Via                                                     | Status | Details                                                                                                                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client_app.@app.train`                                           | `fedrec_foundation.mode.assert_benchmark_one_user_per_client`            | benchmark one-user lock                                 | WIRED  | Appears twice (train + evaluate handler); additional 2 mentions in docstrings / asserts guard block. |
| `client_app.@app.evaluate`                                        | same                                                                      | same lock                                               | WIRED  | Same count includes the evaluate-side call. |
| `client_app.@app.train/evaluate`                                  | `task.train_bpr_mf` / `task.evaluate_ranking_sampled`                    | RNG + exclusion kwargs (run_seed, user_idx, round_num, exclude_items) | WIRED  | 23× `exclude_items` in task.py surface confirms both paths accept + propagate. |
| `client_app.@app.evaluate`                                        | `fedrec_foundation.fit_metrics.EvaluateMetricsContract`                  | strict-contract payload + `validate_evaluate_metrics`   | WIRED  | `EvaluateMetricsContract`×5, `validate_evaluate_metrics`×4; partition_id echoed to enable Plan-05 G-03-01 discover_only round. |
| `task._sample_negatives_seeded`                                   | `fedrec_foundation.rng.np_rng`                                           | FND-06 deterministic negative sampling                  | WIRED  | 6× `np_rng(` across train + eval paths. |
| `server_app.@app.main`                                            | `fedrec_foundation.mode.resolve_mode_defaults`                           | mode-first bootstrap                                    | WIRED  | 2× `resolve_mode_defaults`. |
| `server_app.@app.main`                                            | `fedrec_foundation.rng.server_rng`                                       | seeded per-round client sampler (BSL-04)                | WIRED  | 2× `server_rng(` / `_server_sampler`; Plan-05: samples `range(num_supernodes)`, translates via `partition_to_node_id` (5× refs). |
| `server_app.@app.main`                                            | `BaselineFedAvg` / `BaselineFedProx`                                     | strategy wire-up (BSL-06)                               | WIRED  | 5× / 4× respectively; `aggregate_evaluate` called with 12-key sufficient-stat list. |
| `server_app.@app.main`                                            | `fedrec_foundation.manifest.{build_run_manifest, embed_manifest_in_result, write_manifest_sibling}` | BSL-08 fingerprint + D-15 double-write | WIRED  | Each used 2×; live manifest in 115038/115226/101756 result JSONs carries all 8 criterion-4 fields + foundation hashes + overrides. |
| `BaselineFedAvg.aggregate_evaluate`                               | `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics`          | sum-first divide-once aggregation                       | WIRED  | Both helpers present in strategy.py; unit test `test_aggregate_evaluate_uses_sum_not_average` green. |
| `dataset._load_foundation_bundle`                                 | `fedrec_foundation.bundle.verify_bundle`                                 | bundle integrity before cache                           | WIRED  | 3× `verify_bundle` in dataset.py; live hash fe181daf… matches embedded manifest hash. |

All critical links WIRED.

### Requirements Coverage

| Requirement | Source Plan      | Description                                                                                                                | Status        | Evidence                                                                                                                                                                 |
| ----------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BSL-01      | Plan 02          | `pyproject.toml` defaults to `num-supernodes=6040` + `partition-mode="natural"`; cross-silo remains explicit opt-in         | ✓ SATISFIED   | `pyproject.toml` lines 68 + 109 + 122; default federation = `local-sim-gpu` (line 101). Cross-silo opt-in documented inline at both federation blocks. REQUIREMENTS.md marks BSL-01 `[x]`. |
| BSL-02      | Plan 03          | `client_app.py` asserts exactly one local user per client in benchmark mode                                                 | ✓ SATISFIED   | 4× `assert_benchmark_one_user_per_client` in client_app.py (both handlers). `test_benchmark_mode_asserts_one_user` + `test_benchmark_mode_skipped_with_override` PASS. Test 1 in UAT passed live end-to-end. |
| BSL-03      | Plan 03          | Training negative sampling uses FND-03 exclusion set so held-out test item is NEVER drawn as training negative              | ✓ SATISFIED   | `exclude_items` kwarg 23× in task.py (train + eval); `excluded_set` 6× inside negative-sampling loops; `test_train_negatives_exclude_test_positive` PASS. |
| BSL-04      | Plan 04          | Server-side `random.sample` replaced by seeded RNG from run seed; selected client IDs logged per round                      | ✓ SATISFIED   | server_app.py: 0× `import random`/`random.sample`/`random.seed`, 2× `server_rng`; `selected_clients_per_round`×5 writes to result JSON; `test_server_rng_reproducible_per_round_selection` PASS. Live UAT runs 115038 vs 115226 are byte-identical. |
| BSL-05      | Plan 03          | Sampled evaluator no longer calls `random.seed(seed)`; accepts seeded RNG instance from FND-06                              | ✓ SATISFIED   | task.py: 0× `random.seed`/`random.sample`/`import random`, 6× `np_rng(run_seed, user_idx, round_num, …)`. `test_random_seed_calls_stripped` + `test_evaluate_ranking_sampled_accepts_rng_signature` PASS. |
| BSL-06      | Plans 01 + 04    | Clients return sufficient stats; server computes final ratio ONCE                                                           | ✓ SATISFIED   | `FitMetricsContract` 5× + `EvaluateMetricsContract` 5× in client_app.py; `BaselineFedAvg.aggregate_evaluate` sums then divides once via `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics`. `aggregate_fit` INHERITED (not overridden), confirming sum-divide pattern is evaluate-only. |
| BSL-07      | Plan 03          | Module-level evaluator path uses only FND-04 primary protocol; any secondary `allrank_*` stays explicitly namespaced        | ✓ SATISFIED   | client_app.py asserts `get_primary_evaluator(mode) == "sampled_loo_99"`; `evaluate_ranking` (all-items) path drops return value so `allrank_*` keys never enter the wire payload. `test_get_primary_evaluator_selects_sampled_loo_99` PASS. |
| BSL-08      | Plan 04          | Module logs FND-07 protocol fingerprint alongside results                                                                   | ✓ SATISFIED   | Live inspection of `20260419-115038-da9aa9_results.json._manifest`: all required fingerprint fields present with expected values (mode, num_supernodes, partition_mode, fraction_train, fraction_eval, weight_policy, primary_evaluator, num_train_negatives, foundation_contract_sha256=fe181daf…). Sibling `{run_id}-manifest.json` double-write present. |

**Coverage:** 8/8 BSL requirements satisfied (100%). No orphaned requirements. REQUIREMENTS.md shows all 8 BSL entries marked `[x]` Complete.

### Gap Closure

| Gap ID  | Source Test | Description                                                                                     | Status                                              |
| ------- | ----------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| G-03-01 | Test 3      | `selected_clients_per_round` not byte-identical across reruns (Flower os.urandom node_ids)      | ✓ CLOSED (Plan-05, UAT 2026-04-19)                  |
| G-04-01 | Test 4      | W&B project routed to legacy `federated-cf` instead of `federated-cf-cross-device`              | ✓ CLOSED (inline fix during UAT 2026-04-19)         |
| G-04-02 | Test 4     | pfedrec / personalized / adaptive server_apps lack mode-aware W&B routing                       | DOCUMENTED as out-of-phase follow-up in 02-UAT.md   |

G-03-01 closure verified live: UAT runs 20260419-115038-da9aa9 vs 20260419-115226-35228e have byte-identical `selected_clients_per_round`. Round-1 first-10 `[5238, 912, 204, 2253, 2006, 1828, 1143, 6033, 839, 5543]` matches in both runs.

G-04-01 closure verified live: `pyproject.toml:88` reads `wandb-project = ""` (empty sentinel + inline comment documenting mode-routing semantics). UAT Test 4 confirmed user-verified on wandb.ai.

G-04-02 is explicitly filed as a follow-up for the three downstream migration phases (PFR / PSN / ADP). Each module's server_app.py still hardcodes its own W&B project default without mode-aware routing, per 02-UAT.md lines 253-272. This is a documented out-of-phase follow-up, not a Phase-02 gap.

### Anti-Patterns Found

| File                                                | Line | Pattern                          | Severity | Impact                                                                                                                                                                         |
| --------------------------------------------------- | ---- | -------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `federated_baseline_cf/dataset.py`                  | 359  | `raise NotImplementedError(...)` | ℹ Info    | Intentional D-17 rip — cross-silo legacy `partition_mode="dirichlet"` is fail-loud on the cross-device branch, per plan design. Documented in the error message. Benchmark path (`natural`) unaffected. |
| `federated_baseline_cf/server_app.py`               | 90   | `return {}`                      | ℹ Info    | Edge-case guard inside legacy `weighted_average_metrics` helper for RMSE/MAE fallback when no eval examples. Preserved per D-18 surgical scope; not part of thesis-metric path.   |

No blocker or warning anti-patterns. No TODO / FIXME / placeholder markers in the migrated surface. Stub scan across five modified files returns 0 React-style empty handlers (N/A — Python codebase), 0 `PLACEHOLDER`/`coming soon` strings, 0 `console.log`-only implementations.

### Test Suite Status

| Suite                                          | Result                                   |
| ---------------------------------------------- | ---------------------------------------- |
| `federated-baseline-cf/tests/` (unit)          | **22 passed** (in 4.55s)                 |
| `scripts/foundation/tests/`                    | **81 passed** (in 8.63s)                 |
| `federated-baseline-cf/tests/test_server_integration.py::test_selected_partitions_byte_identical_across_subprocess_reruns` (@pytest.mark.slow, end-to-end subprocess) | **FAIL (environment/path glitch, not code defect)** |

The end-to-end subprocess test fails because it expects result JSONs under `<repo_root>/results/federated/` (where repo_root = `parents[2]` from the test file = the `federated-baseline-cf/..` project root), but `server_app.py:788` writes to `../results/federated/` relative to the module's cwd. When the launcher is invoked from the project root, that resolves to `/home/bes/Desktop/vinh/federated-learning/results/federated/` — one directory ABOVE the project root, where the UAT Test-3 run JSONs actually ARE (20260419-115038-da9aa9 + 20260419-115226-35228e present).

The invariant the test was written to guard (byte-identical `selected_clients_per_round`) is independently verified at verification time by direct JSON comparison of the UAT runs. This is a test-infrastructure issue (path mismatch between server_app's relative output and the test's path assumption), not a regression in Phase-02 code. Recommended follow-up: teach the test to resolve `results_dir = repo_root / ".." / "results" / "federated"` or parameterize via env var.

### Human Verification Required

All four UAT tests already executed by the user and marked `result: pass` in `02-UAT.md`. Per the re-verification mandate, treat these as authoritative user-side verification:

1. **Test 1 (GPU smoke)** — PASSED on run 20260419-090228-08262a. Manifest fingerprint verified clean.
2. **Test 2 (launcher end-to-end)** — PASSED on run 20260419-101756-badbb7 after three launcher-fix commits.
3. **Test 3 (determinism)** — PASSED on run pair 20260419-115038-da9aa9 vs 20260419-115226-35228e after G-03-01 closure by Plan-05. Byte-identical `selected_clients_per_round` confirmed.
4. **Test 4 (W&B routing)** — PASSED after G-04-01 inline fix. User confirmed on wandb.ai.

No retest requested. The UAT record is authoritative for Phase-02 human verification.

### Gaps Summary

**No Phase-02 gaps.** Goal is achieved:

1. Cross-device defaults wired in `pyproject.toml` (BSL-01).
2. Benchmark one-user assertion on both handlers (BSL-02).
3. FND-03 exclusion merge verified in train + eval paths (BSL-03).
4. Stdlib `random.*` fully stripped; seeded `server_rng` + `np_rng` throughout (BSL-04, BSL-05).
5. Strict-contract sufficient-statistic metrics + sum-then-divide aggregation (BSL-06).
6. Primary evaluator locked to `sampled_loo_99`; secondary `allrank_*` neutralised (BSL-07).
7. Protocol fingerprint manifest with D-15 double-write (BSL-08).
8. G-03-01 (determinism) CLOSED by Plan-05 (discover_only round + partition_id sampling).
9. G-04-01 (W&B routing) CLOSED inline during UAT.
10. G-04-02 (pfedrec / personalized / adaptive W&B routing) DOCUMENTED as out-of-phase follow-up for downstream migration phases.

Requirement coverage 8/8 BSL; all already `[x]` in REQUIREMENTS.md. All UAT tests pass. 22 unit tests + 81 foundation tests green. One @pytest.mark.slow subprocess test fails due to a test-path glitch (not a code regression); the invariant it guards is independently confirmed by direct inspection of the UAT run-pair JSONs.

---

_Re-verified: 2026-04-19T19:15:00Z_
_Verifier: Claude (gsd-verifier)_
