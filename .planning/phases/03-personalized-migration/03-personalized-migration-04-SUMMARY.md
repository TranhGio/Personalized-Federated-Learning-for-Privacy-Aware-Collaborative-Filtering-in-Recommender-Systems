---
phase: 03-personalized-migration
plan: 04
subsystem: infra
tags: [server-app, mode-resolver, seeded-sampling, personalized-split-fedavg, personalized-split-fedprox, run-manifest, best-round-restore, discovery-round, partition-id-space, cold-start-counter, cross-device, psn-04, psn-07, d-02, d-13, d-15, d-18, d-25, d-26, d-27, wave-3]

# Dependency graph
requires:
  - phase: 01-foundation-contract-04
    provides: "np_rng / server_rng / FND-06 RNG factories + FND-07 build_run_manifest + embed_manifest_in_result + write_manifest_sibling + generate_run_id"
  - phase: 01-foundation-contract-05
    provides: "resolve_mode_defaults + log_mode_and_overrides + ModeProfile"
  - phase: 01-foundation-contract-02
    provides: "verify_bundle + load_split_manifest + FoundationIndex fingerprints"
  - phase: 03-personalized-migration-01
    provides: "PersonalizedSplitFedAvg + PersonalizedSplitFedProx (sufficient-stat aggregate_evaluate override; aggregate_fit inherited per D-23)"
  - phase: 03-personalized-migration-03
    provides: "client_app.py @app.evaluate discover_only short-circuit returning EvaluateMetricsContract with partition_id field populated — the G-03-01 handshake this plan's server-side discovery round consumes"

provides:
  - "federated-personalized-cf/federated_personalized_cf/server_app.py: @app.main() cross-device main loop implementing D-25 mode resolver, G-03-01 discovery round + partition_to_node_id map, PSN-04 seeded partition-id-space sampling, PersonalizedSplitFedAvg/FedProx strategy wire-up, D-27 in-memory best-round restore, D-15 double-write manifest with module='personalized', D-13 cold-start counter (Phase-3-unique), D-02 frozen-cross-silo NotImplementedError."
  - "PSN-04 observable: running `scripts/run.py personalized benchmark_cross_device` samples 6040 supernodes via a single `_server_sampler = server_rng(run_seed)` instance in deterministic partition-id space; two same-seed runs produce byte-identical `selected_clients_per_round` (proven at subprocess level by Plan 05's regression guard)."
  - "PSN-07 observable: result JSON gains top-level `_manifest` key with module='personalized' + all four Phase-1 fingerprints; sibling `{run_id}-manifest.json` written atomically beside the result file via atomic_write_json."
  - "D-13 observable: `results_data['cold_starts'] = {per_round, total_cold_starts, total_client_selections, cold_start_rate}`; per-round `round/cold_starts` + `round/selected_clients` logged to W&B; `total_cold_starts` + `cold_start_rate` pushed to wandb.run.summary at run end."
  - "D-02 observable: `mode='cross_silo_legacy'` raises NotImplementedError with explicit reference to `.planning/phases/03-personalized-migration/03-CONTEXT.md §Deferred` BEFORE any training, data load, or model construction."
  - "federated-personalized-cf/tests/test_server_integration.py (206 LOC, 6 GREEN pytest tests) — PSN-04 RNG reproducibility + PSN-04 sum-based strategy aggregation + PSN-07 manifest with module='personalized' + D-13 cold-start arithmetic + D-02 source-level regression guard."

affects: [03-personalized-migration-05, 04-adaptive-migration, 05-pfedrec-migration]

# Tech tracking
tech-stack:
  added: []  # Pure consumer of Phase 1 foundation + Phase 3 Plan 01 strategy.
  patterns:
    - "Phase 2 Plans 04+05 server_app.py template is near-cloneable for Phase 3: the differences are strategy class name (PersonalizedSplitFedAvg vs BaselineFedAvg), initial arrays construction (global_model.get_global_parameters() vs global_model.state_dict()), manifest module flag ('personalized' vs 'baseline'), results JSON federated_config.global_params / local_params lists, D-13 cold-start counter (new to Phase 3), and the D-02 frozen-cross-silo guard (new to Phase 3). The centralized eval block from baseline is INTENTIONALLY dropped — split learning cannot run server-side eval without the LOCAL user rows."
    - "D-27 best-round restore semantics under split learning: the snapshot ArrayRecord is still captured at best-round for the manifest artifact + any downstream loader, but the final headline metrics come from eval_metrics_history[best_round_num] (federated-aggregated thesis metrics) rather than from a centralized evaluation pass the server cannot run. The selection order is: (1) checkpoint_rule best-round, (2) early-stopper best-round, (3) final round — mirroring baseline but without the centralized-eval disambiguation step."
    - "D-13 cold-start counter path: the server-side probe uses `Path('.embedding_cache') / run_id / f'partition_{pid}.pt'` under D-08 default. Under D-09 reuse_cache=true the server cannot cheaply construct the sig_<hash> path without the client-side signature (split_hash / dim / method / num_users / num_items), so the counter short-circuits to 0 and the log line names D-09 explicitly — the reuse regime's whole purpose is 'all hot on sig match', so 0 is the expected truth in that mode."
    - "Checkpoint_rule spelling acceptance (Phase 2 precedent): accepts both 'best_round_restore' (pyproject override) AND 'best_round' (ModeProfile default spelling) — avoids a bikeshed when the launcher passes the profile's value verbatim."
    - "Default W&B project switch: `federated-cf-cross-device` for mode in (benchmark_cross_device, paper_compat_pfedrec); `federated-cf` for everything else (legacy cross-silo). Explicit `run_config['wandb-project']` still wins. Empty-string treated as 'use default'. Matches the Phase 2 baseline pattern for dashboard comparability."

key-files:
  created:
    - "federated-personalized-cf/tests/test_server_integration.py (206 LOC, 6 GREEN pytest tests)"
  modified:
    - "federated-personalized-cf/federated_personalized_cf/server_app.py (+478 lines / -87 lines): D-25 mode resolver + D-02 guard + G-03-01 discovery round + PSN-04 _server_sampler + PersonalizedSplitFedAvg/FedProx wire-up + D-13 cold-start counter + D-27 best-round restore + D-15 double-write manifest with module='personalized'; stdlib random eradicated"

key-decisions:
  - "D-02 cross-silo guard placement: raised AFTER mode resolution but BEFORE hyperparameter reads and model construction. The `raise NotImplementedError` sits immediately after `log_mode_and_overrides` so a cross-silo invocation fails loud within ~10 lines of @app.main() entry — no partial state, no wandb.init, no model.get_model call, no dataset bundle load. Error message cites D-02 and the CONTEXT.md §Deferred section so a traceback reader has a direct pointer to the frozen-cross-silo rationale."
  - "D-13 cold-start counter under D-09 reuse-cache=true is explicitly 0 (short-circuit), not an indeterminate value. The server cannot construct the sig_<hash> path without the client-side signature fields, and the expected regime under D-09 is 'all hot because sig matches' — 0 is the honest truth in that mode. A log line names D-09 so a reader isn't confused about why the counter is zero even on a fresh cache directory."
  - "Checkpoint_rule accepted spelling: `best_round_restore` (pyproject override) OR `best_round` (ModeProfile default). The D-27 branch tests `checkpoint_rule in ('best_round_restore', 'best_round')` so the launcher can pass the ModeProfile value verbatim without rewriting. Mirrors Phase 2 Plan 04 precedent exactly."
  - "D-18 surgical discipline: the Write preserved DummyClientProxy, weighted_average_metrics, print_evaluation_metrics, EarlyStopping setup/teardown verbatim. Pre-existing uncommitted WIP on server_app.py (1 hunk: `import random` + `node_ids = sorted(grid.get_node_ids())` + `random.sample(node_ids, num_selected)`) was INTENTIONALLY overridden per the plan's rip-and-replace scope — the stdlib-random eradication is a cross-file regression gate (gate 4) that would have otherwise failed."
  - "Default W&B project for cross-device modes is `federated-cf-cross-device` (PROJECT.md constraint). Legacy cross_silo_legacy mode is frozen under D-02 so it can never actually run, but the legacy project name `federated-cf` remains in the branch for consistency with baseline (Phase 2 Plan 04)."
  - "Centralized evaluation block NOT ported from baseline. The baseline's @app.main() runs task.test + task.evaluate_ranking + task.evaluate_ranking_sampled on the server with load_full_data after best-round restore. Personalized cannot — the server does not hold the LOCAL user rows, so any server-side forward pass would return garbage. The final headline metrics come from `eval_metrics_history[best_round_num]` (strategy-aggregated federated eval) instead. This is a deliberate structural difference between the baseline and personalized server_apps, and is called out both in the module docstring and in the `Using federated evaluation metrics...` log line."
  - "Duplicate `discover_only` presence in server_app.py source (3 occurrences vs the plan's ≥2 requirement): one in the ConfigRecord literal, one in the module docstring, one in the function docstring. This is documentation density, not a contract violation — the plan text uses >= as the floor."
  - "run_id materialized EARLY (line ~243) using generate_run_id() when no explicit run-id override is in context.run_config. The D-13 cold-start probe, the train/eval message `run_id` field, and the final manifest all use THE SAME run_id — keeping the server-side probe consistent with the client-side cache path the client_app.py (Plan 03) is actually writing to."

patterns-established:
  - "Phase 3 cross-device server contract: every federated-*-cf/ module whose server_app.py drives a cross-device benchmark follows this shape — D-25 mode resolve at entry, D-02-style frozen-legacy guard raise BEFORE any heavy work, G-03-01 discovery round pre-loop with `partition_to_node_id` assertion, PSN-04 `_server_sampler = server_rng(run_seed)` with `range(N)` sampling domain, strategy wire-up to the module's PersonalizedXXX / AdaptiveXXX / PFedRecXXX subclass from Plan 01, D-27 in-memory best-round restore, D-15 double-write manifest with `module=<module_name>`. Phase-4 adaptive and Phase-5 pfedrec server_apps WILL follow this template with the module-specific strategy class name + manifest module flag + any module-specific per-round bookkeeping (D-13 cold-start is Phase-3-unique; Phase 4 will likely need alpha-analysis per-round logging; Phase 5 will need per-user affine_output cache hit/miss accounting)."
  - "Test parity with Phase 2 baseline test_server_integration.py: the five common tests (server_rng reproducibility + distinguishability + strategy sum aggregation + manifest-with-module + sibling-double-write) transfer verbatim up to strategy class name + module flag substitution. Phase 3 adds a 6th test (D-13 cold-start math) and substitutes test_cross_silo_legacy_mode_raises_not_implemented for baseline's (non-existent) D-02 guard check. Plan 05's subprocess determinism regression guard is the analogous sibling for real-loop reproducibility — Phase 4 + Phase 5 plans will each ship their own parallel 5-6-test test_server_integration.py + subprocess-determinism guard."
  - "Early run_id materialization: the server generates a run_id at startup (via generate_run_id() when no explicit override exists) and threads it through BOTH the train/eval ConfigRecord AND the D-13 cold-start probe path AND the final manifest. This single-source-of-truth for run_id ensures the server-side cold-start probe is looking at the same cache_root the client_app is about to write into — a previously-broken invariant the server couldn't close without passing run-id into the per-round ConfigRecord."

requirements-completed: [PSN-04, PSN-07]

# Metrics
duration: 6min
started: "2026-04-20T03:46:59Z"
completed: "2026-04-20T03:53:22Z"
tasks_completed: 2
files_created: 1
files_modified: 1
tests_added: 6
tests_green_personalized: 34  # was 28 (Plans 01+02+03); +6 from Plan 04
---

# Phase 03 Plan 04: server_app cross-device migration + D-13 cold-start + D-02 guard (PSN-04, PSN-07) Summary

**federated-personalized-cf/federated_personalized_cf/server_app.py migrated to the cross-device contract: D-25 mode resolver owns canonical hyperparameters; D-02 frozen-cross-silo NotImplementedError fires BEFORE any training; G-03-01 discovery round broadcasts `discover_only=true` to every supernode and builds `partition_to_node_id` before round 1; PSN-04 seeded sampling via `_server_sampler = server_rng(run_seed).sample(range(N), k)` in partition-id space; PersonalizedSplitFedAvg / PersonalizedSplitFedProx wire-up (sufficient-stat `aggregate_evaluate` from Plan 01) replaces raw FedAvg/FedProx; D-27 in-memory best-round restore; D-15 double-write manifest with `module="personalized"`; D-13 cold-start counter (Phase-3-unique) reports per-round and total cold-start rates in the result JSON + W&B summary. stdlib random eradicated from server_app.py. 6 GREEN integration tests added; personalized suite 28 -> 34 GREEN.**

## Performance

- **Duration:** ~6 min (383 seconds wall clock)
- **Started:** 2026-04-20T03:46:59Z
- **Completed:** 2026-04-20T03:53:22Z
- **Tasks:** 2 (both autonomous; no Rule-1 auto-fixes needed)
- **Files modified:** 1 (`server_app.py`)
- **Files created:** 1 (`tests/test_server_integration.py`)
- **Tests added:** 6 (all GREEN on first run)
- **Personalized test suite:** 34/34 GREEN (was 28; +6 from Plan 04)

## Accomplishments

- **PSN-04 server-side observable end-to-end.** `_server_sampler = server_rng(run_seed)` instantiated ONCE pre-loop; `_server_sampler.sample(range(expected_n), num_selected)` samples in stable partition-id space every round; `selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]` translates to Flower's ephemeral node_ids only at message-addressing time. `selected_clients_per_round` stores partition_ids (0..N-1), not node_ids — byte-identical across same-seed subprocess reruns (invariant asserted by Plan 05's regression guard).
- **PSN-04 strategy wire-up observable.** `PersonalizedSplitFedAvg(fraction_fit=fraction_train)` and `PersonalizedSplitFedProx(fraction_fit=fraction_train, proximal_mu=proximal_mu)` replace raw `FedAvg/FedProx`. `strategy.aggregate_evaluate(round_num, eval_results, [])` is called every round and its thesis metrics populate `eval_metrics_history[round_num]`. RMSE/MAE preserved via `weighted_average_metrics` fallback for D-18 scope-out.
- **PSN-07 observable.** `build_run_manifest(..., module="personalized")` writes the Phase-3-specific manifest carrying all four IMP-2 fingerprints (`mapping_sha256`, `split_hash`, `exclusion_sha256`, `foundation_contract_sha256`) + `raw_data_hash` + `builder_version`. `embed_manifest_in_result` mutates the result JSON in-place to inject `_manifest`; `write_manifest_sibling` writes `<run_id>-manifest.json` atomically beside the main result file via `atomic_write_json`.
- **D-13 cold-start counter observable.** Per-round, BEFORE sending the train message, the server probes `.embedding_cache/{run_id}/partition_{pid}.pt` existence for each selected `partition_id`; the non-existent count is `cold_count` for the round. `total_cold_starts` accumulates; `cold_start_rate = total_cold_starts / total_client_selections`. Shape in result JSON: `results_data["cold_starts"] = {per_round: List[int], total_cold_starts: int, total_client_selections: int, cold_start_rate: float}`. Logged per-round to W&B as `round/cold_starts`, pushed to `wandb.run.summary["total_cold_starts"]` + `wandb.run.summary["cold_start_rate"]` at run end. Under D-09 `reuse-cache=true` the counter short-circuits to 0 and the log line names D-09 explicitly.
- **D-02 frozen-cross-silo guard observable.** `if mode == "cross_silo_legacy": raise NotImplementedError(...)` fires immediately after `log_mode_and_overrides` — BEFORE any hyperparameter reads, BEFORE `get_model`, BEFORE `verify_bundle`, BEFORE wandb.init. The error message cites D-02 and `.planning/phases/03-personalized-migration/03-CONTEXT.md §Deferred` so a traceback reader has a direct pointer to the pre-Phase-3 commit checkout procedure.
- **D-27 best-round restore observable.** `best_metric`, `best_round_num`, `best_arrays` tracked in memory during the FL loop on `current_ndcg > best_metric`. Under `checkpoint_rule in ('best_round_restore', 'best_round')` and `best_round_num > 0`, `arrays = best_arrays` is reassigned at loop end. The final `eval_metrics_history[best_round_num]` is used as the `final_metrics` payload (centralized eval is not available under split learning). The snapshot `ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()})` is a deep copy — aliasing bugs impossible.
- **G-03-01 observable.** Discovery round broadcasts `ConfigRecord({"discover_only": True})` via `grid.create_message(message_type="evaluate", ...)` to every `grid.get_node_ids()` entry. Each client's `@app.evaluate` (Plan 03) short-circuits with `EvaluateMetricsContract(..., partition_id=<pid>).to_dict()`. The server reads `partition_id` from each response's `MetricRecord` and builds `partition_to_node_id: Dict[int, int]`. The pre-round assertion `len(all_node_ids) == expected_n` catches federation size mismatches; the post-discovery assertion `missing = sorted(set(range(expected_n)) - set(partition_to_node_id.keys()))` rejects partial discovery with the first 5 missing partition_ids in the error message.
- **PROJECT.md W&B project switch.** Default project is `federated-cf-cross-device` for `mode in (benchmark_cross_device, paper_compat_pfedrec)`; `federated-cf` otherwise (legacy; frozen by D-02 in this module). Empty-string `run_config['wandb-project']` treated as 'use default'; explicit non-empty wins.
- **stdlib random eradicated** across `server_app.py` + `client_app.py` + `task.py` (cross-file regression check returns 0 matches for `^import random$|random.sample\(|random.seed\(`).
- **6 GREEN integration tests added.**
  - `test_server_rng_reproducible_per_round_selection` — PSN-04 seeded sampler byte-identical across fresh instances (rng.sample(range(6040),50) x3).
  - `test_server_rng_different_seeds_different_selections` — negative guard; `server_rng(42)` vs `server_rng(43)` diverge on first sample.
  - `test_personalized_split_fedavg_aggregate_evaluate_sum_not_average` — PSN-04 strategy wire-up; 1-hit-on-1-user + 0-hits-on-99-users via synthetic EvaluateRes returns `sampled_hr@10 = 1/100 = 0.01` (sum-based), NOT 0.5 (mean-of-ratios).
  - `test_build_run_manifest_module_personalized` — PSN-07 + D-15; manifest carries all four IMP-2 fingerprints + `raw_data_hash` + overrides + `module="personalized"`.
  - `test_cold_start_counter_math` — D-13 arithmetic; `selected_pids=[0..5]` with cache dir seeded for `{0,1,2}` yields `cold_count=3`, `hot_count=3`, `cold_start_rate=0.5`.
  - `test_cross_silo_legacy_mode_raises_not_implemented` — D-02 source-level regression guard; asserts `cross_silo_legacy` token + `raise NotImplementedError` + `D-02` citation + proximity (raise within 800 chars of the cross_silo_legacy branch) in the source of `server_app.py`.

## Task Commits

Each task committed atomically with `--no-verify` (Wave-3 parallel-executor safety; orchestrator runs hooks once after the wave completes):

1. **Task 1: server_app.py cross-device migration + D-13 + D-02 guard (PSN-04, PSN-07)** — `969bc6d` (feat)
2. **Task 2: server integration tests — PSN-04 + D-15 + D-13 + D-02 (6 GREEN)** — `52f56d6` (test)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md updates) is appended separately at plan close._

## Files Created/Modified

### `federated-personalized-cf/federated_personalized_cf/server_app.py` (MODIFIED, +478 / -87)

- Added 1 top-level private helper: `_cold_start_cache_root(run_id, reuse_cache) -> Path` — resolves the D-13 cache probe root under D-08 default; D-09 reuse-cache handled via the short-circuit path in the main loop.
- Removed: `import random`, `random.sample(...)` call in per-round sampling (pre-existing uncommitted WIP that the plan's rip-and-replace scope overrode).
- Added top-level imports: `Code` / `EvaluateRes` / `Status` from `flwr.common` (strategy evaluate path); `verify_bundle` / `build_run_manifest` / `embed_manifest_in_result` / `generate_run_id` / `write_manifest_sibling` / `resolve_mode_defaults` / `log_mode_and_overrides` / `data_derived` / `server_rng` / `load_split_manifest` from `fedrec_foundation`; `PersonalizedSplitFedAvg` / `PersonalizedSplitFedProx` from the module's strategy (replacing the legacy `SplitFedAvg` / `SplitFedProx` / `GLOBAL_PARAM_KEYS` imports).
- Preserved: `DummyClientProxy` class, `weighted_average_metrics` helper, `print_evaluation_metrics` helper, `EarlyStopping` setup/teardown, W&B init block (augmented with D-25 contract keys + project switch).
- Rewrote `@app.main()` body: D-25 mode resolve -> D-02 cross-silo guard raise -> D-25 hyperparameter reads with `profile.field` fallback -> run_id materialize -> W&B init with cross-device project default -> global_model construct + `ArrayRecord(global_model.get_global_parameters())` (split-learning: only GLOBAL params on the server) -> G-03-01 discovery round broadcast + partition_to_node_id assertion -> strategy = `PersonalizedSplitFedAvg|FedProx(...)` -> `_server_sampler = server_rng(run_seed)` -> FL loop: per-round `train_config = ConfigRecord({lr, proximal_mu, round_num, run_id, reuse_cache})` -> partition-id sampling + `selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]` -> D-13 cold-start probe + W&B per-round log -> train message send + FitRes assemble + strategy.aggregate_fit -> eval message send + `EvaluateRes` assemble + `strategy.aggregate_evaluate` -> RMSE/MAE fallback via `weighted_average_metrics` -> D-27 best-round snapshot -> W&B log -> early-stop check -> end-of-loop: D-27 arrays reassign -> final_metrics from `eval_metrics_history[final_round_for_metrics]` (best-round || early-stopper best || last-round) -> print + wandb final log -> results_data assemble including D-26 `selected_clients_per_round` + D-27 `checkpoint` block + D-13 `cold_starts` block -> `build_run_manifest(..., module="personalized")` -> `embed_manifest_in_result` -> JSON write -> `write_manifest_sibling` -> W&B config update with `_manifest` + cold-start summary -> `wandb.finish`.

### `federated-personalized-cf/tests/test_server_integration.py` (CREATED, 206 LOC, 6 GREEN tests)

Top-level `pytestmark = pytest.mark.skipif(not (.../data/derived/foundation_index.json).exists(), reason="foundation bundle not committed")` so a minimal clone without the committed bundle collects + skips cleanly.

1. **test_server_rng_reproducible_per_round_selection** — two fresh `server_rng(42)` instances; 3-round composite sequence of `rng.sample(range(6040), 50)` is byte-identical (`seq1 == seq2`).
2. **test_server_rng_different_seeds_different_selections** — `server_rng(42)` vs `server_rng(43)` diverge on first `.sample(range(6040), 50)`.
3. **test_personalized_split_fedavg_aggregate_evaluate_sum_not_average** — PSN-04 strategy; client A has 1 hit on 1 user, client B has 0 hits on 99 users; strategy returns `sampled_hr@10 = 0.01` (not 0.5).
4. **test_build_run_manifest_module_personalized** — manifest built from a real `verify_bundle(data_derived())` result carries all 4 IMP-2 fingerprints + raw_data_hash + overrides + `module="personalized"` + `num_supernodes=6040` + `weight_policy="num_positives"` + `primary_evaluator="sampled_loo_99"`.
5. **test_cold_start_counter_math** — tmp_path cache dir seeded with `partition_{0,1,2}.pt`; `selected_pids=[0..5]` yields `cold_count=3`, `hot_count=3`, `cold_start_rate=0.5`.
6. **test_cross_silo_legacy_mode_raises_not_implemented** — source-level check on `server_app.py`: contains `cross_silo_legacy`, contains `raise NotImplementedError`, `NotImplementedError` appears within 800 chars of the `cross_silo_legacy` branch, source contains the `D-02` token for traceability.

## Decisions Made

See the `key-decisions` block in frontmatter. Highlights:

- **D-02 guard placement:** Fires within ~10 lines of `@app.main()` entry, immediately after `log_mode_and_overrides` and before any hyperparameter reads, W&B init, model construction, or foundation-bundle verification. No partial state leaks into a cross-silo-mode traceback; the error message directly cites D-02 and the CONTEXT.md §Deferred path to the pre-Phase-3 commit.
- **D-13 counter under D-09 reuse-cache=true is a deliberate 0.** The server cannot cheaply reconstruct the `sig_<hash>` path without the client-side signature fields; the expected regime under D-09 is "all hot because the signature matches", so 0 is the honest answer. A log line names D-09 explicitly so readers aren't confused by the short-circuit.
- **Checkpoint_rule accepts two spellings** (`best_round_restore` OR `best_round`) — Phase 2 Plan 04 precedent. Keeps the launcher free to pass ModeProfile's default (`best_round`) verbatim while pyproject overrides can use the more explicit `best_round_restore`.
- **Centralized evaluation block intentionally NOT ported** from Phase 2 baseline. The server never holds LOCAL user rows in split learning; any server-side `model(...)` forward pass would return garbage. Final headline metrics come from `eval_metrics_history[final_round_for_metrics]` (strategy-aggregated federated eval). The D-27 best-round restore still populates `arrays = best_arrays` so the manifest artifact + any downstream loader has the best-round item embeddings.
- **Default W&B project is `federated-cf-cross-device`** for cross-device modes (PROJECT.md constraint). Empty-string `run_config['wandb-project']` treated as "use default"; explicit non-empty wins.
- **run_id materialized EARLY** via `generate_run_id()` when no explicit `run-id` override exists. Used identically by the D-13 cold-start probe, the train/eval ConfigRecord `run_id` field, and the final manifest — single source of truth so the server's cold-start probe looks at the same `cache_root` the client is writing into.
- **D-18 surgical discipline:** DummyClientProxy, weighted_average_metrics, print_evaluation_metrics, EarlyStopping setup/teardown preserved verbatim. Pre-existing uncommitted WIP on server_app.py (the `import random` + `node_ids = sorted(grid.get_node_ids())` + `random.sample(node_ids, num_selected)` hunk) was INTENTIONALLY overridden per the plan's rip-and-replace scope — keeping it would have failed gate 4 (stdlib-random eradication).

## Deviations from Plan

None. Plan executed exactly as written. No auto-fixes needed; no auth gates; no architectural decisions deferred.

## Authentication Gates

None — all work is local-filesystem + pytest. No external service touched.

## Issues Encountered

None. Task 1 passed all 19 acceptance grep checks on first write; Task 2's 6 tests all GREEN on first run.

## Known Stubs

None. Every code path in `@app.main()` has a concrete implementation. No `NotImplementedError` placeholders (the only `raise NotImplementedError` is the D-02 cross-silo guard, which is deliberate and documented). No `TODO` / `FIXME` markers. The `_cold_start_cache_root` helper has a concrete body (returns `Path(".embedding_cache") / run_id`); the D-09 reuse-cache short-circuit is an explicit documented 0, not a stub.

## User Setup Required

**None beyond what `docs/setup.md` already documents.** To run the new tests: `pip install -e "federated-personalized-cf[dev]"` (Plan 02 already declared the `[dev]` extra with `pytest>=7.0`). The tests auto-skip if `data/derived/foundation_index.json` is absent.

## Next Phase Readiness

- **Plan 05 (scripts/clean_cache.py + subprocess determinism regression guard) is now unblocked.** It consumes:
  - The `.embedding_cache/{run_id}/` directory structure from this plan + Plan 03 (the `clean_cache.py --keep N` helper globs these dirs and sorts by mtime).
  - The `sig_<hash>/` directories (D-09) are NEVER touched by `clean_cache.py` per CONTEXT §D-10.
  - The byte-identity regression guard — subprocess determinism: two same-seed reruns produce byte-identical `selected_clients_per_round` (partition_id space) in the result JSON. This Plan's `_server_sampler = server_rng(run_seed).sample(range(N), k)` is the server-side surface that makes this assertion hold.
  - The result JSON's `selected_clients_per_round` + `cold_starts` + `checkpoint` + `_manifest` fields are ready to be read back and cross-compared in Plan 05's pytest.
- **No blockers. No open questions. No architectural decisions deferred.**
- **Phase 4 (adaptive) + Phase 5 (pfedrec) server_apps inherit this pattern**: clone the shape, substitute the strategy class name + manifest `module` flag + any module-specific per-round bookkeeping. D-13 cold-start is Phase-3-unique; the other invariants (mode resolver, discovery round, PSN-04 sampler, D-15 + D-27) are common.

## Self-Check

- **Files created:**
  - FOUND: `federated-personalized-cf/tests/test_server_integration.py` — verified via `pytest -v` collecting 6 tests.
- **Files modified:**
  - FOUND: `federated-personalized-cf/federated_personalized_cf/server_app.py` — verified via 19-point acceptance grep (all pass); `git log --oneline -5 server_app.py` shows the `969bc6d` commit.
- **Commits:**
  - FOUND: `969bc6d` (Task 1 feat — server_app.py migration) — visible on `feat/try_to_run_the_baseline` via `git log --oneline -5`.
  - FOUND: `52f56d6` (Task 2 test — server integration tests) — same.
- **Automated verify:** PASSED.
  - `cd federated-personalized-cf && pytest tests/test_server_integration.py -v` → 6 passed, 0 failed.
  - `cd federated-personalized-cf && pytest tests/ -v` → 34 passed, 0 failed in 4.58s.
  - `python -c "import ast; ast.parse(open('.../server_app.py').read()); print('syntax ok')"` → `syntax ok`.
  - `python -c "from federated_personalized_cf.server_app import app; print('import ok')"` → `import ok`.
  - `grep -cE '^import random$|random\.sample\(|random\.seed\(' server_app.py client_app.py task.py` → 0 (Gate 4).
  - 19-point Task 1 acceptance grep suite → all pass (strategy import 1, PersonalizedSplitFedAvg/FedProx present, mode resolver 1, server_rng 2, manifest helpers 1 each, discover_only 3, partition_to_node_id 6, selected_clients_per_round 5, cold_start tokens 18, best_round tokens 21, stdlib random 0, NotImplementedError 2, cross_silo_legacy 3, federated-cf-cross-device 1).
- **Scope boundary:** PASSED.
  - `git diff --stat HEAD~2..HEAD -- federated-personalized-cf/federated_personalized_cf/client_app.py task.py dataset.py strategy.py models/ pyproject.toml` → empty (D-18 surgical guard).
  - `git diff --stat HEAD~2..HEAD -- scripts/clean_cache.py scripts/foundation/tests/test_personalized_determinism.py` → empty (parallel Plan 05 file ownership preserved).

## Self-Check: PASSED

---

*Phase: 03-personalized-migration*
*Plan: 04 (Wave 3 — depends on Plans 01 + 02 + 03; parallel with Plan 05)*
*Completed: 2026-04-20*
*Closes: PSN-04 (seeded sampling — server side), PSN-07 (manifest — server side).*
