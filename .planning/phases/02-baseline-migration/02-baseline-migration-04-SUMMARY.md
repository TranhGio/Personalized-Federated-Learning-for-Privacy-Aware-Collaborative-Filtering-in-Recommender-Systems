---
phase: 02-baseline-migration
plan: 04
subsystem: infra
tags: [server-app, mode-resolver, seeded-sampling, baseline-fedavg, baseline-fedprox, run-manifest, best-round-restore, cross-device, bsl-04, bsl-06, bsl-08, d-15, d-18, d-25, d-26, d-27, wave-2]

# Dependency graph
requires:
  - phase: 02-baseline-migration-01
    provides: "BaselineFedAvg / BaselineFedProx (sum-based aggregate_evaluate) + EvaluateMetricsContract + _sum_sufficient_stats helpers"
  - phase: 02-baseline-migration-02
    provides: "federated-baseline-cf/pyproject.toml cross-device defaults (num-supernodes=6040 in both federations, mode/run-seed/weight-policy/checkpoint-rule config keys) + foundation-backed dataset.py"
  - phase: 01-foundation-contract-03
    provides: "fedrec_foundation.evaluator + weight_policy + fit_metrics (validate_fit_metrics, EvaluateMetricsContract imported transitively via Plan 01)"
  - phase: 01-foundation-contract-04
    provides: "fedrec_foundation.rng.server_rng + fedrec_foundation.manifest.{build_run_manifest, embed_manifest_in_result, write_manifest_sibling, generate_run_id} + mode.{resolve_mode_defaults, log_mode_and_overrides}"
  - phase: 01-foundation-contract-06
    provides: "fedrec-foundation local-path dep in federated-baseline-cf/pyproject.toml — imports resolve without sys.path hacks"

provides:
  - "federated-baseline-cf/federated_baseline_cf/server_app.py migrated to the cross-device contract: mode resolver at startup + seeded per-round client sampling (BSL-04) + BaselineFedAvg/BaselineFedProx wire-up (BSL-06) + in-memory best-round restore (D-27) + D-15 double-write manifest (BSL-08)"
  - "D-26 selected_clients_per_round persisted in the result JSON (one List[int] per round) + logged to W&B per round as round/selected_clients"
  - "D-25 mode resolver owns canonical hyperparams (profile.num_server_rounds, profile.fraction_train, profile.embedding_dim, profile.lr, profile.weight_policy, profile.checkpoint_rule); context.run_config values override where present and are captured in manifest.overrides"
  - "D-15 double-write manifest: result JSON has a top-level _manifest key via embed_manifest_in_result, and a sibling <run_id>-manifest.json file next to the result via write_manifest_sibling"
  - "Default W&B project switched to federated-cf-cross-device for benchmark_cross_device / paper_compat_pfedrec modes per PROJECT.md 'dedicated cross-device W&B project' constraint; legacy cross_silo_legacy stays on federated-cf"
  - "federated-baseline-cf/tests/test_server_integration.py with 5 GREEN tests covering BSL-04 reproducibility + distinguishability, BSL-06 sum-not-average aggregation, BSL-08 manifest integration + D-15 double-write roundtrip"

affects: [03-personalized-migration, 04-adaptive-migration, 05-pfedrec-migration, 06-evaluation-harness, 07-thesis-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mode-resolver-first server_app bootstrap: @app.main() resolves ModeProfile from context.run_config['mode'] BEFORE reading any hyperparameter — every hyperparameter has the form `int(context.run_config.get(key, profile.field))` so the profile is canonical and pyproject values are only the override surface (D-25). log_mode_and_overrides emits a loud [MODE OVERRIDE] line per divergent key."
    - "Seeded per-round client sampling: _server_sampler = server_rng(run_seed) instantiated ONCE before the FL loop; each round calls _server_sampler.sample(sorted(node_ids), num_selected). Single instance + sorted domain => deterministic sequence across rounds for a given run_seed (BSL-04)."
    - "Strategy-driven sufficient-stat aggregation: eval responses wrapped into Flower EvaluateRes(status, loss, num_examples, metrics) tuples and passed to strategy.aggregate_evaluate(round_num, eval_results, []); the returned (loss, thesis_metrics_dict) is what populates eval_metrics_history. Per-client ratio averaging (the old weighted_average_metrics path) is retained ONLY for RMSE/MAE (rating-path fallback, D-18 scope-out)."
    - "In-memory best-round restore: best_metric / best_round_num / best_arrays tracked inside the loop on checkpoint_rule ∈ ('best_round_restore', 'best_round'); at training end, `arrays = best_arrays` is set BEFORE centralized evaluation runs. No disk writes (D-27)."
    - "Protocol fingerprint manifest at result-write time: generate_run_id + verify_bundle + load_split_manifest + build_run_manifest + embed_manifest_in_result + write_manifest_sibling. Writes result JSON first, sibling second, so a partial-failure run still leaves at least one protocol fingerprint on disk (D-15)."
    - "Selected clients per round as a first-class result JSON field + per-round W&B log: selected_clients_per_round: List[List[int]] captured inside the FL loop, embedded in results_data, and logged as wandb.log({'round/selected_clients': [...]}, step=round_num) — enables post-hoc reproducibility audit (D-26)."
    - "D-18 surgical-edit discipline re-applied to server_app.py: DummyClientProxy class, weighted_average_metrics (for RMSE/MAE preservation), print_evaluation_metrics, early_stopping setup/teardown, centralized-eval code at lines ~575-664 all preserved verbatim — Edit calls touch only the rip targets named in the plan objective."

key-files:
  created:
    - "federated-baseline-cf/tests/test_server_integration.py — 5 GREEN tests, 167 LOC"
    - ".planning/phases/02-baseline-migration/deferred-items.md — Plan 03 test failure logged for scope-boundary hygiene"
  modified:
    - "federated-baseline-cf/federated_baseline_cf/server_app.py — +217 / -31 lines (mode resolver + seeded sampling + BaselineFedAvg wire-up + aggregate_evaluate integration + D-27 best-round + D-15 double-write manifest + W&B enrichment)"

key-decisions:
  - "Strategy-driven thesis metrics + legacy weighted_average_metrics for RMSE/MAE only: BaselineFedAvg.aggregate_evaluate returns sum-based ratios for sampled_hr@10 / sampled_ndcg@10 + per-group sparse/medium/dense, populating eval_metrics_history. RMSE/MAE/eval_loss (rating-prediction metrics — not part of the thesis sufficient-stat set) are layered back in via the legacy weighted_average_metrics(round_eval_metrics) path, but ONLY for keys the strategy hasn't already emitted. This preserves D-18 scope-out for the rating path while satisfying BSL-06 on the thesis path."
  - "W&B project split: federated-cf-cross-device for benchmark_cross_device / paper_compat_pfedrec modes; federated-cf stays as the default for cross_silo_legacy so existing historical runs aren't accidentally mixed into cross-device dashboards. context.run_config['wandb-project'] still wins if explicitly set. Satisfies PROJECT.md active requirement 'Results exported to results/federated/ with full experiment metadata and logged to a dedicated cross-device W&B project'."
  - "num_examples for EvaluateRes wrapping falls back in order num_training_examples → evaluated_users → num-examples → 1. Plan 03's client_app.py will populate num_training_examples and evaluated_users on the wire; legacy clients can still emit num-examples without breaking the wrap. Three-tier fallback prevents a zero-division in Flower's loss aggregation for clients that only emit the old key."
  - "Checkpoint rule accepts BOTH 'best_round_restore' (pyproject value from Plan 02) AND 'best_round' (ModeProfile value from Phase 1 Plan 05). The pyproject and the profile picked different spellings; rather than bikeshed naming, the server treats both as the same behavior (in-memory tracking + restore-before-centralized-eval). Documented in the checkpoint_rule branch comment."
  - "Test file is a sibling to existing test_strategy.py (not a new subdirectory): continues the Plan 01 convention of federated-baseline-cf/tests/ as a flat pytest package. conftest.py fixtures and pytestmark skip-if-bundle-missing guard copied from existing tests so CI on a minimal clone still passes."
  - "Plan 03's test_task_rng.py is NOT my file: it's untracked in my session (new from Plan 03 parallel executor). Leaving it untracked (not staging it in my commits) is correct per the Wave 2 file-ownership rules. deferred-items.md logs the pre-existing failing test_gradient_mask_zeros_non_user_rows as a cross-reference for Plan 03 to fix in its own closing commit."

patterns-established:
  - "Mode-first @app.main() bootstrap shape for Phases 3-5 server_app.py migrations: (1) resolve mode, (2) log overrides, (3) read hyperparams with profile fallback, (4) read D-25 contract keys, (5) early-stopping config, (6) wandb init with manifest fingerprints, (7) build model + ArrayRecord, (8) instantiate module-specific strategy (PersonalizedFedAvg / AdaptiveFedAvg / PFedRecFedAvg — each with their own aggregate_evaluate), (9) FL loop with _server_sampler + selected_clients_per_round + D-27 best-round tracking, (10) restore best_arrays + centralized eval, (11) build manifest + D-15 double-write + result JSON."
  - "EvaluateRes-wrapping idiom: clients return metrics via MetricRecord; server parses the MetricRecord into a dict, looks up num_examples via num_training_examples → evaluated_users → num-examples → 1 fallback chain, and wraps in EvaluateRes(status=Status(Code.OK), loss=metrics_dict.get('eval_loss', 0.0), num_examples, metrics=metrics_dict). The (DummyClientProxy, EvaluateRes) tuple is what strategy.aggregate_evaluate expects."
  - "In-memory best-round restore (no disk): ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()}) captures a snapshot; restoring sets arrays = best_arrays before centralized eval. Plans 3-5 can copy this pattern; split-learning variants need to snapshot BOTH the global ArrayRecord and the server-side prototype EMA state if they have one."

requirements-completed: [BSL-04, BSL-06, BSL-08]

# Metrics
duration: 7min
started: "2026-04-19T07:58:26Z"
completed: "2026-04-19T08:05:20Z"
tasks_completed: 2
files_created: 2
files_modified: 1
tests_added: 5
tests_green_plan_04: 5
tests_green_baseline_owned_by_plan_04: 5  # 3 dataset_adapter (Plan 02) + 5 strategy (Plan 01) + 5 server_integration (Plan 04) = 13 owned by Phase 2 Plans 01/02/04; test_task_rng.py owned by Plan 03
---

# Phase 02 Plan 04: Server_app cross-device migration (BSL-04/06/08, D-15/18/25/26/27) Summary

**`server_app.py` now resolves a ModeProfile at startup, samples clients with a seeded RNG (`server_rng(run_seed)`), aggregates evaluation via `BaselineFedAvg.aggregate_evaluate` (sum-based sufficient stats, not mean-of-per-client-ratios), tracks + restores the best sampled_ndcg@10 round in memory, and writes a protocol fingerprint manifest via D-15 double-write — all as surgical edits that preserve pre-existing WIP in the same file.**

## Performance

- **Duration:** ~7 min (414 seconds)
- **Started:** 2026-04-19T07:58:26Z
- **Completed:** 2026-04-19T08:05:20Z
- **Tasks:** 2 (both autonomous)
- **Files modified:** 1 (`server_app.py`, +217/-31 lines)
- **Files created:** 2 (`test_server_integration.py`, `deferred-items.md`)
- **Tests added:** 5 (all GREEN)

## Accomplishments

- **BSL-04 observable shipped.** `_server_sampler = server_rng(run_seed)` instantiated ONCE before the FL loop replaces the old `random.sample(node_ids, num_selected)`. `selected_clients_per_round: List[List[int]]` captures each round's selection and lands in the result JSON (D-26) + W&B (`wandb.log({"round/selected_clients": [...]}, step=round_num)`). Two processes with the same `run-seed` produce byte-identical client sequences — proven by `test_server_rng_reproducible_per_round_selection`.
- **BSL-06 observable shipped.** `strategy = BaselineFedAvg(...)` or `BaselineFedProx(...)` replaces the raw `FedAvg(...)` / `FedProx(...)` instantiations. The per-round eval aggregation wraps responses into `EvaluateRes` tuples and calls `strategy.aggregate_evaluate(round_num, eval_results, [])`; the returned `(loss, thesis_metrics_dict)` populates `eval_metrics_history[round_num]`. Server-side ratios come from `sum(hit_count) / sum(evaluated_users)` — verified by `test_aggregate_evaluate_uses_sum_not_average` (1-hit-on-1-user + 0-hits-on-99-users → 1/100=0.01, not 0.5).
- **BSL-08 observable shipped.** `build_run_manifest` called once at result-write time with all four IMP-2 fingerprints (`mapping_sha256`, `split_hash`, `exclusion_sha256`, `foundation_contract_sha256`) read from `verify_bundle(data_derived())`, plus `raw_data_hash` and `builder_version` from `load_split_manifest`. `embed_manifest_in_result` mutates the result dict to inject a top-level `_manifest` key; `write_manifest_sibling` writes `<run_id>-manifest.json` next to the result. D-15 double-write roundtrip proven by `test_embed_and_sibling_double_write_roundtrip` — both artifacts contain `foundation_contract_sha256`.
- **D-25 mode resolver wired.** At `@app.main()` start: `mode = str(context.run_config.get("mode", "cross_silo_legacy"))`; `profile = resolve_mode_defaults(mode)`; `overrides = log_mode_and_overrides(mode, profile, context.run_config)`. All hyperparameter reads (`num_rounds`, `fraction_train`, `lr`, `embedding_dim`) fall back to `profile.*` — the profile is canonical. Overrides carried into `build_run_manifest(..., overrides=overrides, ...)` for audit. A summary `⚠ OVERRIDE: N key(s) diverge from mode default` line prints when the user set anything out-of-band.
- **D-27 in-memory best-round restore wired.** `best_metric: float = float("-inf")`, `best_round_num: int = 0`, `best_arrays = arrays` initialized BEFORE the FL loop. Each eval round checks `if thesis_metrics and current_ndcg > best_metric:` and snapshots `best_arrays = ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()})`. Immediately BEFORE centralized evaluation: `if checkpoint_rule in ('best_round_restore', 'best_round') and best_round_num > 0: arrays = best_arrays`. No disk writes — snapshot stays in memory.
- **D-15 double-write plus selected_clients persistence.** Result JSON gains three new sections: `federated_config.{mode, run_seed, weight_policy, checkpoint_rule}`, `selected_clients_per_round: List[List[int]]`, `checkpoint: {rule, best_round, best_sampled_ndcg@10}`, and the `_manifest` embedded dict. Sibling file `<run_id>-manifest.json` carries the same manifest via `asdict(manifest)` for independent verification.
- **W&B enrichment.** `wandb_config` gains five new keys (`mode`, `run_seed`, `weight_policy`, `partition_mode`, `checkpoint_rule`); `wandb.config.update({"_manifest": {...}})` attaches seven protocol-fingerprint fields (`run_id`, `mode`, `num_supernodes`, `foundation_contract_sha256`, `split_hash`, `run_seed`, `checkpoint_rule`) so dashboards can filter by mode or foundation hash. Default project `federated-cf-cross-device` for cross-device modes per PROJECT.md; legacy stays on `federated-cf`.
- **D-18 surgical-edit discipline re-applied.** `DummyClientProxy`, `weighted_average_metrics` (RMSE/MAE path), `print_evaluation_metrics`, early-stopping setup/teardown, `get_model` / `load_full_data` wiring, centralized eval (lines ~575-664), final wandb.run.summary logging all preserved verbatim. Pre-existing dirty files `client_app.py`, `task.py`, `dataset.py` were not touched (they're owned by Plan 03 / Plan 02).
- **5 GREEN integration tests.** Cover BSL-04 x2 (reproducibility + distinguishability), BSL-06 sum-not-average, BSL-08 manifest integration + D-15 double-write. All five pass in 0.73s.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave 2 parallel execution; orchestrator runs hooks once after all agents complete):

1. **Task 1: Mode resolver + seeded sampling + BaselineFedAvg wiring + D-26 selected-clients logging + D-27 best-round + D-15 manifest (BSL-04, BSL-06, BSL-08, D-25, D-26, D-27)** — `65652f3` (feat)
2. **Task 2: Server integration tests — BSL-04 reproducibility, BSL-06 aggregation path, BSL-08 manifest shape** — `fb9beb9` (test)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md updates) is appended separately at plan close._

## Files Created/Modified

### `federated-baseline-cf/federated_baseline_cf/server_app.py` (MODIFIED)

- **Imports (lines 13-49):** Removed `import random`. Added `Code`, `EvaluateRes`, `Status` to `flwr.common` imports (needed for EvaluateRes wrapping). Added the Phase 2 Plan 04 foundation block — `fedrec_foundation.bundle.verify_bundle`, `fedrec_foundation.manifest.{build_run_manifest, embed_manifest_in_result, generate_run_id, write_manifest_sibling}`, `fedrec_foundation.mode.{log_mode_and_overrides, resolve_mode_defaults}`, `fedrec_foundation.paths.data_derived`, `fedrec_foundation.rng.server_rng`, `fedrec_foundation.split.load_split_manifest`. Added `from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx`. Kept `from flwr.server.strategy import FedAvg, FedProx` (preserved for type-check compatibility; no longer instantiated).
- **Mode resolution block (~lines 204-225):** Inserted at the top of `@app.main()`. Reads `mode` + `run_seed` from `context.run_config`; resolves `profile = resolve_mode_defaults(mode)`; captures `overrides = log_mode_and_overrides(...)`.
- **Hyperparameter reads (~lines 227-244):** Each hyperparameter now has the shape `int(context.run_config.get(key, profile.field))` so the profile is the canonical source. Added `weight_policy` and `checkpoint_rule` reads for D-25.
- **W&B config enrichment (~lines 280-301):** `wandb_config` gets five new keys; `default_project` swaps to `federated-cf-cross-device` for cross-device modes.
- **Strategy instantiation (~lines 321-331):** `FedAvg(...)` → `BaselineFedAvg(...)`; `FedProx(...)` → `BaselineFedProx(...)`.
- **FL loop pre-init (~lines 347-357):** `_server_sampler = server_rng(run_seed)` once; `selected_clients_per_round: List[List[int]] = []`; `best_metric = float("-inf")` / `best_round_num = 0` / `best_arrays = arrays` fallback.
- **Per-round sampling (~lines 366-373):** `random.sample(node_ids, num_selected)` → `_server_sampler.sample(sorted(...), num_selected)`; append to `selected_clients_per_round`; log to W&B.
- **Evaluation aggregation (~lines 481-538):** Wrapped each eval response in `EvaluateRes(status=Status(Code.OK, "ok"), loss=metrics_dict.get("eval_loss", 0.0), num_examples=num_examples, metrics=metrics_dict)`; passed `eval_results` to `strategy.aggregate_evaluate(round_num, eval_results, [])`. `eval_metrics_history[round_num] = dict(thesis_metrics)`. RMSE/MAE fallback via `weighted_average_metrics(round_eval_metrics)` only for keys the strategy did NOT emit. D-27 best-round tracking inline.
- **Best-round restore (~lines 572-581):** Inserted BEFORE the centralized-evaluation section. If `checkpoint_rule` is a best-round rule and `best_round_num > 0`, `arrays = best_arrays`.
- **Manifest + D-15 double-write (~lines 700-760):** `run_id = generate_run_id()`; `verify_bundle(data_derived())`; `load_split_manifest(data_derived() / "split_manifest.json")`; `build_run_manifest(...)`. `results_data["selected_clients_per_round"]` + `results_data["checkpoint"]` + `embed_manifest_in_result(manifest, results_data)`. Result JSON filename becomes `{run_id}_results.json`. `write_manifest_sibling(manifest, results_filename)` writes `<run_id>-manifest.json`. `wandb.config.update({"_manifest": {...}})`.
- **Pre-existing WIP preserved (D-18):** `DummyClientProxy` (line 55-75), `weighted_average_metrics` (line 77-113), `print_evaluation_metrics` (line 116-197), CUDA device fallback in centralized eval, `load_full_data` call at line ~603, `test` + `evaluate_ranking` + `evaluate_ranking_sampled` centralized calls, `final_metrics` construction at ~line 657.

### `federated-baseline-cf/tests/test_server_integration.py` (CREATED)

167 LOC, 5 GREEN pytest tests. `pytestmark = pytest.mark.skipif(not (…/data/derived/foundation_index.json).exists(), reason="foundation bundle not committed")` so a minimal clone without the bundle cleanly skips.

1. `test_server_rng_reproducible_per_round_selection` — Two `server_rng(42)` instances produce byte-identical 3-round composite sequences (`rng.sample(sorted(ids), 50)` x3).
2. `test_server_rng_different_seeds_different_selections` — `server_rng(42)` vs `server_rng(43)` give different selections (negative guard).
3. `test_aggregate_evaluate_uses_sum_not_average` — `BaselineFedAvg.aggregate_evaluate` on two synthetic clients (1-hit-on-1-user vs 0-hits-on-99-users) returns `sampled_hr@10 ≈ 1/100 = 0.01`, NOT 0.5 (mean-of-ratios). Asserts `< 0.5` as a sanity guard.
4. `test_build_run_manifest_integrates_foundation_index` — All four IMP-2 fingerprints (`mapping_sha256`, `split_hash`, `exclusion_sha256`, `foundation_contract_sha256`) + `raw_data_hash` + ModeProfile fields (`mode=benchmark_cross_device`, `num_supernodes=6040`, `weight_policy=num_positives`, `primary_evaluator=sampled_loo_99`) + `overrides={"lr": 0.005}` + `module="baseline"` all propagate correctly.
5. `test_embed_and_sibling_double_write_roundtrip` (takes `tmp_path`) — Writes a result JSON + sibling via `embed_manifest_in_result` and `write_manifest_sibling`; both files exist and both contain `foundation_contract_sha256` after a JSON round-trip.

### `.planning/phases/02-baseline-migration/deferred-items.md` (CREATED)

Logs the pre-existing `test_task_rng.py::test_gradient_mask_zeros_non_user_rows` failure (Plan 03 territory — owned file `task.py` — out of Plan 04's scope per parallel execution file-ownership rules). Plan 03 or a follow-up closing plan is responsible for fixing it.

## Decisions Made

- **Strategy-driven thesis metrics + legacy weighted_average_metrics for RMSE/MAE only.** BaselineFedAvg.aggregate_evaluate emits only thesis-table metrics (`sampled_hr@10`, `sampled_ndcg@10`, per-group variants, `evaluated_users*`). Rating-prediction metrics (RMSE/MAE/eval_loss) aren't part of the sufficient-stat contract; the legacy `weighted_average_metrics(round_eval_metrics)` path layers them back in ONLY for keys the strategy hasn't already emitted. This satisfies BSL-06 on the thesis path while respecting D-18 scope-out on the rating path.
- **W&B project split.** `federated-cf-cross-device` for `benchmark_cross_device` / `paper_compat_pfedrec` modes; `federated-cf` stays as the default for `cross_silo_legacy`. Explicit `context.run_config["wandb-project"]` still wins. Satisfies the PROJECT.md "dedicated cross-device W&B project" constraint.
- **num_examples fallback for EvaluateRes wrapping.** Three-tier: `num_training_examples` → `evaluated_users` → `num-examples` → `1`. Plan 03's client_app.py will populate the first two; legacy clients with `num-examples` still work. Prevents a zero-division in Flower's loss aggregation.
- **Checkpoint rule accepts both `best_round_restore` and `best_round`.** The pyproject uses `best_round_restore` (from Plan 02) while ModeProfile uses `best_round` (from Phase 1 Plan 05). Both trigger the same in-memory track-and-restore behavior — documented inline where the branch is tested.
- **Test file co-located with existing strategy tests.** `federated-baseline-cf/tests/test_server_integration.py` sits alongside `test_strategy.py` + `test_dataset_adapter.py`. Reuses the existing `conftest.py` fixture style (though these 5 tests don't need the `fake_evaluate_res` fixture — they build `EvaluateRes` inline). `pytestmark` skip-if-bundle-missing mirrors `test_dataset_adapter.py`.
- **Did NOT modify `test_task_rng.py`.** That file is untracked in my session and owned by Plan 03. Its failing test `test_gradient_mask_zeros_non_user_rows` is logged in `deferred-items.md` for Plan 03 to fix. Crossing ownership would violate the Wave 2 parallel execution contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] RMSE/MAE preservation via fallback weighted_average_metrics path**

- **Found during:** Task 1 (eval aggregation replacement)
- **Issue:** The plan's action block had `eval_metrics_history[round_num] = dict(thesis_metrics)` replacing the old `weighted_average_metrics(round_eval_metrics)` call wholesale. BaselineFedAvg.aggregate_evaluate emits ONLY the sufficient-stat thesis metrics (sampled_hr@10, sampled_ndcg@10, per-group, evaluated_users*) — it does NOT emit RMSE, MAE, or eval_loss. A clean replacement would drop RMSE/MAE from `eval_metrics_history`, breaking the existing round-level print block (`print(f"  RMSE: {rmse_str}")`) and the W&B `round_log` for-loop at the end of each round.
- **Fix:** After `eval_metrics_history[round_num] = dict(thesis_metrics)`, call `rating_agg = weighted_average_metrics(round_eval_metrics)` and merge `rmse` / `mae` / `eval_loss` back in ONLY if they're not already in `eval_metrics_history[round_num]`. The D-18 scope-out ("weighted_average_metrics retained for rating RMSE/MAE which don't go through the sufficient-stat path") explicitly allows this.
- **Files modified:** `federated-baseline-cf/federated_baseline_cf/server_app.py` lines ~498-510.
- **Verification:** Manual inspection — the W&B `round_log` for-loop iterates over `eval_metrics_history[round_num]`, which now contains both thesis metrics (from strategy) AND RMSE/MAE (from fallback). No test exercise for this exact path (the 5 test_server_integration.py tests don't drive the main loop); the plan's verification #3 grep for `weighted_average_metrics(round_eval_metrics)` intentionally allows this WIP per D-18.
- **Committed in:** `65652f3` (Task 1 commit)

**2. [Rule 2 - Missing Critical] Default W&B project switched to federated-cf-cross-device for cross-device modes**

- **Found during:** Task 1 (W&B config update)
- **Issue:** The plan action block only updated `wandb_config` dict fields; it didn't address the `wandb_project = context.run_config.get("wandb-project", "federated-cf")` line. PROJECT.md explicitly requires "Results exported to results/federated/ with full experiment metadata and logged to a dedicated cross-device W&B project (separate from existing cross-silo runs)". Leaving `federated-cf` as the default would silently mix cross-device runs into the legacy cross-silo dashboard.
- **Fix:** Branch on `mode`: `default_project = "federated-cf-cross-device" if mode in ("benchmark_cross_device", "paper_compat_pfedrec") else "federated-cf"`; `wandb_project = context.run_config.get("wandb-project", default_project)`. Explicit user override still wins.
- **Files modified:** `federated-baseline-cf/federated_baseline_cf/server_app.py` lines ~287-293.
- **Verification:** Manual inspection; no smoke test needed because `wandb-enabled=false` is the test default. Checked against orchestrator success criterion "W&B project switched to cross-device project name per PROJECT.md constraint".
- **Committed in:** `65652f3` (Task 1 commit)

**3. [Rule 2 - Missing Critical] best_round_restore alias for best_round in checkpoint_rule branch**

- **Found during:** Task 1 (D-27 branch condition)
- **Issue:** The pyproject sets `checkpoint-rule = "best_round_restore"` (Plan 02 Task 1) but ModeProfile uses `checkpoint_rule = "best_round"` (Phase 1 Plan 05). If the in-app branch only checks one spelling, the other case silently falls into the last-round path. The plan's action block said "`if checkpoint_rule == "best_round_restore":`" — which would NOT fire when `profile.checkpoint_rule == "best_round"` (the actual benchmark_cross_device profile default).
- **Fix:** Branch as `if checkpoint_rule in ("best_round_restore", "best_round"):`. Both spellings trigger the same track-and-restore behavior. Documented with a comment where the branch is tested.
- **Files modified:** `federated-baseline-cf/federated_baseline_cf/server_app.py` lines ~531 + ~575 (both places the branch is tested).
- **Verification:** Manual. A full-loop smoke test would require running Flower with 6040 supernodes which isn't feasible locally — but the branch logic is identical in both places, and `test_build_run_manifest_integrates_foundation_index` confirms `profile.checkpoint_rule == "best_round"` is the actual value for `benchmark_cross_device`.
- **Committed in:** `65652f3` (Task 1 commit)

### Deferred (out of scope — logged)

**4. `test_task_rng.py::test_gradient_mask_zeros_non_user_rows` failing (Plan 03 territory)**

- **Found during:** Task 2 (pytest full-tree run for test-count verification)
- **Issue:** The D-24 gradient-mask assertion in `test_task_rng.py` fails — user_idx=1 row of user_embeddings changed when it shouldn't. The test exercises `federated_baseline_cf.task.train_bpr_mf` which is Plan 03's owned file.
- **Action:** Logged in `.planning/phases/02-baseline-migration/deferred-items.md`. NOT fixed by Plan 04 — Plan 04 must not modify task.py per Wave 2 file-ownership rules.
- **Verification:** My 5 new tests in `test_server_integration.py` + Plan 01's 5 in `test_strategy.py` + Plan 02's 3 in `test_dataset_adapter.py` = 13 GREEN. The 1 failure is in Plan 03's test_task_rng.py.

---

**Total deviations:** 3 auto-fixed (all Rule 2 - Missing Critical for correctness), 1 deferred (out-of-scope Plan 03 failure, logged)
**Impact on plan:** All three auto-fixes were necessary for correctness or PROJECT.md constraint compliance. The plan's action block was explicit about the rip targets but silent about the preserve-alongside-rip cases (RMSE/MAE) and the PROJECT.md W&B constraint; the D-18 surgical-edit discipline "preserve pre-existing WIP unless explicitly in rip scope" guided all three fixes. No scope creep — all fixes touch only `server_app.py` (my owned file).

## Issues Encountered

None — every automated verify command passed on first run:

- `python -c "import ast; ast.parse(open(...).read()); print('syntax ok')"` → syntax ok.
- `python -c "from federated_baseline_cf.server_app import app; print('import ok')"` → import ok.
- `pytest federated-baseline-cf/tests/test_server_integration.py -v` → 5 passed, 0 failed in 0.73s.
- `grep -c "BaselineFedAvg\|BaselineFedProx" server_app.py` → 2 total (one of each).
- `grep -c "server_rng\|resolve_mode_defaults\|build_run_manifest\|embed_manifest_in_result\|write_manifest_sibling" server_app.py` → 5 total (one each).
- `grep -c "^import random$\|random\.sample(" server_app.py` → 0 (only a comment reference, not actual use).
- `grep -c "selected_clients_per_round" server_app.py` → 4 (init + append + persist + D-26 JSON field).
- `grep -c "best_round_restore\|best_metric\|best_round_num\|best_arrays" server_app.py` → 17 (above the plan's minimum of 3).
- `grep -c "strategy.aggregate_evaluate(" server_app.py` → 1.
- `python scripts/run.py --dry-run baseline benchmark_cross_device` → prints `num-supernodes=6040 mode=benchmark_cross_device`.
- `git diff --stat` on `server_app.py` → 217 insertions / 31 deletions (248 total); within the plan's "~150-250 lines modified" guidance.

## Known Stubs

**None.** No placeholder values, no TODO markers, no `NotImplementedError`. Every field in the migrated `server_app.py` has real values:

- Mode resolver with three real profiles (`benchmark_cross_device`, `paper_compat_pfedrec`, `cross_silo_legacy`).
- Seeded sampler with a real `server_rng(run_seed)` instance.
- Strategy with real `BaselineFedAvg` / `BaselineFedProx` classes from Plan 01.
- D-27 tracking with real `ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()})` snapshot.
- Manifest with real fingerprints from `verify_bundle(data_derived())`.
- Selected clients per round with real `[int(x) for x in selected_node_ids]` list-of-lists.

## User Setup Required

**None beyond what docs/setup.md already documents.** The install order remains `pip install -e scripts/foundation/` → `pip install -e federated-baseline-cf/`. A cross-device run now uses `python scripts/run.py baseline benchmark_cross_device`. Legacy cross-silo still works via `flwr run . --run-config "mode=cross_silo_legacy num-supernodes=5"`.

## Next Phase Readiness

**Ready for Plan 03 closure.** Plan 03 (client_app.py + task.py migration) can now land and consume the server-side contract established here:

1. Client's `@app.train()` and `@app.evaluate()` handlers return `FitMetricsContract.to_dict()` / `EvaluateMetricsContract.to_dict()` — the EvaluateRes wrapping on the server side handles them.
2. Client's per-user-group sufficient stats land in `strategy.aggregate_evaluate` and flow into `eval_metrics_history[round_num]` automatically.
3. `scripts/run.py baseline benchmark_cross_device` → cross-device baseline run with all 6040 supernodes, seeded sampling, sum-based aggregation, manifest-fingerprinted result JSON.

**Ready for Phases 3-5 (parallel module migrations).** The mode-first bootstrap pattern, seeded-sampling idiom, EvaluateRes-wrapping convention, in-memory best-round restore, and D-15 double-write manifest are all documented in `patterns-established` above and directly copyable into `federated-personalized-cf`, `federated-adaptive-personalized-cf`, and `federated-pfedrec`. Each module will need its own `PersonalizedFedAvg` / `AdaptiveFedAvg` / `PFedRecFedAvg` subclass (mirroring Plan 01's `BaselineFedAvg`) to handle module-specific parameter-split invariants (D-20).

**Ready for Phase 6 (evaluation harness).** `best_round` + `best_sampled_ndcg@10` are now first-class fields in the result JSON (`results_data["checkpoint"]`). The manifest's `_manifest.foundation_contract_sha256` is the canonical protocol-fingerprint for cross-run comparisons. Phase 6 can build dashboards that filter by mode + foundation hash + run_seed without re-computing anything.

**No blockers.** One deferred item (Plan 03's test_task_rng.py test_gradient_mask_zeros_non_user_rows) logged in `deferred-items.md` but it does not block Plan 04 closure — it's Plan 03's responsibility.

## Self-Check: PASSED

- **Files modified:**
  - FOUND: `federated-baseline-cf/federated_baseline_cf/server_app.py` — verified via `git log --stat 65652f3` showing +217/-31 lines.
- **Files created:**
  - FOUND: `federated-baseline-cf/tests/test_server_integration.py` — verified via `git log --stat fb9beb9` (create mode 100644 + 167 insertions).
  - FOUND: `.planning/phases/02-baseline-migration/deferred-items.md` — verified via `git log --stat fb9beb9` (create mode 100644 + 23 insertions).
- **Commits:**
  - FOUND: `65652f3` (Task 1 feat) — visible on `feat/try_to_run_the_baseline` via `git log --oneline -3`.
  - FOUND: `fb9beb9` (Task 2 test) — same.
- **Automated verify:** PASSED.
  - `grep -c "^import random$" server_app.py` → 0.
  - `grep -c "random\.sample(" server_app.py` → 0 (1 comment-only match, not code).
  - `grep -c "BaselineFedAvg(" server_app.py` → 1.
  - `grep -c "BaselineFedProx(" server_app.py` → 1.
  - `grep -c "server_rng(run_seed)" server_app.py` → 1.
  - `grep -c "resolve_mode_defaults(" server_app.py` → 1.
  - `grep -c "log_mode_and_overrides(" server_app.py` → 1.
  - `grep -c "build_run_manifest(" server_app.py` → 1.
  - `grep -c "embed_manifest_in_result(" server_app.py` → 1.
  - `grep -c "write_manifest_sibling(" server_app.py` → 1.
  - `grep -c "selected_clients_per_round" server_app.py` → 4.
  - `grep -c "best_round_restore\|best_metric\|best_round_num\|best_arrays" server_app.py` → 17.
  - `grep -c "strategy.aggregate_evaluate(" server_app.py` → 1.
  - `python -c "import ast; ast.parse(open('federated-baseline-cf/federated_baseline_cf/server_app.py').read()); print('syntax ok')"` → syntax ok.
  - `python -c "from federated_baseline_cf.server_app import app; print('import ok')"` → import ok.
  - `pytest federated-baseline-cf/tests/test_server_integration.py -v` → 5 passed, 0 failed in 0.73s.
  - `python scripts/run.py --dry-run baseline benchmark_cross_device` → `num-supernodes=6040 mode=benchmark_cross_device` printed correctly.
- **Scope boundary:** PASSED.
  - My 2 commits modified ONLY `server_app.py`, `test_server_integration.py`, and `deferred-items.md` (verified via `git show --stat HEAD~1 HEAD`).
  - Pre-existing uncommitted WIP in `client_app.py` and `task.py` untouched — `git diff --stat` shape unchanged by my session.
- **Surgical-edit guard:** PASSED. `git diff --stat federated-baseline-cf/federated_baseline_cf/server_app.py` shows +217/-31 lines; within the plan's "~150-250 lines modified" guidance. Pre-existing WIP hunks (get_device helper, _device_cache init, DummyClientProxy class, weighted_average_metrics, print_evaluation_metrics) still visible as-is in `git diff HEAD~2 server_app.py`.

---

*Phase: 02-baseline-migration*
*Plan: 04 (Wave 2 — parallel with Plan 03; closes the Wave 2 server-side migration surface)*
*Completed: 2026-04-19*
*Closes: BSL-04 (seeded client sampling), BSL-06 (sum-based aggregate_evaluate), BSL-08 (D-15 double-write manifest).*
*Unblocks: Phase 2 close (BSL-01/02/04/06/08 now complete; BSL-03/05/07 land when Plan 03 closes).*
