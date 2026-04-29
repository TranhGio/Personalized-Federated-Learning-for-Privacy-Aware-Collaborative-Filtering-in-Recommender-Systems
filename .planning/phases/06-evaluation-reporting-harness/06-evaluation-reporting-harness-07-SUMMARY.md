---
phase: 06-evaluation-reporting-harness
plan: "07"
subsystem: testing
tags: [wandb, sweep, regression-guard, d-09, evl-05, evl-06, pitfall-7]

# Dependency graph
requires:
  - phase: 06-evaluation-reporting-harness
    plan: "03"
    provides: "baseline server_app best/last W&B namespaces + D-09 test (partial)"
  - phase: 06-evaluation-reporting-harness
    plan: "04"
    provides: "personalized server_app best/last W&B namespaces + D-09 test (partial)"
  - phase: 06-evaluation-reporting-harness
    plan: "05"
    provides: "adaptive server_app best/last W&B namespaces + D-09 test (full)"
  - phase: 06-evaluation-reporting-harness
    plan: "06"
    provides: "pfedrec server_app best/last W&B namespaces + D-09 test (partial) + slash-delimiter key deviation documented"
provides:
  - "federated-adaptive-personalized-cf/sweep.yaml metric.name migrated final/ -> best/ (Pitfall 7 closure)"
  - "NEW test_wandb_summary_keys.py: 5 test items pinning sweep.yaml structured parse + all 4 module server_app namespace migrations"
  - "All 4 test_server_integration.py files strengthened to full canonical D-09 required_keys set"
affects:
  - "Phase 7 thesis evaluation: wandb sweep agent now reads correct best/sampled_ndcg@10 metric"
  - "Future regression guards: any revert of final/* namespace in any server_app will trip test_wandb_summary_keys.py"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "yaml.safe_load structured parse for sweep.yaml assertions (not substring grep) — comment-proof"
    - "Parametrized pytest across 4 server_app.py paths for cross-module source regression guard"
    - "required_keys issubset pattern for D-09 per-group exposure checks"

key-files:
  created:
    - "federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py — 2 test functions (5 items): sweep YAML structured parse + 4-module namespace parametrize"
  modified:
    - "federated-adaptive-personalized-cf/sweep.yaml — line 18: name: final/sampled_ndcg@10 -> name: best/sampled_ndcg@10"
    - "federated-baseline-cf/tests/test_server_integration.py — strengthened D-09: added evaluated_users to required_keys set"
    - "federated-personalized-cf/tests/test_server_integration.py — strengthened D-09: added evaluated_users to required_keys set"
    - "federated-pfedrec/tests/test_server_integration.py — strengthened D-09: added evaluated_users (overall) assertion alongside slash-delimiter per-group keys"

key-decisions:
  - "yaml.safe_load structured parse (not substring grep): a comment like '# was final/sampled_ndcg@10' would satisfy a substring check but cannot satisfy structured loading loaded['metric']['name']. This is the plan-checker MAJOR fix from iteration 1."
  - "Both f-string and raw-string final/* patterns guarded: test_summary_keys_use_best_last_namespace checks both wandb.run.summary[f\"final/ and wandb.run.summary[\"final/ forms per plan-checker MINOR fix."
  - "Pfedrec slash-delimiter deviation acknowledged: PFedRecSplitFedAvg emits evaluated_users/sparse (slash) not evaluated_users_sparse (underscore). Strengthening added evaluated_users (overall, no suffix) while leaving the slash per-group keys intact."
  - "Adaptive test already at full strength: Plan 05 already checked all four keys including evaluated_users. No changes needed for adaptive."

patterns-established:
  - "required_keys issubset idiom: replace N individual assert X in metrics with a set-difference check that reports ALL missing keys in one failure message — cleaner for future extension"
  - "Parametrized source-grep test pattern: @pytest.mark.parametrize('server_app_path', _SERVER_APPS, ids=lambda p: p.parts[-3]) gives one test item per module with the module name as the test id for immediate diagnostic locality"

requirements-completed: [EVL-03, EVL-05, EVL-06]

# Metrics
duration: ~8min
completed: 2026-04-29
---

# Phase 6 Plan 07: Cross-Cutting W&B Namespace + D-09 Exposure Guards Summary

**Pitfall 7 closed: sweep.yaml metric.name migrated final/ -> best/sampled_ndcg@10 via yaml.safe_load structured assertion; D-09 per-round exposure regression guard strengthened to full 4-key required_keys set in all four modules.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-29T08:40:00Z
- **Completed:** 2026-04-29T08:46:41Z
- **Tasks:** 2
- **Files modified:** 5 (1 sweep.yaml + 1 new test file + 3 strengthened test files)

## Accomplishments

- **Pitfall 7 closure**: `federated-adaptive-personalized-cf/sweep.yaml` line 18 mutated from `name: final/sampled_ndcg@10` to `name: best/sampled_ndcg@10`. Without this change, the next `wandb agent` run would read a key that no longer exists in W&B summary (Plans 03-06 migrated all thesis metrics to `best/*`), silently receive NaN, and fail to converge the Bayesian sweep.
- **NEW `test_wandb_summary_keys.py`**: 2 test functions producing 5 test items. `test_sweep_yaml_metric_name_uses_best_namespace` uses `yaml.safe_load` + `loaded["metric"]["name"]` structured navigation — comment-proof per plan-checker iteration 1 MAJOR fix. `test_summary_keys_use_best_last_namespace` parametrized across all 4 module server_apps checks both f-string (`wandb.run.summary[f"best/`) and positive surfaces (best/*, last/*) plus both f-string and raw-string `final/*` negative surfaces — per plan-checker iteration 1 MINOR fix.
- **D-09 strengthening**: Plans 03, 04, 06 each added a weaker version of the per-round exposure guard: baseline and personalized only checked `evaluated_users_sparse/medium/dense` (3 keys), pfedrec checked only the slash-delimiter variants. This plan strengthens all three to the canonical `required_keys = {evaluated_users, evaluated_users_sparse, evaluated_users_medium, evaluated_users_dense}` set. Adaptive (Plan 05) was already full — no changes needed.

## Task Commits

1. **Task 1: sweep.yaml migration + test_wandb_summary_keys.py** — `20a5879` (feat)
2. **Task 2: D-09 strengthening in 3 test files** — `9a7b66e` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `federated-adaptive-personalized-cf/sweep.yaml` (modified, 1 line): `name: final/sampled_ndcg@10` → `name: best/sampled_ndcg@10`
- `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` (created, 92 lines): 2 test functions — sweep YAML structured parse + 4-module server_app namespace parametrize
- `federated-baseline-cf/tests/test_server_integration.py` (modified, +10/-7 lines): D-09 guard strengthened — individual 3-key asserts replaced with `required_keys` issubset check (4 keys)
- `federated-personalized-cf/tests/test_server_integration.py` (modified, +11/-6 lines): same strengthening as baseline
- `federated-pfedrec/tests/test_server_integration.py` (modified, +5/-0 lines): `evaluated_users` (overall) assertion added alongside existing slash-delimiter per-group key assertions

## Decisions Made

- **yaml.safe_load structured parse over substring grep**: The acceptance criteria explicitly forbade a substring check because a comment `# was final/sampled_ndcg@10` would have satisfied it. Structured parse navigates `loaded["metric"]["name"]` — comment-proof and future-schema-change-visible.
- **Both f-string AND raw-string negative assertions**: Plan-checker iteration 1 MINOR fix required guarding both `wandb.run.summary[f"final/` (loop form) and `wandb.run.summary["final/` (literal form). Plan 06 removed both forms from pfedrec; this test guards against either regressing.
- **Pfedrec slash-delimiter deviation preserved**: `PFedRecSplitFedAvg.aggregate_evaluate` uses slash-delimiter for per-group keys (`evaluated_users/sparse`) — documented in Plan 06 SUMMARY. The strengthening added `evaluated_users` (overall, no slash) which the strategy also emits. Slash-delimiter per-group keys left intact.
- **Adaptive unchanged**: Plan 05's test already covered all four keys including `evaluated_users`. Adding a duplicate check would violate the acceptance criterion of exactly 1 definition per file.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Strengthening] D-09 tests in baseline, personalized, and pfedrec were weaker than canonical required_keys**
- **Found during:** Task 2 inspection (Step 1: grep confirmed all 4 tests existed; Step 2: read each test body revealed 3 of 4 were missing `evaluated_users` key)
- **Issue:** Baseline and personalized checked only `evaluated_users_{sparse,medium,dense}` (3 keys); pfedrec checked only `evaluated_users/{sparse,medium,dense}` (slash-delimiter, 3 keys). All three missing the overall `evaluated_users` count from the canonical required_keys = {evaluated_users, evaluated_users_sparse, evaluated_users_medium, evaluated_users_dense}.
- **Fix:** Replaced individual 3-key asserts in baseline and personalized with `required_keys` issubset pattern (4 keys). Added `evaluated_users` assertion to pfedrec while preserving slash-delimiter per-group keys.
- **Files modified:** baseline/tests/test_server_integration.py, personalized/tests/test_server_integration.py, pfedrec/tests/test_server_integration.py
- **Verification:** Each strengthened module test passes individually; all four module non-slow suites green (26/38/73/41).
- **Committed in:** 9a7b66e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical coverage in upstream plan tests)
**Impact on plan:** Strengthening was the intended purpose of this plan per Task 2 action step 3 ("if a module's upstream plan added a weaker version, STRENGTHEN it"). No scope creep.

## Test Results

- `test_wandb_summary_keys.py`: **5/5 PASSED** (1 sweep YAML + 4 parametrized server_app)
- `federated-baseline-cf/tests/test_server_integration.py::test_round_metrics_history_carries_per_group_exposure`: **PASSED**
- `federated-personalized-cf/tests/test_server_integration.py::test_round_metrics_history_carries_per_group_exposure`: **PASSED**
- `federated-adaptive-personalized-cf/tests/test_server_integration.py::test_round_metrics_history_carries_per_group_exposure`: **PASSED** (already full strength; unchanged)
- `federated-pfedrec/tests/test_server_integration.py::test_round_metrics_history_carries_per_group_exposure`: **PASSED**
- Full non-slow suites: **baseline 26, personalized 38, adaptive 73, pfedrec 41** — zero regressions

## D-18 Surgical Scope Verification

This plan modified:
- `federated-adaptive-personalized-cf/sweep.yaml` (1 line in config file)
- `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` (NEW test file)
- `federated-baseline-cf/tests/test_server_integration.py` (test-only edit)
- `federated-personalized-cf/tests/test_server_integration.py` (test-only edit)
- `federated-pfedrec/tests/test_server_integration.py` (test-only edit)

NOT modified: any `server_app.py`, `strategy.py`, `client_app.py`, `dataset.py`, `task.py`, or `models/` files. Wave-3 file-disjointness maintained.

## Known Stubs

None — all data paths verified. sweep.yaml metric.name directly matches the `best/sampled_ndcg@10` key written by all four module server_app.py W&B summary loops.

## Self-Check: PASSED

- FOUND: `federated-adaptive-personalized-cf/sweep.yaml` — `name: best/sampled_ndcg@10` at line 18
- FOUND: `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` — created (92 lines)
- FOUND: all 4 `test_server_integration.py` files — `test_round_metrics_history_carries_per_group_exposure` count=1 in each
- FOUND: commit `20a5879` — Task 1 (sweep.yaml + test_wandb_summary_keys.py)
- FOUND: commit `9a7b66e` — Task 2 (3 strengthened test files)
- GREP: `grep -c "name: best/sampled_ndcg@10" sweep.yaml` = 1
- GREP: `grep -c "name: final/sampled_ndcg@10" sweep.yaml` = 0
- GREP: `grep -c "yaml.safe_load(_SWEEP_YAML.read_text())" test_wandb_summary_keys.py` = 1
- GREP: `grep -c 'loaded["metric"]["name"]' test_wandb_summary_keys.py` = 1
- GREP: `grep -c '"name: best/sampled_ndcg@10" in text' test_wandb_summary_keys.py` = 0
- All 5 test_wandb_summary_keys.py tests: PASSED
- All 4 per-module D-09 tests: PASSED
- Full non-slow suites: 26+38+73+41 = 178 tests PASSED, 0 failed

---
*Phase: 06-evaluation-reporting-harness*
*Plan: 07 — Cross-cutting W&B namespace + D-09 exposure guards (final plan in Phase 6)*
*Completed: 2026-04-29*
