---
phase: 06-evaluation-reporting-harness
plan: 04
subsystem: federated-personalized-cf
tags: [server-app, results-path, extra-eval-round, nested-final-metrics, wandb, d-06, d-07, evl-01, evl-02, evl-03, evl-04, evl-06]

# Dependency graph
requires:
  - phase: 06-evaluation-reporting-harness
    provides: "Plan 01: module_run_results_dir(module, run_id) helper at fedrec_foundation.paths"
  - phase: 06-evaluation-reporting-harness
    provides: "Plan 02: RunManifest schema v2 with final_eval_round_index + metrics fields + sibling_name kwarg"
  - phase: 03-personalized-migration
    provides: "Plan 04: personalized server_app.py with D-27 best-round restore + D-15 manifest double-write"
provides:
  - "federated-personalized-cf/server_app.py with Phase-6 harness: per-run-dir layout, D-06 extra-eval-round, nested final_metrics, best/last W&B namespaces"
  - "4 NEW integration tests in federated-personalized-cf/tests/test_server_integration.py pinning D-02/D-06/D-07/D-09"
  - "Updated path probe in scripts/foundation/tests/test_personalized_determinism.py for Phase-6 per-run-dir layout"
affects:
  - "Phase 7 thesis evaluation run — consumes results/federated/personalized/<run_id>/results.json"
  - "06-evaluation-reporting-harness Plan 06/07 — same pattern for adaptive + pfedrec modules"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-06 extra-eval-round after best-arrays restore: broadcast all partition_to_node_id nodes, aggregate via strategy.aggregate_evaluate(final_eval_round_index, ...), result populates final_metrics['best'] instead of stale eval_metrics_history[best_round_num]"
    - "D-07 nested final_metrics schema: {best, last, best_round, last_round, final_eval_round_index}"
    - "Pitfall-9 last_round = max(eval_metrics_history.keys()) guards against early-stopping edge cases"
    - "Pitfall-8 cross-silo coexistence: if mode in ('benchmark_cross_device', 'paper_compat_pfedrec') -> per-run-dir; else -> legacy flat layout"
    - "np.float64 JSON-safe coercion at best_round_metrics assignment site (float(v) if isinstance(v, (int, float)))"
    - "dataclass_replace(manifest, final_eval_round_index=N, metrics=...) post-build mutation pattern (Plan 02 contract)"
    - "atomic_write_json replaces json.dump at results write site"
    - "W&B summary: final/* -> best/* + last/* namespace migration"

key-files:
  created:
    - "federated-personalized-cf/tests/test_server_integration.py — 4 NEW tests appended (test_results_path_repo_root_anchored, test_extra_eval_round_replaces_history_lookup, test_canonical_artifact_carries_best_and_last_blocks, test_round_metrics_history_carries_per_group_exposure)"
  modified:
    - "federated-personalized-cf/federated_personalized_cf/server_app.py — Phase-6 harness wired (+~90 lines, -25 lines)"
    - "scripts/foundation/tests/test_personalized_determinism.py — path probe updated to personalized/*/results.json layout"

key-decisions:
  - "D-06 extra-eval-round uses partition_to_node_id (all nodes, no sampling) for reproducibility; the broadcast result overwrites eval_metrics_history[best_round_num] lookup (the D-06-forbidden pattern) — correctness over latency"
  - "Pitfall-9 last_round from max key chosen over actual_rounds to correctly handle early-stopping cases where actual_rounds may not equal the last round with eval metrics"
  - "np.float64 coercion applied at best_round_metrics assignment site (path b) rather than at JSON write site — ensures the nested dict is Python-native before dataclass_replace and any downstream access"
  - "test_round_metrics_history_carries_per_group_exposure uses a live strategy.aggregate_evaluate() call (not source grep) to prove per-group counts actually flow through the aggregation pipeline; source-level check verifies eval_metrics_history storage"
  - "sibling_name='manifest.json' inlined directly into the cross-device write_manifest_sibling call (not via **dict unpacking) so the acceptance criterion grep matches the literal kwarg form"

patterns-established:
  - "D-06 extra-eval-round pattern: after arrays = best_arrays restore, broadcast evaluate to sorted(partition_to_node_id.values()), pass final_eval_round_index as the round number, aggregate via strategy.aggregate_evaluate — same shape for Plans 05/06"
  - "D-07 nested final_metrics pattern: best from extra-eval (or last_block fallback), last from max(eval_metrics_history.keys()), both blocks + round indices in one dict — reuse for adaptive + pfedrec"
  - "Mode-conditional results path: cross-device -> module_run_results_dir + clean filenames; legacy -> repo_root/results/federated flat layout — Pitfall 8 must be replicated in all four modules"

requirements-completed: [EVL-01, EVL-02, EVL-03, EVL-04, EVL-06]

# Metrics
duration: 7min
completed: 2026-04-29
---

# Phase 6 Plan 04: Personalized Server App Phase-6 Harness Summary

**Wired D-06 extra-eval-round + D-07 nested final_metrics + per-run-dir path migration into federated-personalized-cf/server_app.py, closing the D-06-forbidden eval_metrics_history[best_round_num] lookup and migrating W&B summary to best/last namespaces.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-29T08:01:52Z
- **Completed:** 2026-04-29T08:08:23Z
- **Tasks:** 1 (TDD: RED + GREEN, no refactor needed)
- **Files modified:** 3

## Accomplishments

- `server_app.py`: Added `module_run_results_dir`, `atomic_write_json`, `dataclass_replace` imports + `_MODULE: str = "personalized"` constant.
- D-06 extra-eval-round block inserted after `arrays = best_arrays` restore: broadcasts all `partition_to_node_id.values()` nodes, aggregates via `strategy.aggregate_evaluate(final_eval_round_index, ...)`, result populates `best_round_metrics`. The previously-forbidden `eval_metrics_history.get(final_round_for_metrics, {})` lookup is gone.
- D-07 nested `final_metrics = {best, last, best_round, last_round, final_eval_round_index}` replaces the old flat dict. Pitfall-9 closure: `last_round = max(eval_metrics_history.keys())` not `actual_rounds`.
- W&B summary keys migrated from `final/*` to `best/*` + `last/*`.
- Manifest mutated via `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` between `build_run_manifest` and `embed_manifest_in_result`.
- Results path: cross-device modes write to `module_run_results_dir(_MODULE, run_id) / "results.json"` + sibling `manifest.json`; legacy cross-silo writes flat `<run_id>_results.json` (D-03 + Pitfall 8 preserved).
- `atomic_write_json` replaces legacy `json.dump` at the write site.
- np.float64 JSON-safe coercion at `best_round_metrics` assignment (path b from plan-checker iteration 1).
- 4 NEW integration tests: all GREEN; full personalized suite 38/38 GREEN.
- `test_personalized_determinism.py` path probe updated from legacy `*_results.json` to `personalized/*/results.json`.

## Task Commits

TDD task; RED + GREEN committed separately:

1. **Task 1 (RED): 4 failing tests** — `4a2f546` (test) — 4 new test functions appended to test_server_integration.py; all fail against unmodified server_app.py.
2. **Task 1 (GREEN): Phase-6 server_app wiring** — `b69e48f` (feat) — All edits to server_app.py + determinism test path probe. All 38 personalized tests GREEN.

## Files Created/Modified

- `federated-personalized-cf/federated_personalized_cf/server_app.py` (modified, +~90/-25 lines) — Phase-6 harness: 5 new imports, `_MODULE` constant, D-06 extra-eval-round block, D-07 nested final_metrics, Pitfall-8/9 closures, W&B namespace migration, manifest dataclass_replace, mode-conditional results path, atomic_write_json.
- `federated-personalized-cf/tests/test_server_integration.py` (modified, +260 lines) — 4 NEW test functions covering D-02 path migration (test_results_path_repo_root_anchored), D-06 extra-eval-round wiring (test_extra_eval_round_replaces_history_lookup), D-07 nested schema (test_canonical_artifact_carries_best_and_last_blocks), D-09 per-group exposure (test_round_metrics_history_carries_per_group_exposure).
- `scripts/foundation/tests/test_personalized_determinism.py` (modified, +8/-4 lines) — `_run_personalized` path probe updated: primary glob now `_RESULTS_DIR.glob("personalized/*/results.json")`, filtered to run_id; legacy flat pattern and mtime-newest fallbacks preserved for pre-Phase-6 compatibility.

## Decisions Made

- **Inline `sibling_name="manifest.json"` kwarg** rather than `**dict` unpacking — the acceptance criterion grep looks for the literal `sibling_name="manifest.json"` as a keyword arg form. The dict form `{"sibling_name": "manifest.json"}` uses colon not equals so the grep would not match. Inlining is cleaner anyway.
- **Source-level check in test_round_metrics_history_carries_per_group_exposure** changed from explicit `evaluated_users_sparse` token search to checking `eval_metrics_history[round_num] = dict(thesis_metrics)` storage line — the per-group keys flow through `dict(thesis_metrics)` without being named individually in server_app.py; a token search would always fail and would be the wrong invariant to pin.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] strategy.aggregate_evaluate call had newline before first arg**
- **Found during:** Task 1 GREEN (test_extra_eval_round_replaces_history_lookup assertion)
- **Issue:** The acceptance criterion grep for `strategy.aggregate_evaluate(final_eval_round_index` failed because the original multi-line call split the first arg onto the next line.
- **Fix:** Collapsed to single-line `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])`.
- **Files modified:** federated-personalized-cf/federated_personalized_cf/server_app.py
- **Verification:** Test passed after fix.
- **Committed in:** b69e48f (Task 1 GREEN commit)

**2. [Rule 1 - Bug] dataclass_replace(manifest...) had newline breaking acceptance criterion grep**
- **Found during:** Task 1 GREEN (edit-order check)
- **Issue:** `dataclass_replace(\n    manifest,` — acceptance criterion looks for `dataclass_replace(manifest` as a contiguous string. Newline broke the match.
- **Fix:** Moved `manifest` arg to same line as the function call: `dataclass_replace(manifest,`.
- **Files modified:** federated-personalized-cf/federated_personalized_cf/server_app.py
- **Verification:** Edit-order check `python -c "... assert idx_replace > idx_final ..."` passed.
- **Committed in:** b69e48f (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — call formatting to match acceptance criterion string patterns)
**Impact on plan:** Both trivial formatting fixes with no semantic change. The code logic was correct; only the line formatting needed adjustment to satisfy the plan's literal grep tests.

## Issues Encountered

None beyond the two auto-fixed formatting deviations above.

## User Setup Required

None — pure server_app.py + test changes; no external service configuration required.

## Next Phase Readiness

- **Wave-2 personalized harness complete:** `results/federated/personalized/<run_id>/results.json` + `manifest.json` will be written for all `benchmark_cross_device` and `paper_compat_pfedrec` runs.
- **Plans 05 (adaptive) and 06 (pfedrec)** follow the same pattern; they can copy the D-06 extra-eval-round block, D-07 nested final_metrics structure, and mode-conditional path logic from this plan.
- **D-18 surgical scope:** Only `federated-personalized-cf/federated_personalized_cf/server_app.py`, `federated-personalized-cf/tests/test_server_integration.py`, and `scripts/foundation/tests/test_personalized_determinism.py` were modified by this plan. `strategy.py`, `client_app.py`, `dataset.py`, `task.py` remain untouched.

## Known Stubs

None — all data paths are wired. The extra-eval-round block will produce an empty `best_round_metrics` dict if `checkpoint_rule` is not `best_round_restore`/`best_round` (collapses to `last_block`), which is the correct behavior per D-07.

## Self-Check: PASSED

- FOUND: federated-personalized-cf/federated_personalized_cf/server_app.py (modified, all edits present)
- FOUND: federated-personalized-cf/tests/test_server_integration.py (modified, 4 new tests)
- FOUND: scripts/foundation/tests/test_personalized_determinism.py (modified, path probe updated)
- FOUND: .planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-04-SUMMARY.md (this file)
- FOUND: commit 4a2f546 (RED — 4 failing tests)
- FOUND: commit b69e48f (GREEN — server_app implementation)
- pytest federated-personalized-cf/tests/ -q -m "not slow": 38 passed, 0 failed

---
*Phase: 06-evaluation-reporting-harness*
*Plan: 04 — Personalized server_app Phase-6 harness (EVL-01/02/03/04/06)*
*Completed: 2026-04-29*
