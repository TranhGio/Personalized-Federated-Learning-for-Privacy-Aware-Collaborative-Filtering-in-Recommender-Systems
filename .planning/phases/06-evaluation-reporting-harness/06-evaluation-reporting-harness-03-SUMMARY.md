---
phase: 06-evaluation-reporting-harness
plan: 03
subsystem: baseline
tags: [server_app, results-path, extra-eval-round, nested-final-metrics, d-02, d-06, d-07, evl-01, evl-02, evl-03, evl-04, evl-06]

# Dependency graph
requires:
  - phase: 06-evaluation-reporting-harness
    plan: 01
    provides: "module_run_results_dir(module, run_id) -> repo-root-anchored Path"
  - phase: 06-evaluation-reporting-harness
    plan: 02
    provides: "RunManifest schema v2 (final_eval_round_index, metrics fields) + sibling_name kwarg"
provides:
  - "federated-baseline-cf/federated_baseline_cf/server_app.py wired with D-02 path, D-06 extra-eval-round, D-07 nested final_metrics, best/last W&B namespaces"
  - "4 NEW integration tests pinning EVL-01/02/03/04/06 invariants"
  - "test_baseline_subprocess_determinism.py slow regression guard re-enabling folded phase2 path-bug todo"
affects:
  - "Phase 7 thesis evaluation: results/federated/baseline/<run_id>/results.json is the canonical artifact"
  - "W&B dashboards: best/* and last/* namespaces replace legacy final/*"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-02 path migration: module_run_results_dir(_MODULE, run_id) replaces Path('../results/federated')"
    - "D-06 extra-eval-round: broadcast to ALL partition_to_node_id nodes after best_arrays restore"
    - "D-07 nested final_metrics: {best, last, best_round, last_round, final_eval_round_index}"
    - "Pitfall-9 closure: last_round = max(eval_metrics_history.keys()), not actual_rounds"
    - "Pitfall-10 closure: centralized eval feeds last block diagnostics; best block is federated-only"
    - "np.float64 coercion at best_round_metrics assignment site (plan-checker MAJOR fix)"
    - "dataclass_replace before embed_manifest_in_result — ordering enforced by source-level test"

key-files:
  created:
    - "scripts/foundation/tests/test_baseline_subprocess_determinism.py - @pytest.mark.slow regression guard (163 lines)"
  modified:
    - "federated-baseline-cf/federated_baseline_cf/server_app.py - +158 lines: D-02 path, D-06 extra eval, D-07 nested schema, best/last W&B namespaces, manifest post-build mutation"
    - "federated-baseline-cf/tests/test_server_integration.py - +299 lines: 4 NEW tests + BLOCKER slow test migration"

key-decisions:
  - "D-02 closed at server_app.py call site: module_run_results_dir(_MODULE, run_id) with _MODULE='baseline' local constant. Single repo-root-anchored path. Resolves folded phase2-baseline-determinism-path-bug.md todo."
  - "D-06: extra-eval-round broadcasts to ALL nodes in partition_to_node_id.values() (sorted). No sampling — reproducibility over latency per CONTEXT.md D-06 decision. Extra-eval result becomes final_metrics['best']."
  - "D-07 + Pitfall-9: last_round = max(eval_metrics_history.keys()) guards against early-stopping edge cases where actual_rounds != last recorded eval round."
  - "Pitfall-10: centralized eval block (RMSE/MAE/ranking_metrics/sampled_metrics) feeds final_metrics['last'] diagnostics ONLY; final_metrics['best'] comes exclusively from the federated extra-eval-round."
  - "D-03 + Pitfall-8: cross-silo legacy path preserved via mode branch — only 'benchmark_cross_device'/'paper_compat_pfedrec' use the new per-run-dir layout."
  - "np.float64 coercion path (b) chosen: best_round_metrics dict comprehension applies float(v) cast at assignment site so downstream dataclass_replace + atomic_write_json never see np.float64."
  - "Edit ordering: dataclass_replace(manifest, ...) BEFORE embed_manifest_in_result so _manifest._final_eval_round_index and _manifest.metrics are populated in the result JSON."
  - "Parallel Wave-2 execution: Plan 04 agent committed Plan 03 files (server_app.py + test_server_integration.py + test_baseline_subprocess_determinism.py) together in commit b69e48f since Plan 03 had no prior commit at the time."

patterns-established:
  - "Post-build manifest mutation pattern: dataclasses.replace(manifest, field=value) after build_run_manifest, before embed_manifest_in_result — ensures schema-v2 fields are populated in both the embedded _manifest key and the sibling JSON."
  - "Extra-eval-round pattern: after best_arrays restore, broadcast @evaluate to ALL partition_to_node_id nodes; wrap responses as EvaluateRes; call strategy.aggregate_evaluate(final_eval_round_index, extra_results, []). Reusable by Plans 04/05/06."

requirements-completed: [EVL-01, EVL-02, EVL-03, EVL-04, EVL-06]

# Metrics
duration: ~7min
completed: 2026-04-29
---

# Phase 6 Plan 03: Baseline server_app Phase-6 wiring Summary

**Wired module_run_results_dir + D-06 extra-eval-round + nested final_metrics + best/last W&B namespaces into baseline server_app.py, closing EVL-01/02/03/04/06 for the baseline module and resolving the folded phase2-baseline-determinism-path-bug.md todo.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-29T08:01:46Z
- **Completed:** 2026-04-29T08:08:41Z
- **Tasks:** 1 (multi-edit single task with 10 surgical edits)
- **Files modified:** 2 (server_app.py, test_server_integration.py)
- **Files created:** 1 (test_baseline_subprocess_determinism.py)

## Accomplishments

- **Edit 1-2**: Added `from fedrec_foundation.paths import module_run_results_dir, repo_root`, `from fedrec_foundation.atomic import atomic_write_json`, `from dataclasses import replace as dataclass_replace`, `from typing import Any` and `_MODULE: str = "baseline"` local constant.
- **Edit 3 (D-06)**: Inserted extra-eval-round block after best_arrays restore, broadcasting to `sorted(partition_to_node_id.values())` — ALL nodes, no sampling. Calls `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])`. Result populates `best_round_metrics` with np.float64 coercion applied inline.
- **Edit 4 (D-07)**: Restructured `final_metrics` from flat dict to nested `{best, last, best_round, last_round, final_eval_round_index}`. `last` merges federated in-loop metrics with centralized diagnostics (Pitfall-10). `best` comes exclusively from the extra-eval-round result. Pitfall-9 closed: `last_round = max(eval_metrics_history.keys())`.
- **Edit 5**: Migrated W&B summary from `final/*` to `best/*` + `last/*` namespaces. `final/*` completely removed.
- **Edit 6**: Post-build manifest mutation via `dataclass_replace(manifest, final_eval_round_index=N, metrics=results_data["final_metrics"])`. Placed BEFORE `embed_manifest_in_result` so `_manifest` in result JSON carries schema-v2 fields.
- **Edit 7 (D-01/D-02/D-03)**: Results path conditional: cross-device runs write to `module_run_results_dir(_MODULE, run_id) / "results.json"` with `sibling_name="manifest.json"` (D-04). Cross-silo writes to legacy `repo_root() / "results" / "federated" / f"{run_id}_results.json"`. `atomic_write_json` replaces `json.dump`.
- **Edit 8**: Created `scripts/foundation/tests/test_baseline_subprocess_determinism.py` with `@pytest.mark.slow` + `FEDREC_SKIP_SLOW=1` escape hatch. Contains `test_selected_partitions_byte_identical_across_subprocess_reruns` probing `_RESULTS_DIR.glob("*/results.json")` (Phase-6 per-run-dir layout, not legacy flat).
- **Edit 9**: 4 NEW integration tests added to `federated-baseline-cf/tests/test_server_integration.py`: `test_results_path_repo_root_anchored`, `test_extra_eval_round_after_best_arrays_restore`, `test_canonical_artifact_carries_best_and_last_blocks`, `test_round_metrics_history_carries_per_group_exposure`.
- **Edit 10 (BLOCKER)**: Migrated in-tree slow test glob patterns from `*_results.json` (legacy flat) to `baseline/*/results.json` (Phase-6 per-run-dir). Migrated schema lookup from `final_metrics.get("sampled_ndcg@10")` to `final_metrics["best"].get("sampled_ndcg@10")`.

## Task Commits

1. **Task 1**: All edits committed in `b69e48f` (parallel Wave-2: Plan 04 agent committed Plan 03 files since Plan 03 had no prior commit at wave execution time. Commit message: `feat(06-04): wire module_run_results_dir + D-06 extra-eval-round + nested final_metrics into personalized server_app`)

## Test Results

- 4 new integration tests: **4/4 PASSED** (`test_results_path_repo_root_anchored`, `test_extra_eval_round_after_best_arrays_restore`, `test_canonical_artifact_carries_best_and_last_blocks`, `test_round_metrics_history_carries_per_group_exposure`)
- Full baseline non-slow suite: **26 passed, 1 deselected** — zero regressions
- Full foundation non-slow suite: **100 passed, 4 deselected** — zero regressions

## Files Created/Modified

- `federated-baseline-cf/federated_baseline_cf/server_app.py` (modified, +158 lines): All 10 edits applied. Import block extended; `_MODULE` constant added; D-06 extra-eval block inserted; flat final_metrics restructured to nested; W&B summary migrated; manifest post-build mutation added; results path conditional branching.
- `federated-baseline-cf/tests/test_server_integration.py` (modified, +299 lines): 4 NEW Phase-6 tests + BLOCKER slow-test migration (2 glob patterns + 2 schema lookups).
- `scripts/foundation/tests/test_baseline_subprocess_determinism.py` (created, 163 lines): `@pytest.mark.slow` subprocess regression guard probing Phase-6 per-run-dir layout.

## Decisions Made

- **_MODULE constant placement**: Inside `@app.main()` immediately after mode resolve (not at module level) to mirror the Phase-3/4/5 pattern of keeping mode-dependent constants near their first use.
- **np.float64 coercion**: Path (b) chosen per plan-checker iteration 1 — inline at `best_round_metrics` dict comprehension. Downstream `dataclass_replace` and `atomic_write_json` both see pure Python `float`.
- **Edit ordering**: `dataclass_replace` placed BEFORE `embed_manifest_in_result`. The acceptance criterion python one-liner verifies this at the source level (`idx_final < idx_replace`).
- **Inline sibling_name kwarg**: `write_manifest_sibling(manifest, results_filename, sibling_name="manifest.json")` used as explicit kwarg (not via `**sibling_kwarg` dict) so the acceptance grep `sibling_name="manifest.json"` returns 1 cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing import] Added `Any` to typing imports**
- **Found during:** Edit 3 (extra-eval-round block uses `Dict[str, Any]`)
- **Issue:** `Any` was missing from `from typing import Dict, List, Tuple, Optional`
- **Fix:** Added `Any` to the typing import line
- **Files modified:** `federated-baseline-cf/federated_baseline_cf/server_app.py`
- **Commit:** b69e48f

**2. [Rule 1 - Bug] edit ordering — embed_manifest_in_result called before dataclass_replace**
- **Found during:** Post-edit review of the results section
- **Issue:** Initial sequencing had `embed_manifest_in_result` before `dataclass_replace`, meaning `_manifest` in the JSON would lack schema-v2 fields
- **Fix:** Swapped order — `dataclass_replace` now runs first, then `embed_manifest_in_result`
- **Files modified:** `federated-baseline-cf/federated_baseline_cf/server_app.py`
- **Commit:** b69e48f

**3. [Rule 1 - Bug] Inline sibling_name kwarg for grep acceptance**
- **Found during:** Post-edit acceptance grep check (`sibling_name="manifest.json"` returned 0)
- **Issue:** Used `**sibling_kwarg` dict unpacking which doesn't match the acceptance grep literal
- **Fix:** Refactored to use explicit inline `sibling_name="manifest.json"` kwarg in the cross-device branch
- **Files modified:** `federated-baseline-cf/federated_baseline_cf/server_app.py`
- **Commit:** b69e48f

## Known Stubs

None — all Phase-6 wiring is fully connected. `best_round_metrics` falls back to `last_block_federated` when `checkpoint_rule` is `last_round` (documented collapse behavior, not a stub).

## Self-Check: PASSED

- FOUND: `federated-baseline-cf/federated_baseline_cf/server_app.py` — `module_run_results_dir`, `dataclass_replace`, `_MODULE`, `final_eval_round_index`, `best_round_metrics`, `sibling_name="manifest.json"`, `atomic_write_json`, `max(eval_metrics_history.keys())`, `if mode in ("benchmark_cross_device"` present.
- FOUND: `federated-baseline-cf/tests/test_server_integration.py` — 4 new tests present + 2 `baseline/*/results.json` globs + 2 `final_metrics"]["best"]` lookups.
- FOUND: `scripts/foundation/tests/test_baseline_subprocess_determinism.py` — `test_selected_partitions_byte_identical_across_subprocess_reruns` + `@pytest.mark.slow` present.
- FOUND: commit b69e48f — contains all 3 Plan 03 files.
- 4 new integration tests: PASSED (4/4)
- Full baseline non-slow suite: PASSED (26/26)
- Full foundation non-slow suite: PASSED (100/100)
- `grep -c 'Path("\.\./results/federated")' server_app.py` returns 0 (D-02 hard cutover).
- `grep -c 'json.dump(results_data' server_app.py` returns 0 (atomic_write_json replaces it).

---
*Phase: 06-evaluation-reporting-harness*
*Completed: 2026-04-29*
