---
phase: 06-evaluation-reporting-harness
plan: "06"
subsystem: federated-pfedrec
tags: [pfedrec, server_app, pitfall-1, d-02, d-06, d-07, evl-01, evl-02, evl-03, evl-04, evl-06, wandb-migration, nested-schema, extra-eval-round, path-migration]

dependency_graph:
  requires:
    - 06-evaluation-reporting-harness-01  # foundation paths primitive (module_run_results_dir)
    - 06-evaluation-reporting-harness-02  # RunManifest schema v2 + write_manifest_sibling sibling_name kwarg
    - 05-pfedrec-migration-reproduction-04  # pfedrec server_app D-14 hook + D-27 best-round restore
  provides:
    - "pfedrec server_app wired with D-02/D-06/D-07 + Pitfall-1 PFR-08 hook consuming final_metrics['best']"
    - "EVL-01/02/03/04/06 closed for federated-pfedrec module"
    - "5 new integration tests covering results path, extra-eval-round, nested schema, per-group exposure, Pitfall-1 headline guard"
  affects:
    - scripts/foundation/tests/test_pfedrec_subprocess_determinism.py  # path probe updated

tech_stack:
  added: []
  patterns:
    - "D-07 nested final_metrics: {best, last, best_round, last_round, final_eval_round_index}"
    - "Pitfall-1 closure: _emit_pfr_08_verification(final_metrics['best'], ...) — ONLY the call-site arg changes, not the hook signature"
    - "D-06 extra-eval-round after arrays = best_arrays: strategy.aggregate_evaluate(final_eval_round_index, ...)"
    - "np.float64 JSON-safe coercion in best_round_metrics comprehension"
    - "dataclass_replace(manifest, ...) BEFORE embed_manifest_in_result (edit-order invariant)"
    - "Pitfall-8 mode branch: cross-device/paper_compat -> module_run_results_dir; cross_silo_legacy -> legacy repo_root()/.../pfedrec"
    - "Pitfall-9 last_round = max(eval_metrics_history.keys())"
    - "slash-delimiter keys in PFedRecSplitFedAvg.aggregate_evaluate output (evaluated_users/sparse)"

key_files:
  created: []
  modified:
    - federated-pfedrec/federated_pfedrec/server_app.py
    - federated-pfedrec/tests/test_server_integration.py
    - scripts/foundation/tests/test_pfedrec_subprocess_determinism.py

decisions:
  - "Pitfall-1 closure: the hook _emit_pfr_08_verification receives final_metrics['best'] (positional), not final_metrics (flat). Hook ORDER unchanged: AFTER embed_manifest_in_result, BEFORE W&B summary write. Hook signature itself NOT touched."
  - "Per-group exposure keys use slash delimiter (evaluated_users/sparse) not underscore (evaluated_users_sparse) — matches PFedRecSplitFedAvg.aggregate_evaluate actual output; test assertions corrected to match"
  - "D-06 extra-eval-round fires only when checkpoint_rule in ('best_round_restore', 'best_round') and best_round_num > 0; else best block falls back to last block"
  - "cross_silo_legacy else-branch NOT replaced with NotImplementedError — legacy write path preserved per D-03 + PROJECT.md backwards-compat constraint; sibling_kwarg={} for default <run_id>-manifest.json"
  - "W&B PFR-08 audit surface: top-level keys pfr08 / pfr08_delta_hr_pts / pfr08_delta_ndcg_pts (no namespace prefix — independent of best/last thesis metrics)"

metrics:
  duration: "~30 min (multi-session: RED in prior session, GREEN in this session)"
  completed_date: "2026-04-29"
  tasks_completed: 1
  files_modified: 3
---

# Phase 6 Plan 06: PFedRec Server App Phase-6 Wiring Summary

PFedRec server_app.py wired with foundation primitives (D-02 path migration, D-06 extra-eval-round, D-07 nested schema) and Pitfall-1 closed: the D-14 PFR-08 auto-verify hook now receives `final_metrics["best"]` instead of flat `final_metrics`, preventing NaN-delta false-FAILED stamps under the new nested schema.

## What Was Built

### Pitfall-1 Closure (HEADLINE)

The D-14 PFR-08 hook `_emit_pfr_08_verification` reads `sampled_hr@10` and `sampled_ndcg@10`. Under the Phase-6 D-07 nested schema, those keys live at `final_metrics["best"][...]`, not `final_metrics[...]`. The call site (line 1141) was rewired from:

```python
_emit_pfr_08_verification(final_metrics, reference_path=..., tolerance_pts=2.0)
```

to:

```python
_emit_pfr_08_verification(final_metrics["best"], reference_path=..., tolerance_pts=2.0)
```

The hook signature, hook function body, hook ORDER (after `embed_manifest_in_result`, before W&B summary), and the `results_data["_manifest"]["pfr08_verification"] = pfr08_audit` post-embed mutation are all preserved verbatim. Only the input dict at the call site changed.

### D-02: Path Migration

Replaced `Path("../results/federated/pfedrec")` with `module_run_results_dir(_MODULE, run_id)` for cross-device and `paper_compat_pfedrec` modes. For `cross_silo_legacy`, the original write path is preserved via an else-branch (`legacy_dir = repo_root() / "results" / "federated" / "pfedrec"`). `atomic_write_json` replaces `json.dump`. `sibling_name="manifest.json"` passes D-04 clean artifact naming for the cross-device path.

### D-06: Extra Eval Round

After `arrays = best_arrays` (the D-27 best-round restore), a new block broadcasts an evaluate message to all `partition_to_node_id` nodes with `final_eval_round_index = actual_rounds + 1`. The responses are aggregated via `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])`. The result becomes `best_round_metrics` — the canonical `final_metrics["best"]` block.

### D-07: Nested Final Metrics

`final_metrics` is now:

```python
final_metrics = {
    "best": best_round_metrics or last_block,
    "last": last_block,
    "best_round": best_round_num if best_round_num > 0 else last_round,
    "last_round": last_round,
    "final_eval_round_index": final_eval_round_index,
}  # type: Dict[str, Any]
```

Pitfall-9 closure: `last_round = max(eval_metrics_history.keys())`.

### W&B Summary Migration

Thesis metrics migrated from `final/*` to `best/*` + `last/*` namespaces. PFR-08 audit keys migrated from `final/pfr08*` to top-level `pfr08` / `pfr08_delta_hr_pts` / `pfr08_delta_ndcg_pts` (independent surface, no namespace).

### Manifest Mutation

`dataclass_replace(manifest, final_eval_round_index=..., metrics=results_data["final_metrics"])` inserted AFTER `build_run_manifest` and BEFORE `embed_manifest_in_result` (edit-order invariant — `dataclass_replace` must precede embed so the manifest fields carry the updated values).

### test_pfedrec_subprocess_determinism.py

Path probe updated from flat `*_results.json` glob to `pfedrec/*/results.json` (per-run-dir layout). Existing `_manifest.pfr08_verification` byte-identity invariant preserved.

## Integration Tests (5 New)

| Test | What it pins |
|------|-------------|
| `test_results_path_repo_root_anchored` | D-02 import checks, no legacy path literal, `module_run_results_dir` call, `sibling_name` kwarg, `atomic_write_json` |
| `test_extra_eval_round_after_best_arrays_restore` | D-06 ordering (extra-eval AFTER best_arrays), `final_eval_round_index` count>=5, Pitfall-9 max(keys()), Pitfall-8 mode branch, legacy_dir + sibling_kwarg |
| `test_canonical_artifact_carries_best_and_last_blocks` | D-07 nested keys, W&B best/last namespaces, no final/* legacy, schema_version==2, dataclass_replace import, pfr08_verification post-embed mutation, np.float64 coercion, edit-order invariant |
| `test_round_metrics_history_carries_per_group_exposure` | D-09 live PFedRecSplitFedAvg call with slash-delimiter key assertions (evaluated_users/sparse, /medium, /dense); summed values 4/8/3 |
| `test_pfr08_hook_consumes_nested_best_block` | HEADLINE Pitfall-1 regression guard: (a) positive path — final_metrics["best"] with paper-anchor values returns passed=True, no NaN deltas, our_hr==0.7287; (b) negative path — flat dict with no "best" key raises KeyError; (c) source assertion — final_metrics["best"] present in server_app.py, legacy flat-input pattern absent |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test assertions used wrong key format for PFedRecSplitFedAvg per-group output**

- **Found during:** Task 1 (GREEN phase)
- **Issue:** Test `test_round_metrics_history_carries_per_group_exposure` asserted `"evaluated_users_sparse" in metrics` (underscore format) but `PFedRecSplitFedAvg.aggregate_evaluate` emits `"evaluated_users/sparse"` (slash format). The strategy was not changed — the test assertions were wrong.
- **Fix:** Changed all 6 per-group assertions to slash-delimiter form (`evaluated_users/sparse`, `evaluated_users/medium`, `evaluated_users/dense`) with `pytest.approx(4.0)` comparisons.
- **Files modified:** `federated-pfedrec/tests/test_server_integration.py`
- **Commit:** `4eaff85`

### None — Plan executed as written for all other edits.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| `37731ac` | test | RED: 5 failing Phase-6 tests for pfedrec server_app wiring |
| `4eaff85` | feat | GREEN: wire foundation primitives + Pitfall-1 closure (all 41 tests pass) |

## Known Stubs

None. All wired paths produce concrete artifacts. The D-06 extra-eval-round has a graceful fallback (`best block falls back to last block`) for degenerate cases but does not stub the final_metrics schema.

## Self-Check: PASSED

Files exist:
- `federated-pfedrec/federated_pfedrec/server_app.py` — FOUND (modified)
- `federated-pfedrec/tests/test_server_integration.py` — FOUND (extended: 8+5=13 tests)
- `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` — FOUND (path probe updated)

Commits exist:
- `37731ac` — FOUND
- `4eaff85` — FOUND

Tests: 41/41 pass (`cd federated-pfedrec && pytest tests/ -q -m "not slow"` exits 0)
