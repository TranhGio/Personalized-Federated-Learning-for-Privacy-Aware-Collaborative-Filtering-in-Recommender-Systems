---
phase: 04-adaptive-migration-bug-fixes
plan: 01
subsystem: infra
tags: [strategy, adaptive-split-fedavg, adaptive-split-fedprox, sufficient-stats, best-prototype-snapshot, prototype-ema, adp-03, adp-06, d-05, d-08, d-20, d-23, tdd, wave-1]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: EvaluateMetricsContract 12 sufficient-stat keys + FND-06 logging helper (log WARNING) that the aggregate_evaluate override + snapshot_best_prototype consume
  - phase: 03-personalized-migration
    provides: PersonalizedSplitFedAvg sufficient-stat aggregator shape + D-23 split-learning invariant + tests/ conftest fixtures template
provides:
  - AdaptiveSplitFedAvg + AdaptiveSplitFedProx subclasses with sufficient-stat aggregate_evaluate (12-key sum) + prototype-EMA aggregate_fit override + best_prototype snapshot helper
  - Module-level GLOBAL_PARAM_KEYS = {item_embeddings.weight, item_bias.weight, global_bias} + LOCAL_PARAM_KEYS_BASE = {user_embeddings.weight, user_bias.weight} frozensets (BASE only; model dynamically expands with _logit_alpha / _item_perturbation / personal_mlp.* / fusion_*)
  - snapshot_best_prototype(round_num, embedding_dim) -> None helper on both subclasses — D-05 snapshot-on-best-round + D-08 zero-vector degenerate fallback with WARNING
  - federated-adaptive-personalized-cf/tests/ pytest package with conftest.py (fake_evaluate_res + fake_client_proxy fixtures) + test_strategy.py (8 GREEN) + test_dual_model.py (3 GREEN)
  - ADP-02 enable-before-load fingerprint: 3 GREEN tests pin the DualPersonalizedBPRMF._LOCAL_PARAMS + get/set_local_parameters contract that Plan 03 will target when it reorders client_app.py
affects: [04-plan-03-client-app, 04-plan-04-task-py, 04-plan-05-server-app, 04-plan-06-regression-guard]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sufficient-stat server-side aggregation (sum(hit)/sum(evaluated) over per-group totals) — cloned from Phase 3 personalized with base-LOCAL-set reduced to user_* keys (adaptive keys appended dynamically at the model layer)"
    - "aggregate_fit OVERRIDDEN (not pure-inherited like Phase 3): super().aggregate_fit runs first to preserve D-23, then _aggregate_prototypes updates server EMA. Duplicated across FedAvg + FedProx subclasses to avoid diamond inheritance with BaseFedProx — same 4-line duplication Phase 3 used for aggregate_evaluate on FedProx."
    - "D-05 best-round snapshot: self.best_prototype: Optional[np.ndarray] = None on both subclasses; snapshot_best_prototype(round_num, embedding_dim) copies self._global_prototype (deep copy so later EMA mutations don't leak) or falls back to np.zeros + WARNING when no prior aggregation exists (D-08)."
    - "Pytest package scaffolding for adaptive module mirrors Phase 3: empty __init__.py marker + conftest.py (fake_evaluate_res + fake_client_proxy fixtures copied verbatim, only module docstring adjusted) + topical test files."

key-files:
  created:
    - federated-adaptive-personalized-cf/tests/__init__.py
    - federated-adaptive-personalized-cf/tests/conftest.py
    - federated-adaptive-personalized-cf/tests/test_strategy.py
    - federated-adaptive-personalized-cf/tests/test_dual_model.py
  modified:
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py

key-decisions:
  - "aggregate_fit OVERRIDDEN rather than inherited (diverges from Phase 3): the adaptive module has server-side state beyond GLOBAL params (the EMA _global_prototype), so pure inheritance à la Phase 3 is insufficient. D-23 is preserved via an explicit super().aggregate_fit(...) call at the top of the override, verified by a unittest.mock.patch spy test."
  - "best_prototype field placed on the strategy object (not on the ArrayRecord sidecar): server_app.py (Plan 05) will call strategy.snapshot_best_prototype(round_num, embedding_dim) alongside its existing best_arrays snapshot at the same current_ndcg > best_metric moment. Symmetry with Phase 2 D-27 best_arrays keeps the best-round restore fan-out simple."
  - "D-08 degenerate case ships WITH the strategy, not in server_app.py: snapshot_best_prototype internally picks np.zeros(embedding_dim, dtype=np.float32) and logs WARNING via flwr.common.logger.log. Keeps the server_app.py call site a one-liner and makes the degenerate-zero-vector fallback testable at the strategy-unit level (test 7)."
  - "LOCAL_PARAM_KEYS_BASE (frozenset) named with _BASE suffix to make it impossible to accidentally use as the runtime key list: the model's DualPersonalizedBPRMF._LOCAL_PARAMS property owns dynamic expansion with _logit_alpha / _item_perturbation / personal_mlp.* / fusion_*. The strategy layer's frozenset only declares the base cross-module invariant (user_* stays LOCAL)."
  - "AdaptiveSplitFedProx.aggregate_evaluate is an EXACT COPY (not super()) of AdaptiveSplitFedAvg.aggregate_evaluate — mirrors Phase 2 baseline + Phase 3 personalized precedent. aggregate_fit override is ALSO duplicated across subclasses (same 4-line duplication) to avoid diamond inheritance with BaseFedProx when calling super().aggregate_fit."

patterns-established:
  - "Pytest package parity with Phase 3: conftest.py fixtures copied verbatim (only module docstring tweaked); test_strategy.py reuses the 5 sufficient-stat tests with strategy class names substituted and adds 3 adaptive-specific tests (aggregate_fit super-call spy + 2 best_prototype snapshot tests). Phase 5 pfedrec tests SHOULD follow the same shape."
  - "best_prototype snapshot helper pattern: subclass owns snapshot_best_prototype(round_num, embedding_dim) method; no-prior-aggregation fallback to np.zeros + WARNING with substrings 'Prototype snapshot at best round' AND 'zero vector' — caller (server_app.py Plan 05) never needs to check for None."
  - "Fingerprint tests for untouched modules: test_dual_model.py pins ADP-02 enable-before-load contract against the UNMODIFIED DualPersonalizedBPRMF class. Three-test pattern: (1) flags-off baseline, (2) flags-on key presence, (3) round-trip restore with sentinel values. Plan 03 will target test 3 as its acceptance anchor when it reorders client_app.py."

requirements-completed: [ADP-03, ADP-06]

# Metrics
duration: 3min
completed: 2026-04-20
---

# Phase 04 Plan 01: Adaptive Split Strategies + Best-Prototype Snapshot + ADP-02 Fingerprint Tests Summary

**AdaptiveSplitFedAvg/FedProx with sufficient-stat aggregate_evaluate, prototype-EMA aggregate_fit override, best_prototype snapshot with D-08 zero-vector fallback, and 3 GREEN fingerprint tests pinning the enable-before-load contract against the untouched DualPersonalizedBPRMF class.**

## Performance

- **Duration:** ~3 min (between Task 1 and Task 2 commits)
- **Started:** 2026-04-20T08:25:31Z
- **Completed:** 2026-04-20T08:28:34Z (approximate; Task 2 commit timestamp)
- **Tasks:** 2 (both TDD)
- **Files modified:** 5 (1 source + 4 test)

## Accomplishments

- `AdaptiveSplitFedAvg` + `AdaptiveSplitFedProx` shipped in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` with sufficient-stat `aggregate_evaluate` (ADP-06 server half, mirrors Phase 3 shape).
- `aggregate_fit` OVERRIDDEN on both subclasses: `super().aggregate_fit(...)` runs the weighted-average of GLOBAL params (D-23 preserved) then `_aggregate_prototypes(results)` updates the server EMA prototype. Verified by a `unittest.mock.patch.object(BaseFedAvg, "aggregate_fit")` spy test.
- `self.best_prototype: Optional[np.ndarray] = None` field added to both subclasses, plus `snapshot_best_prototype(round_num, embedding_dim) -> None` helper: deep-copies the live EMA when available, else falls back to `np.zeros(embedding_dim, dtype=np.float32)` and logs a WARNING per D-08.
- Module-level frozensets declare the split-learning base boundary: `GLOBAL_PARAM_KEYS = {item_embeddings.weight, item_bias.weight, global_bias}`, `LOCAL_PARAM_KEYS_BASE = {user_embeddings.weight, user_bias.weight}`. Dynamic LOCAL expansion (`_logit_alpha.weight`, `_item_perturbation.weight`, `personal_mlp.*`, `fusion_*`) remains the responsibility of `DualPersonalizedBPRMF._LOCAL_PARAMS` property (unchanged by Phase 4).
- New pytest package at `federated-adaptive-personalized-cf/tests/` with 11 GREEN tests: 8 strategy tests (5 sufficient-stat clones from Phase 3 + super() spy + 2 best_prototype snapshot + 1 frozenset contract guard) + 3 dual-model enable-before-load fingerprint tests.
- ADP-02 enable-before-load contract pinned at the model-unit level: 3 GREEN tests prove that calling `enable_per_user_alpha` + `enable_item_perturbation` BEFORE `set_local_parameters` restores cached `_logit_alpha.weight` + `_item_perturbation.weight` sentinel values through a save/load round-trip. Plan 03 will target this contract when it reorders `client_app.py`.
- Phase 3 full test suite (34/34) continues to pass — no Phase 3 regression.
- Wave-1 write-race safety preserved: `git diff HEAD~2 HEAD` on Plan 02/03/04/05-owned files (pyproject.toml, dataset.py, client_app.py, server_app.py, task.py, models/*) returns empty for Plan 01's two commits.

## Task Commits

Each task was committed atomically:

1. **Task 1: AdaptiveSplitFedAvg + AdaptiveSplitFedProx + best_prototype snapshot + tests/ scaffolding** — `05d8ee3` (feat)
2. **Task 2: DualPersonalizedBPRMF enable-before-load fingerprint tests** — `9269477` (test)

Both commits used `--no-verify` per Wave-1 parallel-execution rule (Plan 02 committed in parallel at `9b9d1f8`).

## Files Created/Modified

- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` — rip-and-replace: class names `SplitFedAvg/SplitFedProx` → `AdaptiveSplitFedAvg/AdaptiveSplitFedProx`; constant `LOCAL_PARAM_KEYS` → `LOCAL_PARAM_KEYS_BASE` (BASE suffix enforces that runtime expansion lives in the model); added sufficient-stat aggregate_evaluate + best_prototype field + snapshot_best_prototype helper.
- `federated-adaptive-personalized-cf/tests/__init__.py` — package marker (empty).
- `federated-adaptive-personalized-cf/tests/conftest.py` — `fake_evaluate_res` + `fake_client_proxy` fixtures copied verbatim from Phase 3, only module docstring reworded.
- `federated-adaptive-personalized-cf/tests/test_strategy.py` — 8 GREEN tests (sufficient-stat sum + per-group ratios + zero-division + FedProx inherit + aggregate_fit super-call spy + 2 best_prototype snapshots + frozenset contract guard).
- `federated-adaptive-personalized-cf/tests/test_dual_model.py` — 3 GREEN tests pinning ADP-02 enable-before-load contract against the UNMODIFIED `DualPersonalizedBPRMF`.

## Decisions Made

See `key-decisions` in frontmatter. Highlights:

- **aggregate_fit OVERRIDDEN (not inherited like Phase 3).** The adaptive module genuinely needs server-side prototype aggregation beyond the GLOBAL-param weighted average. D-23 is preserved via an explicit `super().aggregate_fit(...)` call at the top of the override, verified by the Task 1 Test 5 `unittest.mock.patch.object(BaseFedAvg, "aggregate_fit")` spy.
- **best_prototype lives on the strategy object** (paralleling the Phase 2 D-27 `best_arrays`). Server_app.py (Plan 05) will snapshot both at the same `current_ndcg > best_metric` moment, keeping the best-round restore a single symmetrical step.
- **D-08 degenerate fallback is strategy-internal.** `snapshot_best_prototype` handles the zero-vector case + WARNING itself, so the Plan 05 call site remains a one-liner and the behavior is testable at the strategy-unit level.
- **LOCAL_PARAM_KEYS_BASE suffix is a load-bearing naming choice.** The `_BASE` suffix makes it impossible to accidentally use this frozenset as the runtime local-key list — dynamic expansion (`_logit_alpha`, `_item_perturbation`, `personal_mlp.*`, `fusion_*`) is the model's responsibility via `DualPersonalizedBPRMF._LOCAL_PARAMS` property.
- **FedProx subclass duplicates (does not inherit) aggregate_fit, aggregate_evaluate, and snapshot_best_prototype.** This mirrors the Phase 2 baseline + Phase 3 personalized precedent — exact-copy duplication avoids diamond inheritance with `BaseFedProx` when calling `super()`. Module-level `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics` helpers keep duplication minimal (4 lines per subclass).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added test_frozensets_match_contract as an 8th test**

- **Found during:** Task 1 test authoring
- **Issue:** The plan specified 7 strategy tests. While implementing, the explicit contract that `GLOBAL_PARAM_KEYS` and `LOCAL_PARAM_KEYS_BASE` match the exact expected frozensets was an acceptance criterion of the plan (the `python -c "assert GLOBAL_PARAM_KEYS == frozenset({...})"` smoke test), but only enforced by the external acceptance command — not as a first-class in-suite regression guard.
- **Fix:** Added `test_frozensets_match_contract` that asserts `GLOBAL_PARAM_KEYS == frozenset({"item_embeddings.weight", "item_bias.weight", "global_bias"})` and `LOCAL_PARAM_KEYS_BASE == frozenset({"user_embeddings.weight", "user_bias.weight"})`. Future refactors that accidentally drift the frozenset contents fail fast in the unit suite.
- **Files modified:** `federated-adaptive-personalized-cf/tests/test_strategy.py`
- **Verification:** `pytest tests/test_strategy.py::test_frozensets_match_contract -v` passes.
- **Committed in:** `05d8ee3` (Task 1 commit)

**2. [Rule 1 - Polish] Added from __future__ import annotations to test_strategy.py and test_dual_model.py**

- **Found during:** Test file authoring (both tasks)
- **Issue:** Phase 3 test files (the cloning template) use `from __future__ import annotations`. Under Python 3.9 (the project floor per `CLAUDE.md`), this is a no-op but keeps annotation-forward consistency across the test suite.
- **Fix:** Added the import at the top of both new test files.
- **Files modified:** `federated-adaptive-personalized-cf/tests/test_strategy.py`, `federated-adaptive-personalized-cf/tests/test_dual_model.py`
- **Verification:** All 11 tests pass under Python 3.13; no runtime behavior change.
- **Committed in:** `05d8ee3` + `9269477`

---

**Total deviations:** 2 auto-fixed (1 missing-critical regression guard, 1 polish)
**Impact on plan:** Both auto-fixes are additive regression guards; no scope creep. The 8-test strategy suite is a strict superset of the 7 planned tests.

## Issues Encountered

None. TDD RED → GREEN on Task 1 (initial pytest collection failed as expected with `ImportError: cannot import name 'AdaptiveSplitFedAvg'`; implementation was a one-shot GREEN). Task 2 was GREEN on first run since it targeted the untouched `DualPersonalizedBPRMF` class directly.

## User Setup Required

None — no external service configuration required. `pytest` is already installed system-wide and is being declared as a dev-dep by Plan 02's `pyproject.toml` edit (Wave-1 parallel commit `9b9d1f8`).

## Next Phase Readiness

- **Plan 03 (client_app.py enable-before-load fix + D-04..D-10 manifest-sidecar cache at schema_version=2)** is unblocked. It will target:
  - `AdaptiveSplitFedAvg` frozenset contract (`GLOBAL_PARAM_KEYS`, `LOCAL_PARAM_KEYS_BASE`) for the GLOBAL/LOCAL split at the client boundary.
  - `DualPersonalizedBPRMF._LOCAL_PARAMS` runtime expansion — the enable-before-load contract test (`test_enable_before_load_restores_cached_alpha`) is the acceptance anchor Plan 03 will hit after reordering the `enable_per_user_alpha` + `enable_item_perturbation` calls to BEFORE `load_local_user_embeddings`.
- **Plan 04 (task.py FND-06 RNG + FND-03 exclusion + cold-round branch)** is unblocked by Plan 01's sufficient-stat contract.
- **Plan 05 (server_app.py migration + best-round restore)** is unblocked:
  - Drop `AdaptiveSplitFedAvg`/`AdaptiveSplitFedProx` into the strategy slot (replacing the raw `FedAvg`/`FedProx`).
  - Call `strategy.snapshot_best_prototype(round_num=server_round, embedding_dim=embedding_dim)` at the same `current_ndcg > best_metric` moment where `best_arrays` is already snapshotted in Phase 2 D-27.
  - Before the final centralized/broadcast evaluation, set `strategy._global_prototype = strategy.best_prototype` (D-07) so clients receive the restored prototype via the `train_config_dict["global_prototype"]` field.
- **Plan 06 (regression-guard subprocess test)** is unblocked: the pytest package + fixtures exist; Plan 06 just adds a new `tests/test_subprocess_determinism.py` file.

No blockers. Phase 3 full test suite (34/34) continues to pass — no cross-phase regression introduced.

---
*Phase: 04-adaptive-migration-bug-fixes, Plan 01*
*Completed: 2026-04-20*

## Self-Check: PASSED

All 5 created/modified files and both task commits (`05d8ee3`, `9269477`) verified present.
