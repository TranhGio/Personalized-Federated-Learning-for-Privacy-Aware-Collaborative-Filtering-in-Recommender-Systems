---
phase: 04-adaptive-migration-bug-fixes
plan: 03
subsystem: infra
tags: [client-app, task, rng-threading, exclusion-set, benchmark-assertion, enable-before-load, fit-metrics-contract, evaluate-metrics-contract, per-group-metrics, embedding-cache-manifest-v2, schema-version-2, cold-start-branch, alpha-zero-override, contrastive-skip, d-24-gradient-isolation-ghost-table, adp-02, adp-04, adp-05, adp-06, d-03, d-04, d-09, d-13, d-14, d-15, d-16]

# Dependency graph
requires:
  - phase: 04-adaptive-migration-bug-fixes-01
    provides: "AdaptiveSplitFedAvg/FedProx strategy with best_prototype helper + EvaluateMetricsContract reader"
  - phase: 04-adaptive-migration-bug-fixes-02
    provides: "pyproject cross-device defaults + dataset.py foundation adapter (natural partition)"
  - phase: 03-personalized-migration-03
    provides: "manifest-sidecar cache v1 template + FND-06 RNG wiring + FND-03 exclusion threading"
  - phase: 02-baseline-migration-03
    provides: "D-24 snapshot/restore pattern for per-user-indexed ghost tables"
provides:
  - "ADP-02 enable-before-load ordering fix wired into @app.train + @app.evaluate (PRIMARY Phase-4 BUG FIX)"
  - "task.py FND-06 RNG + FND-03 exclusion + D-13/D-14 cold-round + D-24 ghost-table gradient isolation"
  - "schema_version=2 manifest-sidecar cache (12 signature fields, D-04 loud mismatch with rm -rf hint)"
  - "D-16 alpha diagnostics computed client-side and routed via alpha_diagnostics MetricRecord sidecar"
  - "G-03-01 discover_only short-circuit + partition_id echo on both Fit + Evaluate contracts"
affects:
  - 04-adaptive-migration-bug-fixes-05 (server_app can now drop AdaptiveSplitFedAvg + discovery round + partition-space sampling + best_prototype snapshot call)
  - 04-adaptive-migration-bug-fixes-06 (subprocess determinism regression guard extends to v2 cache)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ADP-02 enable-before-load ordering (enable_per_user_alpha + enable_item_perturbation BEFORE _load_local_user_state)"
    - "schema_version=2 manifest-sidecar cache (extends Phase-3 v1 with 6 Phase-4 fingerprint fields)"
    - "D-13 cold-round try/finally alpha override bracket in train_dual_personalized"
    - "D-24 snapshot_non_user_rows/restore_non_user_rows bracket around optimizer.step for user_embeddings + user_bias + _logit_alpha (item_perturbation NOT protected — item-indexed)"
    - "D-16 sidecar MetricRecord for per-user alpha diagnostics (strict FitMetricsContract forbids inline extras)"

key-files:
  created:
    - "federated-adaptive-personalized-cf/tests/test_task_rng.py"
    - "federated-adaptive-personalized-cf/tests/test_client_assertion.py"
    - "federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py"
  modified:
    - "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py"
    - "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py"

key-decisions:
  - "ADP-02 ordering fix placed in @app.train Step 5 + mirrored in @app.evaluate Step 5 — both handlers compute the effective enable flags via _resolve_enable_flags(context, is_benchmark_mode), compute per-user alphas, then invoke _apply_enable_before_load() BEFORE any call to _load_local_user_state(). Source-proximity guard (enable_per_user_alpha index < _load_local_user_state index) asserted by automated check."
  - "D-03 ablation-only override semantics: under benchmark_cross_device mode, enable flags default ON when run_config lacks the key; an explicit False in run_config flips them OFF (genuine ablation). Non-benchmark modes honor the run_config value verbatim with default False (backwards compatible)."
  - "D-16 alpha diagnostics routed via a SEPARATE MetricRecord keyed 'alpha_diagnostics' rather than extending FitMetricsContract — the Phase-2 D-21 strict contract rejects free-form extras, and extending the dataclass would require touching scripts/foundation/. The sidecar pattern sits inside the Flower RecordDict and is server-side-readable via msg.content['alpha_diagnostics']."
  - "D-24 scope answered per RESEARCH Open Question 4: user_embeddings + user_bias + _logit_alpha are protected (all user-indexed ghost tables); _item_perturbation is NOT protected (item-indexed, legitimately full-table updated). _D24_PROTECTED_EMBEDDINGS tuple in task.py lists exactly those three names."
  - "D-13/D-14 implemented as a try/finally bracket in train_dual_personalized: save current alpha, force 0.0, zero contrastive_lambda_eff, delegate to train_bpr_mf, restore saved alpha on unwind. Tests verify both the override AND the restore (spy captures call_args list)."
  - "Schema-v1 -> v2 rejection implemented as the first comparison key in _load_local_user_state's all_keys tuple — Phase-3 caches encountering Phase-4 code hit the same loud RuntimeError path as any other field mismatch."

patterns-established:
  - "Enable-before-load ordering: explicitly placing LOCAL-key-creating calls (enable_per_user_alpha, enable_item_perturbation) BEFORE the cache load so _LOCAL_PARAMS is complete at load time. Any future adaptive extension that adds a new optional LOCAL tensor must enable it in the same sequence."
  - "Ghost-table D-24 with per-embedding selective masking: the _D24_PROTECTED_EMBEDDINGS constant drives snapshot/restore; adding a new ghost table is a one-line tuple entry. Item-indexed tensors are explicitly excluded."
  - "Sidecar MetricRecord for diagnostics that don't fit the strict contract — precedent set for future thesis diagnostics (per-user alpha evolution, item-perturbation norm distribution, etc.)."

requirements-completed: [ADP-02, ADP-04, ADP-05, ADP-06]

# Metrics
duration: 45min
completed: 2026-04-20
---

# Phase 4 Plan 03: client_app.py + task.py Cross-Device Migration with ADP-02 Enable-Before-Load Fix Summary

**ADP-02 enable-before-load ordering fix + schema_version=2 manifest-sidecar cache + D-13/D-14 cold-round branch + D-16 alpha diagnostics sidecar + D-24 ghost-table isolation — the adaptive client/task half of the Phase-4 cross-device contract**

## Performance

- **Duration:** ~45 minutes
- **Started:** 2026-04-20
- **Completed:** 2026-04-20
- **Tasks:** 2 (both GREEN on first implementation pass after RED tests)
- **Files modified:** 2 (client_app.py, task.py) + 3 new test files
- **Total LOC (modified + created):** 2,752

## Accomplishments
- **Primary Phase-4 bug fix (ADP-02)**: `enable_per_user_alpha` + `enable_item_perturbation` are now called BEFORE `_load_local_user_state` in both `@app.train()` and `@app.evaluate()`. Cached `_logit_alpha.weight` + `_item_perturbation.weight` tensors are restored correctly every round instead of silently re-initialized from the heuristic (CONCERNS.md §enable-after-load).
- **ADP-04 benchmark-mode one-user assertion** fires before any training/eval in both handlers via `assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)`.
- **ADP-05 FND-03 exclusion threading**: `exclude_items = bundle["exclusion"].for_user(partition_id)` folded into training-negative AND eval-negative candidate pools at both the client and task levels.
- **ADP-06 FND-06 RNG wiring**: stdlib-random seeding / sampling stripped from `task.py` (previously at lines 952-953 + 1012); replaced with `np_rng(run_seed, user_idx, round_num, purpose)` instances. Cross-file guard test covers both `task.py` and `client_app.py`.
- **Schema_version=2 manifest-sidecar cache** — 12 signature fields: 6 Phase-3 carry-forwards + `alpha_method`, `fusion_type`, `mlp_hidden_dims`, `per_user_alpha_enabled`, `item_perturbation_enabled`, `contrastive_lambda`. Any mismatch on load raises `RuntimeError` with per-field delta + literal `rm -rf` hint (D-04). Phase-3 schema_version=1 caches also loudly rejected.
- **D-09 reuse-cache opt-in** routes path to `.embedding_cache/sig_{sha256(fields)[:16]}/` so two runs with identical signature silently share the cache directory.
- **D-13/D-14 cold-round branch** in `train_dual_personalized`: `is_cold_round=True` sets `model.set_alpha(0.0)` (prototype-only blend) and zeros `contrastive_lambda_eff`; try/finally restores the saved alpha on unwind. Cold-round signal derived client-side from a cache-exists probe.
- **D-16 alpha diagnostics** — 6 scalar fields (`alpha_mean`, `alpha_std`, `alpha_p25`, `alpha_p50`, `alpha_p75`, `alpha_clip_hit_rate`) computed via `_compute_alpha_diagnostics(model)` after training and routed via a sibling `MetricRecord` keyed `"alpha_diagnostics"` (strict FitMetricsContract rejects inline extras).
- **D-24 gradient isolation** restored for the Phase-4 ghost-table setup: `_snapshot_non_user_rows` + `_restore_non_user_rows` bracket `optimizer.step()` in `train_bpr_mf` (and therefore in `train_dual_personalized`), protecting non-active rows of `user_embeddings`, `user_bias`, and `_logit_alpha`. `_item_perturbation` is item-indexed and NOT protected.
- **G-03-01 carry-forward**: `discover_only=True` short-circuit in `@app.evaluate()` returns a zero-suffstats `EvaluateMetricsContract` with `partition_id` populated and no data/model load. Every contract build — train AND evaluate — echoes `partition_id=partition_id`.

## Task Commits

Each task was committed atomically (with --no-verify per Wave-2 parallel-execution rule):

1. **Task 1: task.py FND-06 RNG + FND-03 exclusion + D-13/D-14 cold-round + D-24 ghost-table isolation** — `0123621` (feat)
2. **Task 2: client_app ADP-02 enable-before-load + schema-v2 cache + alpha diagnostics** — `65f063c` (feat)

## Files Created/Modified

**Modified:**
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` — Rip-and-replace migration. FND-06 RNG threaded through `train` / `train_dual_personalized` / `train_bpr_mf` / `train_basic_mf` / `evaluate_ranking_sampled`; `_sample_negatives_seeded` helper; D-24 `_snapshot_non_user_rows` + `_restore_non_user_rows` helpers plus `_D24_PROTECTED_EMBEDDINGS` constant tuple; D-13/D-14 cold-round bracket in `train_dual_personalized`; stdlib random eradicated (module body, docstrings, comments — all clean per cross-file regression test).
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` — Rip-and-replace migration. Mode resolver + one-user assert + ADP-02 enable-before-load ordering (via `_apply_enable_before_load`); schema_version=2 cache helpers (`_signature_fields_v2`, `_cache_dir_for_run`, `_save_local_user_state`, `_load_local_user_state`); D-16 alpha diagnostics helper (`_compute_alpha_diagnostics`); alpha-config factory (`_build_alpha_configs`) + enable-flag resolver (`_resolve_enable_flags`); discover_only short-circuit; strict FitMetricsContract + EvaluateMetricsContract wire payloads; user_prototype + alpha_diagnostics routed as sibling MetricRecords.

**Created:**
- `federated-adaptive-personalized-cf/tests/test_task_rng.py` — 5 GREEN tests (parametrized cross-file random-strip regression = 2 tests + exclusion + RNG signature + cold-round alpha=0 override).
- `federated-adaptive-personalized-cf/tests/test_client_assertion.py` — 5 GREEN tests (benchmark one-user assert, override bypass, primary evaluator selection, FitMetrics payload + alpha diagnostics sidecar shape, EvaluateMetrics payload with partition_id + free-form rejection).
- `federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py` — 5 GREEN tests (v2 sidecar written+loaded with 12 fields, loud mismatch with rm -rf hint, reuse-cache sig path, extended LOCAL key round-trip byte-identical, v1 manifest rejection under v2 code).

## Decisions Made

See frontmatter `key-decisions` — six decisions recorded:
1. ADP-02 ordering placement (Step 5 in both handlers; source-proximity guard).
2. D-03 ablation-only override semantics (benchmark default ON; explicit False flips OFF; non-benchmark default False).
3. D-16 alpha diagnostics routed as sibling MetricRecord (strict contract forbids inline).
4. D-24 scope — user_embeddings + user_bias + _logit_alpha only (item_perturbation excluded per Research Open Question 4).
5. D-13/D-14 try/finally bracket placement in train_dual_personalized.
6. Schema-v1 rejection via the first-key comparison in all_keys tuple.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reworded docstring text to avoid substring collision with RNG-strip regex**
- **Found during:** Task 1 (test_random_seed_calls_stripped failing despite source being clean)
- **Issue:** The `evaluate_ranking_sampled` docstring referenced the old code as `"random.seed(seed)"` + `"random.sample(...)"` inside prose. The test uses a literal `"random.seed(" not in src` check; docstring substrings were false-positive matches.
- **Fix:** Reworded to `"the old seed-call at line 952-953 and the stdlib sample-call at line 1012 — both against the random module"`. Semantically equivalent documentation without the literal substrings.
- **Files modified:** `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` (one docstring block)
- **Verification:** `grep -cE "random\\.seed\\(|random\\.sample\\(" task.py` returned 0; cross-file regression test passed.
- **Committed in:** `0123621` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — test regex false-positive on docstring prose; documentation-only fix, no behavior change)
**Impact on plan:** Plan executed exactly as specified. The one-line docstring reword was a pure text-hygiene fix to satisfy the acceptance test literal-match rule.

## Issues Encountered

None during implementation. The two tasks landed GREEN on first implementation pass after the RED tests (modulo the docstring false-positive noted above).

## Known Stubs

None. Every LOCAL tensor persisted by `_save_local_user_state` round-trips byte-identical on load (verified by `test_extended_local_key_payload_shape`). The InfoNCE no-op under 1-user-per-client (RESEARCH Open Question 2) is a DOCUMENTED property, not a stub — the `contrastive_lambda=0.1` benchmark default remains the thesis-config value and the fix is explicitly deferred to Phase 4.5 per the plan's critical-notes block.

## Test Results

- Plan 01 + Plan 02 pre-existing: 20 tests (all passing)
- Plan 03 Task 1 (test_task_rng.py): 5 tests (all GREEN)
- Plan 03 Task 2 (test_client_assertion.py + test_embedding_cache_manifest_v2.py): 10 tests (all GREEN)
- Plan 04 tests (test_alpha_factory.py, owned by parallel executor): 18 tests (all passing)
- **Total:** 53/53 GREEN in `federated-adaptive-personalized-cf/`.

## Next Phase Readiness

Plan 05 (server_app.py migration) now has everything it needs:
- Strategy layer shipped in Plan 01 (AdaptiveSplitFedAvg + FedProx with sufficient-stat aggregate_evaluate and best_prototype snapshot branch).
- Client-side contract shipped in this plan: strict FitMetricsContract + EvaluateMetricsContract with partition_id echo, discover_only short-circuit, user_prototype sidecar, alpha_diagnostics sidecar, ArrayRecord(global_params) reply.
- Dataset adapter in place from Plan 02; pyproject benchmark defaults shipped.
- Phase-4 cache layout v2 is operational; Plan 06's subprocess determinism regression guard only needs to pin the v2 signature set.

Plan 05 can now:
- Adopt the Phase-2+Phase-3 server_app template (mode resolver, seeded client sampling in partition-id space via G-03-01 discovery round, aggregate_evaluate consuming sufficient stats, best-round restore via BSL-07 + D-27).
- Wire the best_prototype snapshot call alongside self.best_arrays.
- Consume the alpha_diagnostics MetricRecord sidecar for W&B round logs.
- Embed best_prototype in the result-JSON `_manifest` field (D-06 double-write).

## Self-Check: PASSED

- File `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py`: FOUND (991 lines → now 1163)
- File `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py`: FOUND (645 lines → now 991)
- File `federated-adaptive-personalized-cf/tests/test_task_rng.py`: FOUND (new, 198 lines)
- File `federated-adaptive-personalized-cf/tests/test_client_assertion.py`: FOUND (new, 163 lines)
- File `federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py`: FOUND (new, 237 lines)
- Commit `0123621`: FOUND (Task 1)
- Commit `65f063c`: FOUND (Task 2)
- `pytest tests/` in `federated-adaptive-personalized-cf/`: 53 passed, 0 failed, 0 skipped
- ADP-02 source-proximity guard: `enable_per_user_alpha` index < `_load_local_user_state` index → TRUE
- D-18 scope boundary: `git diff --stat` of strategy.py / dataset.py / models/ / pyproject.toml → empty
- Plan 04 file (`tests/test_alpha_factory.py`) untouched by this plan's commits

---

*Phase: 04-adaptive-migration-bug-fixes*
*Plan: 03*
*Completed: 2026-04-20*
