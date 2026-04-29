---
phase: 07-thesis-evaluation-run
plan: 02
subsystem: server-app-wiring
tags: [thesis-tagging, manifest-mutation, mode-tuple-gates, run-config, dataclass-replace]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: dataclass_replace import already present in all 4 server_apps; embed_manifest_in_result + write_manifest_sibling pattern
  - phase: 06-evaluation-reporting-harness
    provides: Phase 6 D-07 manifest mutation site (final_eval_round_index + metrics) — Phase 7 D-22 thesis kwargs land in the SAME dataclass_replace call
  - phase: 07-thesis-evaluation-run-01
    provides: _THESIS_CROSSDEVICE_MAIN ModeProfile + RunManifest schema v3 (thesis_run_label / ablation_dimension / ablation_value fields with safe defaults)
provides:
  - "All 4 server_app.py files route to federated-cf-cross-device W&B project + module_run_results_dir for thesis_crossdevice_main mode (BOTH mode-tuple gates extended to 3-tuples)"
  - "All 4 server_app.py files mutate manifest via dataclass_replace with thesis_run_label / ablation_dimension / ablation_value read from context.run_config BEFORE embed_manifest_in_result (Pitfall 2 closure)"
  - "All 4 pyproject.toml [tool.flwr.app.config] blocks declare safe defaults for the 3 new keys so flwr's fuse_dicts validation accepts orchestrator overrides"
  - "4 new test_thesis_label_in_manifest source-level wiring tests (one per module) pinning Pitfall 2 + Pitfall 3 invariants"
affects: [07-thesis-evaluation-run-03, 07-thesis-evaluation-run-04, 07-thesis-evaluation-run-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Manifest mutation extension: thesis kwargs land in the EXISTING Phase-6 dataclass_replace call (single mutation site, not a second replace call), preserving the post-build mutation pattern from Phase 6 D-07"
    - "Mode-tuple gate extension: 3-tuple form (benchmark_cross_device, thesis_crossdevice_main, paper_compat_pfedrec) replaces 2-tuple in BOTH gate sites per server_app — defensive coverage for both W&B project routing and per-run-dir results layout"
    - "Run-config sentinel defaults: empty-string thesis-run-label / ablation-value, none ablation-dimension — non-thesis runs are detectable by sentinel inspection at the aggregator layer (Plan 04)"

key-files:
  created: []
  modified:
    - "federated-baseline-cf/federated_baseline_cf/server_app.py (line 295 W&B gate, line 895 manifest mutation, line 909 results-path gate)"
    - "federated-personalized-cf/federated_personalized_cf/server_app.py (line 382 W&B gate, line 980 manifest mutation, line 994 results-path gate)"
    - "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py (line 490 W&B gate, line 1275 manifest mutation, line 1298 results-path gate)"
    - "federated-pfedrec/federated_pfedrec/server_app.py (line 514 W&B gate, line 1118 manifest mutation, line 1165 results-path gate)"
    - "federated-baseline-cf/pyproject.toml (lines 102-104 thesis-run-label / ablation-dimension / ablation-value)"
    - "federated-personalized-cf/pyproject.toml (lines 105-107 thesis-run-label / ablation-dimension / ablation-value)"
    - "federated-adaptive-personalized-cf/pyproject.toml (lines 210-212 thesis-run-label / ablation-dimension / ablation-value)"
    - "federated-pfedrec/pyproject.toml (lines 58-60 thesis-run-label / ablation-dimension / ablation-value)"
    - "federated-baseline-cf/tests/test_server_integration.py (test_thesis_label_in_manifest + _server_app_src helper at line 572-622; schema_version literal bump to v3 at line 501)"
    - "federated-personalized-cf/tests/test_server_integration.py (test_thesis_label_in_manifest at lines 472+; schema_version literal bump to v3 at line 400; existing test_server_integration source-string assertion already 3-tuple form at line 324)"
    - "federated-adaptive-personalized-cf/tests/test_server_integration.py (test_thesis_label_in_manifest + _server_app_src helper at lines 644+)"
    - "federated-pfedrec/tests/test_server_integration.py (test_thesis_label_in_manifest at lines 620+; schema_version literal bump to v3 at line 397; existing source-string assertion already 3-tuple form at line 321)"
    - ".planning/phases/07-thesis-evaluation-run/deferred-items.md (logged Plan 07-02 out-of-scope baseline subprocess slow test)"

key-decisions:
  - "Plan A (extend the 2-tuples to 3-tuples) chosen over Plan B (orchestrator emits wandb-project override per cell) per RESEARCH.md primary recommendation: 8 line edits + matches existing Phase-6 pattern + keeps orchestrator script independent of per-module server_app internals."
  - "Single dataclass_replace mutation call carries BOTH Phase-6 (final_eval_round_index, metrics) AND Phase-7 (thesis_run_label, ablation_dimension, ablation_value) kwargs — preserves the 'one mutation point' invariant from Phase 6 D-07 and avoids drift between the two mutation sites that the Pitfall 2 ordering check (idx_thesis_kwarg < idx_embed) protects against."
  - "Static source-string assertion (Path read_text + substring + .find ordering) chosen over live subprocess Flower run for test_thesis_label_in_manifest: 30+ minute live runs incompatible with fast suite; same pattern as existing pfedrec source-string assertions; smoke-run validation deferred to Plan 05 gate."
  - "Pitfall 3 closure verified by negative grep (`grep -rln '\"benchmark_cross_device\", \"paper_compat_pfedrec\"[^,]'` returns 0 matches across all 4 module test directories) — proves all old 2-tuple literals have been upgraded to 3-tuples without manually inspecting every line."

patterns-established:
  - "Pattern: When extending an existing post-build manifest mutation (Phase 6 D-07 dataclass_replace), append new kwargs to the SAME call rather than creating a second mutation. Preserves single-source-of-truth invariant and the idx_kwarg < idx_embed regression guard."
  - "Pattern: When extending mode-gate tuples, do BOTH sites per file (W&B project + results-path gate). Pitfall-3 grep guard catches drift between planning and execution."
  - "Pattern: pyproject.toml run-config keys for orchestrator overrides MUST be declared with default sentinels in [tool.flwr.app.config] so flwr's fuse_dicts validation doesn't reject the --run-config override at CLI time."

requirements-completed: [THS-01, THS-02]

# Metrics
duration: ~10min
completed: 2026-04-29
---

# Phase 7 Plan 02: Server-App + Manifest Wiring for Thesis Tagging Summary

**4 server_apps + 4 pyproject.toml files + 4 source-level wiring tests wired for `thesis_crossdevice_main` mode (Pitfall 3) and `thesis_run_label` / `ablation_dimension` / `ablation_value` manifest fields (Pitfall 2) — orchestrator (Plan 03) can now fire `flwr run` cells whose results.json carries thesis-tagging metadata that Plan 04 aggregator filters on.**

## Performance

- **Duration:** ~10 min total (Task 1 was already committed pre-execution as `fa11d8a`; Task 2 + Rule 1 auto-fix landed in this session as `f5845d8`)
- **Started:** 2026-04-29T15:30:00Z
- **Completed:** 2026-04-29T15:40:00Z
- **Tasks:** 2 (Task 1 = source edits; Task 2 = source-level wiring tests)
- **Files modified:** 12 (4 server_app + 4 pyproject + 4 test_server_integration) + 1 deferred-items.md

## Accomplishments

- **Pitfall 3 closure (mode-tuple gates):** All 4 `server_app.py` files now branch on the 3-tuple `("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")` in BOTH the W&B-project-default expression AND the per-run-dir results-path branch. Each module has exactly 2 occurrences of `"thesis_crossdevice_main"` in source (one per gate site).
- **Pitfall 2 closure (manifest mutation):** All 4 `server_app.py` files now extend the existing Phase-6 D-07 `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` call with three new kwargs read from `context.run_config`: `thesis_run_label`, `ablation_dimension`, `ablation_value`. The mutation executes BEFORE `embed_manifest_in_result(manifest, results_data)` so the embedded `_manifest` dict in `results.json` carries the thesis-tagging fields end-to-end.
- **fuse_dicts contract closure (pyproject keys):** All 4 `pyproject.toml` `[tool.flwr.app.config]` blocks declare safe-default values for the 3 new keys so flwr's `fuse_dicts` validator accepts `--run-config` overrides without "Key not present" errors at CLI time.
- **Source-level wiring tests:** 4 new `test_thesis_label_in_manifest` tests (one per module) assert (a) all 5 kwargs (Phase-6 + Phase-7) coexist in the dataclass_replace call, (b) the mutation precedes `embed_manifest_in_result` (idx_kwarg < idx_embed), (c) BOTH mode-tuple gates contain `"thesis_crossdevice_main"` (count >= 2). Tests run in <0.05s each; do NOT spawn live Flower runs.
- **Existing 2-tuple → 3-tuple upgrades verified:** pfedrec line 321 + personalized line 324 source-string assertions (pre-existing pre-Plan-02 tests pinning the OLD 2-tuple form) were already upgraded to 3-tuple form in the prior `fa11d8a` commit; baseline + adaptive `test_server_integration.py` files do not contain the literal (their `test_client_assertion.py` files use a different 3-element tuple `("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy")` which stays untouched per the plan).
- **Rule 1 auto-fix:** 3 stale `RUN_MANIFEST_SCHEMA_VERSION == 2` / `_manifest.schema_version == 2` literals (introduced in Phase 6 Plans 03-06 when the schema was at v2) bumped to `== 3` so the canonical-artifact tests don't false-positive against Plan 01's v2→v3 schema bump. Same auto-fix class as Plan 01's `test_run_manifest_backward_compat_v1` literal bump.

## Task Commits

Each task committed atomically with `--no-verify` per parallel-executor protocol:

1. **Task 1: server_app.py + pyproject.toml edits** — `fa11d8a` (feat)
   - 8 mode-tuple gate edits (2 per module x 4 modules) — both W&B-project gate + results-path gate
   - 4 manifest-mutation patches (1 per module) — append `thesis_run_label`, `ablation_dimension`, `ablation_value` kwargs to existing Phase-6 dataclass_replace call
   - 12 pyproject.toml key declarations (3 keys per module x 4 modules) — `thesis-run-label = ""`, `ablation-dimension = "none"`, `ablation-value = ""`
   - 2 existing source-string assertion upgrades from 2-tuple to 3-tuple form (pfedrec line 321, personalized line 324)
2. **Task 2: source-level wiring tests + Rule 1 auto-fix** — `f5845d8` (test)
   - 4 new `test_thesis_label_in_manifest` test functions (one per module) — Pitfall 2 + Pitfall 3 source-level wiring assertions
   - 3 stale schema_version=2 literal bumps to 3 (Rule 1 auto-fix) — baseline `test_canonical_artifact_carries_best_and_last_blocks` line 499 + personalized line 399 + pfedrec line 396
   - deferred-items.md entry for the pre-existing slow `test_selected_partitions_byte_identical_across_subprocess_reruns` failure (out of Plan 02 scope)

**Plan metadata commit:** Will be created with this SUMMARY.md + STATE.md + ROADMAP.md updates.

_Note: Task 1 = `feat`, Task 2 = `test`. Both used `--no-verify` per `<parallel_execution>` protocol._

## Files Created/Modified

### Source

#### server_app.py mode-tuple gates (post-edit line numbers)

| Module | W&B project gate (site #1) | Results-path gate (site #2) |
| --- | --- | --- |
| baseline | line 295 | line 909 |
| personalized | line 382 | line 994 |
| adaptive | line 490 | line 1298 |
| pfedrec | line 514 | line 1165 |

#### server_app.py manifest mutation kwargs (post-edit line numbers)

| Module | thesis_run_label kwarg in dataclass_replace |
| --- | --- |
| baseline | line 895 |
| personalized | line 980 |
| adaptive | line 1275 |
| pfedrec | line 1118 |

(Each kwarg site is followed immediately by `ablation_dimension=...` and `ablation_value=...` on the next two lines, then closed off, then `embed_manifest_in_result(...)` is called several lines below.)

#### pyproject.toml thesis keys (post-edit line numbers)

| Module | thesis-run-label / ablation-dimension / ablation-value lines |
| --- | --- |
| baseline | 102 / 103 / 104 |
| personalized | 105 / 106 / 107 |
| adaptive | 210 / 211 / 212 |
| pfedrec | 58 / 59 / 60 |

### Tests

| Module | test_thesis_label_in_manifest location | _server_app_src helper status |
| --- | --- | --- |
| baseline | new at end of `tests/test_server_integration.py` | NEW helper added (was not present) |
| personalized | new at end of `tests/test_server_integration.py` | reused existing `_server_app_src` helper from earlier in file |
| adaptive | new at end of `tests/test_server_integration.py` | NEW helper added (was not present) |
| pfedrec | new at end of `tests/test_server_integration.py` | reused existing `_pfedrec_server_app_src` helper from earlier in file |

#### Existing source-string assertion 2-tuple → 3-tuple upgrades

| File | Before | After |
| --- | --- | --- |
| federated-pfedrec/tests/test_server_integration.py:321 | `assert 'if mode in ("benchmark_cross_device", "paper_compat_pfedrec")' in src` | `assert 'if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")' in src` |
| federated-personalized-cf/tests/test_server_integration.py:324 | `assert 'if mode in ("benchmark_cross_device", "paper_compat_pfedrec")' in src` | `assert 'if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")' in src` |

(Baseline + adaptive `test_server_integration.py` files do not contain the 2-tuple literal — verified by grep at planning + execution time. Pitfall 3 closure verified via `grep -rln '"benchmark_cross_device", "paper_compat_pfedrec"[^,]' federated-*-cf/tests/test_server_integration.py` returning 0 matches.)

### Planning

- `.planning/phases/07-thesis-evaluation-run/deferred-items.md` — Logged out-of-scope discovery: `test_selected_partitions_byte_identical_across_subprocess_reruns` in baseline fails under live `flwr run` (same family as Plan 07-01 deferred adaptive + personalized failures; unrelated to Plan 07-02 scope; no Plan-02-touched symbol referenced).

## Test Suite Counts

| Suite | Pre-Plan-02 | Post-Plan-02 | Delta |
| --- | --- | --- | --- |
| baseline tests/ (-m "not slow") | 26 passed | 27 passed | +1 (new test_thesis_label_in_manifest) |
| personalized tests/ | 38 passed | 39 passed | +1 (new test_thesis_label_in_manifest) |
| adaptive tests/ | 73 passed | 74 passed | +1 (new test_thesis_label_in_manifest) |
| pfedrec tests/ | 41 passed | 42 passed | +1 (new test_thesis_label_in_manifest) |
| **Total fast tests across 4 modules** | 178 passed | **182 passed** | **+4 net** (4 new tests) |

## Decisions Made

1. **Plan A (extend mode tuples in server_app.py) over Plan B (orchestrator-side wandb-project override):** Per RESEARCH.md primary recommendation. 8 line edits + matches existing Phase-6 mode-gate pattern + keeps the orchestrator script (Plan 03) independent of per-module server_app internals. Plan B would have required the orchestrator to know about each module's per-run-dir gating logic — an unnecessary coupling.
2. **Append thesis kwargs to the existing Phase-6 dataclass_replace call (not a second replace):** Single mutation point preserves the Phase-6 D-07 invariant and gives the test a stable `idx_kwarg < idx_embed` ordering check. A second replace call would have introduced the possibility of one mutation landing without the other.
3. **Static source-string wiring test (not a live Flower run):** 30+ minute live runs are incompatible with the fast suite. The Pitfall 2 (ordering) and Pitfall 3 (mode-tuple) invariants are statically detectable from server_app.py source. Plan 05's smoke-run gate is the canonical end-to-end verification.
4. **`_server_app_src` helper added in baseline + adaptive but not personalized + pfedrec:** Personalized and pfedrec test files already had the helper from earlier Phase 6 work; adding a duplicate would have raised a `NameError` collision or been bikeshed. Baseline + adaptive needed the helper added (was not previously present).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stale `RUN_MANIFEST_SCHEMA_VERSION == 2` literal in 3 test files**
- **Found during:** Task 2 verification (full per-module test suite)
- **Issue:** Phase 7 Plan 01 D-22 bumped the schema version constant 2 → 3, but 3 existing tests in `test_canonical_artifact_carries_best_and_last_blocks` (baseline + personalized + pfedrec) still asserted the old `== 2` literal. These tests fail under Plan 01's bumped constant — exactly the same regression class Plan 01 SUMMARY described and auto-fixed for `test_run_manifest_backward_compat_v1`.
- **Fix:** Bumped 3 stale literals to `== 3` while keeping the test's intent (the canonical artifact carries the current schema version). Added Phase-7-D-22 + Phase-6-D-07 inline comments.
- **Files modified:** `federated-baseline-cf/tests/test_server_integration.py:499`, `federated-personalized-cf/tests/test_server_integration.py:399-400`, `federated-pfedrec/tests/test_server_integration.py:396-397`
- **Verification:** All 3 previously-failing tests PASS after the bump (re-ran each individually).
- **Committed in:** `f5845d8` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary correction caused by Plan 01's schema bump — same auto-fix class Plan 01 already applied. No scope creep. The fix preserves the test's intent (canonical artifact carries the live schema version).

## Issues Encountered

- **Pre-existing slow subprocess determinism failure in baseline:** `test_selected_partitions_byte_identical_across_subprocess_reruns` fails with `AssertionError: No result JSON found after launcher run_id=...` under full `pytest tests/ -ra` (without `-m "not slow"` filter). This is the same family as the Plan 07-01 deferred adaptive + personalized failures — the slow test invokes `scripts/run.py` end-to-end, which forks `flwr run`, which fails to materialize a `results.json` in this environment. Confirmed unrelated to Plan 07-02 scope by grep — the failing test does NOT reference `thesis_crossdevice_main`, `thesis_run_label`, `ablation_dimension`, or any Plan-02-touched symbol. Logged in `.planning/phases/07-thesis-evaluation-run/deferred-items.md` for Plan 02 follow-up. The fast suite (182 tests across 4 modules) is fully GREEN.

## User Setup Required

None — no external service configuration required.

## Self-Check: PASSED

Verified before STATE update:

- All 4 server_app.py files have count `"thesis_crossdevice_main"` >= 2 (verified: each = 2 — both gates) (FOUND)
- All 4 server_app.py files have count `thesis_run_label=str(context.run_config.get("thesis-run-label"` exactly == 1 (verified: each = 1) (FOUND)
- All 4 pyproject.toml files declare `thesis-run-label = ""`, `ablation-dimension = "none"`, `ablation-value = ""` (verified by `grep -E '^(thesis-run-label|ablation-dimension|ablation-value)'`) (FOUND)
- 4 PASSED: `test_thesis_label_in_manifest` in all 4 modules (verified by per-module pytest invocation) (FOUND)
- ZERO matches: `grep -rln '"benchmark_cross_device", "paper_compat_pfedrec"[^,]' federated-*-cf/tests/test_server_integration.py` (Pitfall 3 closure proven) (FOUND)
- All 4 module fast test suites GREEN under `-m "not slow"`: baseline 27, personalized 39, adaptive 74, pfedrec 42 = 182 passed (FOUND)
- 3 previously-failing `test_canonical_artifact_carries_best_and_last_blocks` tests now PASS after Rule 1 auto-fix (FOUND)
- Commit `fa11d8a` (feat: server_app + pyproject + 3-tuple-assertion edits) — present in `git log` (FOUND)
- Commit `f5845d8` (test: 4 new tests + Rule 1 auto-fix) — present in `git log` (FOUND)

## Next Phase Readiness

- **Plan 03 ready:** Orchestrator `scripts/thesis/run_thesis_sweep.py` can now invoke `python scripts/run.py {module} thesis_crossdevice_main --run-config "thesis-run-label=main run-seed=42 ..."` with confidence that:
  - The `mode in (...)` branches in every server_app route to the cross-device W&B project AND per-run-dir results path (Pitfall 3 closure).
  - The `--run-config` thesis kwargs are accepted by flwr's `fuse_dicts` validator (pyproject defaults declared).
  - The resulting `results.json`'s `_manifest` block carries `thesis_run_label`, `ablation_dimension`, `ablation_value` with non-default values (Pitfall 2 closure — manifest mutation site lands BEFORE embed_manifest_in_result).
- **Plan 04 ready:** Aggregator can filter `results.json` files by `_manifest.thesis_run_label != ""` and group by `_manifest.ablation_dimension` / `_manifest.ablation_value`. The D-20 hard-fail-on-missing-cells check has the provenance metadata it needs.
- **Plan 05 ready:** Manual runbook can document the canonical incantation `python scripts/run.py adaptive thesis_crossdevice_main --run-config "thesis-run-label=main run-seed=42 num-server-rounds=2 fraction-train=0.001 wandb-enabled=false"` as the smoke-run gate. The smoke run will exercise the Pitfall-2 + Pitfall-3 wiring end-to-end and produce the very `results.json` that the slow subprocess determinism tests (currently deferred) require.

No blockers. The pre-existing slow subprocess failure (`deferred-items.md` Plan 07-02 entry) is independent of Plan 07-02 scope and does not gate Plans 03-05.

---
*Phase: 07-thesis-evaluation-run*
*Plan: 02*
*Completed: 2026-04-29*
