---
phase: 06-evaluation-reporting-harness
plan: 01
subsystem: foundation
tags: [paths, results-dir, repo-root, fedrec-foundation, evl-04, d-01, d-02, pitfall-6]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: "scripts/foundation/fedrec_foundation/paths.py::repo_root walk-up; tests/conftest.py fixtures"
provides:
  - "module_run_results_dir(module: str, run_id: str) -> Path helper at fedrec_foundation.paths"
  - "_ALLOWED_MODULES = frozenset({'baseline','personalized','adaptive','pfedrec'}) typo-guard whitelist"
  - "13 GREEN test items pinning D-01 layout, D-02 anchoring under chdir, Pitfall-6 typo enforcement"
affects:
  - "06-evaluation-reporting-harness Plans 03/04/05/06 (per-module server_app cross-device migration)"
  - "Phase 7 thesis evaluation run (consumes the canonical <module>/<run_id>/ artifact layout)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Repo-root-anchored per-run directory: <repo>/results/federated/<module>/<run_id>/ (D-01 + D-02)"
    - "Closed-enum whitelist as a frozenset constant + ValueError on miss (Pitfall 6 typo guard)"
    - "mkdir(parents=True, exist_ok=True) inside the resolver so callers never see ENOENT"

key-files:
  created:
    - "scripts/foundation/tests/test_paths.py - 3 test functions, 13 parametrized items"
  modified:
    - "scripts/foundation/fedrec_foundation/paths.py - +49 lines: _ALLOWED_MODULES + module_run_results_dir helper appended after ml1m_dir"

key-decisions:
  - "EVL-04 D-02 closed at the foundation layer: every Wave-2/3 server_app imports module_run_results_dir(module, run_id) instead of computing module-relative `Path('../results/federated/<module>')`. Single source of truth eliminates the four ad-hoc sites and resolves the folded phase2-baseline-determinism-path-bug.md todo."
  - "_ALLOWED_MODULES whitelist content is pinned to the literal four strings in manifest.py:80 RunManifest.module docstring. Any future module name change requires updating both files in lockstep; the test suite's `repr(bad_name) in msg` assertion catches drift."
  - "No env-var override on module_run_results_dir (unlike data_derived's FEDREC_FOUNDATION_DATA_DIR) — results paths are runtime-driven (per-run, per-module) not config-driven. Override surface intentionally omitted."
  - "Helper creates the dir eagerly (mkdir parents=True, exist_ok=True) so Wave-2/3 callers can immediately write results.json + manifest.json without staging a separate ensure-parent call."

patterns-established:
  - "Closed-enum-on-string-input pattern: validate `module not in _ALLOWED_MODULES` before any I/O; ValueError message echoes both `repr(name)` AND `'Expected one of <sorted_whitelist>'` so debug logs name the typo and the valid set in one line."
  - "Pre-flight mkdir-in-resolver pattern: any path-resolver returning a write target should mkdir(parents=True, exist_ok=True) before returning, so callers can drop ensure-parent boilerplate."

requirements-completed: [EVL-04]

# Metrics
duration: 2min
completed: 2026-04-29
---

# Phase 6 Plan 01: Foundation Helper module_run_results_dir Summary

**Single-helper Wave-1 foundation primitive: `module_run_results_dir(module, run_id)` resolves repo-root-anchored per-run directories with closed-enum typo guard, unblocking all four Wave-2/3 server_app cross-device path migrations and resolving the folded phase-2 baseline determinism path bug.**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-29T06:42:48Z
- **Completed:** 2026-04-29T06:45:05Z
- **Tasks:** 1 (TDD: RED + GREEN, no refactor needed)
- **Files modified:** 2 (1 created, 1 extended)

## Accomplishments

- New `module_run_results_dir(module: str, run_id: str) -> Path` helper appended to `scripts/foundation/fedrec_foundation/paths.py` mirroring the shape of the existing `data_derived()` / `ml1m_dir()` helpers.
- New `_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})` constant matching the literal value of `manifest.py:80` `RunManifest.module` docstring.
- New `scripts/foundation/tests/test_paths.py` file with 3 parametrized test functions (13 total test items) pinning D-01 layout, D-02 repo-root anchoring under `monkeypatch.chdir(tmp_path)`, and Pitfall-6 whitelist enforcement against 8 typo variants.
- Helper creates the per-run directory eagerly (`parents=True, exist_ok=True`) so Wave-2/3 callers writing `results.json` + `manifest.json` never see `FileNotFoundError`.

## Task Commits

1. **Task 1 (RED): test_paths.py 3 failing tests** - `e76ea21` (test) — 64-line test file collecting 13 items, fails with `ImportError: cannot import name '_ALLOWED_MODULES'` against unmodified paths.py.
2. **Task 1 (GREEN): module_run_results_dir helper** - `a464a78` (feat) — 49-line addition to paths.py: `_ALLOWED_MODULES` constant + helper function + NumPy-style docstring citing D-01, D-02, Pitfall 6. All 13 test items pass.

_Note: TDD task; RED + GREEN committed separately. No refactor commit needed — implementation matched the plan template line-for-line._

## Files Created/Modified

- `scripts/foundation/tests/test_paths.py` (created, 64 lines) — 3 parametrized test functions covering D-01 layout, D-02 repo-root anchoring, and Pitfall-6 whitelist enforcement. Uses `monkeypatch.chdir(tmp_path)` for cwd-robustness assertion and `shutil.rmtree` cleanup so tests don't pollute the real repo `results/` tree.
- `scripts/foundation/fedrec_foundation/paths.py` (modified, +49 lines) — appended `_ALLOWED_MODULES` frozenset + `module_run_results_dir` helper after the existing `ml1m_dir()` function. Helper raises `ValueError` on unknown module name with message `"Unknown module {module!r}. Expected one of {sorted(_ALLOWED_MODULES)}."`. Other helpers (`repo_root`, `data_derived`, `ml1m_dir`) untouched (D-18 surgical scope).

## Decisions Made

- **Pin which test pins which decision:** The 3-test layout was chosen so each decision has exactly one regression guard. Test 1 pins D-02 (repo-root anchoring under chdir). Test 2 pins D-01 (per-module per-run layout, parametrized over the 4 allowed modules). Test 3 pins Pitfall 6 (whitelist enforcement, parametrized over 8 typo variants including case-mismatch like `"Baseline"`, missing-letter like `"basline"`, abbreviation like `"adapt"`, semantic-but-wrong like `"thesis"`, and the empty string).
- **No env-var override:** Unlike `data_derived()` which honors `FEDREC_FOUNDATION_DATA_DIR`, `module_run_results_dir()` exposes no override surface. Results paths are runtime-driven (per-run, per-module) not config-driven, so an override would be ill-defined for which module's results to redirect.
- **mkdir-in-resolver:** The helper calls `out.mkdir(parents=True, exist_ok=True)` before returning. This intentionally couples directory creation to path resolution so Wave-2/3 callers can immediately write `results.json` + `manifest.json` without a separate ensure-parent call. Tested via `assert path.is_dir()` post-call.
- **Path absolute-and-resolved:** Returned via `repo_root() / "results" / "federated" / module / run_id`. `repo_root()` resolves to an absolute path (it walks up from `__file__`), so the result is always absolute and cwd-independent. Test 1 explicitly asserts `path.is_absolute()` under `monkeypatch.chdir(tmp_path)` to nail this property.

## Deviations from Plan

None - plan executed exactly as written.

The plan's `<action>` block contained the helper code and test file as load-bearing literal text. Both were inserted line-for-line (no shape changes, no signature drift). The only stylistic adjustment was replacing the docstring's en-dashes (`—`) with ASCII double-hyphens (`--`) inside the docstring text, since the surrounding `paths.py` uses ASCII throughout — but this is invisible to all callers and the test suite.

## Issues Encountered

None during Plan 01 execution.

**Cross-cutting note (out of scope):** The full foundation suite shows 4 failing tests in `tests/test_manifest.py` (`test_run_manifest_schema_version_2`, `test_run_manifest_backward_compat_v1`, `test_run_manifest_carries_final_eval_round_index`, `test_write_manifest_sibling_custom_name`). These are RED-step tests committed by the parallel Wave-1 sibling agent (Plan 02, commit `d39af08`) and will land GREEN when Plan 02's `manifest.py` schema-v2 implementation lands. They are out of Plan 01's scope per D-18 surgical-scope discipline (Plan 02 owns `manifest.py`).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Wave 2/3 ready:** Plans 03/04/05/06 (per-module server_app cross-device migration) can now `from fedrec_foundation.paths import module_run_results_dir` and call `module_run_results_dir(module="<module>", run_id=run_id)` instead of computing `Path("../results/federated/<module>")`. The repo-root anchoring resolves the folded `phase2-baseline-determinism-path-bug.md` todo at the same time the migration lands.
- **Cross-phase contract:** The `_ALLOWED_MODULES` whitelist content (`baseline`, `personalized`, `adaptive`, `pfedrec`) is now mechanically enforced at every per-run write site. A future module rename would surface as a `ValueError` at the first server_app invocation, not as a silent results-tree drift.
- **D-18 surgical scope held:** `git diff --stat` over Plan 01's two commits shows ONLY `scripts/foundation/fedrec_foundation/paths.py` (+49 lines) and `scripts/foundation/tests/test_paths.py` (+64 lines created). Nothing else touched.

## Self-Check: PASSED

- FOUND: scripts/foundation/fedrec_foundation/paths.py (`module_run_results_dir` + `_ALLOWED_MODULES` present)
- FOUND: scripts/foundation/tests/test_paths.py (13 GREEN test items)
- FOUND: .planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-01-SUMMARY.md
- FOUND: commit e76ea21 (RED — failing tests)
- FOUND: commit a464a78 (GREEN — helper implementation)

---
*Phase: 06-evaluation-reporting-harness*
*Completed: 2026-04-29*
