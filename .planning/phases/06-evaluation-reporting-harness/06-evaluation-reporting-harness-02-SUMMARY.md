---
phase: 06-evaluation-reporting-harness
plan: 02
subsystem: foundation
tags: [manifest, dataclass, schema-version, evl-01, evl-06, d-04]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: RunManifest dataclass + write_manifest_sibling/embed_manifest_in_result helpers (FND-07/IMP-2/D-15)
provides:
  - RUN_MANIFEST_SCHEMA_VERSION = 2 (bumped from 1)
  - RunManifest.final_eval_round_index (sentinel 0 = no extra eval; >=1 = post-restore broadcast round index — EVL-01)
  - RunManifest.metrics (typed mirror of results_data["final_metrics"] block — EVL-06)
  - write_manifest_sibling(sibling_name=...) optional kwarg (D-04 enabler for clean per-run-dir filenames)
affects: [06-evaluation-reporting-harness-04, 06-evaluation-reporting-harness-05, 06-evaluation-reporting-harness-06, 06-evaluation-reporting-harness-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Schema bump with safe-default-only field additions — Pitfall 3 (existing v1 fixtures construct without TypeError under v2)"
    - "Optional kwarg with default-None preserving legacy behavior — kwarg surface extends without breaking callers"
    - "Post-build mutation via dataclasses.replace — Wave-2/3 server_app plans inject final_eval_round_index + metrics after build_run_manifest returns"

key-files:
  created: []
  modified:
    - "scripts/foundation/fedrec_foundation/manifest.py — schema v2, new fields, sibling_name kwarg"
    - "scripts/foundation/tests/test_manifest.py — 5 NEW tests (schema_version=2, v1 backward-compat, post-build mutation, default sibling filename, sibling_name override)"

key-decisions:
  - "Schema field additions placed at the END of the RunManifest dataclass with safe defaults (int=0, field(default_factory=dict)) so v1 callers and test fixtures construct unchanged — Pitfall 3 closure"
  - "write_manifest_sibling sibling_name=Optional[str]=None: when None, legacy filename <run_id>-manifest.json preserved; Wave-2/3 callers pass 'manifest.json' for D-04 clean per-run-dir layout"
  - "Constant comment cites Phase 6 specifically (`# Phase 6: adds final_eval_round_index + metrics fields`) so future schema bumps have a paper trail"
  - "Kept duck-typed mode_profile: Any in build_run_manifest unchanged — Phase 1 Plan 04 contract preserved"

patterns-established:
  - "Pitfall-3 safe-default schema bump: any future RunManifest field addition that lands BEFORE all consumers are migrated MUST carry a documented sentinel (or `field(default_factory=...)` for mutables) — this lets Wave-N plans land surgically without forcing simultaneous Wave-(N+1) consumer changes."
  - "D-04 clean filename via opt-in kwarg: write_manifest_sibling never breaks legacy <run_id>-manifest.json behavior; Phase-6 callers explicitly opt into 'manifest.json' per the new per-run-dir layout."

requirements-completed: [EVL-01, EVL-06]

# Metrics
duration: ~3min
completed: 2026-04-29
---

# Phase 6 Plan 02: RunManifest Schema v2 (EVL-01 + EVL-06 enabler) Summary

**Bumped RunManifest schema 1->2 with two safe-default fields (`final_eval_round_index`, `metrics`) and one optional `sibling_name` kwarg on `write_manifest_sibling`, unblocking Wave-2/3 server_app plans to attach post-restore broadcast indices and best/last metrics blocks via `dataclasses.replace`.**

## Performance

- **Duration:** ~3 min (Wave-1 parallel executor agent)
- **Started:** 2026-04-29T06:42:58Z
- **Completed:** 2026-04-29T06:45:36Z
- **Tasks:** 1 (TDD: RED + GREEN, no REFACTOR needed)
- **Files modified:** 2

## Accomplishments

- `RUN_MANIFEST_SCHEMA_VERSION` bumped from 1 to 2 with phase-tagged comment.
- `RunManifest.final_eval_round_index: int = 0` field added — sentinel `0` = no extra eval ran (mode is `last_round`, or `best_round` with no best-round recorded); values `>= 1` mean a fresh post-restore evaluation populated `metrics["best"]`. Closes **EVL-01**.
- `RunManifest.metrics: Dict[str, Any] = field(default_factory=dict)` field added — typed mirror of `results_data["final_metrics"]` carrying `best`, `last`, `best_round`, `last_round`, `final_eval_round_index` keys with per-group HR/NDCG sub-dicts. Closes **EVL-06**.
- `write_manifest_sibling(..., sibling_name: Optional[str] = None)` extended; default behavior preserved (`<run_id>-manifest.json`); Wave-2/3 callers pass `sibling_name="manifest.json"` for **D-04** clean per-run-dir filenames.
- Pitfall 3 closure: existing v1 test fixture (`_StubProfile` + `_build()`) constructs unchanged under v2 because both new fields are defaulted; the `test_run_manifest_backward_compat_v1` test pins this explicitly.

## Task Commits

Each TDD step committed atomically:

1. **Task 1 RED: 5 failing tests** — `d39af08` (test)
2. **Task 1 GREEN: schema v2 + sibling_name kwarg implementation** — `5a8098f` (feat)

No refactor commit needed — implementation landed clean on first pass; surgical edits to imports, schema constant, dataclass body, and one function signature only.

## Files Created/Modified

- `scripts/foundation/fedrec_foundation/manifest.py` — Imports extended (`field`, `Optional`); `RUN_MANIFEST_SCHEMA_VERSION` bumped 1->2 with Phase-6 comment; `RunManifest` dataclass extended with two defaulted fields (each fully docstring'd in NumPy style); `write_manifest_sibling` signature extended with `sibling_name: Optional[str] = None`, body honors the override, docstring documents the new kwarg. (+35/-6 lines)
- `scripts/foundation/tests/test_manifest.py` — 5 NEW tests appended at the file end with a banner block: `test_run_manifest_schema_version_2`, `test_run_manifest_backward_compat_v1`, `test_run_manifest_carries_final_eval_round_index`, `test_write_manifest_sibling_default_filename`, `test_write_manifest_sibling_custom_name`. Imports for `dataclass_replace` and `Dict, Any` typing kept inside the new section so the existing v1-era imports remain untouched. (+193/-0 lines)

## Decisions Made

- **Field placement at the END of RunManifest body** (after `git_commit: str`): required for safe defaults, since dataclass fields without defaults cannot follow fields with defaults. This maintains the historical 23-field v1 ordering byte-for-byte and adds the two new fields cleanly at the tail.
- **Comment-tagged constant bump** (`# Phase 6: adds final_eval_round_index + metrics fields` next to the `RUN_MANIFEST_SCHEMA_VERSION: int = 2` line): future plans bumping to v3 will see the v2 cause documented inline, no archaeology needed.
- **Kept all 6 existing tests untouched** — backward-compatibility is asserted by the existing `test_all_fields_populated` test plus the new `test_run_manifest_backward_compat_v1` test, which together prove (a) the v1 23-field surface still constructs and (b) the new defaults are the documented sentinels.
- **Did NOT touch `build_run_manifest`** — duck-typed mode_profile contract from Phase 1 Plan 04 preserved verbatim; Wave-2/3 plans extend the call site via post-build `dataclasses.replace(manifest, final_eval_round_index=N, metrics=results_data['final_metrics'])`.

## Deviations from Plan

None — plan executed exactly as written. The plan's 5-test count, surgical-edit list (6 numbered items in `<action>`), and acceptance criteria (8 grep / python checks + 2 pytest invocations) all landed verbatim. The plan's example test bodies were ported with minor cosmetic adjustments (using `Path` typing-hint on `tmp_path` parameter to match existing test_manifest.py convention, and inlining the imports inside the new banner block rather than at the file top to keep the existing import chunk untouched).

## Issues Encountered

None. The plan's test design (using direct `RunManifest(...)` construction in `test_run_manifest_backward_compat_v1` instead of going through `build_run_manifest`) sidestepped any potential ordering issues with the duck-typed `mode_profile` argument and was the right pattern for surfacing the Pitfall-3 defaults check.

## Verification

- **Manifest tests:** `pytest tests/test_manifest.py -x -v` — **11 passed** (6 existing + 5 new) in 0.70s.
- **Full foundation suite (-m "not slow"):** `pytest tests/ -q -m "not slow"` — **100 passed, 3 deselected** (slow markers) in 7.82s. Zero regressions across the 16 foundation test files.
- **Acceptance criteria grep checks** — all 6 grep counts match the plan's expected values (1 / 0 / 1 / 1 / 1 / 1).
- **Python API smoke checks:**
  - `from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION; assert RUN_MANIFEST_SCHEMA_VERSION == 2` — passes.
  - `from fedrec_foundation.manifest import RunManifest; ... 'final_eval_round_index' in fnames; 'metrics' in fnames` — passes.

## User Setup Required

None — pure foundation-package change; consumers still receive an editable install via `pip install -e scripts/foundation/`.

## Cross-Phase Contract for Wave 2/3

Wave-2/3 server_app plans (Plans 04/05/06/07) MUST extend their existing manifest call site as follows. Existing call sites continue to work today because both new fields default and the new kwarg is optional — these are additive extensions, not mandatory migrations.

```python
# After the main FL loop and post-restore extra-eval-round (D-06):
manifest = build_run_manifest(...)  # unchanged
manifest = dataclasses.replace(
    manifest,
    final_eval_round_index=extra_eval_round_idx,  # 0 if checkpoint_rule == "last_round"
    metrics=results_data["final_metrics"],         # carries best/last + per-group sub-dicts
)
embed_manifest_in_result(manifest, results_data)  # unchanged
write_manifest_sibling(
    manifest,
    results_path,
    sibling_name="manifest.json",  # D-04 clean filename
)
```

Cross-silo legacy callers (if any remain after Phase 6) omit the `sibling_name` kwarg to keep the historical `<run_id>-manifest.json` shape.

## Next Phase Readiness

- **Wave-2/3 unblocked** — every server_app.py plan in Wave 2 (Plan 04 baseline, Plan 05 personalized) and Wave 3 (Plan 06 adaptive, Plan 07 pfedrec) can now import and use the schema-v2 fields without any further manifest-side work.
- **No blockers.** Plan 01 (`paths.py` repo-root resolver) committed in parallel under the same Wave-1 hook-skip protocol; both Wave-1 deliverables are in.

---
*Phase: 06-evaluation-reporting-harness*
*Plan: 02 — RunManifest schema v2 (EVL-01, EVL-06, D-04 enabler)*
*Completed: 2026-04-29*

## Self-Check: PASSED

- `scripts/foundation/fedrec_foundation/manifest.py` — exists, modified.
- `scripts/foundation/tests/test_manifest.py` — exists, modified.
- Commits: `d39af08` (RED test), `5a8098f` (GREEN feat) — both present in `git log`.
- 5 new tests in test_manifest.py: confirmed PASSED (11/11 manifest tests green).
- Full foundation suite (not slow): 100 passed, 3 deselected — zero regressions.
