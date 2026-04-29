---
phase: 07-thesis-evaluation-run
plan: 01
subsystem: foundation
tags: [foundation, mode-profile, manifest-schema, atomic-write, launcher, thesis]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: ModeProfile dataclass + _REGISTRY, RunManifest schema + atomic_write_json + dataclass_replace pattern, scripts/run.py launcher with MODE_NUM_SUPERNODES dict
  - phase: 06-evaluation-reporting-harness
    provides: schema v2 with safe-default field-extension precedent (final_eval_round_index + metrics), Pitfall-3 backward-compat invariant
provides:
  - "_THESIS_CROSSDEVICE_MAIN ModeProfile registered in _REGISTRY (byte-for-byte clone of _BENCHMARK_CROSS_DEVICE)"
  - "RunManifest schema v3 with 3 thesis-tagging fields (thesis_run_label, ablation_dimension, ablation_value)"
  - "atomic_write_text() function for markdown/CSV writes by Plan 04 aggregator"
  - "scripts/run.py MODE_NUM_SUPERNODES dict with thesis_crossdevice_main: 6040 entry (argparse choices auto-extended)"
  - "9 new pytest functions covering all 4 invariants (1 mode + 1 mode-registration extension + 3 manifest + 3 atomic + 1 launcher)"
affects: [07-thesis-evaluation-run-02, 07-thesis-evaluation-run-03, 07-thesis-evaluation-run-04, 07-thesis-evaluation-run-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ModeProfile cloning via dataclasses.replace (mode-name as the discriminant)"
    - "Manifest schema bump with safe-default fields (preserves v1/v2 fixture construction)"
    - "atomic_write_text companion to atomic_write_json (UTF-8 + tempfile + os.replace)"

key-files:
  created:
    - "scripts/foundation/tests/test_atomic.py"
    - ".planning/phases/07-thesis-evaluation-run/deferred-items.md"
  modified:
    - "scripts/foundation/fedrec_foundation/mode.py"
    - "scripts/foundation/fedrec_foundation/manifest.py"
    - "scripts/foundation/fedrec_foundation/atomic.py"
    - "scripts/run.py"
    - "scripts/foundation/tests/test_mode.py"
    - "scripts/foundation/tests/test_manifest.py"
    - "scripts/foundation/tests/test_launcher.py"

key-decisions:
  - "_THESIS_CROSSDEVICE_MAIN clones _BENCHMARK_CROSS_DEVICE byte-for-byte (only mode name differs); D-04 enforces this — the mode name is the provenance discriminant the aggregator filters on, not a structural variation."
  - "checkpoint_rule='best_round' inherited from _BENCHMARK_CROSS_DEVICE; functionally equivalent to pyproject 'best_round_restore' per Phase 2 Plan 04 dual-spelling tolerance (verified: each of the 4 server_app.py files contains >=19 occurrences of best_round/best_round_restore in the checkpoint-rule branch)."
  - "RunManifest schema v3 fields appended after v2 fields with safe defaults (thesis_run_label='', ablation_dimension='none', ablation_value=''); build_run_manifest UNTOUCHED — server_app populates via dataclasses.replace post-build mutation, mirroring Phase 6 D-07."
  - "Backward-compat test test_run_manifest_backward_compat_v1 bumped to schema_version=3 because the v3 fields' safe defaults make the v1-fixture-shape construction valid under v3."

patterns-established:
  - "Pattern: Mode profile cloning — D-04 says mode-name IS the provenance tag; new modes that differ only in name+downstream-routing should clone via dataclasses.replace(parent, mode='new_name') rather than re-listing all 16 fields. Test enforces byte-identity via _replace assertion."
  - "Pattern: Manifest schema bump — append fields with safe defaults BELOW existing default-bearing fields; never touch build_run_manifest; populate via dataclasses.replace at the call site. Pre-existing v1/v2 fixtures continue to construct without TypeError."
  - "Pattern: atomic_write_text — UTF-8 file writes use the same tempfile + os.replace + cleanup-on-exception pattern as atomic_write_json; suffix='.txt' for tempfile differentiation; encoding='utf-8' explicitly set on fdopen."

requirements-completed: [THS-01]

# Metrics
duration: 25min
completed: 2026-04-29
---

# Phase 7 Plan 01: Foundation Extensions for Thesis Evaluation Run Summary

**Cross-device thesis-evaluation foundation: `_THESIS_CROSSDEVICE_MAIN` ModeProfile (byte-clone of `_BENCHMARK_CROSS_DEVICE`), RunManifest schema v3 with 3 thesis-tagging fields, `atomic_write_text` companion, and launcher mode dict — all 4 building blocks Plans 02-05 depend on.**

## Performance

- **Duration:** ~25 min (Task 1 source edits 6 min + Task 2 tests 12 min + verification 4 min + summary 3 min)
- **Started:** 2026-04-29T14:35:00Z
- **Completed:** 2026-04-29T15:00:00Z
- **Tasks:** 2
- **Files modified:** 8 (4 source + 4 test) plus 2 created (test_atomic.py + deferred-items.md)

## Accomplishments

- **Foundation Mode Profile Extension** — `_THESIS_CROSSDEVICE_MAIN` registered in `fedrec_foundation.mode._REGISTRY` between `benchmark_cross_device` and `paper_compat_pfedrec`. `MODE_NAMES` now reports 4 modes; `resolve_mode_defaults('thesis_crossdevice_main')` returns the byte-for-byte clone of `_BENCHMARK_CROSS_DEVICE` with only the `mode` string differing.
- **Manifest Schema v3** — `RUN_MANIFEST_SCHEMA_VERSION` bumped 2→3; three new thesis-tagging fields appended (`thesis_run_label: str = ""`, `ablation_dimension: str = "none"`, `ablation_value: str = ""`). All carry safe defaults so v1/v2 fixtures continue to construct without TypeError; `build_run_manifest` UNTOUCHED.
- **`atomic_write_text` API** — Companion to `atomic_write_json`, mirrors the tempfile + `os.replace` pattern with `suffix=".txt"` + `encoding="utf-8"`. Phase 7 Plan 04 aggregator will use this for `main_comparison.md` / `main_comparison.csv` writes.
- **Launcher Extension** — `scripts/run.py` `MODE_NUM_SUPERNODES` dict gained `thesis_crossdevice_main: 6040` entry. `argparse choices=sorted(MODE_NUM_SUPERNODES.keys())` automatically picked up the new mode (Pitfall 5 closure); `python scripts/run.py adaptive thesis_crossdevice_main --dry-run` exits 0 emitting `mode="thesis_crossdevice_main"`.
- **Test Coverage** — 9 new pytest functions added across 4 test files covering all 4 invariants. Foundation suite went from 100 → 107 fast tests (107 passed, 4 slow-deselected, 0 fast-suite failures).
- **Warning 4 closure** — Verified all 4 `server_app.py` files contain `best_round|best_round_restore` matches (≥19 each: baseline 19, personalized 22, adaptive 21, pfedrec 24); `_THESIS_CROSSDEVICE_MAIN.checkpoint_rule="best_round"` is functionally equivalent to pyproject `"best_round_restore"` downstream.

## Task Commits

Each task was committed atomically with `--no-verify` per parallel-executor protocol:

1. **Task 1: Foundation source edits** — `3c3116f` (feat)
   - mode.py: `_THESIS_CROSSDEVICE_MAIN` ModeProfile + `_REGISTRY` extension
   - manifest.py: schema v2→v3 bump + 3 thesis-tagging fields with safe defaults
   - atomic.py: new `atomic_write_text(path, content)` function
   - scripts/run.py: `MODE_NUM_SUPERNODES` dict gained `thesis_crossdevice_main: 6040`
2. **Task 2: Foundation tests** — `4645d3d` (test)
   - test_mode.py: rename `test_all_three_modes_registered` → `test_all_four_modes_registered`; new `test_thesis_crossdevice_main_profile`
   - test_manifest.py: rename `test_run_manifest_schema_version_2` → `test_run_manifest_schema_version_3`; new `test_run_manifest_backward_compat_v2` + `test_run_manifest_carries_thesis_fields`
   - test_atomic.py (NEW): 3 tests covering UTF-8 round-trip, parent-dir auto-create, idempotent overwrite
   - test_launcher.py: append `test_thesis_mode_dry_run`

**Plan metadata commit:** Will be created with this SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md updates.

_Note: Task 1 = `feat`, Task 2 = `test`. Both used `--no-verify` per `<parallel_execution>` protocol._

## Files Created/Modified

### Source

- `scripts/foundation/fedrec_foundation/mode.py` — `_THESIS_CROSSDEVICE_MAIN` ModeProfile (16 fields cloned from `_BENCHMARK_CROSS_DEVICE` with `mode="thesis_crossdevice_main"`); `_REGISTRY` dict gained `"thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN` entry between benchmark and paper_compat keys.
- `scripts/foundation/fedrec_foundation/manifest.py` — `RUN_MANIFEST_SCHEMA_VERSION: int = 3` (was 2). `RunManifest` dataclass extended with `thesis_run_label: str = ""`, `ablation_dimension: str = "none"`, `ablation_value: str = ""` (all NumPy-style docstrings, all safe defaults).
- `scripts/foundation/fedrec_foundation/atomic.py` — New `atomic_write_text(path: str, content: str) -> None` function added between `atomic_write_json` and `_json_default`. Mirrors `atomic_write_json` body with `suffix=".txt"`, `encoding="utf-8"`, no JSON serialization.
- `scripts/run.py` — `MODE_NUM_SUPERNODES` dict gained `"thesis_crossdevice_main": 6040,` entry. `argparse choices` auto-picks up via `sorted(MODE_NUM_SUPERNODES.keys())`.

### Tests

- `scripts/foundation/tests/test_mode.py` — `test_all_three_modes_registered` renamed to `test_all_four_modes_registered` with `thesis_crossdevice_main` in the asserted set. New `test_thesis_crossdevice_main_profile` test pins all 16 fields and uses `dataclasses.replace` to assert byte-identity vs `_BENCHMARK_CROSS_DEVICE` (mode-only diff).
- `scripts/foundation/tests/test_manifest.py` — `test_run_manifest_schema_version_2` renamed to `test_run_manifest_schema_version_3` (asserts constant == 3, dataclass instance == 3, embedded surface == 3). `test_run_manifest_backward_compat_v1` updated to `schema_version=3` literal. Two new tests added: `test_run_manifest_backward_compat_v2` (constructs without thesis kwargs, asserts default sentinels) and `test_run_manifest_carries_thesis_fields` (uses `dataclass_replace` to populate the 3 thesis fields and asserts they roundtrip through `embed_manifest_in_result`).
- `scripts/foundation/tests/test_atomic.py` (NEW) — 3 tests: `test_atomic_write_text` (UTF-8 round-trip + no `.tmp-*` leftovers), `test_atomic_write_text_creates_parent_dirs` (deeply-nested parent auto-creation), `test_atomic_write_text_overwrites_existing` (idempotent re-aggregation).
- `scripts/foundation/tests/test_launcher.py` — Appended `test_thesis_mode_dry_run` asserting `python scripts/run.py adaptive thesis_crossdevice_main --dry-run` exits 0, stdout contains `mode="thesis_crossdevice_main"` and `federated-adaptive-personalized-cf`, with regression guard that `num-supernodes` does NOT appear in `--run-config`.

### Planning

- `.planning/phases/07-thesis-evaluation-run/deferred-items.md` (NEW) — Logs out-of-scope discovery: 2 pre-existing slow subprocess determinism tests (`test_adaptive_determinism_subprocess_byte_identical`, `test_personalized_determinism_subprocess_byte_identical`) fail with `"No result JSON found after launcher run_id=..."` under live `flwr run`. Confirmed unrelated to Plan 07-01 scope by grep — neither test references any symbol Plan 07-01 touched.

## Final Field Values (all 4 building blocks)

### `_THESIS_CROSSDEVICE_MAIN` (16 fields)

```python
_THESIS_CROSSDEVICE_MAIN = ModeProfile(
    mode="thesis_crossdevice_main",       # ONLY field that differs from _BENCHMARK_CROSS_DEVICE
    num_supernodes=6040,
    partition_mode="natural",
    weight_policy="num_positives",
    primary_evaluator="sampled_loo_99",
    fraction_train=0.1,
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=64,
    optimizer="adam",
    lr=0.001,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",          # equivalent to pyproject "best_round_restore"
    assert_one_user_per_client=True,
)
```

### Schema v2→v3 (3 new fields, all safe defaults)

```python
RUN_MANIFEST_SCHEMA_VERSION: int = 3  # was 2

@dataclass
class RunManifest:
    # ... 25 prior fields (schema v1 + v2) ...
    thesis_run_label: str = ""             # "" = non-thesis sentinel; "main" = main run; "ablation_<knob>=<value>" = ablation
    ablation_dimension: str = "none"       # one of {none, alpha_method, per_user_alpha, item_perturbation, contrastive_lambda, fusion_type}
    ablation_value: str = ""               # specific value of ablated knob; "" for main runs
```

### `atomic_write_text` signature

```python
def atomic_write_text(path: str, content: str) -> None:
    """Write a UTF-8 text string atomically via tempfile + os.replace.

    Companion to atomic_write_json for plain-text payloads
    (markdown, CSV, etc.). Phase 7 aggregator uses this for
    main_comparison.md / main_comparison.csv writes.
    """
```

### `MODE_NUM_SUPERNODES` dict (4 entries)

```python
MODE_NUM_SUPERNODES = {
    "benchmark_cross_device": 6040,
    "thesis_crossdevice_main": 6040,    # Phase 7 D-04 (NEW)
    "paper_compat_pfedrec": 6040,
    "cross_silo_legacy": 5,
}
```

## Test Suite Counts

| Suite | Pre-Plan-01 | Post-Plan-01 | Delta |
|-------|-------------|--------------|-------|
| Fast suite (`-m "not slow"`) | 100 passed | **107 passed** | +7 net |
| Slow suite (subprocess determinism) | 4 collected | 4 collected (deselected in fast run) | 0 |
| **Total** | 104 collected | 111 collected | +7 |

The 9 plan-required test functions break down as:
- **+1 net in test_mode.py** (rename `test_all_three_*` → `test_all_four_*` is structural; new `test_thesis_crossdevice_main_profile` is +1 net)
- **+2 net in test_manifest.py** (rename `*_schema_version_2` → `*_schema_version_3` is structural; `test_run_manifest_backward_compat_v2` and `test_run_manifest_carries_thesis_fields` are +2 net)
- **+3 net in test_atomic.py** (NEW file with 3 tests)
- **+1 net in test_launcher.py** (`test_thesis_mode_dry_run`)
- Total: 1 + 2 + 3 + 1 = **+7 net** (matches observed delta).

## Decisions Made

1. **Mode profile cloning via field-level enumeration vs `dataclasses.replace`:** Followed plan's literal `ModeProfile(...)` enumeration (per the `<action>` block). The test `test_thesis_crossdevice_main_profile` asserts `dataclasses.replace`-equivalence as a sanity guard, but the source itself enumerates all 16 fields for explicitness. This matches Phase 2 conventions (every existing profile is enumerated, not replace-derived).
2. **Backward-compat test bump:** `test_run_manifest_backward_compat_v1` was updated from `schema_version=2` literal to `schema_version=3` because under v3 the safe defaults for the 3 new thesis fields trivially satisfy the v1 backward-compat invariant. Without the bump, the test would still pass on the constant assertion but would assert a stale schema_version literal that doesn't match `RUN_MANIFEST_SCHEMA_VERSION=3`.
3. **No edits to existing schema-v2-literal manifest tests:** `test_run_manifest_carries_final_eval_round_index`, `test_write_manifest_sibling_default_filename`, and `test_write_manifest_sibling_custom_name` continue to construct manifests with `schema_version=2` and assert the embedded surface returns `2`. They are testing v2 backward-compat in spirit (the dataclass surface returns whatever was written, not the build-time constant), and pass cleanly under v3 because the safe defaults flow in correctly. Bumping their literals would have been scope creep.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Backward-compat test stale `schema_version=2` literal**
- **Found during:** Task 2 (extend test_manifest.py)
- **Issue:** Plan's `<action>` block specified renaming `test_run_manifest_schema_version_2` → `_3` and adding 2 new tests, but did not call out updating `test_run_manifest_backward_compat_v1`'s `schema_version=2` literal. Under v3 the constant changes, so the test asserted the dataclass instance's `2` literal which no longer matches the build-time `RUN_MANIFEST_SCHEMA_VERSION`. Strictly speaking the test still asserted what it was constructed with (v1 fixture compatibility), but the literal `2` would mislead future readers.
- **Fix:** Bumped the literal to `schema_version=3` while keeping the test's spirit (v1 fixture without `final_eval_round_index`/`metrics` kwargs constructs without TypeError under the current schema).
- **Files modified:** `scripts/foundation/tests/test_manifest.py`
- **Verification:** All 13 manifest tests pass; the spirit of v1 backward-compat (Pitfall 3) preserved.
- **Committed in:** `4645d3d` (Task 2 commit)

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary to keep the backward-compat tests semantically honest under v3. No scope creep.

## Issues Encountered

- **Slow subprocess determinism tests fail under live `flwr run`** (`test_adaptive_determinism_subprocess_byte_identical`, `test_personalized_determinism_subprocess_byte_identical`): `AssertionError: No result JSON found after launcher run_id=...`. Confirmed unrelated to Plan 07-01 scope by grep — neither test references `thesis_crossdevice_main`, `atomic_write_text`, or `RUN_MANIFEST_SCHEMA_VERSION`. Logged in `.planning/phases/07-thesis-evaluation-run/deferred-items.md` for Plan 02 follow-up. The fast suite (107 tests) is fully GREEN.

## User Setup Required

None — no external service configuration required.

## Self-Check: PASSED

Verified before STATE update:

- `scripts/foundation/fedrec_foundation/mode.py` — `_THESIS_CROSSDEVICE_MAIN = ModeProfile(` present + `"thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN,` in `_REGISTRY` (FOUND)
- `scripts/foundation/fedrec_foundation/manifest.py` — `RUN_MANIFEST_SCHEMA_VERSION: int = 3` + 3 new fields (FOUND: `thesis_run_label`, `ablation_dimension`, `ablation_value` all present in `RunManifest` source)
- `scripts/foundation/fedrec_foundation/atomic.py` — `def atomic_write_text(path: str, content: str) -> None:` (FOUND)
- `scripts/run.py` — `"thesis_crossdevice_main": 6040,` in `MODE_NUM_SUPERNODES` (FOUND)
- `scripts/foundation/tests/test_atomic.py` — exists with 3 tests (FOUND)
- Commit `3c3116f` (feat: foundation primitives) — present in `git log` (FOUND)
- Commit `4645d3d` (test: foundation tests) — present in `git log` (FOUND)
- All 9 plan-required tests GREEN (FOUND)
- Stale `test_all_three_modes_registered` removed (`grep -c` returns 0 — FOUND)

## Next Phase Readiness

- **Plan 02 ready:** `_THESIS_CROSSDEVICE_MAIN` is registered; `RUN_MANIFEST_SCHEMA_VERSION=3` with 3 thesis fields; 4 server_app.py files can now branch on the new mode and emit `dataclass_replace(manifest, thesis_run_label=..., ablation_dimension=..., ablation_value=...)` before `embed_manifest_in_result`.
- **Plan 03 ready:** Orchestrator `scripts/thesis/run_thesis_sweep.py` can invoke `python scripts/run.py {module} thesis_crossdevice_main --run-config "..."` cleanly; argparse will accept the new mode.
- **Plan 04 ready:** Aggregator can `from fedrec_foundation.atomic import atomic_write_text` for `main_comparison.md` / `main_comparison.csv` writes.
- **Plan 05 ready:** Manual runbook can document the canonical incantation `python scripts/run.py adaptive thesis_crossdevice_main` as the entry point.

No blockers. The pre-existing slow subprocess determinism failures (deferred-items.md) are independent of Plan 07-01 scope and don't gate any downstream Phase 7 plan.

---
*Phase: 07-thesis-evaluation-run*
*Plan: 01*
*Completed: 2026-04-29*
