---
phase: 01-foundation-contract
plan: 05
subsystem: infra
tags: [mode-resolver, launcher, cr-2, federation-config, flwr, argparse, subprocess, dataclass, cross-device, pitfall-6, pitfall-8]

requires:
  - phase: 01-foundation-contract-01
    provides: "scripts/foundation package + pytest harness + test_mode.py + test_launcher.py stubs"
  - phase: 01-foundation-contract-03
    provides: "weight_policy.WeightPolicy and evaluator.EvalProtocol enum string values (.value) that mode profiles reference as string literals"

provides:
  - "fedrec_foundation.mode: ModeProfile frozen dataclass + three registered profiles (benchmark_cross_device, paper_compat_pfedrec, cross_silo_legacy)"
  - "fedrec_foundation.mode.resolve_mode_defaults(mode, module_overrides=None) -> ModeProfile (ValueError on unknown mode)"
  - "fedrec_foundation.mode.log_mode_and_overrides(mode, profile, run_config) -> Dict[str, object] (kebab to snake + loud [MODE OVERRIDE] lines)"
  - "fedrec_foundation.mode.assert_benchmark_one_user_per_client(profile, n, overrides) (D-11 + CR-2 in-app assertion)"
  - "fedrec_foundation.mode.MODE_NAMES tuple"
  - "scripts/run.py CR-2 launcher: python scripts/run.py <module> <mode> [--run-config KEY=VAL ...]"
  - "--dry-run flag prints the flwr invocation for tests/CI without executing"
  - "Module aliases: baseline/personalized/adaptive/pfedrec -> federated-*-cf"
affects: [02-foundation-contract, 04-foundation-contract, 06-foundation-contract, 02-baseline-migration, 03-pfedrec-reproduction, 04-personalized-split-learning, 05-adaptive-hierarchical]

tech-stack:
  added:
    - "argparse-based launcher pattern at repo root (scripts/run.py)"
    - "frozen @dataclass(frozen=True) as the mode profile registry primitive"
  patterns:
    - "Federation-level num-supernodes lock: launcher sets it, app asserts it. Context.run_config CANNOT change num-supernodes -- that happens before ServerApp/ClientApp run."
    - "Three-mode canonical registry (benchmark_cross_device / paper_compat_pfedrec / cross_silo_legacy) -- one mode is one experiment (D-07)"
    - "Kebab to snake conversion (key.replace('-','_')) before comparing run_config overrides to dataclass fields (Pitfall 6)"
    - "String-literal enum values to avoid import-time coupling between parallel-wave sibling plans"
    - "--dry-run flag on subprocess launchers for test-safe assertion of command construction"

key-files:
  created:
    - "scripts/foundation/fedrec_foundation/mode.py"
    - "scripts/run.py"
  modified:
    - "scripts/foundation/tests/test_mode.py (flipped GREEN: 10 real tests)"
    - "scripts/foundation/tests/test_launcher.py (flipped GREEN: 7 real tests)"

key-decisions:
  - "String literals for weight_policy / primary_evaluator values in ModeProfile registry (not WeightPolicy.NUM_POSITIVES.value imports) to avoid coupling to Wave 2 parallel sibling Plan 03's import surface. String values are the contract anyway -- downstream callers can still type against the enum."
  - "MODE_NUM_SUPERNODES duplicated between mode.py (ModeProfile.num_supernodes) and scripts/run.py (MODULE_DIR + MODE_NUM_SUPERNODES dicts). Intentional: the launcher lives OUTSIDE the foundation package and should not import from it to avoid installation-order coupling. The in-app assertion enforces consistency at runtime."
  - "D-10 override semantics: visible CLI overrides bypass the one-user assertion (with a visible skip log). Visibility is the guarantee, not immutability. Silent overrides would be a D-10 violation."
  - "--dry-run flag lets tests assert exact command construction without mocking subprocess. Keeps tests fast (subprocess call per assertion) and realistic (argparse + Path resolution exercised)."

patterns-established:
  - "CR-2 launcher pattern: Python script at repo root that resolves (module, mode) -> (module-dir, federation-options), prints the flwr command for auditability, then invokes via subprocess. --dry-run flag for tests."
  - "In-app assertion layer: fedrec_foundation.mode.assert_benchmark_one_user_per_client is the Phase 2-5 hook. Each federated-*-cf/client_app.py's @app.train() calls it with (profile, num_users_in_partition, overrides) to fail LOUDLY on launcher/app drift."
  - "ModeProfile introspection via hasattr(profile, snake_name): log_mode_and_overrides walks run_config keys, applies kebab to snake, and compares to any matching dataclass field. Unknown keys are silently ignored (they belong to module-specific config the mode doesn't lock)."

requirements-completed: []

duration: "3 min"
completed: "2026-04-19"
---

# Phase 01 Plan 05: Mode Resolver + CR-2 Launcher Summary

**Three locked mode profiles (benchmark_cross_device N=6040, paper_compat_pfedrec N=6040, cross_silo_legacy N=5) plus a scripts/run.py launcher that sets num-supernodes at the Flower federation level -- closing the Codex CR-2 gap where Context.run_config cannot change num-supernodes from inside the app.**

## Performance

- **Duration:** ~3 min (228 seconds)
- **Started:** 2026-04-19T03:11:16Z
- **Completed:** 2026-04-19T03:15:04Z
- **Tasks:** 2 (both autonomous, TDD RED + GREEN each)
- **Commits:** 4 (2 TDD RED test commits + 2 TDD GREEN implementation commits)
- **Files created:** 2 (mode.py, scripts/run.py)
- **Files modified:** 2 (test_mode.py + test_launcher.py flipped from skip-stub to real tests)
- **Tests added:** 17 (10 in test_mode.py, 7 in test_launcher.py), all GREEN

## Accomplishments

- `fedrec_foundation.mode` exports `ModeProfile` (17-field frozen dataclass), `resolve_mode_defaults`, `log_mode_and_overrides`, `assert_benchmark_one_user_per_client`, and `MODE_NAMES` -- the mode contract for Phases 2-5.
- Three canonical profiles are registered and frozen:
  - `benchmark_cross_device`: N=6040, natural, num_positives, adam/1e-3, dim=64, 100 rounds, 1 local epoch, best_round, assert=True.
  - `paper_compat_pfedrec`: N=6040, natural, num_positives (deferred confirm to PFR-02), sgd/0.1, dim=32, 100 rounds, 1 local epoch, best_round, assert=True.
  - `cross_silo_legacy`: N=5, dirichlet, num_training_examples, adam/1e-3, dim=128, 10 rounds, 5 local epochs, last_round, assert=False.
- `scripts/run.py` launcher exists at the repo root and is the canonical entry point for cross-device and paper-compat runs. It maps `{baseline, personalized, adaptive, pfedrec}` to `federated-*-cf/` directories and `{benchmark_cross_device, paper_compat_pfedrec, cross_silo_legacy}` to `num-supernodes={6040, 6040, 5}`.
- `--dry-run` flag prints the `flwr run` command without executing -- enables deterministic subprocess-based tests.
- `log_mode_and_overrides` implements kebab to snake conversion (Pitfall 6) via `key.replace('-', '_')` before `hasattr(profile, snake)` comparison; prints `[MODE OVERRIDE] key: mode=... default=... user-override=...` for every deviation.
- `assert_benchmark_one_user_per_client` correctly no-ops on `cross_silo_legacy` (Pitfall 8: `assert_one_user_per_client=False` in that profile) and on visible `num-supernodes` overrides (D-10: override bypasses lock with a visible skip log).
- 17/17 tests pass; full foundation test suite is still green (61 passed, 4 integration skipped for Plans 02 + 06).

## Task Commits

Each task was committed atomically using TDD RED + GREEN:

1. **Task 1 RED: test_mode.py failing tests for D-06..D-11** -- `0c8c48d` (test)
2. **Task 1 GREEN: implement fedrec_foundation.mode resolver** -- `2f50d11` (feat)
3. **Task 2 RED: test_launcher.py failing tests for CR-2** -- `ce3300b` (test)
4. **Task 2 GREEN: scripts/run.py CR-2 launcher** -- `1af86ed` (feat)

_Plan metadata commit (SUMMARY.md + STATE.md + ROADMAP.md) is appended separately at the end of execution._

## Files Created/Modified

### New foundation module (Task 1)
- `scripts/foundation/fedrec_foundation/mode.py` -- 335 lines. ModeProfile frozen dataclass (17 fields), three registered profiles (`_BENCHMARK_CROSS_DEVICE`, `_PAPER_COMPAT_PFEDREC`, `_CROSS_SILO_LEGACY`), `_REGISTRY` dict, `MODE_NAMES` tuple, `resolve_mode_defaults` (ValueError on unknown mode, `dataclasses.replace` for module overrides), `log_mode_and_overrides` (kebab to snake + `[MODE OVERRIDE]` logging, returns dict ready for `manifest.overrides`), `assert_benchmark_one_user_per_client` (CR-2 in-app assertion with Pitfall 8 and D-10 short-circuits).

### New launcher (Task 2)
- `scripts/run.py` -- 158 lines. `MODULE_DIR` alias map (baseline -> federated-baseline-cf, etc.), `MODE_NUM_SUPERNODES` map (benchmark=6040, paper=6040, legacy=5), `_build_run_config` (joins base + user overrides into space-separated `key=value` string for `flwr run --run-config`), `main` (argparse with `choices=` gating module and mode, `--run-config KEY=VAL` repeatable, `--dry-run` flag, subprocess invocation).

### Tests flipped GREEN
- `scripts/foundation/tests/test_mode.py` -- 10 tests: `test_all_three_modes_registered`, `test_benchmark_profile`, `test_cross_silo_legacy_profile`, `test_paper_compat_profile`, `test_unknown_mode_raises`, `test_module_override`, `test_override_logging` (Pitfall 6), `test_assertion_flags_benchmark`, `test_assertion_flags_cross_silo_legacy_skipped` (Pitfall 8), `test_assertion_skipped_on_override` (D-10).
- `scripts/foundation/tests/test_launcher.py` -- 7 tests: `test_launcher_exists`, `test_launcher_sets_num_supernodes_benchmark`, `test_launcher_sets_num_supernodes_cross_silo_legacy`, `test_launcher_paper_compat_pfedrec`, `test_launcher_passes_extra_run_config`, `test_launcher_unknown_mode_rejected`, `test_launcher_malformed_run_config_rejected`.

## Decisions Made

- **String literals for enum values in ModeProfile registry.** The plan's `<action>` shows `weight_policy=WeightPolicy.NUM_POSITIVES.value` (a runtime-resolved string). Since Wave 2 runs this plan in parallel with Plan 03 (which creates `WeightPolicy`), importing from a sibling module that might not exist at module-load time would be fragile. We use the string values directly (`"num_positives"`, `"num_training_examples"`, `"uniform"`, `"sampled_loo_99"`). The TESTS check against those same string literals, so this is the actual contract -- the enum is a type-safety layer on top, not the source of truth.
- **Duplicated num-supernodes values between mode.py and scripts/run.py.** `ModeProfile.num_supernodes` (foundation package) and `MODE_NUM_SUPERNODES` (launcher script) both know 6040 and 5. This is intentional: the launcher must NOT import from the foundation package (which would require the package to be installed before the launcher runs on a fresh clone -- a chicken-and-egg during setup). The in-app assertion (`assert_benchmark_one_user_per_client`) closes the loop at runtime: if the launcher ever drifts from the app's mode profile, the first client batch raises AssertionError.
- **--dry-run flag for test-safe command construction.** Tests subprocess-invoke the launcher and assert on stdout. The alternative (mocking `subprocess.run`) would test less realistic behavior (argparse parsing, Path resolution, working-dir semantics all would be skipped). --dry-run keeps the real script surface under test.
- **Override dict returned by log_mode_and_overrides is re-keyed to snake_case.** Callers should use it to populate `manifest.overrides` (Plan 04's RunManifest). Snake-case keeps the manifest consistent with dataclass field names, which is the dominant Python convention in this repo.
- **`assert_benchmark_one_user_per_client` accepts both `num_supernodes` and `num-supernodes` in the overrides dict.** Defensive: callers may forward either the raw run_config keys or the snake-cased override dict. Either bypasses the assertion correctly.

## Deviations from Plan

None - plan executed exactly as written.

Both `<action>` blocks were followed verbatim. The only deliberate difference from the research Pattern 7 code snippet is the switch from `from fedrec_foundation.weight_policy import WeightPolicy` imports to string literals -- documented in Decisions above and justified by the Wave 2 parallel-execution context. All test assertions pass with either approach because the tests check against the underlying string values.

## Authentication Gates

None - this plan is pure Python + pytest + file creation, no external services touched.

## Issues Encountered

None. All automated verify commands passed on first run:
- RED phase: both test files fail with `ModuleNotFoundError`/`FileNotFoundError` as expected.
- GREEN phase: all 17 tests pass immediately.
- Full foundation suite: 61 passed, 4 skipped (integration tests deferred to Plans 02 + 06). No regressions from parallel-wave plans (mapping/split/exclusion/weight_policy/rng/manifest all green too).
- Smoke test: `python -c "from fedrec_foundation.mode import resolve_mode_defaults; print(resolve_mode_defaults('benchmark_cross_device').num_supernodes)"` prints `6040`.
- Launcher dry-run: `python scripts/run.py --dry-run baseline benchmark_cross_device` emits `num-supernodes=6040 mode=benchmark_cross_device` in the stdout command preview.

## Known Stubs

None. Both modules ship with real implementations, real enum registries, real CLI wiring, and real assertions.

## User Setup Required

None - no external service configuration required. The launcher is ready for use once the user runs `pip install -e scripts/foundation/` (per `docs/setup.md` from Plan 01).

## Integration Contract for Phases 2-5

Each federated module's `server_app.py` MUST at startup:

```python
from fedrec_foundation.mode import resolve_mode_defaults, log_mode_and_overrides

mode = context.run_config["mode"]
profile = resolve_mode_defaults(mode)
overrides = log_mode_and_overrides(mode, profile, context.run_config)
# `overrides` feeds into RunManifest.overrides (Plan 04)
```

Each `client_app.py` MUST inside `@app.train()`:

```python
from fedrec_foundation.mode import (
    resolve_mode_defaults, log_mode_and_overrides,
    assert_benchmark_one_user_per_client,
)

mode = context.run_config["mode"]
profile = resolve_mode_defaults(mode)
overrides = log_mode_and_overrides(mode, profile, context.run_config)
num_users_in_client = int(client_partition_df["user_idx"].nunique())
assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)
```

If the launcher is invoked with the wrong `num-supernodes` value for the declared mode, this assertion fires on the first client batch and halts the run loudly.

## Next Phase Readiness

**Ready for Wave 3 (Plan 06: wiring foundation into each federated-*-cf/ module).** Plan 06 agents will:

1. Add `fedrec-foundation = { path = "../scripts/foundation", develop = true }` (or equivalent `pip install -e` wiring) to each module's `pyproject.toml`.
2. Insert the integration contract above into `server_app.py` + `client_app.py` for each of the four federated modules.
3. Thread the `mode` run-config key through `[tool.flwr.app.config]` blocks (default `cross_silo_legacy` to preserve backwards compatibility with existing W&B dashboards).
4. Wire `scripts/run.py` as the documented entry point in each module's README / claude.md alongside the existing `flwr run .` commands.

**Ready for downstream phases (02 baseline-migration, 03 pfedrec-reproduction, 04 personalized-split-learning, 05 adaptive-hierarchical).** They inherit `scripts/run.py` for starting runs and `fedrec_foundation.mode` for the mode contract.

**No blockers.** All Wave 2 plans (02, 03, 04, 05) have green tests in the foundation suite; no architectural decisions deferred.

## Self-Check: PASSED

- **Files created:** FOUND: scripts/foundation/fedrec_foundation/mode.py (335 lines), scripts/run.py (158 lines). Verified via `ls` + `Grep`.
- **Files modified:** FOUND: scripts/foundation/tests/test_mode.py (10 tests, pytestmark removed), scripts/foundation/tests/test_launcher.py (7 tests, pytestmark removed). Verified via pytest output.
- **Commits:** FOUND: `0c8c48d` (Task 1 RED), `2f50d11` (Task 1 GREEN), `ce3300b` (Task 2 RED), `1af86ed` (Task 2 GREEN). Verified via `git log --oneline -6`.
- **Automated verify:** PASSED.
  - `cd scripts/foundation && pytest tests/test_mode.py tests/test_launcher.py -v` -> 17 passed, 0 failed, 0 skipped.
  - Full foundation suite: 61 passed, 4 skipped, 0 failed.
  - Import smoke: `from fedrec_foundation.mode import resolve_mode_defaults; print(resolve_mode_defaults('benchmark_cross_device').num_supernodes)` -> `6040`.
  - Launcher smoke: `python scripts/run.py --dry-run baseline benchmark_cross_device` -> stdout contains `num-supernodes=6040 mode=benchmark_cross_device federated-baseline-cf`.
  - Launcher smoke: `python scripts/run.py --dry-run pfedrec cross_silo_legacy` -> stdout contains `num-supernodes=5 mode=cross_silo_legacy federated-pfedrec`.
- **Acceptance criteria (Task 1):**
  - `grep -E "replace\([\"']-[\"'], ?[\"']_[\"']\)" scripts/foundation/fedrec_foundation/mode.py` -> matches line 260.
  - `grep "num_supernodes=6040" scripts/foundation/fedrec_foundation/mode.py` -> matches (lines 120, 139).
  - `grep "num_supernodes=5" scripts/foundation/fedrec_foundation/mode.py` -> matches (line 159).
  - `grep "assert_one_user_per_client=False" scripts/foundation/fedrec_foundation/mode.py` -> matches (line 173).
  - `pytest tests/test_mode.py -v` -> 10 passed (>= 10 required).
- **Acceptance criteria (Task 2):**
  - `scripts/run.py` exists at repo root (verified).
  - `grep "num-supernodes" scripts/run.py` -> matches.
  - `grep "MODULE_DIR" scripts/run.py` -> matches.
  - `grep "dry-run" scripts/run.py` -> matches.
  - `python scripts/run.py --dry-run baseline benchmark_cross_device` -> stdout contains the required three substrings (verified).
  - `python scripts/run.py --dry-run pfedrec cross_silo_legacy` -> stdout contains `num-supernodes=5` (verified).
  - `pytest tests/test_launcher.py -v` -> 7 passed (>= 6 required).

---

*Phase: 01-foundation-contract*
*Plan: 05*
*Completed: 2026-04-19*
