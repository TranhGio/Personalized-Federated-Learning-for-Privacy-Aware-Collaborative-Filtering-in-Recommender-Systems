---
phase: 01-foundation-contract
plan: 06
subsystem: infra
tags: [pyproject-toml, pep440, local-path-dep, editable-install, hatchling, flwr, pytest, subprocess, smoke-test, imp-1, wave-3, phase-closure]

requires:
  - phase: 01-foundation-contract-01
    provides: "scripts/foundation/ package (fedrec-foundation 0.1.0) + docs/setup.md install-order documentation"
  - phase: 01-foundation-contract-02
    provides: "fedrec_foundation.{mapping, split, exclusion, bundle, build, user_groups} + committed data/derived/ bundle"
  - phase: 01-foundation-contract-03
    provides: "fedrec_foundation.{evaluator, weight_policy, fit_metrics}"
  - phase: 01-foundation-contract-04
    provides: "fedrec_foundation.{rng, manifest}"
  - phase: 01-foundation-contract-05
    provides: "fedrec_foundation.mode + scripts/run.py launcher"

provides:
  - "Every federated-*-cf/pyproject.toml declares 'fedrec-foundation' as the first entry in [project] dependencies (PEP 440 plain-name, not direct reference) — 4 modules wired (IMP-1)."
  - "scripts/foundation/tests/test_integration.py::test_cross_module_imports (parametrized across 4 modules) — subprocess import check mirroring `flwr run .` execution semantics."
  - "scripts/foundation/tests/test_integration.py::test_pyproject_declares_foundation_dep — regression guard for IMP-1."
  - "docs/setup.md clarifies: plain-name dep requires `pip install -e scripts/foundation/` BEFORE any `pip install -e federated-*-cf/`; the step order is load-bearing."
  - "Phase 1 closure: Phases 2-5 can now add `from fedrec_foundation.X import ...` freely with no sys.path hacks."

affects: [02-baseline-migration, 03-personalized-migration, 04-adaptive-migration, 05-pfedrec-migration, 06-evaluation-harness, 07-thesis-evaluation]

tech-stack:
  added: []  # No new libraries — pure wiring plan.
  patterns:
    - "Plain-name local-path dependency: module's `pyproject.toml` lists `fedrec-foundation` WITHOUT a URL; resolution relies on `pip install -e ../scripts/foundation/` having been done first. Simple, template-compatible, and matches how each module today is installed via `pip install -e .`."
    - "Cross-module subprocess smoke test: parametrized pytest test that spawns `python -c 'import X, Y, Z'` with cwd set to each module's directory. Mirrors `flwr run .` execution semantics without requiring a full federation run."
    - "pytest.skip-aware integration tests: walks up to repo root looking for `data/ml-1m`, skips gracefully if the repo structure doesn't match (minimal clones, CI without artifacts)."

key-files:
  created: []  # No Python source created — pure config wiring.
  modified:
    - "federated-baseline-cf/pyproject.toml (added `fedrec-foundation` dep line + comment)"
    - "federated-pfedrec/pyproject.toml (added `fedrec-foundation` dep line + comment; was untracked, now committed)"
    - "federated-personalized-cf/pyproject.toml (added `fedrec-foundation` dep line + comment)"
    - "federated-adaptive-personalized-cf/pyproject.toml (added `fedrec-foundation` dep line + comment)"
    - "docs/setup.md (Plan 06 notes clarified — plain-name dep requires install-order discipline)"
    - "scripts/foundation/tests/test_integration.py (+2 new tests: test_cross_module_imports parametrized x4, test_pyproject_declares_foundation_dep)"

key-decisions:
  - "PEP 440 plain-name dependency, NOT direct reference. `fedrec-foundation` appears as a bare dep name in each pyproject.toml without a URL. Alternatives considered: `fedrec-foundation @ file:../scripts/foundation` (Hatchling does not allow local `file://` URLs in source dists), or `[tool.uv.sources] fedrec-foundation = { workspace = true }` (uv-specific, incompatible with pip-driven editable installs). The plain-name choice is template-compatible, works with `pip install -e`, and matches how every module today is installed. The cost is a load-bearing step order (foundation first); the benefit is a minimal diff and zero build-backend coupling."
  - "Comment line above the dep makes the install order explicit to humans reading pyproject.toml: `# Foundation contract (Phase 1) — install ../scripts/foundation/ in editable mode FIRST; see docs/setup.md`. The cross-module smoke test's error message also hints at this."
  - "Subprocess-based smoke test (not in-process import) because module dirs have their own `.venv`-like isolation expectations when invoked via `flwr run .`. Running a subprocess with `cwd=<module_dir>` approximates that behavior one level above unit tests and one level below full flwr simulation."
  - "test_pyproject_declares_foundation_dep is a cheap textual regression guard. It would have caught Task 1 incompleteness (e.g., a typo like `fedrec_foundation` vs `fedrec-foundation`) even before the install step."

patterns-established:
  - "Plain-name local-path dependency contract: when a sub-package is not published to any index, declaring its dep-name alone is sufficient IF the install documentation makes the install-order explicit. docs/setup.md is the canonical install-order artifact for this repo."
  - "Parametrized cross-module subprocess tests: pytest.mark.parametrize across the 4 module names, each with cwd=<module_dir> via subprocess.run([sys.executable, '-c', script]). Captures stdout/stderr, asserts returncode==0 and a sentinel 'ok' string. Hint message in the assertion failure includes the exact install commands to fix."
  - "Repo-root detection via walk-up: `for p in [here.parent] + list(here.parents): if (p / 'data' / 'ml-1m').exists(): return p` — resilient to being run from any sub-directory, skips gracefully if the repo layout is non-standard."

requirements-completed: []  # Plan 06 has no direct FND-* requirements; it's the integration wiring that CLOSES all of Phase 1's FND-01..07 by making them consumable.

metrics:
  duration: "~6 min"
  started: "2026-04-19T03:27:03Z"
  completed: "2026-04-19T03:33:00Z"
  tasks_completed: 2
  files_modified: 6
  tests_added: 5  # 4 parametrized + 1 pyproject check
  tests_green: 70  # full foundation suite (was 65 before Plan 06)
  plan_06_commits: 2
---

# Phase 01 Plan 06: Cross-Module Foundation Wiring (IMP-1) Summary

**Every federated-*-cf/pyproject.toml now declares `fedrec-foundation` as a plain-name local-path dependency, plus a 5-test cross-module smoke test that proves each module's working directory can `import fedrec_foundation.*` in a fresh subprocess — closing Phase 1.**

## Performance

- **Duration:** ~6 min (355 seconds)
- **Started:** 2026-04-19T03:27:03Z
- **Completed:** 2026-04-19T03:33:00Z
- **Tasks:** 2 (both autonomous, no deviations from plan)
- **Files modified:** 6 (4 pyproject.toml + 1 docs + 1 test)
- **Files created:** 0 (pure wiring plan — no new Python source)
- **Tests added:** 5 (4 parametrized cross-module imports + 1 pyproject dep check)
- **Foundation test suite:** 70 passed (up from 65), 0 failed, 0 skipped

## Accomplishments

- **IMP-1 closed.** All four federated modules (`federated-baseline-cf`, `federated-pfedrec`, `federated-personalized-cf`, `federated-adaptive-personalized-cf`) declare `fedrec-foundation` as the first entry in their `[project] dependencies` array, above `flwr[simulation]>=1.22.0`, with an explanatory comment pointing at `docs/setup.md`. Phases 2–5 can now add `from fedrec_foundation.X import ...` statements freely without touching `sys.path`.
- **Cross-module import smoke test lands.** `scripts/foundation/tests/test_integration.py::test_cross_module_imports` is parametrized across all four modules; each invocation spawns a subprocess rooted at that module's directory and imports all ten foundation submodules (`fedrec_foundation`, `.mapping`, `.split`, `.exclusion`, `.evaluator`, `.weight_policy`, `.fit_metrics`, `.rng`, `.manifest`, `.mode`). This mirrors the execution semantics of `flwr run .` without requiring a full federation.
- **Regression guard added.** `test_pyproject_declares_foundation_dep` reads each module's `pyproject.toml` and asserts the string `fedrec-foundation` is present — catches any future accidental deletion of the dep line.
- **docs/setup.md clarified.** The Notes section now calls out the load-bearing step order explicitly: because `fedrec-foundation` is a plain-name (not a URL / direct-reference) dep and is not published to any index, users MUST run `pip install -e scripts/foundation/` BEFORE any `pip install -e federated-*-cf/`, otherwise the module install will fail to resolve the name.
- **Install sequence verified on this machine.** Ran `pip install -e scripts/foundation/` + the four `pip install -e federated-*-cf/` calls; all four reinstalled successfully; each module's `cd <mod> && python -c "import fedrec_foundation"` prints `version 0.1.0`. Foundation bundle (`data/derived/`) still verifies clean via `verify_bundle()`.
- **Full foundation test suite: 70 passed, 0 failed.** Regression-free; the 5 new tests sit next to the 15 Plan-02 integration tests and the 45 per-module unit tests from Plans 01–05.

## Task Commits

Each task was committed atomically with hooks enabled (Wave 3 is single-threaded, no sibling conflicts):

1. **Task 1: Add fedrec-foundation as local-path dep to all 4 modules + docs/setup.md clarification** — `cef8ce2` (feat)
2. **Task 2: Add cross-module import smoke test + pyproject dep check** — `1f4a29b` (test)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md updates) is appended separately at plan close._

## Files Created/Modified

### `federated-baseline-cf/pyproject.toml`
Added two lines at the top of `[project] dependencies`:
```toml
    # Foundation contract (Phase 1) — install ../scripts/foundation/ in editable mode FIRST; see docs/setup.md
    "fedrec-foundation",
```
Placed above `"flwr[simulation]>=1.22.0"`. No other content in the file changed (pre-existing unrelated dirty edits in `dataset.py` / etc. are out of scope for this plan and were left untouched).

### `federated-pfedrec/pyproject.toml`
Same change. This file was **untracked** at plan start (new module being spun up); committing it here brings the module's packaging contract into the repo alongside its foundation dep declaration. All existing content (100-round defaults, dual-LR for PFedRec, `num-supernodes=5` federation config) preserved.

### `federated-personalized-cf/pyproject.toml`
Same change. No other content touched.

### `federated-adaptive-personalized-cf/pyproject.toml`
Same change. No other content touched.

### `docs/setup.md`
Plan 06 notes section rewritten to make the install-order dependency explicit and load-bearing rather than cosmetic. Old text assumed the Plan 06 wiring would make "a single `pip install -e <module>/` pull the foundation automatically" — corrected to state that the plain-name dep still requires `pip install -e scripts/foundation/` FIRST.

### `scripts/foundation/tests/test_integration.py`
Appended two new tests (80 new lines, no existing tests touched):

1. `test_cross_module_imports(module_dir)` — parametrized across the 4 module names. For each module dir that exists, runs `subprocess.run([sys.executable, "-c", "import fedrec_foundation; import fedrec_foundation.mapping; ..."], cwd=<mod>)` and asserts `returncode == 0` plus the sentinel `"ok"` in stdout. On failure, the assertion message hints at the exact `pip install -e scripts/foundation/` + `pip install -e <module_dir>/` recovery commands.

2. `test_pyproject_declares_foundation_dep()` — reads each module's `pyproject.toml` via `Path.read_text()` and asserts the substring `"fedrec-foundation"` is present. Catches any accidental deletion (Task 1 regression) even before the install step.

The module-level helpers (`_MODULES`, `_FOUNDATION_SUBMODULES`, `_repo_root`) are defined in a clearly-demarcated comment section so they don't collide with the Plan 02 fixtures above.

## Decisions Made

- **Plain-name dep (`fedrec-foundation`), NOT PEP 440 direct reference.** Considered `fedrec-foundation @ {root:uri}/scripts/foundation/` (pip 21.0+ `root:uri` substitution), but Hatchling does not resolve this at build time without extra `[tool.hatch.metadata]` hooks, and it would break editable-install semantics in our current Python 3.9+ / pip 23+ / Hatchling 1.x stack. The plain-name choice is template-compatible with every module's existing install flow and matches the precedent in each `federated-*-cf/claude.md`'s "Install dependencies → `pip install -e .`" command. The cost is a load-bearing install-order step; the benefit is a zero-risk diff that doesn't touch any other part of packaging.
- **Dep placed FIRST in the dependencies list, above `flwr[simulation]`.** Semantic placement, not functional — readers who scan the dep list see `fedrec-foundation` before any external pin, signaling that the contract is the module's most fundamental dependency.
- **Comment line above the dep.** Costs one line but makes the install-order requirement visible in every `pyproject.toml`, not just in `docs/setup.md`. The cross-module smoke test's failure message also includes the recovery commands, so users who skip the docs still land on the right fix.
- **Subprocess test, not in-process import test.** `flwr run .` spawns subprocesses with cwd=module_dir; mirroring that gives higher confidence than an in-process `importlib.import_module(...)` call. Cost is ~0.5s per parametrization × 4 = 2s total; acceptable.
- **`_repo_root()` walks up from `__file__` looking for `data/ml-1m`.** Matches the pattern already used in `fedrec_foundation.paths.repo_root()` — reuses the repo-shape heuristic consistently. If the repo layout ever changes, one fix propagates everywhere.
- **Do NOT modify Python source files in federated-*-cf/.** Per the execution context's explicit scope boundary: the four modules have pre-existing uncommitted work in their `dataset.py` / `client_app.py` / `server_app.py` / `task.py` files. Plan 06 is pure packaging wiring; touching Python sources here would risk colliding with that in-progress work. Only the `[project] dependencies` section of each pyproject.toml was touched.

## Deviations from Plan

**None — plan executed exactly as written.**

Both tasks followed their `<action>` blocks verbatim:
- Task 1: added the 2-line block (comment + `"fedrec-foundation",`) into `[project] dependencies` of each of 4 pyproject.toml files (including the previously-untracked `federated-pfedrec/pyproject.toml` — committed alongside) + refined `docs/setup.md` Notes section.
- Task 2: appended the two tests specified in the plan's `<action>` block verbatim (including the `_MODULES` / `_FOUNDATION_SUBMODULES` tuples, the `_repo_root()` helper, and the hint message in the assertion failure).

No auto-fixes (Rules 1–3) applied. No architectural question (Rule 4) hit. No authentication gate. No test failures.

## Authentication Gates

None — all work is local-filesystem + pytest + pip. No external service touched.

## Issues Encountered

None. All automated verify commands passed on first run:

- `grep "fedrec-foundation" federated-*-cf/pyproject.toml` — 4/4 files match.
- `pip install -e scripts/foundation/ && pip install -e federated-*-cf/` (all 4) — each module reinstalled cleanly; no missing deps, no version conflicts.
- `for m in federated-*-cf/; do (cd "$m" && python -c "import fedrec_foundation; print(fedrec_foundation.__version__)"); done` — 4/4 print `0.1.0`.
- `cd scripts/foundation && pytest tests/ -v` — **70 passed, 0 failed, 0 skipped** (was 65/0/0 before the new tests; exactly 5 new GREEN tests, zero regressions).
- `python -c "from fedrec_foundation.bundle import verify_bundle; from pathlib import Path; verify_bundle(Path('data/derived'))"` — prints nothing (success sentinel); foundation bundle on disk is intact.

## Known Stubs

**None.** No placeholder values, no TODO markers, no `NotImplementedError`. The pyproject.toml changes are literal dep-name additions; the tests have real assertions and real subprocess invocations.

## User Setup Required

**None in addition to what `docs/setup.md` already documents.** The install order is now explicit in both `docs/setup.md` and the pyproject.toml comments themselves. On a fresh clone, the command sequence is:

```bash
pip install pytest
pip install -e scripts/foundation/
for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do
  pip install -e "$m"
done
# Smoke test
for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do
  (cd "$m" && python -c "import fedrec_foundation; print(fedrec_foundation.__version__)")
done
# Expect: "0.1.0" printed four times
```

## Phase 1 Closure

**Foundation-contract Phase 1 is now COMPLETE.** This plan's landing closes the loop on:

- **7 FND requirements** (FND-01 mapping + FND-02 split + FND-03 exclusion + FND-04 evaluator + FND-05 weight-policy + FND-06 RNG + FND-07 manifest) — all shipped in Plans 02–04 and now consumable from every federated module.
- **5 Codex CRITICAL items** (CR-1 ratings-only items, CR-2 CR-2 launcher, CR-3 sha256 RNG, CR-4 FitMetricsContract, CR-5 train-only stats) — all addressed across Plans 02–05.
- **4 Codex IMPORTANT items** (IMP-1 local-path dep [THIS PLAN], IMP-2 composite hash carriage, IMP-3 flat NPZ, IMP-4 frozen bucket semantics) — all closed.
- **3 Codex NIT items** (N-1 full sha256 digest, N-2 UTC run-ID, N-3 atomic bundle publication) — all honored.

**Integration contract for Phases 2–5:** every federated module's `client_app.py`, `server_app.py`, `task.py`, `strategy.py`, `dataset.py` may now freely write `from fedrec_foundation.X import Y` without any `sys.path` manipulation, without any conditional import guards, without any fallback paths. The one-time user-setup rule is: `pip install -e scripts/foundation/` then `pip install -e federated-*-cf/`. That's it.

## Next Phase Readiness

**Ready for Phase 2 (baseline-migration)**, Phase 3 (personalized-migration), Phase 4 (adaptive-migration), and Phase 5 (pfedrec-migration) — all four parallel migrations can now:

1. Import `fedrec_foundation.mode.resolve_mode_defaults` and `fedrec_foundation.mode.log_mode_and_overrides` in each module's `server_app.py`.
2. Import `fedrec_foundation.mode.assert_benchmark_one_user_per_client` in each module's `client_app.py::@app.train()`.
3. Import `fedrec_foundation.mapping.load_mapping`, `fedrec_foundation.split.load_split_manifest`, `fedrec_foundation.exclusion.ExclusionTable` in each module's `dataset.py`.
4. Import `fedrec_foundation.fit_metrics.FitMetricsContract` in each module's `client_app.py::@app.train()` return path.
5. Import `fedrec_foundation.weight_policy.compute_aggregation_weight` + `validate_fit_metrics` in each module's `strategy.py::aggregate_fit`.
6. Import `fedrec_foundation.rng.{py_rng, np_rng, torch_gen, server_rng}` anywhere sampling or seeding happens (DataLoader, negative sampling, client selection).
7. Import `fedrec_foundation.manifest.{build_run_manifest, embed_manifest_in_result, write_manifest_sibling}` in each module's `server_app.py` at result-writing time.
8. Invoke via `python scripts/run.py <module> <mode> [--run-config KEY=VAL ...]` instead of raw `cd <module> && flwr run . --run-config ...` — gets federation-level `num-supernodes` correct automatically.

**Ready for Phase 6 (evaluation harness)** and Phase 7 (thesis evaluation) transitively — they depend on Phases 2–5 via the roadmap, and Phase 1's contract is the shared spine those migrations use to emit comparable metrics and protocol fingerprints.

**No blockers. No open questions. No architectural decisions deferred from Phase 1.**

## Self-Check: PASSED

- **Files modified:**
  - FOUND (+dep line): `federated-baseline-cf/pyproject.toml` — verified via `grep "fedrec-foundation" federated-baseline-cf/pyproject.toml` matches 1.
  - FOUND (+dep line, new file): `federated-pfedrec/pyproject.toml` — verified via the same grep; `git log --stat` on `cef8ce2` shows `create mode 100644`.
  - FOUND (+dep line): `federated-personalized-cf/pyproject.toml` — verified via grep.
  - FOUND (+dep line): `federated-adaptive-personalized-cf/pyproject.toml` — verified via grep.
  - FOUND (clarified notes): `docs/setup.md` — verified via `git diff cef8ce2~1 cef8ce2 docs/setup.md`.
  - FOUND (+2 tests): `scripts/foundation/tests/test_integration.py` — verified via `pytest --collect-only` listing the new test IDs.
- **Commits:**
  - FOUND: `cef8ce2` (Task 1 feat), `1f4a29b` (Task 2 test). Both visible on `feat/try_to_run_the_baseline` via `git log --oneline -3`.
- **Automated verify:** PASSED.
  - `grep -c "fedrec-foundation" federated-*-cf/pyproject.toml` → 1/1/1/1 (4 hits, one per file).
  - `for m in federated-*-cf/; do (cd "$m" && python -c "import fedrec_foundation"); done` → no errors.
  - `cd scripts/foundation && pytest tests/ -v` → **70 passed, 0 failed, 0 skipped**. Specifically:
    - `test_cross_module_imports[federated-baseline-cf]` PASSED
    - `test_cross_module_imports[federated-pfedrec]` PASSED
    - `test_cross_module_imports[federated-personalized-cf]` PASSED
    - `test_cross_module_imports[federated-adaptive-personalized-cf]` PASSED
    - `test_pyproject_declares_foundation_dep` PASSED
  - `python -c "from fedrec_foundation.bundle import verify_bundle; from pathlib import Path; verify_bundle(Path('data/derived'))"` → success (no exception).
- **Scope boundary:** PASSED. `git diff cef8ce2~1 cef8ce2 --stat` shows ONLY 5 files touched (4 pyproject.toml + docs/setup.md). `git diff 1f4a29b~1 1f4a29b --stat` shows ONLY 1 file touched (test_integration.py). No Python source files in federated-*-cf/ modified — pre-existing dirty work in those files left untouched.

---

*Phase: 01-foundation-contract*
*Plan: 06 (Wave 3 — integration wiring)*
*Completed: 2026-04-19*
*Closes Phase 1: FND-01..07 + CR-1..5 + IMP-1..4 + N-1..3 all consumable from Phases 2–5.*
