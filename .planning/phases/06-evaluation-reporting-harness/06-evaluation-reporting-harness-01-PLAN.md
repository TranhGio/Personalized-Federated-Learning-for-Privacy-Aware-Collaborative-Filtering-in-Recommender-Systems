---
phase: 06-evaluation-reporting-harness
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/foundation/fedrec_foundation/paths.py
  - scripts/foundation/tests/test_paths.py
autonomous: true
requirements: [EVL-04]
must_haves:
  truths:
    - "module_run_results_dir(module, run_id) returns <repo>/results/federated/<module>/<run_id>/ as an absolute Path (D-01 + D-02)"
    - "module_run_results_dir creates the directory (parents=True, exist_ok=True) so callers never see an ENOENT"
    - "module_run_results_dir raises ValueError on a module name outside {baseline, personalized, adaptive, pfedrec} (Pitfall 6 typo guard)"
    - "Function works under cwd != repo_root (Flower subprocess chdir behavior) because it delegates to repo_root() walk-up"
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/paths.py"
      provides: "module_run_results_dir(module, run_id) helper appended after data_derived/ml1m_dir"
      contains: "def module_run_results_dir(module: str, run_id: str) -> Path:"
    - path: "scripts/foundation/tests/test_paths.py"
      provides: "Three GREEN tests covering D-01 layout, D-02 repo-root anchoring, whitelist validation"
      contains: "def test_module_run_results_dir_repo_root_anchored"
  key_links:
    - from: "scripts/foundation/fedrec_foundation/paths.py::module_run_results_dir"
      to: "scripts/foundation/fedrec_foundation/paths.py::repo_root"
      via: "out = repo_root() / 'results' / 'federated' / module / run_id"
      pattern: "repo_root\\(\\) / .results. / .federated."
    - from: "scripts/foundation/fedrec_foundation/paths.py::_ALLOWED_MODULES"
      to: "scripts/foundation/fedrec_foundation/manifest.py::RunManifest.module"
      via: "Whitelist matches RunManifest.module docstring 'one of: baseline | personalized | adaptive | pfedrec'"
      pattern: "frozenset\\(\\{.baseline., .personalized., .adaptive., .pfedrec.\\}\\)"
---

<objective>
Add the single foundation helper `module_run_results_dir(module, run_id)` that resolves repo-root-anchored per-run results directories, plus its three unit tests. This is the Wave 1 foundation primitive every Wave 2/3 server_app plan depends on.

Purpose:
  - Close EVL-04 D-02 at the foundation layer: every server_app (Wave 2/3) imports this helper instead of computing relative paths. Eliminates the four ad-hoc `Path("../results/federated[/<module>]")` sites and resolves the folded `phase2-baseline-determinism-path-bug.md` todo.
  - Encode D-01 layout (`<repo>/results/federated/<module>/<run_id>/`) as the single source of truth.
  - Pin the module-name whitelist (Pitfall 6 typo guard) so `module_run_results_dir(module="basline", ...)` raises immediately at runtime instead of silently writing to `/results/federated/basline/...`.

Output:
  - 12-line addition to `scripts/foundation/fedrec_foundation/paths.py` mirroring the shape of the existing `data_derived()` / `ml1m_dir()` helpers.
  - New `scripts/foundation/tests/test_paths.py` file with 3 GREEN tests.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/fedrec_foundation/paths.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/fedrec_foundation/manifest.py

<interfaces>
<!-- Existing repo_root() walk-up: anchored on `data/ml-1m/` existence (paths.py:16-36). -->
<!-- Existing data_derived() / ml1m_dir() helpers (paths.py:39-53) — module_run_results_dir mirrors this shape. -->
<!-- The new helper does NOT support env-var override (unlike data_derived's FEDREC_FOUNDATION_DATA_DIR) -->
<!-- because results paths are runtime-driven (per-run, per-module) not config-driven. -->

```python
# scripts/foundation/fedrec_foundation/paths.py — current public surface
def repo_root() -> Path
def data_derived() -> Path
def ml1m_dir() -> Path
# NEW (this plan):
def module_run_results_dir(module: str, run_id: str) -> Path
```

```python
# scripts/foundation/fedrec_foundation/manifest.py:80 — whitelist rationale
# RunManifest dataclass field declared as:
module: str  # one of: "baseline" | "personalized" | "adaptive" | "pfedrec"
# That comment is the source of truth for the whitelist used by module_run_results_dir.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Append module_run_results_dir(module, run_id) helper to paths.py + ship test_paths.py with 3 GREEN tests pinning D-01 layout, D-02 repo-root anchoring, and whitelist enforcement</name>
  <files>scripts/foundation/fedrec_foundation/paths.py, scripts/foundation/tests/test_paths.py</files>
  <read_first>
    - scripts/foundation/fedrec_foundation/paths.py — current state (paths.py:16-53, the helper goes BELOW ml1m_dir at line 53)
    - scripts/foundation/fedrec_foundation/manifest.py:80 — `module: str  # one of: "baseline" | "personalized" | "adaptive" | "pfedrec"` (whitelist source of truth)
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-01, D-02
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Pattern 1 + §Common Pitfalls Pitfall 6 (module-string drift) + §Code Examples Example 1
    - scripts/foundation/tests/conftest.py — pytest fixtures available for foundation suite
    - scripts/foundation/tests/test_mapping.py OR test_split.py — existing test-file shape pattern (imports, fixtures, assertion style)
  </read_first>
  <behavior>
    - Test 1 (test_module_run_results_dir_repo_root_anchored): Call `module_run_results_dir("baseline", "20260429-104530-a1b2c3")`; assert returned path is absolute (`.is_absolute()`); assert it equals `repo_root() / "results" / "federated" / "baseline" / "20260429-104530-a1b2c3"`; assert the directory now exists on disk (`returned_path.is_dir()`); under `monkeypatch.chdir(tmp_path)` (cwd != repo root), the call still returns the same repo-root-anchored path (D-02 — Flower subprocess chdir robustness).
    - Test 2 (test_module_run_results_dir_layout): For each module in `["baseline", "personalized", "adaptive", "pfedrec"]`, call `module_run_results_dir(module, "test-run-id")`; assert the returned path's parts contain literally `["results", "federated", module, "test-run-id"]` in that order; assert each call creates a distinct dir (D-01: per-module, per-run dir IS the run identifier). Use `tmp_path` if available to avoid polluting the actual repo `results/` tree — accept that the helper writes into the real repo `results/federated/<module>/test-run-id/` if no monkeypatch is in place; in that case the test cleans up after itself with `shutil.rmtree`.
    - Test 3 (test_module_run_results_dir_whitelist): For each invalid module name in `["basline", "Baseline", "BASELINE", "personalize", "adapt", "PFedRec", "thesis", ""]`, call `module_run_results_dir(name, "any-run-id")`; assert it raises `ValueError` whose message contains the literal `repr(name)` substring AND the substring `"Expected one of"` (Pitfall 6 — typos must fail loud at runtime).
  </behavior>
  <action>
Append the helper to `scripts/foundation/fedrec_foundation/paths.py` AFTER the existing `ml1m_dir()` function (at line 53). Add NO env-var override (unlike `data_derived`); results paths are runtime-driven.

Insert this exact block at the end of `paths.py`:

```python


_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})


def module_run_results_dir(module: str, run_id: str) -> Path:
    """Return ``<repo>/results/federated/<module>/<run_id>/`` (creating it).

    Used by every per-module ``server_app.py`` to resolve the canonical write
    path for a Phase-6 cross-device run. The directory is created with
    ``parents=True, exist_ok=True`` so callers never see ``FileNotFoundError``
    on the parent path. The directory IS the run identifier (D-01 — one
    directory per run; results.json + manifest.json live inside).

    The path is repo-root anchored via :func:`repo_root` (D-02). This makes
    the helper safe to call from any cwd — Flower simulation may chdir
    subprocesses; the returned path is independent of cwd.

    Parameters
    ----------
    module : str
        One of ``"baseline"`` / ``"personalized"`` / ``"adaptive"`` /
        ``"pfedrec"``. Matches the literal value passed to
        :func:`fedrec_foundation.manifest.build_run_manifest` ``module=`` kwarg
        (manifest.py:80 comment is the source of truth).
    run_id : str
        Same string as ``RunManifest.run_id`` (from
        :func:`fedrec_foundation.manifest.generate_run_id`).

    Returns
    -------
    pathlib.Path
        Absolute, resolved path to the per-run directory.

    Raises
    ------
    ValueError
        If ``module`` is not in the allowed-modules whitelist (Pitfall 6 —
        typos in literals like ``"basline"`` must fail loud at runtime so
        results never land in ``/results/federated/basline/<run_id>/``).
    """
    if module not in _ALLOWED_MODULES:
        raise ValueError(
            f"Unknown module {module!r}. Expected one of "
            f"{sorted(_ALLOWED_MODULES)}."
        )
    out = repo_root() / "results" / "federated" / module / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out
```

Then create `scripts/foundation/tests/test_paths.py`:

```python
"""Tests for fedrec_foundation.paths — Phase 6 module_run_results_dir helper."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fedrec_foundation.paths import (
    _ALLOWED_MODULES,
    module_run_results_dir,
    repo_root,
)


def test_module_run_results_dir_repo_root_anchored(tmp_path, monkeypatch):
    """D-02: returned path is repo-root anchored, robust under chdir."""
    monkeypatch.chdir(tmp_path)  # Simulate Flower subprocess cwd != repo root
    run_id = "20260429-104530-deadbe"
    try:
        path = module_run_results_dir("baseline", run_id)
        assert path.is_absolute(), f"Expected absolute path, got {path!r}"
        assert path == repo_root() / "results" / "federated" / "baseline" / run_id, (
            f"D-02 anchoring broken: got {path!r}"
        )
        assert path.is_dir(), f"Expected directory to exist after call, got {path!r}"
    finally:
        # Cleanup so the test does not pollute the real repo results/ tree.
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("module", sorted(_ALLOWED_MODULES))
def test_module_run_results_dir_layout(module):
    """D-01: per-module, per-run directory layout."""
    run_id = f"20260429-104530-test{module[:3]}"
    path = module_run_results_dir(module, run_id)
    try:
        # Assert the trailing parts are exactly [results, federated, module, run_id].
        assert path.parts[-4:] == ("results", "federated", module, run_id), (
            f"D-01 layout broken: got parts {path.parts!r}"
        )
        assert path.is_dir()
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize(
    "bad_name",
    ["basline", "Baseline", "BASELINE", "personalize", "adapt", "PFedRec", "thesis", ""],
)
def test_module_run_results_dir_whitelist(bad_name):
    """Pitfall 6: typos in module string must fail loud, not silently write."""
    with pytest.raises(ValueError) as excinfo:
        module_run_results_dir(bad_name, "any-run-id")
    msg = str(excinfo.value)
    assert repr(bad_name) in msg, (
        f"Expected {bad_name!r} in error message, got {msg!r}"
    )
    assert "Expected one of" in msg, (
        f"Expected 'Expected one of' in error message, got {msg!r}"
    )
```

Verify the test passes by running:

```bash
cd scripts/foundation && pytest tests/test_paths.py -x -v
```

All 3 test functions × parametrize expansion (1 + 4 + 8 = 13 test items) MUST pass.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation && pytest tests/test_paths.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "def module_run_results_dir(module: str, run_id: str) -> Path:" scripts/foundation/fedrec_foundation/paths.py` returns 1
    - `grep -c "_ALLOWED_MODULES = frozenset" scripts/foundation/fedrec_foundation/paths.py` returns 1
    - `grep -E "_ALLOWED_MODULES = frozenset\\(\\{.baseline., .personalized., .adaptive., .pfedrec.\\}\\)" scripts/foundation/fedrec_foundation/paths.py` matches at least 1 line (whitelist literal complete)
    - `grep -c "out = repo_root() / .results. / .federated. / module / run_id" scripts/foundation/fedrec_foundation/paths.py` returns 1
    - `grep -c "out.mkdir(parents=True, exist_ok=True)" scripts/foundation/fedrec_foundation/paths.py` returns 1
    - `grep -c "raise ValueError" scripts/foundation/fedrec_foundation/paths.py` returns 1 (whitelist enforcement)
    - `test -f scripts/foundation/tests/test_paths.py` exits 0
    - `python -c "from fedrec_foundation.paths import module_run_results_dir; print(module_run_results_dir.__doc__[:50])"` exits 0 and prints docstring start
    - `cd scripts/foundation && pytest tests/test_paths.py -x -v` exits 0 with all parametrized expansions passing (3 functions, 13 test items)
    - `python -c "from fedrec_foundation.paths import module_run_results_dir; module_run_results_dir('basline', 'x')"` exits 1 with ValueError containing 'basline' and 'Expected one of'
  </acceptance_criteria>
  <done>
    - paths.py extended by ~50 lines (helper + docstring + whitelist constant) below the existing ml1m_dir
    - 3 test functions in test_paths.py pin D-01 layout, D-02 anchoring, Pitfall-6 whitelist
    - All 13 test items pass
    - No changes to repo_root / data_derived / ml1m_dir (D-18 surgical scope held)
  </done>
</task>

</tasks>

<verification>
- Helper imports cleanly: `python -c "from fedrec_foundation.paths import module_run_results_dir, _ALLOWED_MODULES; assert _ALLOWED_MODULES == frozenset({'baseline', 'personalized', 'adaptive', 'pfedrec'}); print('ok')"` prints "ok"
- Whitelist enforcement: `python -c "from fedrec_foundation.paths import module_run_results_dir; module_run_results_dir('basline', 'r')"` raises ValueError
- Layout matches D-01: `python -c "from fedrec_foundation.paths import module_run_results_dir; p = module_run_results_dir('baseline', 'TEST'); print(p.parts[-4:])"` prints `('results', 'federated', 'baseline', 'TEST')`
- Foundation test suite remains green: `cd scripts/foundation && pytest tests/ -q -m "not slow"` exits 0
- D-18 surgical scope: `git diff --stat` shows ONLY changes to scripts/foundation/fedrec_foundation/paths.py + scripts/foundation/tests/test_paths.py; nothing else
</verification>

<success_criteria>
- New `module_run_results_dir(module: str, run_id: str) -> Path` helper appended to paths.py with docstring citing D-01 + D-02 + Pitfall 6
- `_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})` constant matches manifest.py:80 comment
- Helper creates the directory (parents=True, exist_ok=True), returns absolute path, raises ValueError on unknown module name
- 3 GREEN test functions (1 + 4 parametrize + 8 parametrize = 13 test items) covering D-01 layout, D-02 anchoring under chdir, Pitfall-6 typo enforcement
- Foundation suite full green: `pytest scripts/foundation/tests/ -q -m "not slow"` exits 0
- Does NOT touch manifest.py / mode.py / atomic.py / any tests other than the new test_paths.py (D-18 surgical scope; Plan 02 owns manifest.py)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-01-SUMMARY.md` covering:
- module_run_results_dir helper signature, whitelist contents, mkdir semantics
- Test counts and which decisions each test pins (D-01 layout, D-02 anchoring, Pitfall-6 whitelist)
- Cross-phase contract: Wave 2/3 plans (03/04/05/06) import this helper instead of computing relative paths
</output>
</content>
</invoke>