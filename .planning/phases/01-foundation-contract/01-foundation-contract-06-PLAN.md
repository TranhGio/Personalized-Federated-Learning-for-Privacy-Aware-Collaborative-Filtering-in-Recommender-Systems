---
phase: 01-foundation-contract
plan: 06
type: execute
wave: 3
depends_on:
  - 01-foundation-contract-01
  - 01-foundation-contract-02
  - 01-foundation-contract-03
  - 01-foundation-contract-04
  - 01-foundation-contract-05
files_modified:
  - federated-baseline-cf/pyproject.toml
  - federated-pfedrec/pyproject.toml
  - federated-personalized-cf/pyproject.toml
  - federated-adaptive-personalized-cf/pyproject.toml
  - scripts/foundation/tests/test_integration.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "After `pip install -e scripts/foundation/` followed by `pip install -e federated-<name>/` for each of the four modules, `python -c 'import fedrec_foundation'` succeeds from inside EACH module's directory."
    - "Every module's `pyproject.toml` declares `fedrec-foundation` as a local-path dependency so downstream installs resolve the import automatically (IMP-1)."
    - "Cross-module smoke test runs `pytest scripts/foundation/tests/test_integration.py::test_cross_module_imports` and confirms all four modules can import `fedrec_foundation.mapping`, `fedrec_foundation.rng`, `fedrec_foundation.weight_policy`, `fedrec_foundation.fit_metrics`, `fedrec_foundation.manifest`, `fedrec_foundation.mode`, `fedrec_foundation.exclusion`, `fedrec_foundation.split`, `fedrec_foundation.evaluator`."
    - "Installing a module does NOT break existing test scripts (`python test_dataset.py`, `python test_models.py`) for that module — foundation adds imports, does not modify behavior."
  artifacts:
    - path: "federated-baseline-cf/pyproject.toml"
      provides: "local-path dep on fedrec-foundation"
      contains: "fedrec-foundation"
    - path: "federated-pfedrec/pyproject.toml"
      provides: "local-path dep on fedrec-foundation"
      contains: "fedrec-foundation"
    - path: "federated-personalized-cf/pyproject.toml"
      provides: "local-path dep on fedrec-foundation"
      contains: "fedrec-foundation"
    - path: "federated-adaptive-personalized-cf/pyproject.toml"
      provides: "local-path dep on fedrec-foundation"
      contains: "fedrec-foundation"
  key_links:
    - from: "federated-baseline-cf/pyproject.toml"
      to: "fedrec-foundation local-path"
      via: "PEP 440 direct reference or PEP 660 editable install"
      pattern: "fedrec-foundation"
---

<objective>
Wire the `fedrec-foundation` package as a local-path dependency into each of the four federated modules' `pyproject.toml` (Codex IMP-1), then run the end-to-end smoke test that every module can import the foundation. This plan is pure integration: no Python source changes, no FND requirements directly, but it is what makes Plans 02–05 actually consumable from Phases 2–5.

Purpose: IMP-1 flagged that mutating `sys.path` in each module's `dataset.py` is fragile because other files (`server_app.py`, `client_app.py`, `task.py`, `strategy.py`) also need to import the foundation. Proper solution: declare the foundation as a pip-installable local-path dependency in every module's `pyproject.toml` so editable-install resolves the import natively. Exact syntax varies with Flower's hatchling pin — this plan picks ONE syntax and verifies it works for all four modules via a subprocess smoke test.

Output: Four updated `pyproject.toml` files and a new integration test `test_cross_module_imports` that passes. No module source code changes.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-foundation-contract/01-CONTEXT.md
@.planning/phases/01-foundation-contract/01-RESEARCH.md
@.planning/phases/01-foundation-contract/01-VALIDATION.md
@.planning/phases/01-foundation-contract/01-foundation-contract-01-SUMMARY.md
@.planning/phases/01-foundation-contract/01-foundation-contract-05-SUMMARY.md
@docs/setup.md

@federated-baseline-cf/pyproject.toml
@federated-pfedrec/pyproject.toml
@federated-personalized-cf/pyproject.toml
@federated-adaptive-personalized-cf/pyproject.toml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add fedrec-foundation as local-path dep to all four modules' pyproject.toml</name>
  <files>
    federated-baseline-cf/pyproject.toml
    federated-pfedrec/pyproject.toml
    federated-personalized-cf/pyproject.toml
    federated-adaptive-personalized-cf/pyproject.toml
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §IMP-1 (local-path dep syntax; editable install; install order)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Open Questions" item 3 (PEP 440 direct-reference vs relative file URL)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pitfall 5: Foundation module is not on sys.path"
    - federated-baseline-cf/pyproject.toml (current dependencies list and format)
    - federated-pfedrec/pyproject.toml
    - federated-personalized-cf/pyproject.toml
    - federated-adaptive-personalized-cf/pyproject.toml
    - docs/setup.md (install order doc from Plan 01)
  </read_first>
  <behavior>
    - Each of the four modules' `pyproject.toml` adds the line `"fedrec-foundation",` to its `[project] dependencies` list.
    - PEP 440 direct-reference SYNTAX CHOSEN: plain name `fedrec-foundation` (no URL). This relies on the user having done `pip install -e scripts/foundation/` first — documented in `docs/setup.md`. This is the simpler, template-compatible choice and is consistent with how the four modules today are installed.
    - The docstring-or-comment above the dependency list in each pyproject.toml clarifies: `# fedrec-foundation: install via `pip install -e ../scripts/foundation/` BEFORE installing this module (see docs/setup.md).`
    - No other fields in pyproject.toml change. Flower-template fields (`[tool.flwr.app.components]`, federation config, etc.) stay untouched.
  </behavior>
  <action>
For each of the four files, use the Edit tool to insert `"fedrec-foundation",` as a new line into the existing `[project] dependencies = [...]` list. Place the line ABOVE `"flwr[simulation]>=1.22.0",` so it reads as the package's most fundamental dependency. Add a comment line `# Foundation contract (Phase 1) — install ../scripts/foundation/ in editable mode FIRST; see docs/setup.md` immediately above the new dependency line.

Concretely, each of the four pyproject.toml dependencies sections changes from:
```toml
dependencies = [
    "flwr[simulation]>=1.22.0",
    ...
]
```
to:
```toml
dependencies = [
    # Foundation contract (Phase 1) — install ../scripts/foundation/ in editable mode FIRST; see docs/setup.md
    "fedrec-foundation",
    "flwr[simulation]>=1.22.0",
    ...
]
```

Do this exactly once per module. Do not modify any other dependencies or the `[tool.flwr.*]` sections.

Also update `docs/setup.md` to make explicit that the plain `fedrec-foundation` dep-name requires `pip install -e scripts/foundation/` before each `pip install -e federated-*-cf/` — reinforcing what Plan 01's doc already said.
  </action>
  <verify>
    <automated>grep -c "^ *.fedrec-foundation." federated-baseline-cf/pyproject.toml federated-pfedrec/pyproject.toml federated-personalized-cf/pyproject.toml federated-adaptive-personalized-cf/pyproject.toml</automated>
  </verify>
  <acceptance_criteria>
    - `grep "fedrec-foundation" federated-baseline-cf/pyproject.toml` matches.
    - `grep "fedrec-foundation" federated-pfedrec/pyproject.toml` matches.
    - `grep "fedrec-foundation" federated-personalized-cf/pyproject.toml` matches.
    - `grep "fedrec-foundation" federated-adaptive-personalized-cf/pyproject.toml` matches.
    - Each file still contains `flwr[simulation]>=1.22.0` (unchanged).
    - Each file still contains `[tool.flwr.app.components]` (unchanged).
  </acceptance_criteria>
  <done>All four modules declare fedrec-foundation; no other pyproject.toml content changed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add cross-module import smoke test + reinstall + verify</name>
  <files>
    scripts/foundation/tests/test_integration.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-VALIDATION.md (smoke-test row `test_cross_module_imports`)
    - docs/setup.md (install order)
    - Each of the four module dirs' `claude.md` to verify current installability (read the top-level `flwr run .` command section)
  </read_first>
  <behavior>
    - Add a new test `test_cross_module_imports` to `scripts/foundation/tests/test_integration.py` that:
      1. Detects whether `fedrec_foundation` is importable at all.
      2. For each of the four module directories, verifies that a subprocess Python process running with `cwd=<module_dir>` can import each of: `fedrec_foundation`, `fedrec_foundation.mapping`, `fedrec_foundation.split`, `fedrec_foundation.exclusion`, `fedrec_foundation.evaluator`, `fedrec_foundation.weight_policy`, `fedrec_foundation.fit_metrics`, `fedrec_foundation.rng`, `fedrec_foundation.manifest`, `fedrec_foundation.mode`.
      3. Uses `pytest.skip(...)` if the repo's four module dirs aren't present (graceful skip when run in a minimal clone).
    - The test does NOT trigger `pip install` — that is the user/CI's job per `docs/setup.md`. The test simply verifies the foundation is reachable from a subprocess rooted at each module dir, which is what `flwr run .` will effectively do.
    - After the test passes, ALSO execute `pip install -e scripts/foundation/ && for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do pip install -e "$m"; done` once from the repo root to materialize the install state on this machine for downstream phases.
  </behavior>
  <action>
Append to `scripts/foundation/tests/test_integration.py` (keep the existing tests from Plan 02 intact):

```python
# --- Cross-module import smoke test (Plan 06 / IMP-1) ---

import subprocess
import sys
from pathlib import Path

import pytest


_MODULES = (
    "federated-baseline-cf",
    "federated-pfedrec",
    "federated-personalized-cf",
    "federated-adaptive-personalized-cf",
)

_FOUNDATION_SUBMODULES = (
    "fedrec_foundation",
    "fedrec_foundation.mapping",
    "fedrec_foundation.split",
    "fedrec_foundation.exclusion",
    "fedrec_foundation.evaluator",
    "fedrec_foundation.weight_policy",
    "fedrec_foundation.fit_metrics",
    "fedrec_foundation.rng",
    "fedrec_foundation.manifest",
    "fedrec_foundation.mode",
)


def _repo_root() -> Path:
    """Walk up from this test to the repo root (containing data/ml-1m)."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "data" / "ml-1m").exists():
            return p
    pytest.skip("Repo root with data/ml-1m not located")


@pytest.mark.parametrize("module_dir", _MODULES)
def test_cross_module_imports(module_dir: str) -> None:
    """Each of the four federated modules can import every foundation submodule.

    Runs a subprocess with cwd set to the module's directory to mirror
    `flwr run .` behavior. Requires that the user has already run
    `pip install -e scripts/foundation/` (documented in docs/setup.md).
    """
    root = _repo_root()
    mod_path = root / module_dir
    if not mod_path.exists():
        pytest.skip(f"{mod_path} not present")
    import_stmts = "; ".join(f"import {m}" for m in _FOUNDATION_SUBMODULES)
    script = f"{import_stmts}; print('ok')"
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(mod_path),
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, (
        f"Cross-module import failed in {module_dir}:\n"
        f"STDOUT={r.stdout!r}\nSTDERR={r.stderr!r}\n"
        f"Hint: run `pip install -e scripts/foundation/` and "
        f"`pip install -e {module_dir}/` (see docs/setup.md)."
    )
    assert "ok" in r.stdout


def test_pyproject_declares_foundation_dep() -> None:
    """IMP-1: each module's pyproject.toml declares fedrec-foundation as a dep."""
    root = _repo_root()
    for mod in _MODULES:
        pyproject = root / mod / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip(f"{pyproject} not present")
        content = pyproject.read_text()
        assert "fedrec-foundation" in content, (
            f"{mod}/pyproject.toml missing fedrec-foundation dependency"
        )
```

After the test file is updated, execute the documented install sequence once so the test actually passes on this machine:
```
pip install -e scripts/foundation/
pip install -e federated-baseline-cf/
pip install -e federated-pfedrec/
pip install -e federated-personalized-cf/
pip install -e federated-adaptive-personalized-cf/
```
Then run `cd scripts/foundation && pytest tests/test_integration.py::test_cross_module_imports tests/test_integration.py::test_pyproject_declares_foundation_dep -v`.
  </action>
  <verify>
    <automated>pip install -e scripts/foundation/ &amp;&amp; for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do pip install -e "$m"; done &amp;&amp; cd scripts/foundation &amp;&amp; pytest tests/test_integration.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/foundation/tests/test_integration.py` now defines `test_cross_module_imports` (parametrized across 4 modules) and `test_pyproject_declares_foundation_dep`.
    - After running the install sequence, `for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do (cd "$m" && python -c "import fedrec_foundation" ); done` prints no errors.
    - `cd scripts/foundation && pytest tests/test_integration.py -v` prints all tests passing (including the new 4 parametrized imports + pyproject dep test + the Plan 02 bundle tests + the 6040/3706 empirical anchor).
  </acceptance_criteria>
  <done>Every one of the four federated modules can import every fedrec_foundation submodule from its own directory; pyproject.toml of each module declares the foundation dependency.</done>
</task>

</tasks>

<verification>
- `grep "fedrec-foundation" federated-*-cf/pyproject.toml` lists all four files.
- `for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do (cd "$m" && python -c "import fedrec_foundation"); done` prints nothing (no errors).
- `cd scripts/foundation && pytest tests/ -v` — full suite green, no skipped tests remain for the implemented modules (FND-01..07, mode, launcher, bundle).
- `python -c "from fedrec_foundation.bundle import verify_bundle; from pathlib import Path; verify_bundle(Path('data/derived'))"` succeeds — foundation artifacts on disk check out.
</verification>

<success_criteria>
- Every federated module's pyproject.toml declares `fedrec-foundation` as a dependency.
- Every federated module's working directory can `import fedrec_foundation.*` in a subprocess.
- Foundation-contract Phase 1 is COMPLETE: all 7 FND requirements + 5 Codex CRITICAL items + 4 Codex IMPORTANT items + 3 NIT items are addressed and test-verified.
- `cd scripts/foundation && pytest tests/ -v` finishes green; no SKIPPED tests left for implemented modules.
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation-contract/01-foundation-contract-06-SUMMARY.md` — list the four modified pyproject.toml files, the install-order (foundation first, then modules), and confirm the cross-module smoke test passes. Note that this SUMMARY closes Phase 1; any Phase 2-5 `pip install -e .` now auto-pulls `fedrec_foundation` without sys.path manipulation.
</output>
