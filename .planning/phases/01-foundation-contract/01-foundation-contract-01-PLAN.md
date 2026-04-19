---
phase: 01-foundation-contract
plan: 01
type: execute
wave: 0
depends_on: []
files_modified:
  - scripts/foundation/pyproject.toml
  - scripts/foundation/fedrec_foundation/__init__.py
  - scripts/foundation/fedrec_foundation/paths.py
  - scripts/foundation/fedrec_foundation/atomic.py
  - scripts/foundation/fedrec_foundation/hashing.py
  - scripts/foundation/tests/__init__.py
  - scripts/foundation/tests/conftest.py
  - scripts/foundation/tests/test_hashing.py
  - scripts/foundation/tests/test_mapping.py
  - scripts/foundation/tests/test_split.py
  - scripts/foundation/tests/test_exclusion.py
  - scripts/foundation/tests/test_evaluator.py
  - scripts/foundation/tests/test_weight_policy.py
  - scripts/foundation/tests/test_rng.py
  - scripts/foundation/tests/test_mode.py
  - scripts/foundation/tests/test_manifest.py
  - scripts/foundation/tests/test_launcher.py
  - scripts/foundation/tests/test_integration.py
  - docs/setup.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "pytest is installable and discovers the scripts/foundation/tests/ directory"
    - "`pip install -e scripts/foundation/` succeeds in a clean environment"
    - "`python -c 'import fedrec_foundation'` succeeds after editable install"
    - "Every FND-01..07 has at least one Wave-0 test stub in SKIPPED state (waiting for Plans 02-05 to unblock), plus at least one real passing test for the modules Plan 01 implements (hashing)"
  artifacts:
    - path: "scripts/foundation/pyproject.toml"
      provides: "Installable fedrec-foundation package declaration + pytest config"
      contains: "name = \"fedrec-foundation\""
    - path: "scripts/foundation/fedrec_foundation/__init__.py"
      provides: "Package marker with __version__"
    - path: "scripts/foundation/fedrec_foundation/paths.py"
      provides: "Repo-root + data/derived/ + data/ml-1m/ path resolution"
    - path: "scripts/foundation/fedrec_foundation/atomic.py"
      provides: "atomic_write_json helper"
    - path: "scripts/foundation/fedrec_foundation/hashing.py"
      provides: "compute_raw_data_hash + sha256-file helper"
    - path: "scripts/foundation/tests/conftest.py"
      provides: "Shared fixtures: synthetic ratings_df, tmp_derived_dir, PYTHONHASHSEED fixture"
  key_links:
    - from: "scripts/foundation/pyproject.toml"
      to: "[tool.hatch.build.targets.wheel] packages = [\"fedrec_foundation\"]"
      via: "hatchling build backend"
      pattern: "packages = \\[\"fedrec_foundation\"\\]"
    - from: "scripts/foundation/pyproject.toml"
      to: "[tool.pytest.ini_options]"
      via: "pytest auto-discovery"
      pattern: "testpaths = \\[\"tests\"\\]"
---

<objective>
Create the `fedrec-foundation` installable package scaffold and Wave-0 test infrastructure so every downstream plan has a SKIPPED → GREEN TDD loop. This plan produces zero FND requirements on its own — it is the gate that unlocks Plans 02-05.

Purpose: Plans 02-05 implement FND-01..07. They need a real `pytest` setup, a real installable package, and SKIPPED test stubs enumerating the expected behaviors (each later plan flips its `pytestmark = pytest.mark.skip(...)` off and replaces the stub body). Building all three artifacts in one Wave-0 plan avoids any per-requirement plan having to bootstrap its own test harness (which would make Plans 02-05 slower and less focused).

Note on wave label: this plan uses `wave: 0` frontmatter to match the "Wave 0" narrative label used in ROADMAP and VALIDATION.md — bootstrap/setup work that unblocks Wave 1+ parallel plans.

Note on file count: `files_modified` lists 16 files — above the 15-file soft threshold — but ~13 of them are near-identical test stubs (boilerplate). Splitting test-stub generation into a separate plan adds ceremony without reducing cognitive load, so keeping the package scaffold + test stubs together is the deliberate choice.

Output: An installable `fedrec-foundation` package with real paths/atomic/hashing modules, a complete `tests/` directory with 14 test files containing SKIPPED stubs for every behavior in `01-VALIDATION.md`, and a `docs/setup.md` documenting install order.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/01-foundation-contract/01-CONTEXT.md
@.planning/phases/01-foundation-contract/01-RESEARCH.md
@.planning/phases/01-foundation-contract/01-VALIDATION.md
@CLAUDE.md
@.planning/codebase/CONVENTIONS.md

<interfaces>
<!-- Package skeleton. Plans 02-05 implement the stubs; this plan only lays out the import graph. -->

From scripts/foundation/fedrec_foundation/__init__.py (Plan 01 creates this file with only __version__):
```python
__version__ = "0.1.0"
```

From scripts/foundation/fedrec_foundation/paths.py (Plan 01 implements this fully):
```python
from pathlib import Path

def repo_root() -> Path: ...        # walk up from __file__ until finding 'data/ml-1m'
def data_derived() -> Path: ...     # repo_root() / "data" / "derived"
def ml1m_dir() -> Path: ...         # repo_root() / "data" / "ml-1m"
```

From scripts/foundation/fedrec_foundation/atomic.py (Plan 01 implements this fully):
```python
def atomic_write_json(path: str, data: object) -> None: ...
def _json_default(obj): ...   # numpy scalars + Path -> JSON-safe
```

From scripts/foundation/fedrec_foundation/hashing.py (Plan 01 implements this fully):
```python
from pathlib import Path
def sha256_file(path: Path) -> str: ...
def compute_raw_data_hash(ml1m_dir: Path) -> str: ...  # ratings.dat || movies.dat || users.dat
```

From scripts/foundation/pyproject.toml (schema):
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedrec-foundation"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.hatch.build.targets.wheel]
packages = ["fedrec_foundation"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Scaffold fedrec-foundation package + install-order doc</name>
  <files>
    scripts/foundation/pyproject.toml
    scripts/foundation/fedrec_foundation/__init__.py
    scripts/foundation/fedrec_foundation/paths.py
    scripts/foundation/fedrec_foundation/atomic.py
    scripts/foundation/fedrec_foundation/hashing.py
    docs/setup.md
  </files>
  <read_first>
    - CLAUDE.md (notation convention, Python 3.9+ typing style, absolute-imports rule, NumPy-style docstrings)
    - .planning/phases/01-foundation-contract/01-CONTEXT.md (D-01 data/derived/ location, D-05 no-pickle rule)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Recommended Project Structure" AND §"Pattern 9: Atomic Write Helper" AND CODEX PEER REVIEW §IMP-1 (editable-install + local-path dependency)
    - .planning/codebase/CONVENTIONS.md (typing.Dict style, hatchling packaging, no-pickle)
    - /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf/pyproject.toml (reference for hatchling build layout, version pins baseline modules already use)
  </read_first>
  <behavior>
    - `scripts/foundation/pyproject.toml` declares `name = "fedrec-foundation"`, version `0.1.0`, hatchling build backend, wheel target `packages = ["fedrec_foundation"]`, dependencies `numpy>=1.24.0` + `pandas>=2.0.0`, optional dev dep `pytest>=7.0`, and `[tool.pytest.ini_options] testpaths = ["tests"]`.
    - `fedrec_foundation/__init__.py` contains only `__version__ = "0.1.0"`.
    - `fedrec_foundation/paths.py` exports `repo_root()`, `data_derived()`, `ml1m_dir()` — each returns a `pathlib.Path`. `repo_root()` walks up from `__file__` until it finds a directory containing `data/ml-1m/`. Env-var override `FEDREC_FOUNDATION_DATA_DIR` short-circuits `data_derived()` if set.
    - `fedrec_foundation/atomic.py` exports `atomic_write_json(path: str, data: object) -> None` that writes via `tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".json")` then `os.replace`. Sorts keys. Handles numpy scalars + `Path` objects via `_json_default`.
    - `fedrec_foundation/hashing.py` exports `sha256_file(path: Path) -> str` (hex digest, 65536-byte chunks) and `compute_raw_data_hash(ml1m_dir: Path) -> str` (byte-concat hash of `ratings.dat || movies.dat || users.dat` in exactly that order).
    - `docs/setup.md` documents the install order: `pip install pytest`, then `pip install -e scripts/foundation/`, then `for m in federated-*-cf; do pip install -e "$m"; done` — and notes that Plan 06 adds `fedrec-foundation` as a local-path dep to each module so downstream installs pull it automatically.
  </behavior>
  <action>
Create the installable package skeleton. Use these EXACT file contents (typing per CLAUDE.md: `typing.Dict`, `typing.Optional`, etc.):

**`scripts/foundation/pyproject.toml`:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fedrec-foundation"
version = "0.1.0"
description = "Shared cross-device FedRec protocol contract (ID mapping, LOO split, exclusion set, evaluator selector, weight policy, four-tier RNG, mode resolver, run manifest) for the movie-recommendation-system thesis."
license = "Apache-2.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.hatch.build.targets.wheel]
packages = ["fedrec_foundation"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

**`scripts/foundation/fedrec_foundation/__init__.py`:**
```python
"""fedrec-foundation: shared cross-device FedRec protocol contract."""

__version__ = "0.1.0"
```

**`scripts/foundation/fedrec_foundation/paths.py`:**
```python
"""Path helpers for foundation artifacts and raw data.

Locates the repo root by walking up from this file until a directory
containing ``data/ml-1m/`` is found. This keeps the foundation module
usable from any cwd (Flower subprocesses may chdir).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_OVERRIDE = "FEDREC_FOUNDATION_DATA_DIR"


def repo_root() -> Path:
    """Return the repo root by walking up from this file.

    Returns
    -------
    pathlib.Path
        The first ancestor directory that contains ``data/ml-1m``.

    Raises
    ------
    RuntimeError
        If no such ancestor exists.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "data" / "ml-1m").exists():
            return parent
    raise RuntimeError(
        f"Could not locate repo root from {here}. Expected an ancestor "
        f"containing data/ml-1m/."
    )


def data_derived() -> Path:
    """Return the data/derived/ directory (override via env var).

    Env var ``FEDREC_FOUNDATION_DATA_DIR`` overrides the default for CI
    or remote environments.
    """
    override: Optional[str] = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)
    return repo_root() / "data" / "derived"


def ml1m_dir() -> Path:
    """Return the data/ml-1m/ directory (not overridable)."""
    return repo_root() / "data" / "ml-1m"
```

**`scripts/foundation/fedrec_foundation/atomic.py`:**
Copy verbatim the `atomic_write_json` + `_json_default` block from `01-RESEARCH.md` §"Pattern 9: Atomic Write Helper" (lines 1131-1173 of the research file). Use `typing` imports where needed; NumPy-style docstring on `atomic_write_json`.

**`scripts/foundation/fedrec_foundation/hashing.py`:**
```python
"""SHA-256 hashing helpers for foundation artifacts (FND-02, FND-07)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a single file read in 65536-byte chunks.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase 64-character hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_raw_data_hash(ml1m_dir: Path) -> str:
    """SHA-256 of ratings.dat || movies.dat || users.dat (in that order).

    The concatenation order is LOCKED — changing it changes the raw-data
    fingerprint for every committed artifact.

    Parameters
    ----------
    ml1m_dir : pathlib.Path
        Directory containing the three ML-1M .dat files.

    Returns
    -------
    str
        Lowercase 64-character hex digest.
    """
    h = hashlib.sha256()
    for fname in ("ratings.dat", "movies.dat", "users.dat"):
        with open(ml1m_dir / fname, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()
```

**`docs/setup.md`:**
```markdown
# Development Environment Setup

## Install order (REQUIRED)

The shared `fedrec-foundation` package MUST be installed before any of the four federated modules. Each module's `pyproject.toml` declares it as a local-path dependency, but editable-install semantics require foundation to exist on disk first.

```bash
# 1. Test framework (one-off)
pip install pytest

# 2. Shared foundation package
pip install -e scripts/foundation/

# 3. The four federated modules (any order, all four)
for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do
  pip install -e "$m"
done

# 4. Smoke test
for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do
  (cd "$m" && python -c "import fedrec_foundation; print(fedrec_foundation.__version__)")
done
```

## Running foundation tests

```bash
cd scripts/foundation && pytest -x tests/         # fail-fast quick run
cd scripts/foundation && pytest tests/ -v         # full suite
```
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pip install -e . &amp;&amp; python -c "import fedrec_foundation; from fedrec_foundation.paths import repo_root, data_derived, ml1m_dir; from fedrec_foundation.atomic import atomic_write_json; from fedrec_foundation.hashing import sha256_file, compute_raw_data_hash; print(fedrec_foundation.__version__)"</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/pyproject.toml` exists and grep `name = "fedrec-foundation"` matches.
    - File `scripts/foundation/pyproject.toml` contains `packages = ["fedrec_foundation"]`.
    - File `scripts/foundation/pyproject.toml` contains `testpaths = ["tests"]`.
    - File `scripts/foundation/fedrec_foundation/__init__.py` contains `__version__ = "0.1.0"`.
    - File `scripts/foundation/fedrec_foundation/paths.py` defines functions `repo_root`, `data_derived`, `ml1m_dir`.
    - File `scripts/foundation/fedrec_foundation/atomic.py` defines `atomic_write_json`.
    - File `scripts/foundation/fedrec_foundation/hashing.py` defines `sha256_file` and `compute_raw_data_hash`.
    - File `docs/setup.md` contains the literal string `pip install -e scripts/foundation/`.
    - Running `pip install -e scripts/foundation/` then `python -c "import fedrec_foundation"` prints `0.1.0` with no error.
  </acceptance_criteria>
  <done>Package installs cleanly; paths/atomic/hashing modules importable.</done>
</task>

<task type="auto">
  <name>Task 2: Create Wave-0 test stubs (14 test files + conftest)</name>
  <files>
    scripts/foundation/tests/__init__.py
    scripts/foundation/tests/conftest.py
    scripts/foundation/tests/test_hashing.py
    scripts/foundation/tests/test_mapping.py
    scripts/foundation/tests/test_split.py
    scripts/foundation/tests/test_exclusion.py
    scripts/foundation/tests/test_evaluator.py
    scripts/foundation/tests/test_weight_policy.py
    scripts/foundation/tests/test_rng.py
    scripts/foundation/tests/test_mode.py
    scripts/foundation/tests/test_manifest.py
    scripts/foundation/tests/test_launcher.py
    scripts/foundation/tests/test_integration.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-VALIDATION.md (the exact pytest test-ID list — this task converts that table into test file stubs)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Validation Architecture" (pytest test layout + cross-process subprocess test body for RNG)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §"Codex's Added Validation Requirements"
    - CLAUDE.md (typing style: `typing.Dict` not `dict[str, int]`)
  </read_first>
  <action>
Create the complete test tree. Every test file (except `test_hashing.py` which Plan 01 implements) starts in SKIPPED state via a module-level `pytestmark = pytest.mark.skip(reason="Plan NN implements fedrec_foundation.<module>")`. Plans 02-05 flip the skip off and replace the stub bodies with real assertions. SKIPPED tests show up as `s` in pytest output — NOT as failures — but `pytest --collect-only` still enumerates every test ID for contract-level visibility.

**`tests/__init__.py`:** empty.

**`tests/conftest.py`:**
```python
"""Shared fixtures for foundation tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator

import pandas as pd
import pytest


@pytest.fixture
def synthetic_ratings_df() -> pd.DataFrame:
    """Tiny deterministic ratings DataFrame for unit tests.

    5 users, 4 items, 12 interactions — small enough to hand-verify.
    Columns match MovieLens-1M: user_id, movie_id, rating, timestamp.
    """
    rows = [
        # (user_id, movie_id, rating, timestamp)
        (1, 10, 5, 1000),
        (1, 20, 4, 1001),
        (1, 30, 3, 1002),
        (2, 10, 5, 2000),
        (2, 40, 4, 2001),
        (3, 20, 3, 3000),
        (3, 30, 4, 3001),
        (3, 40, 5, 3002),
        (4, 10, 2, 4000),
        (4, 20, 5, 4001),
        (5, 30, 4, 5000),
        (5, 40, 3, 5001),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


@pytest.fixture
def tmp_derived_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp data/derived/ directory and point the env override at it."""
    derived = tmp_path / "derived"
    derived.mkdir()
    monkeypatch.setenv("FEDREC_FOUNDATION_DATA_DIR", str(derived))
    return derived


@pytest.fixture
def pythonhashseed_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force PYTHONHASHSEED=random for the current process.

    Note: PYTHONHASHSEED is read at interpreter startup; setting it here
    only affects CHILD subprocesses spawned after the fixture runs.
    Use this in combination with subprocess.run tests.
    """
    monkeypatch.setenv("PYTHONHASHSEED", "random")
```

**`tests/test_hashing.py`** (this module IS implemented by Plan 01 — tests must be GREEN, not skipped):
```python
"""Tests for fedrec_foundation.hashing (implemented in Plan 01)."""
from __future__ import annotations

from pathlib import Path

from fedrec_foundation.hashing import sha256_file, compute_raw_data_hash


def test_sha256_file_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_raw_data_hash_order_matters(tmp_path: Path) -> None:
    (tmp_path / "ratings.dat").write_bytes(b"R")
    (tmp_path / "movies.dat").write_bytes(b"M")
    (tmp_path / "users.dat").write_bytes(b"U")
    h = compute_raw_data_hash(tmp_path)
    assert len(h) == 64
    # Flip one file; hash must change.
    (tmp_path / "ratings.dat").write_bytes(b"X")
    h2 = compute_raw_data_hash(tmp_path)
    assert h != h2
```

For every OTHER test file (`test_mapping.py`, `test_split.py`, `test_exclusion.py`, `test_evaluator.py`, `test_weight_policy.py`, `test_rng.py`, `test_mode.py`, `test_manifest.py`, `test_launcher.py`, `test_integration.py`), write a stub file with:
1. A module-level `pytestmark = pytest.mark.skip(reason="Plan NN implements fedrec_foundation.<module>")` — replace NN with the plan number per the map below. This produces SKIPPED (not FAILED) — the intent is that `pytest --collect-only` enumerates all test IDs while runs stay green.
2. ONE test function per row in the VALIDATION.md per-task map (names MUST match the automated-command column of that map so the table updates in-place).
3. Each test body contains `raise NotImplementedError("Plan NN fills this in")` — safe to leave because the module-level skip prevents execution. Plans 02-05 delete the skip marker and replace the body in the same edit.

Plan-to-file map:
- `test_mapping.py`: Plan 02 — tests `test_sort_order`, `test_item_mapping_from_ratings_only`, `test_roundtrip` (cover FND-01-a/b/c)
- `test_split.py`: Plan 02 — tests `test_hash_deterministic`, `test_timestamp_tiebreak`, `test_split_lock_refuses_overwrite`, `test_train_only_user_stats` (FND-02-a/b/c/d)
- `test_exclusion.py`: Plan 02 — tests `test_includes_test_item`, `test_safe_load`, `test_indptr_layout`, `test_module_level_exclusion_for` (FND-03-a/b/c + CR-3 module-level helper coverage)
- `test_evaluator.py`: Plan 03 — tests `test_primary_evaluator_all_modes` (FND-04-a)
- `test_weight_policy.py`: Plan 03 — tests `test_num_positives`, `test_unknown_policy_raises`, `test_fit_metrics_contract`, `test_from_dict_missing_required_raises` (FND-05-a/b/c + CR-4 missing-key error handling)
- `test_rng.py`: Plan 04 — tests `test_derive_rng_stable_across_processes`, `test_tuple_uniqueness`, `test_all_three_rng_factories`, `test_torch_generator_reproducible`, `test_sample_reproducible` (FND-06-a/b/c/d/e)
- `test_mode.py`: Plan 05 — tests `test_override_logging`, `test_assertion_flags` (mode-a/b)
- `test_launcher.py`: Plan 05 — tests `test_launcher_sets_num_supernodes` (mode-c)
- `test_manifest.py`: Plan 04 — tests `test_all_fields_populated`, `test_both_writes`, `test_composite_foundation_hash` (FND-07-a/b/c)
- `test_integration.py`: Plan 02 (bundle) + Plan 06 (cross-module import) — tests `test_build_idempotent`, `test_bundle_atomic_publication`, `test_build_creates_all_artifacts`, `test_ml1m_counts_6040_3706` (bundle-a/b + build-e2e + empirical-a)

CRITICAL: `test_rng.py::test_derive_rng_stable_across_processes` body MUST be the subprocess-varying-PYTHONHASHSEED test from `01-RESEARCH.md` lines 800-820 (even though it is skipped now, copy the real body so Plan 04 only has to flip `pytest.mark.skip` to pass it). Keep all other test bodies as `raise NotImplementedError("Plan NN fills this in")`.

Each test file starts with:
```python
"""Tests for fedrec_foundation.<module> (implemented in Plan NN)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan NN implements fedrec_foundation.<module>")
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/ --collect-only -q 2>&amp;1 | tail -5 &amp;&amp; pytest tests/test_hashing.py -v</automated>
  </verify>
  <acceptance_criteria>
    - Files `scripts/foundation/tests/{__init__,conftest,test_hashing,test_mapping,test_split,test_exclusion,test_evaluator,test_weight_policy,test_rng,test_mode,test_manifest,test_launcher,test_integration}.py` all exist (13 files in tests/).
    - `pytest tests/ --collect-only -q` collects AT LEAST 25 tests (count from VALIDATION.md map). Collection must succeed even though most are skipped.
    - `pytest tests/test_hashing.py -v` prints `2 passed` (hashing tests are real, not skipped).
    - `pytest tests/test_mapping.py -v` reports all tests as SKIPPED with reason mentioning "Plan 02" (pytest exits 0; `s` markers in output).
    - `pytest tests/test_rng.py -v` reports all tests as SKIPPED with reason mentioning "Plan 04".
    - `pytest tests/` (full run) exits 0 with some PASSED (hashing) and the rest SKIPPED — no FAILED.
    - `grep -r "PYTHONHASHSEED" scripts/foundation/tests/test_rng.py` finds at least one match (cross-process subprocess test body present even though skipped).
    - `conftest.py` defines fixtures named `synthetic_ratings_df`, `tmp_derived_dir`, `pythonhashseed_random`.
  </acceptance_criteria>
  <done>pytest discovers all tests; test_hashing passes; other tests are SKIPPED waiting for their plan.</done>
</task>

</tasks>

<verification>
- Run `cd scripts/foundation && pip install -e .` — editable install succeeds.
- Run `python -c "import fedrec_foundation; from fedrec_foundation.atomic import atomic_write_json; from fedrec_foundation.hashing import compute_raw_data_hash"` — all imports succeed.
- Run `cd scripts/foundation && pytest tests/ -v` — test_hashing.py has 2 passes; all other tests are SKIPPED with "Plan NN" reason; overall exit 0.
- Run `cd scripts/foundation && pytest tests/ --collect-only -q | tail -1` — shows at least 25 tests collected.
- Grep `docs/setup.md` contains the install-order commands.
</verification>

<success_criteria>
- `fedrec-foundation` is pip-installable as an editable package.
- `pytest` discovers the tests directory and runs a passing test_hashing.py.
- Every FND-01..07 has at least one named SKIPPED test stub that Plans 02-05 will unblock and populate with real assertions.
- `pytest --collect-only` enumerates all ~25+ tests even though most are skipped — VALIDATION.md's test-ID map is enforced at collection time.
- Install-order documented in `docs/setup.md` (foundation before modules).
- No code in Plans 02-05 is blocked on package scaffolding.
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation-contract/01-foundation-contract-01-SUMMARY.md` per the standard summary template — enumerate the 14 test files, the 3 real source files (paths/atomic/hashing), and the install-order doc. List what's left as SKIPPED stubs (mapping, split, exclusion, evaluator, weight_policy, rng, mode, manifest) so subsequent plans know where to pick up and which `pytestmark` line to delete.
</output>
</content>
</invoke>