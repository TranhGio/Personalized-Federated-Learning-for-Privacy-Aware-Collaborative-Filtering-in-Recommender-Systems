"""Tests for fedrec_foundation.paths -- Phase 6 module_run_results_dir helper."""
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
    path = None
    try:
        path = module_run_results_dir("baseline", run_id)
        assert path.is_absolute(), f"Expected absolute path, got {path!r}"
        assert path == repo_root() / "results" / "federated" / "baseline" / run_id, (
            f"D-02 anchoring broken: got {path!r}"
        )
        assert path.is_dir(), f"Expected directory to exist after call, got {path!r}"
    finally:
        # Cleanup so the test does not pollute the real repo results/ tree.
        if path is not None and path.exists():
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
