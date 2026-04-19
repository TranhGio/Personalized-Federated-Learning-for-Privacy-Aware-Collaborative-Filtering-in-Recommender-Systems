"""Tests for scripts/run.py launcher (Codex CR-2)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _launcher_path() -> Path:
    """Locate scripts/run.py at the repo root."""
    # This test file lives at scripts/foundation/tests/test_launcher.py;
    # repo root is three parents up.
    return Path(__file__).resolve().parents[3] / "scripts" / "run.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_launcher_path()), *args],
        capture_output=True, text=True, check=False,
    )


def test_launcher_exists() -> None:
    assert _launcher_path().exists(), f"{_launcher_path()} missing"


def test_launcher_sets_mode_benchmark() -> None:
    r = _run("--dry-run", "baseline", "benchmark_cross_device")
    assert r.returncode == 0, r.stderr
    # Mode string values must be TOML-quoted so flwr's tomli parser accepts them.
    assert 'mode="benchmark_cross_device"' in r.stdout
    assert "federated-baseline-cf" in r.stdout
    # Regression: num-supernodes must NOT appear in --run-config (it's a
    # federation-level option, and flwr's fuse_dicts rejects run-config keys
    # that aren't present in [tool.flwr.app.config]).
    assert "num-supernodes" not in r.stdout


def test_launcher_sets_mode_cross_silo_legacy() -> None:
    r = _run("--dry-run", "pfedrec", "cross_silo_legacy")
    assert r.returncode == 0, r.stderr
    assert 'mode="cross_silo_legacy"' in r.stdout
    assert "federated-pfedrec" in r.stdout
    assert "num-supernodes" not in r.stdout


def test_launcher_paper_compat_pfedrec() -> None:
    r = _run("--dry-run", "pfedrec", "paper_compat_pfedrec")
    assert r.returncode == 0, r.stderr
    assert 'mode="paper_compat_pfedrec"' in r.stdout
    assert "num-supernodes" not in r.stdout


def test_launcher_passes_extra_run_config() -> None:
    r = _run(
        "--dry-run", "adaptive", "benchmark_cross_device",
        "--run-config", "run-seed=999", "--run-config", "lr=0.005",
    )
    assert r.returncode == 0, r.stderr
    assert "run-seed=999" in r.stdout
    assert "lr=0.005" in r.stdout
    assert 'mode="benchmark_cross_device"' in r.stdout


def test_launcher_unknown_mode_rejected() -> None:
    r = _run("--dry-run", "baseline", "not_a_mode")
    assert r.returncode != 0
    assert "invalid choice" in r.stderr.lower() or "invalid" in r.stderr.lower()


def test_launcher_malformed_run_config_rejected() -> None:
    r = _run(
        "--dry-run", "baseline", "benchmark_cross_device",
        "--run-config", "no_equals_sign",
    )
    assert r.returncode != 0
