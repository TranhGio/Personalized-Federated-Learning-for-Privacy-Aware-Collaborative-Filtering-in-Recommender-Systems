"""Phase 5 PFR-01 regression guard: cross-device defaults in pyproject.toml.

Mirrors the Phase-3/Phase-4 Plan-02 in-file regression guard pattern
(``federated-personalized-cf/tests/test_dataset_adapter.py`` predecessor;
``federated-adaptive-personalized-cf/tests/test_pyproject_shape.py`` clone).

D-25 + PFR-01 invariants pinned here:
    - num-supernodes = 6040 in BOTH local-simulation and local-sim-gpu blocks
    - partition-mode = "natural" (D-09 cross-silo FROZEN at the dataset layer)
    - mode = "paper_compat_pfedrec" (D-05: only PFedRec mode shipped)
    - weight-policy = "uniform" (D-24, matches engine.py:81 reference)
    - run-seed = 42 (FND-06 single source of truth)
    - reuse-cache = false (D-18 default; opt-in via --run-config)
    - eval-num-negatives = 99 (NCF protocol, FND-04)
    - checkpoint-rule keys present (best_round_restore or best_round)
    - [project.optional-dependencies] dev includes pytest>=7.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    """Parse the federated-pfedrec/pyproject.toml once per module."""
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_num_supernodes_6040(cfg) -> None:
    """PFR-01: BOTH federation blocks declare num-supernodes=6040."""
    fed = cfg["tool"]["flwr"]["federations"]
    assert fed["local-simulation"]["options"]["num-supernodes"] == 6040
    assert fed["local-sim-gpu"]["options"]["num-supernodes"] == 6040


def test_partition_mode_natural(cfg) -> None:
    """PFR-01 + D-25 contract keys present.

    Asserts the 6 Phase-5 contract keys plus partition-mode = "natural".
    """
    app_cfg = cfg["tool"]["flwr"]["app"]["config"]
    assert app_cfg["partition-mode"] == "natural"
    assert app_cfg["mode"] == "paper_compat_pfedrec"
    assert app_cfg["weight-policy"] == "uniform"
    assert app_cfg["run-seed"] == 42
    assert app_cfg["reuse-cache"] is False
    assert app_cfg["eval-num-negatives"] == 99
    assert app_cfg["checkpoint-rule"] in ("best_round_restore", "best_round")


def test_dev_extra_pytest_present(cfg) -> None:
    """[project.optional-dependencies] dev = ['pytest>=7.0'] (Wave-1 dev pytest dep)."""
    deps = cfg["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("pytest>=7.0") for d in deps), deps
