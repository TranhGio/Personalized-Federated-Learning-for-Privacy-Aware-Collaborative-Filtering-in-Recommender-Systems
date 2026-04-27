"""ADP-01 regression: federated-adaptive-personalized-cf/pyproject.toml cross-device defaults.

Asserts the 11 cross-device-defining keys are present with the expected values:
- 6 Phase-3 carry-forward (mode, run-seed, weight-policy, eval-num-negatives,
  checkpoint-rule, reuse-cache)
- 5 Phase-4 signature-driving (alpha-method, fusion-type, enable-per-user-alpha,
  enable-item-perturbation, contrastive-lambda)
PLUS the 2 federation num-supernodes flips AND the [dev] pytest extra AND the
preserved Phase 1 Plan 06 fedrec-foundation dep.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load():
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_num_supernodes_flipped_in_both_federations():
    d = _load()
    feds = d["tool"]["flwr"]["federations"]
    assert feds["local-simulation"]["options"]["num-supernodes"] == 6040
    assert feds["local-sim-gpu"]["options"]["num-supernodes"] == 6040


def test_phase3_foundation_contract_keys_present():
    d = _load()
    cfg = d["tool"]["flwr"]["app"]["config"]
    assert cfg["partition-mode"] == "natural"
    # Default mode flipped 2026-04-27: cross-device is the canonical thesis path.
    # Cross-silo escape hatch preserved via explicit --run-config "mode=cross_silo_legacy"
    # (which still raises NotImplementedError per D-02). Commit 83b5120.
    assert cfg["mode"] == "benchmark_cross_device"
    assert cfg["run-seed"] == 42
    assert cfg["weight-policy"] == "num_positives"
    assert cfg["eval-num-negatives"] == 99
    assert cfg["checkpoint-rule"] == "best_round_restore"
    assert cfg["reuse-cache"] is False


def test_phase4_signature_keys_at_thesis_defaults():
    d = _load()
    cfg = d["tool"]["flwr"]["app"]["config"]
    assert cfg["model-type"] == "dual"
    assert cfg["alpha-method"] == "hierarchical_conditional"
    assert cfg["fusion-type"] == "concat"
    assert cfg["enable-per-user-alpha"] is True
    assert cfg["enable-item-perturbation"] is True
    assert abs(cfg["contrastive-lambda"] - 0.1) < 1e-9


def test_dev_pytest_extra_declared():
    d = _load()
    assert d["project"]["optional-dependencies"]["dev"] == ["pytest>=7.0"]


def test_fedrec_foundation_dep_preserved():
    d = _load()
    deps = d["project"]["dependencies"]
    assert any("fedrec-foundation" in dep for dep in deps), \
        "Phase 1 Plan 06 dep must be preserved"
