"""Tests for scripts/thesis/run_thesis_sweep.py (Phase 7 Plan 03)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Bootstrap import path for the orchestrator (mirrors how the script runs).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from thesis.run_thesis_sweep import (  # noqa: E402
    ABLATION_KNOBS,
    THESIS_BASE_OVERRIDES,
    THESIS_SEEDS,
    ThesisCell,
    build_ablation_matrix,
    build_main_matrix,
    cell_already_done,
    cell_run_config_string,
    execute_cell,
)


# ============================================================================
# Matrix shape tests
# ============================================================================


def test_main_matrix_size() -> None:
    """Phase 7 D-09: main matrix is 4 modules x 3 seeds = 12 cells.

    3 modules at thesis_crossdevice_main + 1 module (pfedrec) at paper_compat_pfedrec.
    """
    cells = build_main_matrix()
    assert len(cells) == 12, f"Expected 12 main cells, got {len(cells)}"


def test_main_modules_correct() -> None:
    """Phase 7 D-04 + D-06: main matrix covers exactly 4 modules; pfedrec uses paper_compat_pfedrec."""
    cells = build_main_matrix()
    modules = sorted({c.module for c in cells})
    assert modules == ["adaptive", "baseline", "personalized", "pfedrec"]
    pfedrec_cells = [c for c in cells if c.module == "pfedrec"]
    assert all(c.mode == "paper_compat_pfedrec" for c in pfedrec_cells), (
        "Phase 7 D-06: PFedRec runs ONLY at paper_compat_pfedrec mode"
    )
    thesis_cells = [c for c in cells if c.module != "pfedrec"]
    assert all(c.mode == "thesis_crossdevice_main" for c in thesis_cells), (
        "Phase 7 D-04: baseline/personalized/adaptive run at thesis_crossdevice_main"
    )


def test_ablation_matrix_size() -> None:
    """Phase 7 D-13 + D-14: 7 ablation knobs x 3 seeds = 21 ablation cells."""
    cells = build_ablation_matrix()
    assert len(cells) == 21, f"Expected 21 ablation cells, got {len(cells)}"


def test_ablation_module_is_adaptive_only() -> None:
    """Phase 7 D-13: ablations are always module='adaptive' (only adaptive has the knobs)."""
    cells = build_ablation_matrix()
    modules = {c.module for c in cells}
    assert modules == {"adaptive"}, f"Ablation cells must be adaptive-only; got {modules}"
    modes = {c.mode for c in cells}
    assert modes == {"thesis_crossdevice_main"}, (
        "Ablation cells must run at thesis_crossdevice_main; got {}".format(modes)
    )


def test_seeds_are_canonical_set() -> None:
    """Phase 7 D-10: seeds = {42, 1337, 2026} across both matrices."""
    assert THESIS_SEEDS == (42, 1337, 2026)
    main_seeds = {c.run_seed for c in build_main_matrix()}
    ablation_seeds = {c.run_seed for c in build_ablation_matrix()}
    assert main_seeds == {42, 1337, 2026}
    assert ablation_seeds == {42, 1337, 2026}


# ============================================================================
# Skip-on-existing logic (Pitfall 8 mitigation)
# ============================================================================


def _write_synthetic_manifest(
    results_root: Path,
    module: str,
    run_id: str,
    thesis_run_label: str,
    run_seed: int,
    ablation_dimension: str = "none",
    ablation_value: str = "",
) -> Path:
    """Helper: drop a synthetic manifest.json under results_root/federated/<module>/<run_id>/."""
    run_dir = results_root / "federated" / module / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 3,
        "run_id": run_id,
        "module": module,
        "run_seed": run_seed,
        "thesis_run_label": thesis_run_label,
        "ablation_dimension": ablation_dimension,
        "ablation_value": ablation_value,
        "mode": "thesis_crossdevice_main",
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_skip_on_existing_full_tuple(tmp_path: Path) -> None:
    """Phase 7 Pitfall 8: cell_already_done matches on (module, label, seed, dim, value) — NOT on (module, seed) alone.

    Adaptive at seed=42 happens 8 times (1 main + 7 ablations). A naive (module, seed)
    match would skip 7 unrelated cells once any of them completes.
    """
    # Seed the disk with a single adaptive main run at seed=42.
    _write_synthetic_manifest(
        tmp_path, "adaptive", "20260429-100000-aaaaaa",
        thesis_run_label="main", run_seed=42,
        ablation_dimension="none", ablation_value="",
    )
    # The matching main cell is now done.
    main_cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
    )
    assert cell_already_done(main_cell, tmp_path) is True
    # An ablation cell at the SAME (module, seed) is still pending — distinct identity.
    ablation_cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="ablation_fusion_type=add",
        ablation_dimension="fusion_type", ablation_value="add",
    )
    assert cell_already_done(ablation_cell, tmp_path) is False, (
        "Pitfall 8: cell at same (module, seed) but different ablation MUST not be skipped"
    )


def test_skip_on_existing_returns_false_when_no_disk(tmp_path: Path) -> None:
    """Empty results-root means no cells are done — every cell should run."""
    cell = ThesisCell(
        module="baseline", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
    )
    assert cell_already_done(cell, tmp_path) is False


def test_skip_on_existing_ignores_corrupt_manifest(tmp_path: Path) -> None:
    """Corrupt manifest.json (mid-write crash) MUST be tolerated — return False, do not raise."""
    run_dir = tmp_path / "federated" / "baseline" / "20260429-100000-corrupt"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
    cell = ThesisCell(
        module="baseline", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
    )
    assert cell_already_done(cell, tmp_path) is False


# ============================================================================
# Run-config string builder
# ============================================================================


def test_run_config_quoting() -> None:
    """Phase 7 D-22 + D-21: cell_run_config_string emits all required keys.

    Bare-word values pass raw to scripts/run.py; the launcher's _quote_value_for_flwr
    adds TOML quoting downstream.
    """
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
    )
    s = cell_run_config_string(cell)
    assert "run-seed=42" in s
    assert "thesis-run-label=main" in s
    assert "ablation-dimension=none" in s
    assert "ablation-value=" in s
    assert "wandb-run-name=thesis-main-adaptive-seed42" in s


def test_run_config_string_includes_extra_knobs() -> None:
    """Ablation cell's extra_run_config flows into the --run-config string."""
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=1337,
        thesis_run_label="ablation_fusion_type=add",
        ablation_dimension="fusion_type", ablation_value="add",
        extra_run_config={"fusion-type": "add"},
    )
    s = cell_run_config_string(cell)
    assert "fusion-type=add" in s
    # D-21 short-form for W&B: fusion (not fusion_type).
    assert "wandb-run-name=thesis-ablation-adaptive-seed1337-fusion=add" in s


def test_run_config_string_item_perturbation_two_knobs() -> None:
    """item_perturbation cell carries TWO extra knobs: enable-item-perturbation + item-perturbation-reg."""
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=2026,
        thesis_run_label="ablation_item_perturbation=true",
        ablation_dimension="item_perturbation", ablation_value="true",
        extra_run_config={"enable-item-perturbation": "true", "item-perturbation-reg": "0.01"},
    )
    s = cell_run_config_string(cell)
    assert "enable-item-perturbation=true" in s
    assert "item-perturbation-reg=0.01" in s
    # D-21 short-form: ip (not item_perturbation).
    assert "wandb-run-name=thesis-ablation-adaptive-seed2026-ip=true" in s


# ============================================================================
# Dry-run + subprocess avoidance
# ============================================================================


def test_dry_run_no_subprocess() -> None:
    """Phase 7 D-18: --dry-run prints commands but never invokes subprocess.run."""
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
    )
    with patch("thesis.run_thesis_sweep.subprocess.run") as mock_run:
        success, stderr = execute_cell(cell, _REPO_ROOT, dry_run=True)
        assert success is True
        assert stderr == ""
        # Critical invariant: subprocess.run was NEVER called under dry_run.
        mock_run.assert_not_called()


def test_ablation_knobs_shape() -> None:
    """Phase 7 D-13: ABLATION_KNOBS contains exactly 7 entries with the expected dimensions."""
    assert len(ABLATION_KNOBS) == 7
    dimensions = [k[0] for k in ABLATION_KNOBS]
    # Multi-occurrence dimensions: alpha_method (2), fusion_type (2). Singletons: per_user_alpha,
    # item_perturbation, contrastive_lambda.
    assert dimensions.count("alpha_method") == 2
    assert dimensions.count("fusion_type") == 2
    assert dimensions.count("per_user_alpha") == 1
    assert dimensions.count("item_perturbation") == 1
    assert dimensions.count("contrastive_lambda") == 1


# ============================================================================
# BLOCKER 1 + BLOCKER 2: D-02 + D-03 enforcement (per-checker iteration 1)
# ============================================================================


def test_adaptive_main_cell_includes_dual_model_and_hc_alpha() -> None:
    """BLOCKER 1 (D-02 + D-03): adaptive main cell's run-config MUST contain
    strategy=fedavg AND model-type=dual AND alpha-method=hierarchical_conditional.

    Without this, all 12 main + all 21 ablation cells silently run with
    FedProx + whatever-model-type from pyproject.toml defaults, producing
    invalid thesis numbers.
    """
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
        extra_run_config={},
    )
    s = cell_run_config_string(cell)
    assert "strategy=fedavg" in s, "D-03: adaptive main MUST run with fedavg, not fedprox"
    assert "model-type=dual" in s, "D-02: adaptive main MUST use model-type=dual"
    assert "alpha-method=hierarchical_conditional" in s, (
        "D-02: adaptive main MUST use alpha-method=hierarchical_conditional"
    )
    # D-02: next-gen knobs OFF in main config (they default to true/0.1 in pyproject.toml).
    assert "enable-per-user-alpha=false" in s, (
        "D-02: next-gen knob enable-per-user-alpha MUST be OFF in main config"
    )
    assert "enable-item-perturbation=false" in s, (
        "D-02: next-gen knob enable-item-perturbation MUST be OFF in main config"
    )
    assert "contrastive-lambda=0.0" in s, (
        "D-02: next-gen knob contrastive-lambda MUST be 0.0 in main config"
    )


def test_alpha_method_ablation_overrides_base_hc() -> None:
    """BLOCKER 1 (merge precedence): the alpha_method=multi_factor ablation cell
    MUST contain alpha-method=multi_factor (NOT hierarchical_conditional).

    This test PROVES the merge order is correct: cell.extra_run_config wins
    over THESIS_BASE_OVERRIDES[module] on conflicting keys. If the merge were
    reversed, the ablation would silently revert to hierarchical_conditional
    and produce duplicate main-config rows instead of ablation data.
    """
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="ablation_alpha_method=multi_factor",
        ablation_dimension="alpha_method", ablation_value="multi_factor",
        extra_run_config={"alpha-method": "multi_factor"},
    )
    s = cell_run_config_string(cell)
    assert "alpha-method=multi_factor" in s, (
        "BLOCKER 1: ablation cell's alpha-method=multi_factor MUST appear in run-config"
    )
    assert "alpha-method=hierarchical_conditional" not in s, (
        "BLOCKER 1 (merge order): the base override hierarchical_conditional "
        "MUST NOT appear in the run-config — extra_run_config wins on conflict."
    )
    # Strategy + model-type overrides still apply (no conflict — ablation only flips alpha-method).
    assert "strategy=fedavg" in s
    assert "model-type=dual" in s


def test_pfedrec_main_cell_does_not_set_model_type() -> None:
    """BLOCKER 1: PFedRec's pyproject.toml has NO model-type or alpha-method keys.

    THESIS_BASE_OVERRIDES['pfedrec'] MUST NOT include those keys, otherwise
    flwr's fuse_dicts validation rejects the run-config with 'Key not present'
    before the run starts.
    """
    cell = ThesisCell(
        module="pfedrec", mode="paper_compat_pfedrec", run_seed=42,
        thesis_run_label="main", ablation_dimension="none", ablation_value="",
        extra_run_config={},
    )
    s = cell_run_config_string(cell)
    assert "strategy=fedavg" in s, "D-03: pfedrec base override sets strategy=fedavg"
    assert "model-type=" not in s, (
        "BLOCKER 1: pfedrec has no model-type config key; THESIS_BASE_OVERRIDES "
        "MUST NOT inject it (would cause fuse_dicts validation failure)"
    )
    assert "alpha-method=" not in s, (
        "BLOCKER 1: pfedrec has no alpha-method config key; THESIS_BASE_OVERRIDES "
        "MUST NOT inject it (would cause fuse_dicts validation failure)"
    )


def test_fusion_type_ablation_includes_dual_model() -> None:
    """BLOCKER 2 (D-02 amplification): fusion-type ablations only have effect when
    model-type=dual. Without the adaptive base override forcing model-type=dual,
    fusion-type=add ablation runs as plain BPRMF (silently producing results
    identical to the bpr default — wrong ablation data).
    """
    cell = ThesisCell(
        module="adaptive", mode="thesis_crossdevice_main", run_seed=42,
        thesis_run_label="ablation_fusion_type=add",
        ablation_dimension="fusion_type", ablation_value="add",
        extra_run_config={"fusion-type": "add"},
    )
    s = cell_run_config_string(cell)
    # BLOCKER 2: fusion-type knob requires model-type=dual to take effect.
    assert "model-type=dual" in s, (
        "BLOCKER 2: fusion-type ablation MUST include model-type=dual from THESIS_BASE_OVERRIDES "
        "or the fusion-type knob is a silent no-op (run reduces to BPRMF default)"
    )
    assert "fusion-type=add" in s, "Ablation cell sets fusion-type=add"
    # alpha-method=hierarchical_conditional is preserved from base (not flipped by this ablation).
    assert "alpha-method=hierarchical_conditional" in s, (
        "fusion_type ablation does NOT touch alpha-method — base HC value preserved"
    )
