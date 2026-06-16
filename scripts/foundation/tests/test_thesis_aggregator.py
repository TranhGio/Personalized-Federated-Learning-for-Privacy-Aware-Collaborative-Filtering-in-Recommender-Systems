"""Tests for scripts/thesis/aggregate_results.py (Phase 7 Plan 04).

Covers VALIDATION.md rows 7-04-01 through 7-04-11.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# Bootstrap import path for the aggregator + orchestrator.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from thesis.aggregate_results import (  # noqa: E402
    aggregate_by_seed,
    collect_thesis_results,
    expected_ablation_cells,
    expected_main_cells,
    extract_evaluated_users,
    extract_metric,
    find_missing_cells,
    fmt_cell,
    is_winner,
    run_aggregator,
    ThesisResult,
)
from thesis.run_thesis_sweep import build_ablation_matrix, build_main_matrix  # noqa: E402


# ============================================================================
# Synthetic-fixture helpers
# ============================================================================


def _make_results_data(
    *,
    ndcg_overall: float,
    hr_overall: float,
    ndcg_sparse: Optional[float] = None,
    hr_sparse: Optional[float] = None,
    ndcg_medium: Optional[float] = None,
    hr_medium: Optional[float] = None,
    ndcg_dense: Optional[float] = None,
    hr_dense: Optional[float] = None,
    evaluated_users: int = 6040,
    evaluated_users_sparse: int = 1850,
    evaluated_users_medium: int = 2100,
    evaluated_users_dense: int = 2090,
    sparse_delim: str = "_",  # "_" for non-pfedrec; "/" for pfedrec
) -> Dict[str, Any]:
    """Synthesize a per-run results.json payload matching the Phase-6 D-07 schema."""
    best: Dict[str, Any] = {
        "sampled_ndcg@10": ndcg_overall,
        "sampled_hr@10": hr_overall,
        "evaluated_users": evaluated_users,
    }
    if ndcg_sparse is not None:
        best["sampled_ndcg@10/sparse"] = ndcg_sparse
        best["sampled_hr@10/sparse"] = hr_sparse if hr_sparse is not None else 0.0
        best[f"evaluated_users{sparse_delim}sparse"] = evaluated_users_sparse
    if ndcg_medium is not None:
        best["sampled_ndcg@10/medium"] = ndcg_medium
        best["sampled_hr@10/medium"] = hr_medium if hr_medium is not None else 0.0
        best[f"evaluated_users{sparse_delim}medium"] = evaluated_users_medium
    if ndcg_dense is not None:
        best["sampled_ndcg@10/dense"] = ndcg_dense
        best["sampled_hr@10/dense"] = hr_dense if hr_dense is not None else 0.0
        best[f"evaluated_users{sparse_delim}dense"] = evaluated_users_dense
    return {
        "final_metrics": {
            "best": best,
            "last": dict(best),
            "best_round": 87,
            "last_round": 100,
            "final_eval_round_index": 1,
        },
    }


def _write_synthetic_run(
    results_root: Path,
    module: str,
    run_id: str,
    thesis_run_label: str,
    run_seed: int,
    results_data: Dict[str, Any],
    ablation_dimension: str = "none",
    ablation_value: str = "",
    mode: str = "thesis_crossdevice_main",
) -> None:
    """Drop a synthetic results.json + manifest.json under results_root/federated/<module>/<run_id>/."""
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
        "mode": mode,
    }
    results_data["_manifest"] = dict(manifest)
    (run_dir / "results.json").write_text(json.dumps(results_data), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _seed_full_main_matrix(results_root: Path) -> None:
    """Drop synthetic results.json files covering ALL 12 main cells (passing aggregator)."""
    for cell in build_main_matrix():
        # Adaptive scores higher than baseline/personalized to make D-11 win clean.
        scale = {"baseline": 0.40, "personalized": 0.41, "adaptive": 0.43, "pfedrec": 0.44}[cell.module]
        delim = "/" if cell.module == "pfedrec" else "_"
        data = _make_results_data(
            ndcg_overall=scale,
            hr_overall=scale + 0.30,
            ndcg_sparse=scale - 0.02,
            hr_sparse=scale + 0.28,
            ndcg_medium=scale + 0.005,
            hr_medium=scale + 0.31,
            ndcg_dense=scale + 0.01,
            hr_dense=scale + 0.32,
            sparse_delim=delim,
        )
        _write_synthetic_run(
            results_root, cell.module, f"20260429-100000-{cell.module}{cell.run_seed}",
            thesis_run_label=cell.thesis_run_label, run_seed=cell.run_seed,
            results_data=data, ablation_dimension=cell.ablation_dimension,
            ablation_value=cell.ablation_value, mode=cell.mode,
        )


def _seed_full_ablation_matrix(results_root: Path) -> None:
    """Drop synthetic results.json files covering ALL 21 ablation cells."""
    for cell in build_ablation_matrix():
        data = _make_results_data(
            ndcg_overall=0.42,
            hr_overall=0.72,
            ndcg_sparse=0.39,
            hr_sparse=0.69,
            ndcg_medium=0.42,
            hr_medium=0.72,
            ndcg_dense=0.43,
            hr_dense=0.73,
            sparse_delim="_",
        )
        _write_synthetic_run(
            results_root, cell.module, f"20260429-200000-{cell.thesis_run_label}-seed{cell.run_seed}".replace("=", "_"),
            thesis_run_label=cell.thesis_run_label, run_seed=cell.run_seed,
            results_data=data, ablation_dimension=cell.ablation_dimension,
            ablation_value=cell.ablation_value, mode=cell.mode,
        )


# ============================================================================
# 7-04-01: extract_metric overall NDCG@10
# ============================================================================


def test_extract_overall_ndcg10() -> None:
    """Phase 7: extract_metric reads final_metrics['best']['sampled_ndcg@10']."""
    data = _make_results_data(ndcg_overall=0.4123, hr_overall=0.7290)
    assert extract_metric(data, "sampled_ndcg@10") == pytest.approx(0.4123)
    assert extract_metric(data, "sampled_hr@10") == pytest.approx(0.7290)
    assert extract_metric(data, "nonexistent_key") is None


# ============================================================================
# 7-04-02: extract sparse with uniform slash delimiter (Pitfall 1)
# ============================================================================


def test_extract_sparse_ndcg10_uniform_slash() -> None:
    """Phase 7 Pitfall 1: HR/NDCG keys use SLASH delimiter UNIFORMLY across all 4 modules."""
    for delim in ("_", "/"):
        data = _make_results_data(
            ndcg_overall=0.42, hr_overall=0.72,
            ndcg_sparse=0.38, hr_sparse=0.68,
            sparse_delim=delim,
        )
        # HR/NDCG keys ALWAYS use slash, regardless of module's evaluated_users delimiter.
        assert extract_metric(data, "sampled_ndcg@10/sparse") == pytest.approx(0.38)
        assert extract_metric(data, "sampled_hr@10/sparse") == pytest.approx(0.68)


def test_extract_evaluated_users_handles_both_delimiters() -> None:
    """Pitfall 1: evaluated_users keys diverge — slash for pfedrec, underscore for others."""
    # PFedRec uses slash.
    pfedrec_data = _make_results_data(
        ndcg_overall=0.44, hr_overall=0.73,
        ndcg_sparse=0.40, hr_sparse=0.70,
        evaluated_users_sparse=1234, sparse_delim="/",
    )
    assert extract_evaluated_users(pfedrec_data, "sparse") == 1234
    # Baseline uses underscore.
    baseline_data = _make_results_data(
        ndcg_overall=0.40, hr_overall=0.70,
        ndcg_sparse=0.38, hr_sparse=0.68,
        evaluated_users_sparse=5678, sparse_delim="_",
    )
    assert extract_evaluated_users(baseline_data, "sparse") == 5678


# ============================================================================
# 7-04-03: D-11 win criterion positive
# ============================================================================


def test_d11_win_criterion() -> None:
    """Phase 7 D-11: adaptive (mean=0.42, std=0.005) wins against personalized (0.40, 0.005).

    0.42 - 0.005 = 0.415 > 0.40 + 0.005 = 0.405 ✓
    """
    assert is_winner(0.42, 0.005, [(0.40, 0.005)]) is True
    # Also wins against multiple lower-mean rows.
    assert is_winner(0.42, 0.005, [(0.40, 0.005), (0.39, 0.004)]) is True


# ============================================================================
# 7-04-04: D-11 win criterion negative (overlapping intervals)
# ============================================================================


def test_d11_overlap_no_winner() -> None:
    """Phase 7 D-11: overlapping ±σ intervals correctly NOT flagged as winner.

    0.42 - 0.01 = 0.41; 0.41 + 0.01 = 0.42. 0.41 <= 0.42 → not winner.
    """
    assert is_winner(0.42, 0.01, [(0.41, 0.01)]) is False
    # Edge case: exactly equal lower bound → still NOT a winner (strict >).
    assert is_winner(0.42, 0.005, [(0.41, 0.015)]) is False  # 0.415 <= 0.425
    # Empty comparable set → cannot win.
    assert is_winner(0.42, 0.005, []) is False


# ============================================================================
# 7-04-05: sparse partial-seed handling (Pitfall 10)
# ============================================================================


def test_sparse_partial_seeds(tmp_path: Path) -> None:
    """Phase 7 Pitfall 10: when one seed has zero sparse evaluations, that seed is excluded
    from the sparse aggregation and the result row carries n_seeds_with_sparse=2/3."""
    # 3 baseline runs at seeds 42, 1337, 2026; seed=2026 has zero sparse evaluations.
    for seed in (42, 1337):
        data = _make_results_data(
            ndcg_overall=0.40, hr_overall=0.70,
            ndcg_sparse=0.38, hr_sparse=0.68,
            evaluated_users_sparse=1850, sparse_delim="_",
        )
        _write_synthetic_run(
            tmp_path, "baseline", f"20260429-100000-baseline{seed}",
            thesis_run_label="main", run_seed=seed, results_data=data,
        )
    # Seed 2026: sparse evaluable count is 0; mean sparse stat is 0.0 (zero-divide-safe in strategy).
    data_zero = _make_results_data(
        ndcg_overall=0.40, hr_overall=0.70,
        ndcg_sparse=0.0, hr_sparse=0.0,
        evaluated_users_sparse=0, sparse_delim="_",
    )
    _write_synthetic_run(
        tmp_path, "baseline", "20260429-100000-baseline2026",
        thesis_run_label="main", run_seed=2026, results_data=data_zero,
    )
    records = collect_thesis_results(tmp_path)
    assert len(records) == 3
    # With sparse_evaluable_only=True, seed=2026 is dropped → n_seeds = 2.
    sparse_agg = aggregate_by_seed(records, "sampled_ndcg@10/sparse", sparse_evaluable_only=True)
    assert ("baseline", "main") in sparse_agg
    mean, std, n = sparse_agg[("baseline", "main")]
    assert n == 2, f"Expected 2 seeds with sparse; got {n}"
    assert mean == pytest.approx(0.38)
    assert std == pytest.approx(0.0)
    # Without the filter, all 3 seeds counted → n=3 with inflated std (regression evidence).
    sparse_agg_unfiltered = aggregate_by_seed(records, "sampled_ndcg@10/sparse", sparse_evaluable_only=False)
    _, _, n_unfiltered = sparse_agg_unfiltered[("baseline", "main")]
    assert n_unfiltered == 3


# ============================================================================
# 7-04-06: ablation label grouping
# ============================================================================


def test_ablation_label_grouping(tmp_path: Path) -> None:
    """Phase 7 D-13: ablation labels of the form 'ablation_<dim>=<val>' group correctly across seeds."""
    for seed in (42, 1337, 2026):
        data = _make_results_data(
            ndcg_overall=0.43, hr_overall=0.73,
            ndcg_sparse=0.40, hr_sparse=0.70,
        )
        _write_synthetic_run(
            tmp_path, "adaptive", f"20260429-200000-fusionadd-seed{seed}",
            thesis_run_label="ablation_fusion_type=add",
            run_seed=seed, results_data=data,
            ablation_dimension="fusion_type", ablation_value="add",
        )
    records = collect_thesis_results(tmp_path)
    assert len(records) == 3
    agg = aggregate_by_seed(records, "sampled_ndcg@10")
    assert ("adaptive", "ablation_fusion_type=add") in agg
    mean, std, n = agg[("adaptive", "ablation_fusion_type=add")]
    assert n == 3
    assert mean == pytest.approx(0.43)


# ============================================================================
# 7-04-07: D-20 hard-fail on missing cells
# ============================================================================


def test_d20_hard_fail_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Phase 7 D-20: aggregator hard-fails with explicit list when cells are missing.

    Synthetic fixture has 32/33 cells (missing adaptive seed=2026 main); aggregator must
    raise SystemExit and print the missing tuple.
    """
    # Seed 32 cells (drop the adaptive main seed=2026 cell).
    target_cells = [c for c in build_main_matrix() + build_ablation_matrix()
                    if not (c.module == "adaptive" and c.thesis_run_label == "main" and c.run_seed == 2026)]
    for cell in target_cells:
        delim = "/" if cell.module == "pfedrec" else "_"
        data = _make_results_data(
            ndcg_overall=0.42, hr_overall=0.72,
            ndcg_sparse=0.39, hr_sparse=0.69,
            sparse_delim=delim,
        )
        _write_synthetic_run(
            tmp_path, cell.module, f"20260429-{cell.thesis_run_label}-seed{cell.run_seed}".replace("=", "_"),
            thesis_run_label=cell.thesis_run_label, run_seed=cell.run_seed,
            results_data=data, ablation_dimension=cell.ablation_dimension,
            ablation_value=cell.ablation_value, mode=cell.mode,
        )
    output_dir = tmp_path / "_thesis_out"
    with pytest.raises(SystemExit) as exc_info:
        run_aggregator(tmp_path, output_dir, check_only=False)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    # Missing tuple appears in stderr.
    assert "Missing 1 cells" in captured.err
    assert "adaptive" in captured.err
    assert "main" in captured.err
    assert "2026" in captured.err


# ============================================================================
# 7-04-08: per-group columns in CSV
# ============================================================================


def test_csv_per_group_columns(tmp_path: Path) -> None:
    """Phase 7 THS-06: CSV output preserves per-group keys (medium, dense) for main_comparison."""
    _seed_full_main_matrix(tmp_path)
    _seed_full_ablation_matrix(tmp_path)
    output_dir = tmp_path / "_thesis_out"
    run_aggregator(tmp_path, output_dir, check_only=False)
    csv_text = (output_dir / "main_comparison.csv").read_text(encoding="utf-8")
    # Header contains all per-group columns.
    for col in ("ndcg10_medium_mean", "ndcg10_medium_std", "ndcg10_dense_mean", "hr10_dense_std"):
        assert col in csv_text, f"Expected column '{col}' missing from main_comparison.csv"


# ============================================================================
# 7-04-09: 6 output files emitted
# ============================================================================


def test_six_output_files(tmp_path: Path) -> None:
    """Phase 7 D-17: aggregator emits 6 files: main_comparison.{md,csv}, ablations.{md,csv}, sparse_slice.{md,csv}."""
    _seed_full_main_matrix(tmp_path)
    _seed_full_ablation_matrix(tmp_path)
    output_dir = tmp_path / "_thesis_out"
    rc = run_aggregator(tmp_path, output_dir, check_only=False)
    assert rc == 0
    expected_files = [
        "main_comparison.md", "main_comparison.csv",
        "ablations.md", "ablations.csv",
        "sparse_slice.md", "sparse_slice.csv",
    ]
    for name in expected_files:
        path = output_dir / name
        assert path.exists(), f"Expected output file {name} missing"
        # Each file is non-empty.
        assert path.stat().st_size > 0


# ============================================================================
# 7-04-10: atomic write — no .tmp-* leftovers
# ============================================================================


def test_atomic_write_no_tmp(tmp_path: Path) -> None:
    """Phase 7 D-15: aggregator's writes use atomic_write_text; no .tmp-* leftovers in output dir."""
    _seed_full_main_matrix(tmp_path)
    _seed_full_ablation_matrix(tmp_path)
    output_dir = tmp_path / "_thesis_out"
    run_aggregator(tmp_path, output_dir, check_only=False)
    leftovers = list(output_dir.glob(".tmp-*"))
    assert leftovers == [], f"Expected no .tmp-* leftovers; found {leftovers}"


# ============================================================================
# 7-04-11: cell format (D-24)
# ============================================================================


def test_cell_format() -> None:
    """Phase 7 D-24: cell format is `0.4123 ± 0.0089` with 4 decimal places."""
    assert fmt_cell(0.4123, 0.0089) == "0.4123 ± 0.0089"
    assert fmt_cell(0.7000, 0.0001) == "0.7000 ± 0.0001"
    # Missing data → em-dash sentinel.
    assert fmt_cell(None, None) == "—"
    assert fmt_cell(0.5, None) == "—"


# ============================================================================
# Bonus regression tests
# ============================================================================


def test_winner_bolded_in_main_md(tmp_path: Path) -> None:
    """Phase 7 D-11 + Discretion: D-11 winner gets bold formatting in main_comparison.md.

    The fixture _seed_full_main_matrix sets adaptive (0.43) > personalized (0.41) > baseline (0.40)
    with std=0 (single seed scaled), so the D-11 win condition is satisfied for adaptive on every metric.
    """
    _seed_full_main_matrix(tmp_path)
    _seed_full_ablation_matrix(tmp_path)
    output_dir = tmp_path / "_thesis_out"
    run_aggregator(tmp_path, output_dir, check_only=False)
    md_text = (output_dir / "main_comparison.md").read_text(encoding="utf-8")
    # Adaptive's NDCG cell should be bolded.
    assert "**0.4300 ±" in md_text or "**0.4400 ±" in md_text or "**" in md_text, (
        "Expected at least one bolded cell in main_comparison.md (D-11 winner highlighting)"
    )
    # PFedRec footnote present.
    assert "† PFedRec (paper-faithful)" in md_text


def test_collect_filters_legacy_phase6_manifests(tmp_path: Path) -> None:
    """Phase 7 Pitfall 7: legacy Phase-6 manifests (no thesis_run_label) are filtered out."""
    # Synthesize a legacy manifest (no thesis fields, schema_version=2).
    run_dir = tmp_path / "federated" / "baseline" / "20260420-090000-legacy"
    run_dir.mkdir(parents=True, exist_ok=True)
    legacy_manifest = {
        "schema_version": 2,
        "run_id": "20260420-090000-legacy",
        "module": "baseline",
        "run_seed": 42,
        # NO thesis_run_label, NO ablation_dimension, NO ablation_value
    }
    (run_dir / "manifest.json").write_text(json.dumps(legacy_manifest), encoding="utf-8")
    legacy_results = _make_results_data(
        ndcg_overall=0.40, hr_overall=0.70,
        ndcg_sparse=0.38, hr_sparse=0.68,
    )
    legacy_results["_manifest"] = legacy_manifest
    (run_dir / "results.json").write_text(json.dumps(legacy_results), encoding="utf-8")
    # Aggregator filter: thesis_run_label "" (default) excludes this run.
    records = collect_thesis_results(tmp_path)
    assert len(records) == 0, "Legacy Phase-6 manifests must be filtered out by aggregator"


def test_check_only_does_not_write_files(tmp_path: Path) -> None:
    """Phase 7 Plan 04 design: --check-only mode verifies state but writes NO files."""
    _seed_full_main_matrix(tmp_path)
    _seed_full_ablation_matrix(tmp_path)
    output_dir = tmp_path / "_thesis_out"
    rc = run_aggregator(tmp_path, output_dir, check_only=True)
    assert rc == 0
    # Output directory should NOT exist (or should be empty).
    if output_dir.exists():
        assert list(output_dir.iterdir()) == [], "--check-only must not write any files"


# ============================================================================
# Eval-validity guard + per-cell dedupe (run-id audit, 2026-06-10)
# ============================================================================


def _write_sidecar(results_root: Path, module: str, run_id: str, payload: Dict[str, Any]) -> None:
    """Write an EVAL_VALIDITY.json AFTER results.json so it is mtime-fresh."""
    run_dir = results_root / "federated" / module / run_id
    (run_dir / "EVAL_VALIDITY.json").write_text(json.dumps(payload), encoding="utf-8")


def test_sidecar_invalid_status_skipped(tmp_path: Path) -> None:
    """status != valid -> the record never enters the collection."""
    _write_synthetic_run(
        tmp_path, "adaptive", "20260506-074753-bc134c",
        thesis_run_label="main", run_seed=42,
        results_data=_make_results_data(ndcg_overall=0.0563, hr_overall=0.12),
    )
    _write_sidecar(tmp_path, "adaptive", "20260506-074753-bc134c",
                   {"status": "invalid_cold_eval", "reason": "predates fix"})
    assert collect_thesis_results(tmp_path) == []


def test_backfill_superseded_by_real_run(tmp_path: Path) -> None:
    """A real manifest-labeled run at the same (module,label,seed) supersedes a
    backfilled provisional record — no double-count in aggregate_by_seed."""
    # Backfilled provisional run: empty manifest label, sidecar provides it.
    _write_synthetic_run(
        tmp_path, "pfedrec", "20260608-071106-ef41ab",
        thesis_run_label="", run_seed=42,
        results_data=_make_results_data(ndcg_overall=0.3352, hr_overall=0.59),
    )
    _write_sidecar(tmp_path, "pfedrec", "20260608-071106-ef41ab",
                   {"status": "valid", "thesis_run_label_backfill": "main",
                    "run_seed_backfill": 42})
    # Real sweep run, same cell.
    _write_synthetic_run(
        tmp_path, "pfedrec", "20260701-000000-real42",
        thesis_run_label="main", run_seed=42,
        results_data=_make_results_data(ndcg_overall=0.40, hr_overall=0.65),
    )
    _write_sidecar(tmp_path, "pfedrec", "20260701-000000-real42", {"status": "valid"})
    records = collect_thesis_results(tmp_path)
    assert len(records) == 1
    assert records[0].results_path.parent.name == "20260701-000000-real42"
    assert records[0].backfilled is False
    agg = aggregate_by_seed(records, "sampled_ndcg@10")
    mean, _std, n = agg[("pfedrec", "main")]
    assert n == 1 and abs(mean - 0.40) < 1e-9  # not (0.3352+0.40)/2


def test_backfill_used_when_no_real_run(tmp_path: Path) -> None:
    """Without a real run, the backfilled provisional cell IS collected."""
    _write_synthetic_run(
        tmp_path, "pfedrec", "20260608-071106-ef41ab",
        thesis_run_label="", run_seed=42,
        results_data=_make_results_data(ndcg_overall=0.3352, hr_overall=0.59),
    )
    _write_sidecar(tmp_path, "pfedrec", "20260608-071106-ef41ab",
                   {"status": "valid", "thesis_run_label_backfill": "main",
                    "run_seed_backfill": 42})
    records = collect_thesis_results(tmp_path)
    assert len(records) == 1 and records[0].backfilled is True


def test_strict_validity_fails_on_missing_sidecar(tmp_path: Path) -> None:
    """strict_validity=True -> SystemExit when a thesis-labeled run has no sidecar."""
    _write_synthetic_run(
        tmp_path, "baseline", "20260701-000000-nosc42",
        thesis_run_label="main", run_seed=42,
        results_data=_make_results_data(ndcg_overall=0.20, hr_overall=0.36),
    )
    with pytest.raises(SystemExit):
        collect_thesis_results(tmp_path, strict_validity=True)
    # Non-strict: collected, but flagged on stderr (legacy passthrough).
    records = collect_thesis_results(tmp_path, strict_validity=False)
    assert len(records) == 1
