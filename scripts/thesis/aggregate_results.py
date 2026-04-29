#!/usr/bin/env python
"""Phase 7 thesis-result aggregator (D-19, D-20).

Reads per-run ``results/federated/<module>/<run_id>/results.json`` files,
filters to thesis-tagged runs (``_manifest.thesis_run_label != ""`` AND
``_manifest.run_seed in {42, 1337, 2026}``), aggregates mean+/-std across seeds,
and emits 6 output files under ``results/federated/_thesis/``:

- main_comparison.md / main_comparison.csv  (D-17)
- ablations.md / ablations.csv              (D-17)
- sparse_slice.md / sparse_slice.csv        (D-17)

Hard-fails on missing cells per D-20 with an explicit list.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Import path bootstrap.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
_FOUNDATION_PKG = _REPO_ROOT / "scripts" / "foundation"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for p in (str(_FOUNDATION_PKG), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fedrec_foundation.atomic import atomic_write_text  # noqa: E402
from thesis.run_thesis_sweep import (  # noqa: E402
    THESIS_SEEDS,
    ThesisCell,
    build_ablation_matrix,
    build_main_matrix,
)


# ============================================================================
# Constants
# ============================================================================

# D-17 output filenames.
_MAIN_MD = "main_comparison.md"
_MAIN_CSV = "main_comparison.csv"
_ABL_MD = "ablations.md"
_ABL_CSV = "ablations.csv"
_SPARSE_MD = "sparse_slice.md"
_SPARSE_CSV = "sparse_slice.csv"

# Modules in the comparable set (D-05: PFedRec is excluded from win comparison).
_COMPARABLE_MODULES: Tuple[str, ...] = ("baseline", "personalized", "adaptive")

# PFedRec drift thresholds (PFR-08; D-23 RESEARCH Open Question 4).
_PFEDREC_HR10_TARGET = 0.729
_PFEDREC_NDCG10_TARGET = 0.441
_PFEDREC_DRIFT_TOLERANCE = 0.02  # +/- 2 points

# D-08 footnote text (verbatim from CONTEXT.md).
_PFEDREC_FOOTNOTE = (
    "† PFedRec (paper-faithful) — `dim=32, SGD lr=0.1, BCE, fraction-train=1.0; "
    "matches IJCAI-23 reference within ±2 points`. Not counted toward "
    "\"adaptive beats baselines\" claim per Phase 7 D-05."
)


# ============================================================================
# Result record (one results.json file)
# ============================================================================


@dataclass
class ThesisResult:
    """One results.json record matched to a thesis cell."""
    module: str
    thesis_run_label: str
    ablation_dimension: str
    ablation_value: str
    run_seed: int
    results_path: Path
    results_data: Dict[str, Any]


# ============================================================================
# Step 1 — collect (Pattern 4 from RESEARCH)
# ============================================================================


def collect_thesis_results(results_root: Path) -> List[ThesisResult]:
    """Glob ``results_root/federated/<module>/*/results.json`` and filter to thesis runs.

    Filter: ``_manifest.thesis_run_label != ""`` AND ``_manifest.run_seed`` in canonical seed set.
    Pitfall 7 invariant: legacy Phase-6 manifests (schema_version <= 2) are filtered out
    naturally by the empty-string default of ``thesis_run_label``.
    """
    out: List[ThesisResult] = []
    seeds = set(THESIS_SEEDS)
    for module in ("baseline", "personalized", "adaptive", "pfedrec"):
        module_dir = results_root / "federated" / module
        if not module_dir.exists():
            continue
        for results_path in module_dir.glob("*/results.json"):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"[WARN] Skipping unreadable results.json: {results_path} ({e})", file=sys.stderr)
                continue
            manifest = data.get("_manifest", {})
            label = manifest.get("thesis_run_label", "")
            seed = manifest.get("run_seed", -1)
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                continue
            if label and seed_int in seeds:
                out.append(ThesisResult(
                    module=module,
                    thesis_run_label=str(label),
                    ablation_dimension=str(manifest.get("ablation_dimension", "none")),
                    ablation_value=str(manifest.get("ablation_value", "")),
                    run_seed=seed_int,
                    results_path=results_path,
                    results_data=data,
                ))
    return out


# ============================================================================
# Step 2 — extract metric (Pitfall 1 mitigation)
# ============================================================================


def extract_metric(data: Dict[str, Any], metric_key: str) -> Optional[float]:
    """Read ``final_metrics['best'][metric_key]``.

    HR/NDCG keys (including per-group variants like ``sampled_ndcg@10/sparse``) use
    SLASH delimiter UNIFORMLY across all 4 modules — verified at strategy.py level
    in baseline/personalized/adaptive/pfedrec.
    """
    best = data.get("final_metrics", {}).get("best", {})
    val = best.get(metric_key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_evaluated_users(data: Dict[str, Any], group: str) -> int:
    """Pitfall 1: ``evaluated_users[/_]<group>`` — slash for pfedrec, underscore for others.

    Returns the count for the given group ('sparse', 'medium', 'dense'), or 0 if absent.
    """
    best = data.get("final_metrics", {}).get("best", {})
    # Try BOTH delimiters; whichever exists wins.
    for delim in ("/", "_"):
        key = f"evaluated_users{delim}{group}"
        if key in best:
            try:
                return int(best[key])
            except (TypeError, ValueError):
                return 0
    return 0


# ============================================================================
# Step 3 — aggregate across seeds
# ============================================================================


def aggregate_by_seed(
    records: List[ThesisResult],
    metric_key: str,
    sparse_evaluable_only: bool = False,
) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """Group by ``(module, thesis_run_label)``; compute ``(mean, std, n_seeds)``.

    If ``sparse_evaluable_only`` is True, exclude seeds where ``evaluated_users{_,/}sparse == 0``
    (Pitfall 10 mitigation; CONTEXT discretion footnote ``n_seeds_with_sparse=K/3``).
    """
    groups: Dict[Tuple[str, str], List[float]] = {}
    for rec in records:
        if sparse_evaluable_only:
            sparse_count = extract_evaluated_users(rec.results_data, "sparse")
            if sparse_count == 0:
                continue
        val = extract_metric(rec.results_data, metric_key)
        if val is None:
            continue
        key = (rec.module, rec.thesis_run_label)
        groups.setdefault(key, []).append(val)
    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    for key, vals in groups.items():
        if vals:
            out[key] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=0)),  # population std
                len(vals),
            )
    return out


# ============================================================================
# Step 4 — D-20 missing-cell hard-fail
# ============================================================================


def expected_main_cells() -> Set[Tuple[str, str, int]]:
    """D-20 expected cell set for main_comparison: 4 modules x 3 seeds x label='main'.

    Returns ``{(module, thesis_run_label, seed)}`` triples.
    """
    return {(c.module, c.thesis_run_label, c.run_seed) for c in build_main_matrix()}


def expected_ablation_cells() -> Set[Tuple[str, str, int]]:
    """D-20 expected cell set for ablations: 7 ablation cells x 3 seeds (always module='adaptive')."""
    return {(c.module, c.thesis_run_label, c.run_seed) for c in build_ablation_matrix()}


def find_missing_cells(
    records: List[ThesisResult],
    expected: Set[Tuple[str, str, int]],
) -> List[Tuple[str, str, int]]:
    """D-20 hard-fail surface: subtract observed (module, label, seed) from expected; sort for stable output."""
    observed = {(r.module, r.thesis_run_label, r.run_seed) for r in records}
    return sorted(expected - observed)


# ============================================================================
# Step 5 — D-11 win criterion + D-24 cell formatting
# ============================================================================


def fmt_cell(mean: Optional[float], std: Optional[float]) -> str:
    """D-24: ``0.4123 ± 0.0089`` (4 decimal places). Em-dash on missing data."""
    if mean is None or std is None:
        return "—"
    return f"{mean:.4f} ± {std:.4f}"


def is_winner(
    my_mean: float,
    my_std: float,
    others: List[Tuple[float, float]],
) -> bool:
    """D-11 win criterion: ``my (mean - std) > every other's (mean + std)``.

    Returns False if the comparable set is empty (cannot win against nothing).
    """
    if not others:
        return False
    my_lower = my_mean - my_std
    for other_mean, other_std in others:
        if my_lower <= other_mean + other_std:
            return False
    return True


# ============================================================================
# Step 6 — Markdown + CSV rendering
# ============================================================================


def _check_pfedrec_drift(records: List[ThesisResult]) -> Optional[str]:
    """RESEARCH Open Question 4: footnote PFedRec divergence if mean values fall
    outside ±2 points of the IJCAI-23 reference (HR=0.729, NDCG=0.441)."""
    pfedrec_records = [r for r in records if r.module == "pfedrec" and r.thesis_run_label == "main"]
    if not pfedrec_records:
        return None
    hr_vals = [v for v in (extract_metric(r.results_data, "sampled_hr@10") for r in pfedrec_records) if v is not None]
    ndcg_vals = [v for v in (extract_metric(r.results_data, "sampled_ndcg@10") for r in pfedrec_records) if v is not None]
    if not hr_vals or not ndcg_vals:
        return None
    hr_mean = float(np.mean(hr_vals))
    ndcg_mean = float(np.mean(ndcg_vals))
    drift_notes: List[str] = []
    if abs(hr_mean - _PFEDREC_HR10_TARGET) > _PFEDREC_DRIFT_TOLERANCE:
        drift_notes.append(
            f"HR@10 mean {hr_mean:.4f} drifted >{_PFEDREC_DRIFT_TOLERANCE} from target {_PFEDREC_HR10_TARGET:.4f}"
        )
    if abs(ndcg_mean - _PFEDREC_NDCG10_TARGET) > _PFEDREC_DRIFT_TOLERANCE:
        drift_notes.append(
            f"NDCG@10 mean {ndcg_mean:.4f} drifted >{_PFEDREC_DRIFT_TOLERANCE} from target {_PFEDREC_NDCG10_TARGET:.4f}"
        )
    if drift_notes:
        return (
            "**PFedRec reproduction drifted from IJCAI-23 reference**: "
            + "; ".join(drift_notes)
            + ". Investigate before reporting."
        )
    return None


def _build_main_rows(records: List[ThesisResult]) -> List[Dict[str, Any]]:
    """Compute rows for main_comparison: one row per module (filtering label='main').

    Per-row fields: module, ndcg10_mean, ndcg10_std, ndcg10_n,
    hr10_mean, hr10_std, hr10_n, plus _sparse / _medium / _dense variants.
    """
    main_records = [r for r in records if r.thesis_run_label == "main"]
    metrics_with_sparse_check = {"sampled_ndcg@10/sparse", "sampled_hr@10/sparse"}
    rows: List[Dict[str, Any]] = []
    for module in ("baseline", "personalized", "adaptive", "pfedrec"):
        module_records = [r for r in main_records if r.module == module]
        if not module_records:
            continue
        row: Dict[str, Any] = {"module": module}
        for metric_short, metric_key in [
            ("ndcg10",        "sampled_ndcg@10"),
            ("hr10",          "sampled_hr@10"),
            ("ndcg10_sparse", "sampled_ndcg@10/sparse"),
            ("hr10_sparse",   "sampled_hr@10/sparse"),
            ("ndcg10_medium", "sampled_ndcg@10/medium"),
            ("hr10_medium",   "sampled_hr@10/medium"),
            ("ndcg10_dense",  "sampled_ndcg@10/dense"),
            ("hr10_dense",    "sampled_hr@10/dense"),
        ]:
            sparse_only = metric_key in metrics_with_sparse_check
            agg = aggregate_by_seed(module_records, metric_key, sparse_evaluable_only=sparse_only)
            entry = agg.get((module, "main"))
            if entry is None:
                row[f"{metric_short}_mean"] = None
                row[f"{metric_short}_std"] = None
                row[f"{metric_short}_n"] = 0
            else:
                m, s, n = entry
                row[f"{metric_short}_mean"] = m
                row[f"{metric_short}_std"] = s
                row[f"{metric_short}_n"] = n
        rows.append(row)
    return rows


def render_main_md(rows: List[Dict[str, Any]], drift_note: Optional[str]) -> str:
    """D-17 + D-24 + D-11: render main_comparison.md with bold-the-winner.

    NDCG@10 cells are bolded if the row wins under D-11 against the comparable set
    (baseline + personalized + adaptive). HR cells are informational (not bolded).
    PFedRec row is excluded from win comparison (D-05) and gets a footnote marker.
    """
    headers = [
        "Module",
        "NDCG@10",
        "HR@10",
        "Sparse NDCG@10",
        "Sparse HR@10",
        "Medium NDCG@10",
        "Medium HR@10",
        "Dense NDCG@10",
        "Dense HR@10",
    ]
    comparable_rows = [r for r in rows if r["module"] in _COMPARABLE_MODULES]
    pfedrec_rows = [r for r in rows if r["module"] == "pfedrec"]
    lines: List[str] = []
    lines.append("# Phase 7 — Main Comparison Table")
    lines.append("")
    lines.append("Standardized cross-device thesis comparison. Mean ± std over 3 seeds {42, 1337, 2026}.")
    lines.append("Bold = winner under D-11 criterion (mean - std > all other means + std). NDCG@10 cells only.")
    lines.append("")
    if drift_note is not None:
        lines.append(f"> {drift_note}")
        lines.append("")
    # Header.
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    # Rows: comparable first, PFedRec last with footnote marker.
    for row in comparable_rows + pfedrec_rows:
        cells = [row["module"]]
        for metric in (
            "ndcg10", "hr10",
            "ndcg10_sparse", "hr10_sparse",
            "ndcg10_medium", "hr10_medium",
            "ndcg10_dense", "hr10_dense",
        ):
            cell = fmt_cell(row.get(f"{metric}_mean"), row.get(f"{metric}_std"))
            # Bold if NDCG winner under D-11 (only NDCG metrics + only comparable rows).
            if (
                row in comparable_rows
                and metric.startswith("ndcg10")
                and row.get(f"{metric}_mean") is not None
            ):
                others = [
                    (r[f"{metric}_mean"], r[f"{metric}_std"])
                    for r in comparable_rows
                    if r is not row and r.get(f"{metric}_mean") is not None
                ]
                if is_winner(row[f"{metric}_mean"], row[f"{metric}_std"], others):
                    cell = f"**{cell}**"
            # Footnote marker on PFedRec row (D-08).
            if row["module"] == "pfedrec":
                cell = f"{cell} †"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    # Footnote.
    lines.append("")
    lines.append(_PFEDREC_FOOTNOTE)
    lines.append("")
    return "\n".join(lines)


def render_main_csv(rows: List[Dict[str, Any]]) -> str:
    """D-17: CSV with two columns per metric (mean + std) plus per-group variants."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = ["module"]
    for metric in (
        "ndcg10", "hr10",
        "ndcg10_sparse", "hr10_sparse",
        "ndcg10_medium", "hr10_medium",
        "ndcg10_dense", "hr10_dense",
    ):
        header.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    writer.writerow(header)
    for row in rows:
        out: List[Any] = [row["module"]]
        for metric in (
            "ndcg10", "hr10",
            "ndcg10_sparse", "hr10_sparse",
            "ndcg10_medium", "hr10_medium",
            "ndcg10_dense", "hr10_dense",
        ):
            for suffix in ("_mean", "_std", "_n"):
                v = row.get(f"{metric}{suffix}")
                out.append(f"{v:.4f}" if isinstance(v, float) else (v if v is not None else ""))
        writer.writerow(out)
    return buf.getvalue()


def _build_ablation_rows(records: List[ThesisResult]) -> List[Dict[str, Any]]:
    """Compute rows for ablations table: one row per ablation cell + main row at top.

    D-15: only NDCG@10 + Sparse NDCG@10 + matching HR variants. Medium/dense available
    in per-run JSON but NOT in the ablation table.
    """
    # Main reference row (adaptive at thesis_crossdevice_main main).
    rows: List[Dict[str, Any]] = []
    main_records = [r for r in records if r.module == "adaptive" and r.thesis_run_label == "main"]
    if main_records:
        row: Dict[str, Any] = {
            "label": "main (reference)",
            "ablation_dimension": "none",
            "ablation_value": "",
        }
        for metric_short, metric_key in [
            ("ndcg10",        "sampled_ndcg@10"),
            ("hr10",          "sampled_hr@10"),
            ("ndcg10_sparse", "sampled_ndcg@10/sparse"),
            ("hr10_sparse",   "sampled_hr@10/sparse"),
        ]:
            sparse_only = metric_key.endswith("/sparse")
            agg = aggregate_by_seed(main_records, metric_key, sparse_evaluable_only=sparse_only)
            entry = agg.get(("adaptive", "main"))
            if entry is None:
                row[f"{metric_short}_mean"] = None
                row[f"{metric_short}_std"] = None
                row[f"{metric_short}_n"] = 0
            else:
                row[f"{metric_short}_mean"] = entry[0]
                row[f"{metric_short}_std"] = entry[1]
                row[f"{metric_short}_n"] = entry[2]
        rows.append(row)
    # Ablation rows (one per cell).
    ablation_records = [r for r in records if r.thesis_run_label.startswith("ablation_")]
    labels = sorted({r.thesis_run_label for r in ablation_records})
    for label in labels:
        label_records = [r for r in ablation_records if r.thesis_run_label == label]
        if not label_records:
            continue
        row = {
            "label": label,
            "ablation_dimension": label_records[0].ablation_dimension,
            "ablation_value": label_records[0].ablation_value,
        }
        for metric_short, metric_key in [
            ("ndcg10",        "sampled_ndcg@10"),
            ("hr10",          "sampled_hr@10"),
            ("ndcg10_sparse", "sampled_ndcg@10/sparse"),
            ("hr10_sparse",   "sampled_hr@10/sparse"),
        ]:
            sparse_only = metric_key.endswith("/sparse")
            agg = aggregate_by_seed(label_records, metric_key, sparse_evaluable_only=sparse_only)
            entry = agg.get(("adaptive", label))
            if entry is None:
                row[f"{metric_short}_mean"] = None
                row[f"{metric_short}_std"] = None
                row[f"{metric_short}_n"] = 0
            else:
                row[f"{metric_short}_mean"] = entry[0]
                row[f"{metric_short}_std"] = entry[1]
                row[f"{metric_short}_n"] = entry[2]
        rows.append(row)
    return rows


def render_ablation_md(rows: List[Dict[str, Any]]) -> str:
    """D-15 + D-17 + D-24: ablation table with overall + sparse NDCG/HR (medium/dense omitted)."""
    headers = ["Cell", "NDCG@10", "HR@10", "Sparse NDCG@10", "Sparse HR@10"]
    lines: List[str] = []
    lines.append("# Phase 7 — Adaptive Module Ablation Table")
    lines.append("")
    lines.append("One-factor-at-a-time ablations from the main config. 3 seeds {42, 1337, 2026} per cell.")
    lines.append("Columns: Overall NDCG@10 + Sparse NDCG@10 (+ matching HR@10).")
    lines.append("Medium/dense per-group metrics are in per-run JSON artifacts.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [row["label"]]
        for metric in ("ndcg10", "hr10", "ndcg10_sparse", "hr10_sparse"):
            cells.append(fmt_cell(row.get(f"{metric}_mean"), row.get(f"{metric}_std")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_ablation_csv(rows: List[Dict[str, Any]]) -> str:
    """CSV variant of ablation table."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = ["label", "ablation_dimension", "ablation_value"]
    for metric in ("ndcg10", "hr10", "ndcg10_sparse", "hr10_sparse"):
        header.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    writer.writerow(header)
    for row in rows:
        out: List[Any] = [
            row["label"],
            row.get("ablation_dimension", ""),
            row.get("ablation_value", ""),
        ]
        for metric in ("ndcg10", "hr10", "ndcg10_sparse", "hr10_sparse"):
            for suffix in ("_mean", "_std", "_n"):
                v = row.get(f"{metric}{suffix}")
                out.append(f"{v:.4f}" if isinstance(v, float) else (v if v is not None else ""))
        writer.writerow(out)
    return buf.getvalue()


def _build_sparse_rows(records: List[ThesisResult]) -> List[Dict[str, Any]]:
    """Sparse-slice table: rows = main modules + ablation cells; columns = sparse NDCG/HR only."""
    main_rows = _build_main_rows(records)
    ablation_rows = _build_ablation_rows(records)
    out: List[Dict[str, Any]] = []
    # Main module rows.
    for r in main_rows:
        out.append({
            "label": f"main:{r['module']}",
            "module": r["module"],
            "ndcg10_sparse_mean": r.get("ndcg10_sparse_mean"),
            "ndcg10_sparse_std": r.get("ndcg10_sparse_std"),
            "ndcg10_sparse_n": r.get("ndcg10_sparse_n", 0),
            "hr10_sparse_mean": r.get("hr10_sparse_mean"),
            "hr10_sparse_std": r.get("hr10_sparse_std"),
            "hr10_sparse_n": r.get("hr10_sparse_n", 0),
        })
    # Ablation cell rows (skip the "main (reference)" row already covered above).
    for r in ablation_rows:
        if r["label"] == "main (reference)":
            continue
        out.append({
            "label": r["label"],
            "module": "adaptive",
            "ndcg10_sparse_mean": r.get("ndcg10_sparse_mean"),
            "ndcg10_sparse_std": r.get("ndcg10_sparse_std"),
            "ndcg10_sparse_n": r.get("ndcg10_sparse_n", 0),
            "hr10_sparse_mean": r.get("hr10_sparse_mean"),
            "hr10_sparse_std": r.get("hr10_sparse_std"),
            "hr10_sparse_n": r.get("hr10_sparse_n", 0),
        })
    return out


def render_sparse_md(rows: List[Dict[str, Any]]) -> str:
    """D-17 + THS-04 sparse-user slice: dedicated table for sparse-user NDCG/HR.

    Footnotes per-row n_seeds_with_sparse=K/3 if K < 3 (Pitfall 10 / CONTEXT discretion).
    """
    headers = ["Row", "Sparse NDCG@10", "Sparse HR@10", "Footnote"]
    lines: List[str] = []
    lines.append("# Phase 7 — Sparse-User Slice (THS-04 thesis-claim view)")
    lines.append("")
    lines.append("Sparse-user (interactions 0-30) NDCG@10 + HR@10 for every main module + every ablation cell.")
    lines.append("Bold = winner among comparable main rows under D-11.")
    lines.append("")
    # Identify main module rows for D-11 win highlighting.
    main_module_rows = [r for r in rows if r["label"].startswith("main:") and r["module"] in _COMPARABLE_MODULES]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        ndcg_cell = fmt_cell(row.get("ndcg10_sparse_mean"), row.get("ndcg10_sparse_std"))
        hr_cell = fmt_cell(row.get("hr10_sparse_mean"), row.get("hr10_sparse_std"))
        # Bold the NDCG winner among main comparable rows.
        if row in main_module_rows and row.get("ndcg10_sparse_mean") is not None:
            others = [
                (r["ndcg10_sparse_mean"], r["ndcg10_sparse_std"])
                for r in main_module_rows
                if r is not row and r.get("ndcg10_sparse_mean") is not None
            ]
            if is_winner(row["ndcg10_sparse_mean"], row["ndcg10_sparse_std"], others):
                ndcg_cell = f"**{ndcg_cell}**"
        # Partial-seed footnote (Pitfall 10).
        n = row.get("ndcg10_sparse_n", 0)
        footnote = f"n_seeds_with_sparse={n}/3" if 0 < n < 3 else ("none" if n == 0 else "")
        # PFedRec footnote marker.
        if row.get("module") == "pfedrec":
            ndcg_cell = f"{ndcg_cell} †"
        cells = [row["label"], ndcg_cell, hr_cell, footnote]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(_PFEDREC_FOOTNOTE)
    lines.append("")
    return "\n".join(lines)


def render_sparse_csv(rows: List[Dict[str, Any]]) -> str:
    """CSV variant of sparse-slice table."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([
        "label", "module",
        "ndcg10_sparse_mean", "ndcg10_sparse_std", "ndcg10_sparse_n",
        "hr10_sparse_mean", "hr10_sparse_std", "hr10_sparse_n",
    ])
    for row in rows:
        out: List[Any] = [row["label"], row.get("module", "")]
        for metric in ("ndcg10_sparse", "hr10_sparse"):
            for suffix in ("_mean", "_std", "_n"):
                v = row.get(f"{metric}{suffix}")
                out.append(f"{v:.4f}" if isinstance(v, float) else (v if v is not None else ""))
        writer.writerow(out)
    return buf.getvalue()


# ============================================================================
# Step 7 — top-level driver + CLI
# ============================================================================


def run_aggregator(
    results_root: Path,
    output_dir: Path,
    check_only: bool = False,
) -> int:
    """Top-level orchestration. Returns process exit code."""
    print(f"[INFO] Aggregator: results_root={results_root} output_dir={output_dir}")
    records = collect_thesis_results(results_root)
    print(f"[INFO] Collected {len(records)} thesis-tagged result records.")

    # D-20 missing-cell check.
    expected = expected_main_cells() | expected_ablation_cells()
    missing = find_missing_cells(records, expected)
    if missing:
        msg_lines: List[str] = [
            f"[D-20 HARD-FAIL] Missing {len(missing)} cells:",
        ]
        for m in missing:
            msg_lines.append(f"  - {m}")
        msg_lines.append("Run them then re-aggregate:")
        msg_lines.append("  python scripts/thesis/run_thesis_sweep.py --retry-failed")
        msg = "\n".join(msg_lines)
        print(msg, file=sys.stderr)
        raise SystemExit(1)

    if check_only:
        print(f"[INFO] --check-only: {len(records)} records present, expected set complete. No files written.")
        return 0

    # Render rows.
    main_rows = _build_main_rows(records)
    ablation_rows = _build_ablation_rows(records)
    sparse_rows = _build_sparse_rows(records)
    drift_note = _check_pfedrec_drift(records)

    # Write 6 output files atomically.
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(str(output_dir / _MAIN_MD), render_main_md(main_rows, drift_note))
    atomic_write_text(str(output_dir / _MAIN_CSV), render_main_csv(main_rows))
    atomic_write_text(str(output_dir / _ABL_MD), render_ablation_md(ablation_rows))
    atomic_write_text(str(output_dir / _ABL_CSV), render_ablation_csv(ablation_rows))
    atomic_write_text(str(output_dir / _SPARSE_MD), render_sparse_md(sparse_rows))
    atomic_write_text(str(output_dir / _SPARSE_CSV), render_sparse_csv(sparse_rows))

    print(f"[OK] 6 output files written to {output_dir}")
    return 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="aggregate_results.py")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override repo_root/results (defaults to <repo>/results). Used by tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (defaults to <results-root>/federated/_thesis).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify expected-cell-set match WITHOUT writing files (pre-aggregation gate).",
    )
    args = parser.parse_args(list(argv))
    results_root = args.results_root if args.results_root is not None else (_REPO_ROOT / "results")
    output_dir = args.output_dir if args.output_dir is not None else (results_root / "federated" / "_thesis")
    return run_aggregator(
        results_root=results_root,
        output_dir=output_dir,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
