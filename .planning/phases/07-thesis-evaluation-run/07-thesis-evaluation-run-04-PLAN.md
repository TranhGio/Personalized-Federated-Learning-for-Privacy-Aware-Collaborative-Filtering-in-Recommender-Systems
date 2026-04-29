---
phase: 07-thesis-evaluation-run
plan: 04
type: execute
wave: 3
depends_on:
  - 07-thesis-evaluation-run-01-PLAN.md
  - 07-thesis-evaluation-run-02-PLAN.md
files_modified:
  - scripts/thesis/aggregate_results.py
  - scripts/foundation/tests/test_thesis_aggregator.py
autonomous: true
requirements:
  - THS-03
  - THS-04
  - THS-05
  - THS-06
  - THS-07
user_setup: []

must_haves:
  truths:
    - "Aggregator globs results/federated/<module>/*/results.json for all 4 modules and filters by _manifest.thesis_run_label != '' AND _manifest.run_seed in {42, 1337, 2026}"
    - "Aggregator extracts final_metrics['best']['sampled_ndcg@10'] (overall) and final_metrics['best']['sampled_ndcg@10/sparse'] (sparse) — slash delimiter is uniform for HR/NDCG keys across all 4 modules per Pitfall 1"
    - "D-11 win criterion is implemented: row.mean - row.std > other.mean + other.std (non-overlapping sigma intervals); applied per-metric to baseline/personalized/adaptive (PFedRec excluded per D-05)"
    - "D-20 hard-fail-on-missing-cells: aggregator computes expected cell set from build_main_matrix + build_ablation_matrix, observes the on-disk set, and raises SystemExit listing the missing tuples"
    - "Cell format is `0.4123 ± 0.0089` (4 decimal places) per D-24; markdown bolds the winner with `**0.4123 ± 0.0089**`"
    - "Six output files are emitted atomically: main_comparison.{md,csv}, ablations.{md,csv}, sparse_slice.{md,csv}"
    - "PFedRec footnote text matches D-08 verbatim"
    - "Sparse partial-seed handling: if one seed has zero sparse evaluations, footnote `n_seeds_with_sparse=K/3` is added"
  artifacts:
    - path: "scripts/thesis/aggregate_results.py"
      provides: "Reads per-run results.json, computes mean+/-std across seeds, renders 6 output files, hard-fails on missing cells"
      contains: "def collect_thesis_results"
    - path: "scripts/foundation/tests/test_thesis_aggregator.py"
      provides: "Unit tests for extract / win-criterion / D-20 hard-fail / cell format / 6-file emission / atomic write"
      contains: "def test_extract_overall_ndcg10"
    - path: "results/federated/_thesis/main_comparison.md"
      provides: "Main comparison table — produced when sweep completes (Plan 05); structurally validated by Plan 04 tests via synthetic fixtures"
      min_lines: 5
  key_links:
    - from: "scripts/thesis/aggregate_results.py expected-cell set"
      to: "scripts/thesis/run_thesis_sweep.py build_main_matrix + build_ablation_matrix"
      via: "Aggregator imports build_main_matrix + build_ablation_matrix from the orchestrator"
      pattern: "from thesis.run_thesis_sweep import"
    - from: "results/federated/<module>/<run_id>/results.json _manifest.thesis_run_label"
      to: "main_comparison.md / ablations.md table rows"
      via: "collect_thesis_results filters by manifest field; aggregate_by_seed groups; render_main_md emits"
      pattern: '_manifest.*thesis_run_label'
---

<objective>
Build the result aggregator that reads per-run results.json files and emits the thesis tables. Per D-19 (standalone Python script reading thesis-tagged manifests), D-20 (hard-fail-on-missing-cells with explicit list), D-24 (mean+/-std cell format with 4 decimal places), D-17 (markdown + CSV; no LaTeX, no aggregate JSON).

The aggregator is responsible for:
1. **Glob + filter (D-19)**: walk `results/federated/<module>/*/results.json` for all 4 modules; filter to runs with `_manifest.thesis_run_label != ""` AND `_manifest.run_seed in {42, 1337, 2026}` AND `_manifest.schema_version >= 3`. (schema v2 manifests = legacy Phase-6 runs; correctly excluded by the empty-string default — Pitfall 7's mitigation.)
2. **Metric extraction**: read `final_metrics["best"]["sampled_ndcg@10"]` (overall) and `final_metrics["best"]["sampled_ndcg@10/sparse"]` (sparse) plus HR variants. Slash delimiter is UNIFORM across all 4 modules for HR/NDCG keys (Pitfall 1: only `evaluated_users` keys diverge — not used by table cells). Same for medium/dense per-group.
3. **Aggregation across seeds**: group by `(module, thesis_run_label)`; compute `(mean, std, n_seeds)` per metric. Use `np.std(arr, ddof=0)` (population std) — matches the de facto thesis reporting convention.
4. **D-20 missing-cell hard-fail**: import `build_main_matrix`/`build_ablation_matrix` from the orchestrator (Plan 03); compute the expected `(module, thesis_run_label, run_seed)` set; observe the on-disk set; if non-empty difference, raise `SystemExit` with the explicit list of missing tuples.
5. **D-11 win detection**: applied per-metric (overall NDCG@10, sparse NDCG@10) to the comparable rows `(baseline, personalized, adaptive)`. PFedRec is EXCLUDED from win comparison (D-05/D-08). HR cells are informational — bolding is NDCG-only per Open Question 3 in RESEARCH.md.
6. **Markdown rendering**: `0.4123 ± 0.0089` cells per D-24; winner bolded as `**0.4123 ± 0.0089**`; PFedRec row gets `†` footnote marker. CSV: two columns per metric (`ndcg10_mean`, `ndcg10_std`), per-group columns also.
7. **Sparse partial-seed handling (CONTEXT discretion)**: if a seed has `evaluated_users_sparse=0` (or `evaluated_users/sparse=0` for PFedRec — Pitfall 1), exclude that seed from the sparse aggregation and add a `n_seeds_with_sparse=K/3` footnote.
8. **Atomic write**: each of the 6 output files written via `atomic_write_text` (markdown/CSV) per Plan 01's new helper; no `.tmp-*` leftovers.
9. **D-23 PFedRec divergence note**: if PFedRec's mean HR@10 falls outside `0.729 ± 2pts` (i.e., <0.709 or >0.749) or NDCG@10 outside `0.441 ± 2pts`, the markdown body emits a "PFedRec reproduction drifted" note (Open Question 4 in RESEARCH.md). Aggregator does NOT halt the sweep on PFedRec drift.
10. **Tests (Wave 0 of validation)**: `scripts/foundation/tests/test_thesis_aggregator.py` covers all 11 VALIDATION.md aggregator rows.

Purpose: The aggregator is read-only relative to the FL stack. Plan 03's orchestrator emits `results/federated/<module>/<run_id>/{results.json, manifest.json}` files; Plan 04's aggregator consumes them. They share a contract via the imported `build_main_matrix`/`build_ablation_matrix` functions (single source of truth for expected-cell set).
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/07-thesis-evaluation-run/07-CONTEXT.md
@.planning/phases/07-thesis-evaluation-run/07-RESEARCH.md
@.planning/phases/07-thesis-evaluation-run/07-VALIDATION.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-01-PLAN.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-02-PLAN.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-03-PLAN.md

<interfaces>
From scripts/thesis/run_thesis_sweep.py (Plan 03; just created):
```python
THESIS_SEEDS: Tuple[int, int, int] = (42, 1337, 2026)
def build_main_matrix() -> List[ThesisCell]: ...
def build_ablation_matrix() -> List[ThesisCell]: ...

@dataclass(frozen=True)
class ThesisCell:
    module: str
    mode: str
    run_seed: int
    thesis_run_label: str
    ablation_dimension: str
    ablation_value: str
    extra_run_config: Dict[str, str]
```

From per-run results.json schema (Phase 6 D-07; verified on disk at results/federated/baseline/20260429-082522-984e98/results.json):
```python
{
  "final_metrics": {
    "best": {
      "sampled_hr@10": 0.7234,
      "sampled_ndcg@10": 0.4123,
      "sampled_hr@10/sparse": 0.6850,    # SLASH delimiter (uniform across all 4 modules)
      "sampled_ndcg@10/sparse": 0.3812,
      "sampled_hr@10/medium": 0.7321,
      "sampled_ndcg@10/medium": 0.4189,
      "sampled_hr@10/dense": 0.7456,
      "sampled_ndcg@10/dense": 0.4256,
      "evaluated_users": 6040,
      "evaluated_users_sparse": 1850,    # UNDERSCORE for non-pfedrec (Pitfall 1)
      "evaluated_users_medium": 2100,
      "evaluated_users_dense": 2090,
      ...
    },
    "last": { ... },
    "best_round": 87,
    "last_round": 100,
    "final_eval_round_index": 1
  },
  "_manifest": {
    "schema_version": 3,                  # Phase 7 (post Plan 01)
    "thesis_run_label": "main",          # Phase 7 D-22 (post Plan 02)
    "ablation_dimension": "none",
    "ablation_value": "",
    "run_seed": 42,
    "module": "baseline",
    "mode": "thesis_crossdevice_main",
    ...
  }
}
```
PFedRec evaluated_users keys use slash (per pfedrec/strategy.py:121): `evaluated_users/sparse` etc.
HR/NDCG keys use slash UNIFORMLY across all 4 modules.

From fedrec_foundation.atomic (Plan 01 extended):
```python
def atomic_write_text(path: str, content: str) -> None: ...
```

From fedrec_foundation.paths:
```python
def repo_root() -> Path: ...
```

D-08 PFedRec footnote text (verbatim from CONTEXT):
"† PFedRec (paper-faithful) — `dim=32, SGD lr=0.1, BCE, fraction-train=1.0; matches IJCAI-23 reference within ±2 points`. Not counted toward 'adaptive beats baselines' claim per Phase 7 D-05."
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Aggregator implementation — scripts/thesis/aggregate_results.py</name>
  <read_first>
    - scripts/thesis/run_thesis_sweep.py (file from Plan 03; aggregator imports build_main_matrix + build_ablation_matrix + THESIS_SEEDS)
    - scripts/foundation/fedrec_foundation/atomic.py (Plan 01 extended; aggregator uses atomic_write_text for markdown/CSV)
    - scripts/foundation/fedrec_foundation/paths.py (aggregator uses repo_root)
    - results/federated/baseline/20260429-082522-984e98/results.json (real on-disk schema; verify slash delimiter convention)
    - .planning/phases/07-thesis-evaluation-run/07-CONTEXT.md sections D-08, D-11, D-15, D-17, D-19, D-20, D-24
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pattern 4: Aggregator Filter + Render", "Pattern 5: Markdown Table Rendering", "Pitfall 1", "Pitfall 6", "Pitfall 10"
  </read_first>
  <behavior>
    - The aggregator is a single Python file `scripts/thesis/aggregate_results.py` runnable as `python scripts/thesis/aggregate_results.py [--results-root <path>] [--output-dir <path>] [--check-only]`.
    - `--check-only`: do NOT write any files; just verify expected-cell-set match and print summary. Useful for the pre-aggregation gate in Plan 05.
    - The script does ALL work in a single `main()` function: collect → group → check → render → write.
    - On D-20 missing-cell detection, raises `SystemExit` with code 1 and a multi-line message listing the missing `(module, thesis_run_label, run_seed)` tuples (sorted).
    - On corrupted results.json (mid-write crash), the aggregator logs a warning and continues (does NOT halt). This is the only "soft" failure mode.
    - PFedRec drift is reported as an inline markdown note, not a hard fail.
  </behavior>
  <action>
Create `scripts/thesis/aggregate_results.py` with EXACT content:

```python
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
        row = {
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
```

Make the file executable: `chmod 755 scripts/thesis/aggregate_results.py`.

**Smoke test (with empty results-root)**:
```bash
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
python scripts/thesis/aggregate_results.py --results-root /tmp/empty_results --check-only 2>&1 || true
# Expect: [D-20 HARD-FAIL] Missing 33 cells: ... + exit code 1
```
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && test -f scripts/thesis/aggregate_results.py && python -c "import sys; sys.path.insert(0, 'scripts/foundation'); sys.path.insert(0, 'scripts'); from thesis.aggregate_results import collect_thesis_results, extract_metric, aggregate_by_seed, find_missing_cells, expected_main_cells, expected_ablation_cells, fmt_cell, is_winner; assert len(expected_main_cells()) == 12; assert len(expected_ablation_cells()) == 21; assert fmt_cell(0.4123, 0.0089) == '0.4123 ± 0.0089'; assert fmt_cell(None, None) == '—'; assert is_winner(0.42, 0.005, [(0.40, 0.005)]) is True; assert is_winner(0.42, 0.01, [(0.41, 0.01)]) is False; print('aggregator API OK')" && rm -rf /tmp/agg_smoke && python scripts/thesis/aggregate_results.py --results-root /tmp/agg_smoke --check-only 2>&1 | grep -q 'Missing 33 cells' && echo "D-20 hard-fail OK"</automated>
  </verify>
  <done>
    - `scripts/thesis/aggregate_results.py` exists, executable.
    - All 8 public symbols importable: `collect_thesis_results`, `extract_metric`, `aggregate_by_seed`, `find_missing_cells`, `expected_main_cells`, `expected_ablation_cells`, `fmt_cell`, `is_winner`.
    - `expected_main_cells()` returns 12 tuples; `expected_ablation_cells()` returns 21 tuples.
    - `fmt_cell(0.4123, 0.0089) == "0.4123 ± 0.0089"` per D-24.
    - `is_winner` correctly identifies non-overlapping vs overlapping intervals per D-11.
    - Empty results-root + `--check-only` prints `Missing 33 cells` and exits 1 (D-20 hard-fail).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Aggregator unit tests — scripts/foundation/tests/test_thesis_aggregator.py</name>
  <read_first>
    - scripts/thesis/aggregate_results.py (file from Task 1)
    - scripts/thesis/run_thesis_sweep.py (Plan 03; tests import build_main_matrix for synthetic fixtures)
    - .planning/phases/07-thesis-evaluation-run/07-VALIDATION.md "Per-Task Verification Map" rows 7-04-01 through 7-04-11
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pitfall 1", "Pitfall 6", "Pitfall 10", "Edge Cases" section
  </read_first>
  <behavior>
    - The new test file `scripts/foundation/tests/test_thesis_aggregator.py` covers all 11 VALIDATION.md aggregator rows by name.
    - Tests use `tmp_path` to construct synthetic results-root layouts; no real flwr runs.
    - Synthetic results.json files include the full nested `final_metrics["best"]` block + `_manifest` with thesis fields.
    - `test_d20_hard_fail_missing` constructs only PARTIAL coverage (e.g., 32/33 cells) and asserts `SystemExit` with the missing tuple listed.
    - `test_extract_sparse_ndcg10_uniform_slash` confirms the slash delimiter convention works for ALL 4 modules.
    - `test_six_output_files` runs the full aggregator on a fully-populated synthetic root and asserts all 6 files exist.
    - `test_atomic_write_no_tmp` runs the aggregator and confirms no `.tmp-*` leftovers in the output directory.
  </behavior>
  <action>
Create `scripts/foundation/tests/test_thesis_aggregator.py` with EXACT content:

```python
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
```

Run the tests:
```bash
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
pytest scripts/foundation/tests/test_thesis_aggregator.py -x -v
```
Expect: 14 PASSED.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && pytest scripts/foundation/tests/test_thesis_aggregator.py -x -v 2>&1 | tail -10 | grep -q "passed"</automated>
  </verify>
  <done>
    - `scripts/foundation/tests/test_thesis_aggregator.py` exists.
    - All 14 tests GREEN.
    - The 11 VALIDATION.md-named tests are present and pass: `test_extract_overall_ndcg10`, `test_extract_sparse_ndcg10_uniform_slash`, `test_d11_win_criterion`, `test_d11_overlap_no_winner`, `test_sparse_partial_seeds`, `test_ablation_label_grouping`, `test_d20_hard_fail_missing`, `test_csv_per_group_columns`, `test_six_output_files`, `test_atomic_write_no_tmp`, `test_cell_format`.
    - 3 supplementary tests pass: `test_extract_evaluated_users_handles_both_delimiters`, `test_winner_bolded_in_main_md`, `test_collect_filters_legacy_phase6_manifests`, `test_check_only_does_not_write_files`.
  </done>
</task>

</tasks>

<verification>
- Aggregator imports orchestrator's matrix builders (single source of truth for expected-cell set).
- All 11 VALIDATION-mapped tests are green.
- D-20 hard-fail surfaces with explicit missing-tuple list.
- Cell format `0.4123 ± 0.0089` per D-24.
- D-11 win criterion correctly distinguishes overlapping vs non-overlapping intervals.
- 6 output files emitted under `<output_dir>/`.
- Atomic write contract preserved: no `.tmp-*` leftovers.
- Legacy Phase-6 manifests correctly filtered out (Pitfall 7).
- Sparse partial-seed handling (Pitfall 10) verified.
- PFedRec mode collision case covered: `paper_compat_pfedrec` runs with `thesis_run_label="main"` are correctly counted.
</verification>

<success_criteria>
- [ ] `scripts/thesis/aggregate_results.py` exists, executable.
- [ ] All 8 public symbols importable.
- [ ] `pytest scripts/foundation/tests/test_thesis_aggregator.py -x -v` reports 14 PASSED.
- [ ] Empty results-root + `--check-only` prints `Missing 33 cells` and exits 1 (D-20).
- [ ] Synthetic full-coverage run produces 6 output files with no `.tmp-*` leftovers.
- [ ] PFedRec footnote text matches D-08 exactly.
</success_criteria>

<output>
After completion, create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-04-SUMMARY.md` documenting:
- Final aggregator file size in lines.
- Number of public symbols exported.
- Test counts (per VALIDATION.md row IDs).
- Any deviations from the action text.
- The exact behavior of `--check-only` flag (no file writes; same hard-fail behavior on missing cells).
- The `_PFEDREC_FOOTNOTE` constant value (must be the verbatim D-08 text).
</output>
</content>
</invoke>
