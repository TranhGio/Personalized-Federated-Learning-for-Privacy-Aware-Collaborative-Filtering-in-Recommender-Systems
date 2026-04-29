#!/usr/bin/env python
"""Phase 7 thesis-comparison orchestrator (D-18).

Fires ``scripts/run.py <module> <mode>`` per cell of the thesis run matrix
and ablation matrix. Matrix-as-data; idempotent skip-on-existing; skip+log
+continue on cell failure; --retry-failed reads disk presence (D-23).

Usage
-----
    python scripts/thesis/run_thesis_sweep.py --phase=main
    python scripts/thesis/run_thesis_sweep.py --phase=ablation
    python scripts/thesis/run_thesis_sweep.py --phase=all
    python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --dry-run
    python scripts/thesis/run_thesis_sweep.py --retry-failed

Output paths
------------
- ``results/federated/_thesis/_progress.json``      (per-cell progress)
- ``results/federated/_thesis/failed_cells.json``   (cell-failure log; D-23)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import path bootstrap: scripts/thesis/run_thesis_sweep.py runs as a script.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
_FOUNDATION_PKG = _REPO_ROOT / "scripts" / "foundation"
if str(_FOUNDATION_PKG) not in sys.path:
    sys.path.insert(0, str(_FOUNDATION_PKG))

from fedrec_foundation.atomic import atomic_write_json  # noqa: E402

# ============================================================================
# Constants (D-09 + D-10 + D-13 + D-14 + D-21)
# ============================================================================

THESIS_SEEDS: Tuple[int, int, int] = (42, 1337, 2026)
"""D-10 canonical seed set."""

MAIN_MODULES_CROSSDEVICE: Tuple[str, ...] = ("baseline", "personalized", "adaptive")
"""Modules that run at thesis_crossdevice_main mode (D-04)."""

# D-13 ablation knobs: each tuple is (ablation_dimension, ablation_value, extra_run_config_dict).
ABLATION_KNOBS: List[Tuple[str, str, Dict[str, str]]] = [
    ("alpha_method",       "multi_factor",  {"alpha-method": "multi_factor"}),
    ("alpha_method",       "data_quantity", {"alpha-method": "data_quantity"}),
    ("per_user_alpha",     "true",          {"enable-per-user-alpha": "true"}),
    ("item_perturbation",  "true",          {"enable-item-perturbation": "true",
                                              "item-perturbation-reg": "0.01"}),
    ("contrastive_lambda", "0.1",           {"contrastive-lambda": "0.1",
                                              "contrastive-tau": "0.1"}),
    ("fusion_type",        "add",           {"fusion-type": "add"}),
    ("fusion_type",        "gate",          {"fusion-type": "gate"}),
]

# D-21 short-form mapping for W&B run names.
_ABLATION_SHORT_NAME: Dict[str, str] = {
    "alpha_method":       "alpha",
    "per_user_alpha":     "pua",
    "item_perturbation":  "ip",
    "contrastive_lambda": "cl",
    "fusion_type":        "fusion",
}

# Per-module base overrides for thesis runs (D-02, D-03 enforcement).
# These overrides MUST be merged into every thesis cell's run-config BEFORE
# cell.extra_run_config so that ablation cells can override knobs on top of
# the locked base. The merge order is: THESIS_BASE_OVERRIDES[module] -> cell.extra_run_config.
#
# - All 4 modules: strategy=fedavg (D-03 — no FedProx in thesis runs).
# - Adaptive only: model-type=dual + alpha-method=hierarchical_conditional (D-02 — main config).
# - Adaptive only: enable-per-user-alpha=false + enable-item-perturbation=false +
#   contrastive-lambda=0.0 (D-02 — next-gen knobs OFF in main; ablation cells override
#   each one in turn). The adaptive pyproject.toml defaults these to true/0.1, so we
#   MUST disable them explicitly for the main-comparison row.
# - PFedRec / baseline / personalized: NO model-type/alpha-method overrides
#   (those keys do not exist in their pyproject.toml).
THESIS_BASE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "baseline":     {"strategy": "fedavg"},
    "personalized": {"strategy": "fedavg"},
    "adaptive":     {
        "strategy": "fedavg",
        "model-type": "dual",
        "alpha-method": "hierarchical_conditional",
        "enable-per-user-alpha": "false",
        "enable-item-perturbation": "false",
        "contrastive-lambda": "0.0",
    },
    "pfedrec":      {"strategy": "fedavg"},
}


# ============================================================================
# ThesisCell dataclass
# ============================================================================


@dataclass(frozen=True)
class ThesisCell:
    """One cell of the thesis run matrix.

    Attributes
    ----------
    module : str
        One of ``"baseline"``, ``"personalized"``, ``"adaptive"``, ``"pfedrec"``.
    mode : str
        ``"thesis_crossdevice_main"`` (main runs) or ``"paper_compat_pfedrec"`` (PFedRec only — D-06).
    run_seed : int
        One of ``{42, 1337, 2026}`` (D-10).
    thesis_run_label : str
        ``"main"`` or ``"ablation_<knob>=<value>"``.
    ablation_dimension : str
        ``"none"`` for main, knob name for ablation.
    ablation_value : str
        ``""`` for main, specific value for ablation.
    extra_run_config : Dict[str, str]
        Additional ``--run-config`` overrides (e.g., ``{"alpha-method": "multi_factor"}``).
    """
    module: str
    mode: str
    run_seed: int
    thesis_run_label: str
    ablation_dimension: str
    ablation_value: str
    extra_run_config: Dict[str, str] = field(default_factory=dict)

    @property
    def identity(self) -> Tuple[str, str, int, str, str]:
        """Pitfall 8: full cell identity tuple — used for skip-on-existing matching."""
        return (
            self.module,
            self.thesis_run_label,
            self.run_seed,
            self.ablation_dimension,
            self.ablation_value,
        )


# ============================================================================
# Matrix builders (D-13 + D-14)
# ============================================================================


def build_main_matrix() -> List[ThesisCell]:
    """4 modules x 3 seeds = 12 main cells. PFedRec uses paper_compat_pfedrec mode (D-06)."""
    cells: List[ThesisCell] = []
    for seed in THESIS_SEEDS:
        for module in MAIN_MODULES_CROSSDEVICE:
            cells.append(ThesisCell(
                module=module,
                mode="thesis_crossdevice_main",
                run_seed=seed,
                thesis_run_label="main",
                ablation_dimension="none",
                ablation_value="",
                extra_run_config={},
            ))
        # PFedRec calibration reference (D-05/D-06).
        cells.append(ThesisCell(
            module="pfedrec",
            mode="paper_compat_pfedrec",
            run_seed=seed,
            thesis_run_label="main",
            ablation_dimension="none",
            ablation_value="",
            extra_run_config={},
        ))
    return cells


def build_ablation_matrix() -> List[ThesisCell]:
    """7 ablation knobs x 3 seeds = 21 ablation cells. All adaptive at thesis_crossdevice_main (D-13)."""
    cells: List[ThesisCell] = []
    for seed in THESIS_SEEDS:
        for ablation_dim, ablation_val, extra_cfg in ABLATION_KNOBS:
            label = f"ablation_{ablation_dim}={ablation_val}"
            cells.append(ThesisCell(
                module="adaptive",
                mode="thesis_crossdevice_main",
                run_seed=seed,
                thesis_run_label=label,
                ablation_dimension=ablation_dim,
                ablation_value=ablation_val,
                extra_run_config=dict(extra_cfg),
            ))
    return cells


# ============================================================================
# Skip-on-existing (D-18 idempotence + Pitfall 8)
# ============================================================================


def cell_already_done(cell: ThesisCell, results_root: Path) -> bool:
    """D-18 idempotent skip: scan ``results/federated/<module>/*/manifest.json``
    for any run whose ``_manifest`` matches the cell's full identity tuple.

    Pitfall 8: matches on ``(module, thesis_run_label, run_seed, ablation_dimension,
    ablation_value)`` — NOT on ``(module, seed)`` alone (which collides at seed=42
    where adaptive runs 8 times: 1 main + 7 ablations).
    """
    module_dir = results_root / "federated" / cell.module
    if not module_dir.exists():
        return False
    for manifest_path in module_dir.glob("*/manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        # Tuple match — cell.identity vs manifest fields.
        observed = (
            cell.module,
            m.get("thesis_run_label", ""),
            int(m.get("run_seed", -1)),
            m.get("ablation_dimension", "none"),
            m.get("ablation_value", ""),
        )
        if observed == cell.identity:
            return True
    return False


# ============================================================================
# Run-config string builder
# ============================================================================


def _wandb_run_name(cell: ThesisCell) -> str:
    """D-21 W&B run-name pattern."""
    if cell.thesis_run_label == "main":
        return f"thesis-main-{cell.module}-seed{cell.run_seed}"
    # Ablation: thesis-ablation-<module>-seed<N>-<short_knob>=<value>.
    short = _ABLATION_SHORT_NAME.get(cell.ablation_dimension, cell.ablation_dimension)
    return f"thesis-ablation-{cell.module}-seed{cell.run_seed}-{short}={cell.ablation_value}"


def cell_run_config_string(cell: ThesisCell) -> str:
    """Build the ``--run-config`` string for ``scripts/run.py``.

    Order of keys (later keys override earlier ones in flwr's fuse_dicts):
      1. ``run-seed``, ``thesis-run-label``, ``ablation-dimension``,
         ``ablation-value``, ``wandb-run-name``  — provenance metadata.
      2. ``THESIS_BASE_OVERRIDES[cell.module]``  — D-02 + D-03 enforced base
         (strategy=fedavg for all; model-type=dual + alpha-method=hierarchical_conditional
         + next-gen knobs OFF for adaptive).
      3. ``cell.extra_run_config``               — ablation cells override the
         base where they conflict (e.g. ``alpha-method=multi_factor`` overrides
         the adaptive base ``alpha-method=hierarchical_conditional``).

    Bare-word string values are passed RAW; scripts/run.py's
    ``_quote_value_for_flwr`` adds TOML quoting downstream.
    """
    parts: List[str] = [f"run-seed={cell.run_seed}"]
    parts.append(f"thesis-run-label={cell.thesis_run_label}")
    parts.append(f"ablation-dimension={cell.ablation_dimension}")
    parts.append(f"ablation-value={cell.ablation_value}")
    parts.append(f"wandb-run-name={_wandb_run_name(cell)}")
    # D-02 + D-03 base overrides BEFORE extra_run_config so ablation cells win
    # where they conflict (e.g. alpha-method ablation overrides hierarchical_conditional).
    base_overrides = THESIS_BASE_OVERRIDES.get(cell.module, {})
    merged: Dict[str, str] = dict(base_overrides)
    merged.update(cell.extra_run_config)
    for k, v in merged.items():
        parts.append(f"{k}={v}")
    return " ".join(parts)


# ============================================================================
# Cell execution (subprocess wrapper)
# ============================================================================


def execute_cell(
    cell: ThesisCell,
    repo_root: Path,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Fire scripts/run.py for one cell. Return (success, stderr_excerpt).

    Captures stdout+stderr; on non-zero exit, returns (False, stderr_tail_2KB).
    On dry_run, prints the would-be command and returns (True, "").
    """
    cmd: List[str] = [
        sys.executable,
        str(repo_root / "scripts" / "run.py"),
        cell.module,
        cell.mode,
        "--run-config",
        cell_run_config_string(cell),
    ]
    if dry_run:
        # Quote args containing spaces for human readability in the dry-run print.
        printable = " ".join(repr(x) if " " in x else x for x in cmd)
        print(f"[DRY-RUN] {printable}")
        return True, ""
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    return True, ""


# ============================================================================
# Failure log + progress writer (D-23)
# ============================================================================


def _failed_cells_path(results_root: Path) -> Path:
    return results_root / "federated" / "_thesis" / "failed_cells.json"


def _progress_path(results_root: Path) -> Path:
    return results_root / "federated" / "_thesis" / "_progress.json"


def _append_failure(
    results_root: Path,
    cell: ThesisCell,
    stderr_excerpt: str,
) -> None:
    """D-23: append cell-failure record to failed_cells.json (read-modify-write atomic)."""
    path = _failed_cells_path(results_root)
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing = []
    existing.append({
        "module": cell.module,
        "mode": cell.mode,
        "run_seed": cell.run_seed,
        "thesis_run_label": cell.thesis_run_label,
        "ablation_dimension": cell.ablation_dimension,
        "ablation_value": cell.ablation_value,
        "stderr_excerpt": stderr_excerpt,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_write_json(str(path), existing)


def _write_progress(
    results_root: Path,
    completed: int,
    failed: int,
    remaining: int,
    last_cell: Optional[ThesisCell],
    elapsed_sec: float,
) -> None:
    """Write _progress.json after every cell completion."""
    last_identity: Optional[List[Any]] = None
    if last_cell is not None:
        last_identity = list(last_cell.identity)
    payload = {
        "completed": completed,
        "failed": failed,
        "remaining": remaining,
        "last_cell": last_identity,
        "elapsed_sec": round(elapsed_sec, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(str(_progress_path(results_root)), payload)


# ============================================================================
# Sweep driver
# ============================================================================


def _filter_cells(
    cells: List[ThesisCell],
    module: Optional[str],
    seed: Optional[int],
) -> List[ThesisCell]:
    """Apply CLI filters (--module, --seed) to the matrix."""
    out = cells
    if module is not None:
        out = [c for c in out if c.module == module]
    if seed is not None:
        out = [c for c in out if c.run_seed == seed]
    return out


def run_sweep(
    cells: List[ThesisCell],
    repo_root: Path,
    results_root: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> Tuple[int, int, int]:
    """Run all cells sequentially. Returns (completed, failed, skipped)."""
    completed = 0
    failed = 0
    skipped = 0
    started = time.time()
    for idx, cell in enumerate(cells):
        if skip_existing and cell_already_done(cell, results_root):
            skipped += 1
            print(f"[SKIP] cell {idx + 1}/{len(cells)}: {cell.identity} — already on disk")
            continue
        print(f"[RUN]  cell {idx + 1}/{len(cells)}: {cell.identity}")
        success, stderr_excerpt = execute_cell(cell, repo_root, dry_run=dry_run)
        if success:
            completed += 1
        else:
            failed += 1
            _append_failure(results_root, cell, stderr_excerpt)
            print(f"[FAIL] cell {cell.identity} — appended to failed_cells.json")
        if not dry_run:
            _write_progress(
                results_root,
                completed=completed,
                failed=failed,
                remaining=len(cells) - completed - failed - skipped,
                last_cell=cell,
                elapsed_sec=time.time() - started,
            )
    return completed, failed, skipped


def main(argv: Sequence[str]) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="run_thesis_sweep.py")
    parser.add_argument(
        "--phase",
        choices=("main", "ablation", "all"),
        default="all",
        help="Which matrix to run (default: all).",
    )
    parser.add_argument(
        "--module",
        choices=("baseline", "personalized", "adaptive", "pfedrec"),
        default=None,
        help="Smoke-filter: run only cells for this module.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Smoke-filter: run only cells for this seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without invoking subprocess.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only cells whose results.json is still missing on disk (D-23 + D-31 default).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override repo_root/results (defaults to <repo>/results). Used by tests.",
    )
    args = parser.parse_args(list(argv))

    repo_root = _REPO_ROOT
    results_root = args.results_root if args.results_root is not None else (repo_root / "results")

    # Build the requested matrix (or matrices).
    main_cells = build_main_matrix() if args.phase in ("main", "all") else []
    ablation_cells = build_ablation_matrix() if args.phase in ("ablation", "all") else []
    cells = main_cells + ablation_cells
    cells = _filter_cells(cells, args.module, args.seed)

    if not cells:
        print(f"[INFO] No cells to run for phase={args.phase} module={args.module} seed={args.seed}")
        return 0

    print(f"[INFO] About to run {len(cells)} cells (phase={args.phase}, dry_run={args.dry_run})")

    # On --retry-failed, the cell list IS still the full matrix; skip-on-existing
    # naturally filters out cells whose results.json is on disk. This implements
    # D-31 ("filter by disk presence") without an explicit failed_cells.json read.
    skip_existing = True

    completed, failed, skipped = run_sweep(
        cells,
        repo_root=repo_root,
        results_root=results_root,
        dry_run=args.dry_run,
        skip_existing=skip_existing,
    )

    print(
        f"[SUMMARY] completed={completed} failed={failed} skipped={skipped} "
        f"of {len(cells)} cells (dry_run={args.dry_run})"
    )
    if failed > 0:
        print(
            f"[RECOVERY] {failed} cells failed. "
            f"Re-run with: python scripts/thesis/run_thesis_sweep.py --retry-failed"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
