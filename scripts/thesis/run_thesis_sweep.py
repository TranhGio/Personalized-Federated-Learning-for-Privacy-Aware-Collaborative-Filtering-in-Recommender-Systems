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
#
# Protocol pins (sweep-unblocker review, 2026-06-12): the pins below lock the
# C=0.1 thesis protocol for every claim cell — pyproject values override
# ModeProfile defaults, so relying on per-module pyproject.toml alone is not
# protocol-locked.
# - pfedrec keeps mode=paper_compat_pfedrec but at fraction-train=0.1: matches
#   the credited ef41ab bar run (paper-compat fraction=1.0 would be ~6 days/cell).
# - baseline pins embedding-dim=128 to match its credited seed-42 run 1bf513 —
#   PROVISIONAL: this perpetuates a capacity confound (other modules run
#   embedding-dim=64) and is flagged for review.
# - early-stopping-enabled=false sweep-wide so all runs complete 100 rounds and
#   final-eval negatives stay bit-paired across modules/seeds.
# - strict-d06-cache=true makes claim runs fail closed on cold-eval (knob being
#   added in a parallel patch). flwr fuse_dicts REJECTS undeclared keys, so the
#   key is declared (default false) in the personalized/pfedrec/adaptive
#   pyprojects; baseline has no per-user local state, hence no guard and no pin.
THESIS_BASE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "baseline":     {
        "strategy": "fedavg",
        "early-stopping-enabled": "false",
        "embedding-dim": "128",
    },
    "personalized": {
        "strategy": "fedavg",
        "embedding-dim": "64",
        "lr": "0.005",
        "early-stopping-enabled": "false",
        "final-calibration-enabled": "false",
        "strict-d06-cache": "true",
    },
    "adaptive":     {
        "strategy": "fedavg",
        "model-type": "dual",
        "alpha-method": "hierarchical_conditional",
        "enable-per-user-alpha": "false",
        "enable-item-perturbation": "false",
        "contrastive-lambda": "0.0",
        "num-server-rounds": "100",
        "fraction-train": "0.1",
        "local-epochs": "1",
        "embedding-dim": "64",
        "lr": "0.005",
        "early-stopping-enabled": "false",
        "final-calibration-enabled": "true",
        "strict-d06-cache": "true",
    },
    "pfedrec":      {
        "strategy": "fedavg",
        "fraction-train": "0.1",
        "early-stopping-enabled": "false",
        "final-calibration-enabled": "false",
        "strict-d06-cache": "true",
    },
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

    Backfill-aware credit (run-id audit, 2026-06-12): ALSO returns True when a
    run dir's ``EVAL_VALIDITY.json`` sidecar (written by
    ``scripts/thesis/eval_validity.py``) has ``status == "valid"`` AND
    (manifest ``thesis_run_label`` OR sidecar ``thesis_run_label_backfill``)
    matches the cell label AND (manifest ``run_seed`` OR sidecar
    ``run_seed_backfill``) matches the cell seed. This makes the sweep skip
    cells credited by backfilled valid runs (pfedrec ef41ab seed42, baseline
    1bf513 seed42, personalized f18e64 seed42) instead of re-running them —
    mirroring ``aggregate_results.collect_thesis_results`` (manifest label takes
    precedence; seed backfill applies only when the manifest seed is non-canonical).

    Conversely, a manifest identity match whose sidecar explicitly says
    ``status != "valid"`` (e.g. adaptive bc134c — invalid_cold_eval) does NOT
    credit the cell: the aggregator would reject that run anyway, and crediting
    it would deadlock the sweep (cell skipped here, missing at D-20). A run
    with NO sidecar keeps the legacy credit behavior.
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
            # Sidecar veto: an explicitly non-valid run never credits a cell.
            sidecar_path = manifest_path.parent / "EVAL_VALIDITY.json"
            if sidecar_path.exists():
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        if json.load(f).get("status") != "valid":
                            continue
                except (OSError, json.JSONDecodeError):
                    pass  # unreadable sidecar -> legacy behavior (credit)
            return True
    # Pass 2: backfilled-valid credit via EVAL_VALIDITY.json sidecars.
    for sidecar_path in module_dir.glob("*/EVAL_VALIDITY.json"):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if sidecar.get("status") != "valid":
            continue
        if sidecar.get("module", cell.module) != cell.module:
            continue
        manifest: Dict[str, Any] = {}
        manifest_path = sidecar_path.parent / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                manifest = {}
        # Manifest label takes precedence; backfill never overrides a non-empty label.
        label = manifest.get("thesis_run_label", "") or sidecar.get("thesis_run_label_backfill", "")
        if str(label) != cell.thesis_run_label:
            continue
        try:
            seed_int = int(manifest.get("run_seed", -1))
        except (TypeError, ValueError):
            seed_int = -1
        if seed_int not in THESIS_SEEDS and "run_seed_backfill" in sidecar:
            try:
                seed_int = int(sidecar["run_seed_backfill"])
            except (TypeError, ValueError):
                continue
        if seed_int == cell.run_seed:
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

    Streams stdout to the terminal (so per-round server_app prints are visible)
    while capturing stderr for failure diagnostics. On non-zero exit, returns
    (False, stderr_tail_2KB). On dry_run, prints the would-be command and
    returns (True, "").
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
        stdout=None,                # inherit parent stdout — stream live
        stderr=subprocess.PIPE,     # still capture for failure log
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or "")[-2000:]
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
        ts_start = time.strftime("%H:%M:%S")
        print(f"[RUN]  cell {idx + 1}/{len(cells)}: {cell.identity} — started {ts_start}", flush=True)
        cell_started = time.time()
        success, stderr_excerpt = execute_cell(cell, repo_root, dry_run=dry_run)
        cell_secs = time.time() - cell_started
        cell_mins = cell_secs / 60.0
        if success:
            completed += 1
            print(f"[OK]   cell {idx + 1}/{len(cells)}: {cell.identity} — {cell_mins:.1f}m "
                  f"(completed={completed} failed={failed} skipped={skipped} of {len(cells)})", flush=True)
        else:
            failed += 1
            _append_failure(results_root, cell, stderr_excerpt)
            print(f"[FAIL] cell {cell.identity} — {cell_mins:.1f}m, appended to failed_cells.json", flush=True)
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
