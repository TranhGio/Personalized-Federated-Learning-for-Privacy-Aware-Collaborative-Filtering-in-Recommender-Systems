---
phase: 07-thesis-evaluation-run
plan: 03
type: execute
wave: 3
depends_on:
  - 07-thesis-evaluation-run-01-PLAN.md
  - 07-thesis-evaluation-run-02-PLAN.md
files_modified:
  - scripts/thesis/__init__.py
  - scripts/thesis/run_thesis_sweep.py
  - scripts/foundation/tests/test_thesis_orchestrator.py
autonomous: true
requirements:
  - THS-02
  - THS-05
user_setup: []

must_haves:
  truths:
    - "build_main_matrix() returns 12 ThesisCell instances: 4 modules x 3 seeds (baseline/personalized/adaptive at thesis_crossdevice_main; pfedrec at paper_compat_pfedrec)"
    - "build_ablation_matrix() returns 21 ThesisCell instances: 7 ablation knobs x 3 seeds, all module='adaptive' at thesis_crossdevice_main"
    - "cell_already_done() matches on the FULL tuple (module, thesis_run_label, run_seed, ablation_dimension, ablation_value) — not just (module, run_seed)"
    - "cell_run_config_string() produces TOML-quoted strings consumable by scripts/run.py's --run-config parser"
    - "THESIS_BASE_OVERRIDES applies D-02 + D-03 to every cell: strategy=fedavg for all 4 modules; model-type=dual + alpha-method=hierarchical_conditional + next-gen knobs OFF for adaptive; merge order base->extra_run_config so ablation cells override base on conflict"
    - "--dry-run flag prints commands without invoking subprocess"
    - "--retry-failed flag re-runs only cells whose results.json is still missing on disk"
    - "Cell failures are caught, logged to results/federated/_thesis/failed_cells.json with stderr excerpt, and orchestrator continues to next cell (no stop-the-sweep)"
  artifacts:
    - path: "scripts/thesis/__init__.py"
      provides: "Empty package marker for scripts.thesis namespace"
      min_lines: 0
    - path: "scripts/thesis/run_thesis_sweep.py"
      provides: "Matrix-as-data orchestrator: build_main_matrix(), build_ablation_matrix(), execute_cell(), cell_already_done(), cell_run_config_string(), main() with --dry-run + --retry-failed + --phase=main|ablation flags"
      contains: "def build_main_matrix"
    - path: "scripts/foundation/tests/test_thesis_orchestrator.py"
      provides: "Unit tests for matrix builders, skip-on-existing logic, run-config quoting, dry-run smoke test"
      contains: "def test_main_matrix_size"
  key_links:
    - from: "ThesisCell dataclass"
      to: "scripts/run.py invocation via subprocess.run"
      via: "execute_cell builds [sys.executable, scripts/run.py, module, mode, --run-config, ...] and captures stdout/stderr"
      pattern: 'subprocess\.run.*scripts/run\.py'
    - from: "cell_already_done()"
      to: "results/federated/<module>/*/manifest.json"
      via: "glob + json.load + tuple match on (thesis_run_label, run_seed, ablation_dimension, ablation_value)"
      pattern: "manifest.json"
---

<objective>
Build the matrix-driven orchestrator that fires `flwr run` per cell across the thesis run matrix and the ablation matrix. Per D-18 (Python script, not bash; matrix-as-data, not YAML config) and D-23 (skip + log + retry-at-end on cell failure).

The orchestrator is responsible for:
1. **Matrix definition (D-13 + D-14)**: 12 main cells (3 baseline + 3 personalized + 3 adaptive at `thesis_crossdevice_main` + 3 pfedrec at `paper_compat_pfedrec`) + 21 ablation cells (7 ablations x 3 seeds, all `module="adaptive"` at `thesis_crossdevice_main`).
2. **Idempotent skip (D-18)**: `cell_already_done(cell)` checks every results-dir's manifest.json against the cell's full identity tuple `(module, thesis_run_label, run_seed, ablation_dimension, ablation_value)` — Pitfall 8 mitigation (naive `(module, seed)` matching collides because adaptive at seed=42 happens 8 times).
3. **Run config emission**: `cell_run_config_string(cell)` produces a single space-separated TOML-quoted KEY=VAL string consumable by `scripts/run.py --run-config "..."`. Includes `run-seed`, `thesis-run-label`, `ablation-dimension`, `ablation-value`, `wandb-run-name` (D-21 naming pattern), `THESIS_BASE_OVERRIDES[cell.module]` (D-02 + D-03 enforcement: `strategy=fedavg` for all 4 modules; for adaptive ALSO `model-type=dual` + `alpha-method=hierarchical_conditional` + next-gen knobs OFF), plus any extra knobs from `cell.extra_run_config` which override the base where they conflict (e.g. the `alpha_method=multi_factor` ablation cell overrides the adaptive base `alpha-method=hierarchical_conditional`).
4. **Subprocess invocation**: `execute_cell(cell, dry_run=False)` calls `subprocess.run([sys.executable, scripts/run.py, module, mode, --run-config, ...])`, captures stdout+stderr, returns `(success, stderr_excerpt)`.
5. **Failure handling (D-23)**: append failed cells to `results/federated/_thesis/failed_cells.json` with stderr excerpt; continue to the next cell; at the end print summary + `python scripts/thesis/run_thesis_sweep.py --retry-failed` recovery command.
6. **CLI**: `argparse` with `--phase=main|ablation|all`, `--dry-run`, `--retry-failed`, `--module=<one>` (smoke filter), `--seed=<int>` (smoke filter).
7. **Progress emission**: after every cell, write `results/federated/_thesis/_progress.json` with `{"completed": N, "failed": M, "remaining": K, "last_cell": <cell tuple>, "elapsed_sec": ...}` so a long-running sweep can be monitored without `tail -f`.
8. **Tests (Wave 0 of validation)**: `scripts/foundation/tests/test_thesis_orchestrator.py` covers matrix sizes, skip-on-existing tuple matching, run-config quoting, dry-run smoke.

Purpose: The orchestrator is the work-spawner. The aggregator (Plan 04) is what reads what this plan emits. Plan 03 + Plan 04 are independent (orchestrator emits files, aggregator reads files) and parallelize as Wave 3.
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

<interfaces>
From scripts/run.py (Phase 1 Plan 05; verified 2026-04-29):
```python
# CLI: python scripts/run.py <module> <mode> [--run-config "KEY=VAL KEY2=VAL2"] [--federation NAME] [--dry-run]
# module choices = sorted(MODULE_DIR.keys()) = ["adaptive", "baseline", "personalized", "pfedrec"]
# mode choices = sorted(MODE_NUM_SUPERNODES.keys()) = after Plan 01 includes "thesis_crossdevice_main"
# --run-config accepts repeated flags OR a single space-separated KEY=VAL string per flag.
# Bare-word string values are auto-quoted via _quote_value_for_flwr.
```

From fedrec_foundation.paths (Phase 6):
```python
def repo_root() -> Path: ...        # walks up to find PROJECT.md / .git anchor; cwd-independent
def module_run_results_dir(module: str, run_id: str) -> Path: ...  # repo_root/results/federated/<module>/<run_id>/
```

From fedrec_foundation.atomic (Plan 01 extended):
```python
def atomic_write_json(path: str, data: object) -> None: ...
def atomic_write_text(path: str, content: str) -> None: ...
```

D-13 ablation knobs (CONTEXT.md verbatim):
```
- alpha_method = multi_factor   -> {"alpha-method": "multi_factor"}
- alpha_method = data_quantity  -> {"alpha-method": "data_quantity"}
- per_user_alpha = true         -> {"enable-per-user-alpha": "true"}
- item_perturbation = true      -> {"enable-item-perturbation": "true", "item-perturbation-reg": "0.01"}
- contrastive_lambda = 0.1      -> {"contrastive-lambda": "0.1", "contrastive-tau": "0.1"}
- fusion_type = add             -> {"fusion-type": "add"}
- fusion_type = gate            -> {"fusion-type": "gate"}
```

D-21 W&B naming:
- main: `thesis-main-<module>-seed<N>`
- ablation: `thesis-ablation-<module>-seed<N>-<knob_short>=<value>`
  where knob_short is the short form: alpha (alpha_method) | pua (per_user_alpha) | ip (item_perturbation) | cl (contrastive_lambda) | fusion (fusion_type)

Pattern reference (verified):
- federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py — subprocess pattern
- scripts/foundation/tests/test_baseline_subprocess_determinism.py:94-105 — subprocess test pattern with cwd=_REPO_ROOT
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Orchestrator implementation — scripts/thesis/__init__.py + scripts/thesis/run_thesis_sweep.py</name>
  <read_first>
    - scripts/run.py (full file — 207 lines; read so the executor sees the launcher CLI signature, _quote_value_for_flwr behavior, and how --run-config is consumed)
    - scripts/foundation/fedrec_foundation/paths.py (read so the executor sees repo_root() and module_run_results_dir() exact signatures)
    - scripts/foundation/fedrec_foundation/atomic.py (read so the executor sees atomic_write_json + atomic_write_text patterns)
    - federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py (full file; pattern reference for subprocess + matrix-as-data)
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pattern 3: Orchestrator Matrix-as-Data (D-18)" — contains the exact ThesisCell + build_main_matrix + build_ablation_matrix skeletons
    - .planning/phases/07-thesis-evaluation-run/07-CONTEXT.md sections D-13, D-14, D-18, D-21, D-23
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pitfall 8" + "Pitfall 9" (skip-on-existing tuple matching + ablation name conventions)
  </read_first>
  <behavior>
    - `scripts/thesis/__init__.py` is an empty file (D-18 explicitly: "Empty (cross-script imports)").
    - `scripts/thesis/run_thesis_sweep.py` exports the public symbols: `ThesisCell` (frozen dataclass), `THESIS_BASE_OVERRIDES` (per-module D-02 + D-03 enforced overrides), `build_main_matrix()`, `build_ablation_matrix()`, `cell_already_done(cell, results_root)`, `cell_run_config_string(cell)` (merges THESIS_BASE_OVERRIDES BEFORE cell.extra_run_config), `execute_cell(cell, repo_root, dry_run=False) -> Tuple[bool, str]`, `main(argv) -> int`.
    - `main()` uses argparse with: positional/optional `--phase {main,ablation,all}` (default `all`), `--dry-run`, `--retry-failed`, `--module=<one>`, `--seed=<int>`, `--results-root=<path>` (overridable for tests; defaults to `repo_root() / "results"`).
    - On every cell completion, writes `<results_root>/federated/_thesis/_progress.json` atomically.
    - On every cell failure, appends to `<results_root>/federated/_thesis/failed_cells.json` (read-modify-write atomic via atomic_write_json).
    - At end of sweep, prints summary: completed N, failed M, remaining K (skipped because already-done), and the recovery command line.
    - Honors D-23: skip + log + continue (NEVER stop-the-sweep on first failure). NEVER auto-retry within a single sweep run (D-23 declined "auto-retry with exponential backoff").
    - The script is invokable as `python scripts/thesis/run_thesis_sweep.py [args]` AND as `python -m scripts.thesis.run_thesis_sweep [args]`.
  </behavior>
  <action>
**Step 1 — Create `scripts/thesis/__init__.py`** as a literally empty file (no content). Verify via `wc -c scripts/thesis/__init__.py` returns 0.

**Step 2 — Create `scripts/thesis/run_thesis_sweep.py`.** EXACT file content (the executor copies verbatim, then runs the verify command):

```python
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
```

**Step 3 — Make the script executable** (chmod +x via `chmod 755 scripts/thesis/run_thesis_sweep.py`).

**Step 4 — Smoke test the dry-run flow**:
```bash
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
python scripts/thesis/run_thesis_sweep.py --phase=main --dry-run | tee /tmp/thesis_dry.log
grep -c "DRY-RUN" /tmp/thesis_dry.log    # expect 12 (3 baseline + 3 personalized + 3 adaptive + 3 pfedrec)
python scripts/thesis/run_thesis_sweep.py --phase=ablation --dry-run | grep -c "DRY-RUN"  # expect 21
python scripts/thesis/run_thesis_sweep.py --phase=all --module=adaptive --seed=42 --dry-run | grep -c "DRY-RUN"  # expect 8 (1 main + 7 ablations)
```
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && test -f scripts/thesis/__init__.py && test -f scripts/thesis/run_thesis_sweep.py && test "$(wc -c < scripts/thesis/__init__.py)" -eq 0 && grep -n "strategy.*fedavg" scripts/thesis/run_thesis_sweep.py >/dev/null && grep -n "model-type.*dual" scripts/thesis/run_thesis_sweep.py >/dev/null && grep -n "alpha-method.*hierarchical_conditional" scripts/thesis/run_thesis_sweep.py >/dev/null && python -c "import sys; sys.path.insert(0, 'scripts'); from thesis.run_thesis_sweep import build_main_matrix, build_ablation_matrix, ThesisCell, THESIS_BASE_OVERRIDES, cell_run_config_string; main_cells = build_main_matrix(); abl_cells = build_ablation_matrix(); assert len(main_cells) == 12, f'main matrix size {len(main_cells)} != 12'; assert len(abl_cells) == 21, f'ablation matrix size {len(abl_cells)} != 21'; modules = sorted(set(c.module for c in main_cells)); assert modules == ['adaptive', 'baseline', 'personalized', 'pfedrec'], f'main modules {modules}'; abl_modules = sorted(set(c.module for c in abl_cells)); assert abl_modules == ['adaptive'], f'ablation modules must be only adaptive (D-13); got {abl_modules}'; pfedrec_cells = [c for c in main_cells if c.module == 'pfedrec']; assert all(c.mode == 'paper_compat_pfedrec' for c in pfedrec_cells), 'pfedrec cells must use paper_compat_pfedrec mode (D-06)'; thesis_cells = [c for c in main_cells if c.module != 'pfedrec']; assert all(c.mode == 'thesis_crossdevice_main' for c in thesis_cells), 'non-pfedrec main cells must use thesis_crossdevice_main mode'; assert set(THESIS_BASE_OVERRIDES.keys()) == {'baseline', 'personalized', 'adaptive', 'pfedrec'}, 'THESIS_BASE_OVERRIDES must cover all 4 modules'; assert all(THESIS_BASE_OVERRIDES[m]['strategy'] == 'fedavg' for m in THESIS_BASE_OVERRIDES), 'D-03: strategy=fedavg for all modules'; assert THESIS_BASE_OVERRIDES['adaptive']['model-type'] == 'dual', 'D-02: adaptive base = dual'; assert THESIS_BASE_OVERRIDES['adaptive']['alpha-method'] == 'hierarchical_conditional', 'D-02: adaptive base alpha-method'; print('matrix + base-overrides OK')" && python scripts/thesis/run_thesis_sweep.py --phase=main --dry-run 2>&1 | grep -c "DRY-RUN" | xargs -I{} test {} = "12" && python scripts/thesis/run_thesis_sweep.py --phase=ablation --dry-run 2>&1 | grep -c "DRY-RUN" | xargs -I{} test {} = "21" && python scripts/thesis/run_thesis_sweep.py --phase=all --module=adaptive --seed=42 --dry-run 2>&1 | grep -c "DRY-RUN" | xargs -I{} test {} = "8" && echo "Orchestrator OK"</automated>
  </verify>
  <done>
    - `scripts/thesis/__init__.py` exists, 0 bytes.
    - `scripts/thesis/run_thesis_sweep.py` exists and exports `ThesisCell`, `THESIS_BASE_OVERRIDES`, `build_main_matrix`, `build_ablation_matrix`, `cell_already_done`, `cell_run_config_string`, `execute_cell`, `main`.
    - `build_main_matrix()` returns exactly 12 cells (3 baseline + 3 personalized + 3 adaptive at thesis_crossdevice_main; 3 pfedrec at paper_compat_pfedrec).
    - `build_ablation_matrix()` returns exactly 21 cells (7 ablations x 3 seeds, all adaptive at thesis_crossdevice_main).
    - `--phase=main --dry-run` prints 12 `[DRY-RUN]` lines.
    - `--phase=ablation --dry-run` prints 21 `[DRY-RUN]` lines.
    - `--phase=all --module=adaptive --seed=42 --dry-run` prints 8 `[DRY-RUN]` lines (1 main + 7 ablations for adaptive at seed 42).
    - **BLOCKER 1 (D-02 + D-03 enforcement):** `grep -n "strategy.*fedavg" scripts/thesis/run_thesis_sweep.py` returns at least one match (in THESIS_BASE_OVERRIDES). `grep -n "model-type.*dual" scripts/thesis/run_thesis_sweep.py` returns at least one match (adaptive base override). `grep -n "alpha-method.*hierarchical_conditional" scripts/thesis/run_thesis_sweep.py` returns at least one match.
    - **BLOCKER 1 (merge precedence):** `cell_run_config_string()` applies `THESIS_BASE_OVERRIDES[cell.module]` BEFORE `cell.extra_run_config`. Verified by `test_alpha_method_ablation_overrides_base_hc` (Task 2): an `alpha_method=multi_factor` ablation cell's run-config string contains `alpha-method=multi_factor`, NOT `alpha-method=hierarchical_conditional`.
    - **BLOCKER 2 (fusion-type ablation correctness):** Adaptive base override sets `model-type=dual`, so a fusion-type ablation cell's run-config string contains BOTH `model-type=dual` AND `fusion-type=add` (verified by `test_fusion_type_ablation_includes_dual_model`).
    - PFedRec base override does NOT include `model-type` or `alpha-method` (verified by `test_pfedrec_main_cell_does_not_set_model_type`); PFedRec's pyproject.toml has no such keys.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Orchestrator unit tests — scripts/foundation/tests/test_thesis_orchestrator.py</name>
  <read_first>
    - scripts/thesis/run_thesis_sweep.py (file just created in Task 1)
    - scripts/foundation/tests/test_baseline_subprocess_determinism.py (full file; pattern reference for subprocess test layout, _REPO_ROOT helper, tmp_path fixtures)
    - scripts/foundation/tests/test_launcher.py (pattern reference for argparse-driven CLI tests)
    - .planning/phases/07-thesis-evaluation-run/07-VALIDATION.md "Per-Task Verification Map" rows 7-03-01 through 7-03-05
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pitfall 8" (skip-on-existing tuple matching test design)
  </read_first>
  <behavior>
    - The new test file `scripts/foundation/tests/test_thesis_orchestrator.py` lives alongside other foundation tests; pytest discovers it automatically (foundation `pyproject.toml` already declares `testpaths=["tests"]`).
    - Tests use the `tmp_path` fixture for synthetic results-root manifests and DO NOT require any actual flwr run.
    - Test functions are named exactly per VALIDATION.md row IDs: `test_main_matrix_size`, `test_ablation_matrix_size`, `test_skip_on_existing_full_tuple`, `test_run_config_quoting`, `test_dry_run_no_subprocess`. Plus eight more for completeness: `test_main_modules_correct`, `test_ablation_module_is_adaptive_only`, `test_seeds_are_canonical_set`, `test_skip_on_existing_returns_false_when_no_disk`, `test_skip_on_existing_ignores_corrupt_manifest`, `test_run_config_string_includes_extra_knobs`, `test_run_config_string_item_perturbation_two_knobs`, `test_ablation_knobs_shape`.
    - **BLOCKER 1 D-02/D-03 enforcement tests (3 new):** `test_adaptive_main_cell_includes_dual_model_and_hc_alpha`, `test_alpha_method_ablation_overrides_base_hc`, `test_pfedrec_main_cell_does_not_set_model_type`.
    - **BLOCKER 2 fusion-type-ablation correctness test (1 new):** `test_fusion_type_ablation_includes_dual_model` — proves `fusion-type=add` ablation cell carries `model-type=dual` from the base override (without it, the fusion-type knob is a silent no-op).
    - `test_skip_on_existing_full_tuple` constructs synthetic manifests with same (module, seed) but different (thesis_run_label, ablation_dimension, ablation_value) and asserts cell_already_done correctly distinguishes them (Pitfall 8 mitigation test).
    - `test_dry_run_no_subprocess` mocks subprocess.run and asserts execute_cell with dry_run=True returns (True, "") WITHOUT invoking subprocess.run.
    - The test file imports the orchestrator via `sys.path.insert(0, scripts/)` then `from thesis.run_thesis_sweep import ...` (mirrors how the orchestrator runs).
    - Total test count: **15 GREEN** (8 VALIDATION-mapped + supplementary + 4 BLOCKER 1+2 enforcement).
  </behavior>
  <action>
Create the file `scripts/foundation/tests/test_thesis_orchestrator.py` with EXACT content:

```python
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
```

Run the new tests:
```bash
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
pytest scripts/foundation/tests/test_thesis_orchestrator.py -x -v
```
Expect: **15 PASSED** (was 11 pre-revision; +3 BLOCKER 1 D-02/D-03 enforcement tests + 1 BLOCKER 2 fusion-type-ablation test).
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && pytest scripts/foundation/tests/test_thesis_orchestrator.py -x -v 2>&1 | tail -25 | grep -E "passed|FAILED" | grep -q "15 passed"</automated>
  </verify>
  <done>
    - `scripts/foundation/tests/test_thesis_orchestrator.py` exists.
    - All **15 tests GREEN** (was 11 pre-revision; +3 BLOCKER 1 + 1 BLOCKER 2 enforcement tests).
    - The 5 VALIDATION.md-named tests are present and pass: `test_main_matrix_size`, `test_ablation_matrix_size`, `test_skip_on_existing_full_tuple`, `test_run_config_quoting`, `test_dry_run_no_subprocess`.
    - **BLOCKER 1 (D-02/D-03) tests pass:** `test_adaptive_main_cell_includes_dual_model_and_hc_alpha`, `test_alpha_method_ablation_overrides_base_hc`, `test_pfedrec_main_cell_does_not_set_model_type`.
    - **BLOCKER 2 (fusion-type-ablation correctness) test passes:** `test_fusion_type_ablation_includes_dual_model`.
    - 7 supplementary tests pass: `test_main_modules_correct`, `test_ablation_module_is_adaptive_only`, `test_seeds_are_canonical_set`, `test_skip_on_existing_returns_false_when_no_disk`, `test_skip_on_existing_ignores_corrupt_manifest`, `test_run_config_string_includes_extra_knobs`, `test_run_config_string_item_perturbation_two_knobs`, `test_ablation_knobs_shape`.
  </done>
</task>

</tasks>

<verification>
- Orchestrator builds correct matrix sizes (12 main + 21 ablation).
- All 5 VALIDATION-mapped unit tests are green.
- `--dry-run` paths don't invoke subprocess.
- `cell_already_done` correctly distinguishes cells at the same (module, seed) but different ablation knobs (Pitfall 8 mitigation).
- D-13 ablation knob set is exactly 7 entries with the expected dimensions/values.
- D-21 W&B run-name pattern produced for both main and ablation cells.
</verification>

<success_criteria>
- [ ] `scripts/thesis/__init__.py` exists, 0 bytes.
- [ ] `scripts/thesis/run_thesis_sweep.py` exists, exports the 7 public symbols.
- [ ] `python scripts/thesis/run_thesis_sweep.py --phase=main --dry-run` prints 12 `[DRY-RUN]` lines.
- [ ] `python scripts/thesis/run_thesis_sweep.py --phase=ablation --dry-run` prints 21 `[DRY-RUN]` lines.
- [ ] `pytest scripts/foundation/tests/test_thesis_orchestrator.py -x -v` reports **15 PASSED** (was 11 pre-revision).
- [ ] D-23 invariants verified by code review: failure handling appends to failed_cells.json + continues; never auto-retries within a sweep.
- [ ] **BLOCKER 1 (D-02 + D-03 enforcement):** `THESIS_BASE_OVERRIDES` dict declares `strategy=fedavg` for all 4 modules; adaptive ALSO has `model-type=dual`, `alpha-method=hierarchical_conditional`, and next-gen knobs OFF (`enable-per-user-alpha=false`, `enable-item-perturbation=false`, `contrastive-lambda=0.0`).
- [ ] **BLOCKER 1 (merge precedence):** `cell_run_config_string()` merges `THESIS_BASE_OVERRIDES[cell.module]` BEFORE `cell.extra_run_config` so ablation cells override the base on conflict.
- [ ] **BLOCKER 2 (fusion-type-ablation correctness):** Adaptive base override sets `model-type=dual`, ensuring `fusion-type=add`/`fusion-type=gate` ablations are NOT silent no-ops.
</success_criteria>

<output>
After completion, create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-03-SUMMARY.md` documenting:
- Final orchestrator file size in lines.
- Number of public symbols exported.
- Test counts (per VALIDATION.md row IDs).
- Any deviations from the action text (e.g., extra helper functions added for clarity, additional test cases beyond the 11 specified).
- The exact behavior of `--retry-failed` flag (skip_existing always True; D-31 default of "filter by disk presence").
</output>
</content>
</invoke>
