# Phase 7: Thesis Evaluation Run - Research

**Researched:** 2026-04-29
**Domain:** Experiment-orchestration + result-aggregation tooling on top of an already-validated cross-device FL stack (Phases 1-6)
**Confidence:** HIGH (foundation infra exhaustively validated through 278 GREEN tests across the 4 modules + foundation; the unknowns are isolated to the 2 NEW Python scripts being written, not to the existing surfaces they consume)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Standardized comparison config (THS-01)
- **D-01:** Use the existing `benchmark_cross_device` profile values verbatim for the thesis main comparison: `embedding_dim=64`, `optimizer="adam"`, `lr=0.001`, `local_epochs=1`, `num_server_rounds=100`, `fraction_train=0.1`, `num_train_negatives=4`, `weight_policy="num_positives"`, `primary_evaluator="sampled_loo_99"`. Zero churn — these have been validated through Phases 2-6 on baseline / personalized / adaptive.
- **D-02:** Adaptive's main-comparison config = `model-type=dual` + `alpha-method=hierarchical_conditional` ONLY. The "next-gen" knobs (`enable-per-user-alpha`, `enable-item-perturbation`, `contrastive-lambda`) are **OFF** in the main table. They are ablation knobs only. Clean attribution.
- **D-03:** FedAvg only for the main comparison (no FedProx). The thesis claim is about personalization mechanism, not aggregation strategy.
- **D-04:** Add a new mode profile `thesis_crossdevice_main` to `scripts/foundation/fedrec_foundation/mode.py`. Values clone `benchmark_cross_device` verbatim. Provenance tag.

#### PFedRec role (THS-03, THS-04)
- **D-05:** PFedRec is a **calibration reference**, NOT counted toward "adaptive beats baselines". Adaptive must beat **baseline + personalized only**.
- **D-06:** PFedRec runs **only** at `paper_compat_pfedrec` (Phase 5 D-05 honored). No extra appendix row.
- **D-07:** PFedRec uses 3 seeds, ~10 hr wallclock at ~3 hr/run.
- **D-08:** PFedRec is reported as a footnoted row in the same `_thesis/main_comparison.{md,csv}` table.

#### Seeds & statistical comparison (THS-02..THS-04)
- **D-09:** 3 seeds for the main comparison across all four modules. Main: ~10.5 hr (3 modules × 3 seeds × ~1.2 hr). Plus PFedRec 3 seeds × ~3 hr. Main + PFedRec ≈ ~19.5 hr.
- **D-10:** Seeds = `{42, 1337, 2026}`. Same seeds across modules.
- **D-11:** Win criterion = adaptive's mean NDCG@10 strictly greater AND non-overlapping ±1σ intervals vs every baseline. Specifically: `adaptive.mean - adaptive.std > baseline.mean + baseline.std` AND same vs personalized.
- **D-12:** Contingency on failure to win: document negative result + run ablations as recovery runs. If still no win, escalate to thesis-level replanning per PROJECT.md.

#### Ablation scope (THS-05, THS-06)
- **D-13:** One-factor-at-a-time ablation from main config. 7 ablation cells:
    - Alpha method: `multi_factor`, `data_quantity` (2 cells)
    - Per-user alpha: `enable-per-user-alpha=true` (1 cell)
    - Item perturbation: `enable-item-perturbation=true` + `item-perturbation-reg=0.01` (1 cell)
    - Contrastive λ: `contrastive-lambda=0.1` + `contrastive-tau=0.1` (1 cell)
    - Fusion type: `add`, `gate` (2 cells)
- **D-14:** 3 seeds for all ablation cells. 21 ablation runs at ~1.5 hr/run ≈ ~31.5 hr.
- **D-15:** Ablation table columns: Overall NDCG@10 + Sparse NDCG@10 only (plus matching HR@10 columns).
- **D-16:** Run sequence: main runs first; ablations after.

#### Export pipeline (THS-07)
- **D-17:** Export formats: Markdown + CSV. No LaTeX, no aggregate JSON. Output paths under `results/federated/_thesis/`:
    - `main_comparison.md` + `main_comparison.csv`
    - `ablations.md` + `ablations.csv`
    - `sparse_slice.md` + `sparse_slice.csv`
- **D-18:** Orchestrator: Python script `scripts/thesis/run_thesis_sweep.py`. Matrix-as-data, fires `flwr run` per cell, captures stdout/stderr, logs success/failure. Re-runnable to fill gaps.
- **D-19:** Aggregator: standalone Python script `scripts/thesis/aggregate_results.py`. Reads `results/federated/<module>/<run_id>/results.json` for every run matching the thesis filter (`_manifest.thesis_run_label` set AND `_manifest.run_seed ∈ {42, 1337, 2026}`).
- **D-20:** Aggregator missing-cell handling: hard fail with explicit list. No partial tables.

#### Operational details
- **D-21:** W&B project = `federated-cf-cross-device` (same project as main). Distinguished by run name pattern:
    - Main: `thesis-main-<module>-seed<N>`
    - Ablation: `thesis-ablation-<module>-seed<N>-<knob>=<value>`
    - PFedRec: `thesis-main-pfedrec-seed<N>`
- **D-22:** Manifest schema extension. Bump `RUN_MANIFEST_SCHEMA_VERSION` 2→3. Add three fields to `RunManifest`:
    - `thesis_run_label: str = ""` — `"main"` or `"ablation_<knob>=<value>"` or `""` for non-thesis.
    - `ablation_dimension: str = "none"` — one of `{"none", "alpha_method", "per_user_alpha", "item_perturbation", "contrastive_lambda", "fusion_type"}`.
    - `ablation_value: str = ""` — value of the ablated knob.
- **D-23:** Cell failure handling: skip + log + retry at end. Failures logged to `results/federated/_thesis/failed_cells.json` with stderr excerpt. End-of-run summary prints `python scripts/thesis/run_thesis_sweep.py --retry-failed`.
- **D-24:** Table cell format: `0.4123 ± 0.0089` (4 decimal places). Markdown: `| 0.4123 ± 0.0089 |`. CSV: two columns per metric (`ndcg10_mean`, `ndcg10_std`).

### Claude's Discretion
- **Bold-the-winner styling** in markdown tables. Default: bold the row that "beats" all comparable rows under D-11 win criterion.
- **Sparse-user slice fill behavior** when a seed has zero evaluable sparse interactions. Default: emit row from seeds that DID have sparse interactions, with footnote `n_seeds_with_sparse=2/3`.
- **Wandb-summary key naming for thesis runs.** Recommended: keep Phase-6 `best/*` and `last/*` namespaces; add top-level `thesis/run_label` summary field mirroring the manifest.
- **Intermediate result review checkpoints.** Default: emit progress JSON to `_thesis/_progress.json` every cell; full tables only at end via aggregator.
- **Significance markers (asterisks, color).** Default: no special markers; bold-the-winner is enough.
- **`_thesis/` directory creation handling.** First sweep creates the directory. Atomic write via existing `atomic_write_json` (extend with `atomic_write_text` for markdown).
- **Compute parallelism.** Default: serial within a module, between-module serial too (one giant queue).
- **Retry semantics for `--retry-failed`.** Default: filter by disk presence (idempotent).

### Deferred Ideas (OUT OF SCOPE)
- Two-stage ablation (pick best alpha first, then ablate other knobs against the winner) — deferred.
- Full Cartesian ablation matrix (72 cells × 3 seeds = 216 runs) — deferred.
- PFedRec at non-PFedRec hyperparams as an extra row — explicitly declined per Phase 5 D-05.
- 5 seeds (vs 3) — declined for thesis budget.
- LaTeX export format — declined.
- JSON aggregate export (`_thesis/aggregated.json`) — declined.
- W&B Sweeps via `sweep.yaml` — declined.
- Auto-retry on cell failure with exponential backoff — declined.
- Stop-the-sweep-on-first-failure — declined.
- Per-user-group medium / dense columns in main ablation table — declined.
- DP / privacy quantification — out of scope per PROJECT.md.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **THS-01** | Define ONE standardized cross-device comparison config shared by all four modules | New `thesis_crossdevice_main` ModeProfile cloned verbatim from `_BENCHMARK_CROSS_DEVICE` (D-04). PFedRec uses `paper_compat_pfedrec` (D-06). |
| **THS-02** | Run all four modules under the standardized config, ≥3 seeds; produce mean ± std comparison table | Orchestrator fires `scripts/run.py <module> <mode> --run-config "run-seed=N ..."` per cell (D-18). Seeds `{42, 1337, 2026}` (D-10). Aggregator computes mean ± std across seeds (D-19, D-24). |
| **THS-03** | Adaptive beats baseline + personalized on OVERALL NDCG@10 | Aggregator reads `final_metrics["best"]["sampled_ndcg@10"]` from per-run `results.json`; D-11 win criterion (`adaptive.mean - adaptive.std > baseline.mean + baseline.std`) applied at table-render time. |
| **THS-04** | Adaptive beats baseline + personalized on SPARSE-user NDCG@10 | Aggregator reads `final_metrics["best"]["sampled_ndcg@10/sparse"]` (slash delimiter — confirmed in baseline/personalized/adaptive/pfedrec strategy.py). Sparse slice table is its own export (`sparse_slice.{md,csv}` — D-17). |
| **THS-05** | Ablations: hierarchical-conditional vs multi-factor vs data-quantity alpha; per-user alpha on/off; item perturbation on/off; contrastive λ ∈ {0, 0.1}; fusion ∈ {add, gate, concat} | One-factor-at-a-time per D-13 → 7 ablation cells. Each cell flips ONE knob from main config; remaining knobs stay at main-comparison defaults. |
| **THS-06** | Ablations also report per-user-group metrics | Per-run `results.json` already carries `final_metrics["best"]["sampled_ndcg@10/{sparse,medium,dense}"]` from Phase 6. Ablation table surfaces overall + sparse only (D-15); medium/dense remain in per-run JSON for post-hoc inspection. |
| **THS-07** | Export thesis comparison + ablation tables + sparse-user slice as markdown to `results/federated/_thesis/` | Aggregator writes 6 files (3 markdown + 3 CSV) per D-17 via atomic write helpers. |
</phase_requirements>

## Summary

Phase 7 ships **no ML changes**. It is pure orchestration + aggregation Python work on top of a fully-validated cross-device FL stack. The Phase-6 harness already emits everything the aggregator needs: per-run `results/federated/<module>/<run_id>/results.json` with nested `final_metrics = {best, last, best_round, last_round, final_eval_round_index}` carrying `sampled_ndcg@10`, `sampled_hr@10`, plus per-group variants (`/sparse`, `/medium`, `/dense`), plus `_manifest` block carrying mode + seed + foundation fingerprints.

The phase deliverables are:
1. A new `thesis_crossdevice_main` ModeProfile (cloned verbatim from `_BENCHMARK_CROSS_DEVICE`) — D-04.
2. RunManifest schema bump v2 → v3 with three thesis-tagging fields — D-22.
3. Orchestrator `scripts/thesis/run_thesis_sweep.py` — matrix-as-data, fires `scripts/run.py <module> <mode> --run-config ...` per cell, idempotent skip-on-existing, fail-and-log, `--retry-failed`, `--dry-run` — D-18, D-23.
4. Aggregator `scripts/thesis/aggregate_results.py` — globs `results.json` files, filters by `_manifest.thesis_run_label`, computes mean ± std across seeds, hard-fails on missing cells, emits 6 files — D-19, D-20, D-24.
5. Run + produce numbers: 12 main runs + 21 ablation runs ≈ ~50 hr wallclock.

**Primary recommendation:** All 4 server_apps currently gate cross-device-aware behavior (W&B project routing, `module_run_results_dir` write path) on `mode in ("benchmark_cross_device", "paper_compat_pfedrec")`. The new `thesis_crossdevice_main` mode MUST be added to BOTH `mode in (...)` tuples in BOTH branches of all 4 `server_app.py` files, OR the orchestrator must emit `wandb-project=federated-cf-cross-device` overrides per cell. Plan A (extend the tuples) is one-line edits × 4 files × 2 sites = 8 line edits and matches the existing pattern. Plan B (route through `wandb-project` override) is zero source-code edits but requires the orchestrator to know about each module's per-run-dir gating logic. **Recommendation: Plan A.** It's the lowest-friction extension and keeps the orchestrator script independent of per-module server_app internals.

## Standard Stack

### Core (already installed and validated through Phases 1-6 — zero churn)
| Library | Version (verified) | Purpose | Why Standard |
|---------|--------------------|---------|--------------|
| Python  | 3.9+ (per CLAUDE.md) | Language baseline | Locked by existing modules |
| pandas  | 2.3.3 (installed) | Optional table rendering / CSV write helper | Already in foundation deps `pandas>=2.0.0` |
| numpy   | 2.2.6 (installed) | mean/std computation across seeds | Foundation dep `numpy>=1.24.0` |
| pytest  | 7+ (installed)  | Test framework | Standard across foundation tests |

### Supporting (used by orchestrator/aggregator only)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `subprocess` | stdlib | Fire `scripts/run.py <module> <mode>` per cell, capture stdout/stderr | Orchestrator cell execution |
| `argparse` | stdlib | CLI flag parsing (`--retry-failed`, `--dry-run`, `--module=`, `--seed=`) | Orchestrator + aggregator entry points |
| `json` | stdlib | Read `results.json` + `manifest.json`, write `failed_cells.json` + `_progress.json` | Both scripts |
| `pathlib` | stdlib | Path resolution | Both scripts |
| `csv` | stdlib | CSV table writes | Aggregator (avoids pandas dependency for trivial wide-format tables) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Shell script orchestrator (existing `scripts/run_baseline_sweep_loo.sh` pattern) | Python orchestrator | D-18 explicitly chose Python — better cross-platform, easier idempotent skip-on-existing, easier failure capture, easier matrix-as-data |
| W&B Sweeps via `sweep.yaml` | Standalone matrix orchestrator | Bayesian/Hyperband not needed for deterministic 33-cell matrix. CONTEXT explicitly declines W&B Sweeps |
| pandas DataFrames for table rendering | stdlib `csv` + manual markdown | pandas optional — `csv.writer` handles 8-row × 12-col tables fine. pandas is fine if the planner prefers it (already installed) |

**Installation:**
No new packages. Aggregator + orchestrator use stdlib + already-installed pandas/numpy.

**Version verification:**
```bash
python3 -c "import pandas; print(pandas.__version__)"  # 2.3.3 (verified 2026-04-29)
python3 -c "import numpy; print(numpy.__version__)"   # 2.2.6 (verified 2026-04-29)
```

## Architecture Patterns

### Recommended Project Structure

```
scripts/
├── thesis/                              # NEW directory
│   ├── __init__.py                       # Empty (D-18 mentions "if any cross-script imports needed")
│   ├── run_thesis_sweep.py              # Orchestrator (D-18)
│   └── aggregate_results.py             # Aggregator (D-19)
├── foundation/
│   └── fedrec_foundation/
│       ├── mode.py                       # EXTEND: add _THESIS_CROSSDEVICE_MAIN + register
│       ├── manifest.py                   # EXTEND: bump schema 2→3 + 3 fields
│       └── atomic.py                     # EXTEND: add atomic_write_text companion
└── foundation/tests/
    ├── test_mode.py                      # EXTEND: thesis mode resolution tests
    ├── test_manifest.py                  # EXTEND: schema v3 tests
    └── test_thesis_orchestrator.py       # NEW: orchestrator dry-run + matrix tests
    └── test_thesis_aggregator.py         # NEW: aggregator parsing + render tests
```

### Pattern 1: Mode Profile Cloning (D-04)

**What:** Add `_THESIS_CROSSDEVICE_MAIN` `ModeProfile` instance to `mode.py`. Values are byte-for-byte identical to `_BENCHMARK_CROSS_DEVICE` except the `mode` string.

**When to use:** Every time a new "tagged" experiment profile is added (the existing `_PAPER_COMPAT_PFEDREC` is the prior precedent).

**Example (planner copies into `mode.py` immediately after `_BENCHMARK_CROSS_DEVICE`):**
```python
# Source: scripts/foundation/fedrec_foundation/mode.py:118-135 (existing _BENCHMARK_CROSS_DEVICE)

_THESIS_CROSSDEVICE_MAIN = ModeProfile(
    mode="thesis_crossdevice_main",
    num_supernodes=6040,
    partition_mode="natural",
    weight_policy="num_positives",
    primary_evaluator="sampled_loo_99",
    fraction_train=0.1,
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=64,
    optimizer="adam",
    lr=0.001,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",
    assert_one_user_per_client=True,
)


_REGISTRY: Dict[str, ModeProfile] = {
    "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
    "thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN,  # NEW
    "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
    "cross_silo_legacy": _CROSS_SILO_LEGACY,
}
```

**Side-effect: `scripts/run.py` MUST learn the new mode.** The launcher's `MODE_NUM_SUPERNODES` dict (`scripts/run.py:64-68`) currently lists 3 modes. Add `"thesis_crossdevice_main": 6040`. The `argparse` `choices=` clamp on line 167 is `sorted(MODE_NUM_SUPERNODES.keys())` — adding the entry to the dict automatically updates the CLI choice list.

### Pattern 2: Manifest Schema Bump v2 → v3 (D-22)

**What:** Add three fields to `RunManifest` dataclass with safe defaults so v2 callers (and v1 fixtures from `test_run_manifest_backward_compat_v1`) construct unchanged.

**When to use:** Only when adding aggregator-filterable provenance metadata.

**Example (planner copies into `manifest.py`):**
```python
# Source: scripts/foundation/fedrec_foundation/manifest.py:28-29

# BEFORE:
RUN_MANIFEST_SCHEMA_VERSION: int = 2  # Phase 6: adds final_eval_round_index + metrics fields

# AFTER:
RUN_MANIFEST_SCHEMA_VERSION: int = 3  # Phase 7: adds thesis_run_label + ablation_dimension + ablation_value
```

```python
# Source: scripts/foundation/fedrec_foundation/manifest.py:94-103 (after `metrics: Dict[str, Any] = field(default_factory=dict)`)

# Add at the end of the RunManifest dataclass:
    thesis_run_label: str = ""
    """Phase 7 D-22: thesis run provenance tag.

    Sentinel ``""`` (empty string) = non-thesis run (Phase 1-6 backward compat).
    ``"main"`` = main-comparison run.
    ``"ablation_<knob>=<value>"`` = ablation run (e.g., ``"ablation_fusion=add"``).
    """
    ablation_dimension: str = "none"
    """Phase 7 D-22: which knob is being ablated.

    One of ``{"none", "alpha_method", "per_user_alpha", "item_perturbation",
    "contrastive_lambda", "fusion_type"}``. ``"none"`` for main runs.
    """
    ablation_value: str = ""
    """Phase 7 D-22: specific value of the ablated knob.

    Empty for main runs. Examples: ``"add"`` when ``ablation_dimension="fusion_type"``;
    ``"true"`` when ``ablation_dimension="per_user_alpha"``.
    """
```

**Backward-compatibility invariants:**
- The existing `test_run_manifest_backward_compat_v1` test (`test_manifest.py:181-221`) constructs a manifest with **only the v1 fields** and asserts the v2 defaults (`final_eval_round_index == 0`, `metrics == {}`). Adding three new fields with safe defaults must preserve this exact test.
- The schema-version test (`test_run_manifest_schema_version_2`, line 167) must be updated to assert `RUN_MANIFEST_SCHEMA_VERSION == 3`.
- New tests pin: (1) defaults of all 3 new fields, (2) post-build mutation via `dataclasses.replace`, (3) v3 schema version constant.

### Pattern 3: Orchestrator Matrix-as-Data (D-18)

**What:** Define the run matrix as a Python data structure (list of dataclasses or dicts), iterate, fire `scripts/run.py <module> <mode> --run-config "..."` per cell.

**When to use:** Deterministic experiment matrix execution.

**Example skeleton (planner adapts):**
```python
# Source: based on federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py pattern + scripts/run.py launcher signature

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass(frozen=True)
class ThesisCell:
    """One cell of the thesis run matrix.

    Attributes
    ----------
    module : str
        One of ``"baseline"``, ``"personalized"``, ``"adaptive"``, ``"pfedrec"``.
    mode : str
        ``"thesis_crossdevice_main"`` (main runs) or ``"paper_compat_pfedrec"``
        (PFedRec only — D-06).
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


def build_main_matrix() -> List[ThesisCell]:
    """4 modules × 3 seeds = 12 main cells. PFedRec uses different mode (D-06)."""
    cells = []
    seeds = [42, 1337, 2026]
    for seed in seeds:
        for module in ("baseline", "personalized", "adaptive"):
            cells.append(ThesisCell(
                module=module, mode="thesis_crossdevice_main",
                run_seed=seed,
                thesis_run_label="main",
                ablation_dimension="none",
                ablation_value="",
                extra_run_config={},
            ))
        # PFedRec runs only at paper_compat_pfedrec (D-06)
        cells.append(ThesisCell(
            module="pfedrec", mode="paper_compat_pfedrec",
            run_seed=seed,
            thesis_run_label="main",
            ablation_dimension="none",
            ablation_value="",
            extra_run_config={},
        ))
    return cells


def build_ablation_matrix() -> List[ThesisCell]:
    """7 ablation cells × 3 seeds = 21. Always module='adaptive' (D-13)."""
    cells = []
    seeds = [42, 1337, 2026]
    # 7 ablation knobs (D-13)
    ablations = [
        ("alpha_method", "multi_factor", {"alpha-method": "multi_factor"}),
        ("alpha_method", "data_quantity", {"alpha-method": "data_quantity"}),
        ("per_user_alpha", "true", {"enable-per-user-alpha": "true"}),
        ("item_perturbation", "true", {"enable-item-perturbation": "true",
                                        "item-perturbation-reg": "0.01"}),
        ("contrastive_lambda", "0.1", {"contrastive-lambda": "0.1",
                                        "contrastive-tau": "0.1"}),
        ("fusion_type", "add", {"fusion-type": "add"}),
        ("fusion_type", "gate", {"fusion-type": "gate"}),
    ]
    for seed in seeds:
        for ablation_dim, ablation_val, extra_cfg in ablations:
            label = f"ablation_{ablation_dim}={ablation_val}"
            cells.append(ThesisCell(
                module="adaptive", mode="thesis_crossdevice_main",
                run_seed=seed,
                thesis_run_label=label,
                ablation_dimension=ablation_dim,
                ablation_value=ablation_val,
                extra_run_config=extra_cfg,
            ))
    return cells


def cell_already_done(cell: ThesisCell, results_root: Path) -> bool:
    """D-18 idempotent skip: scan results/federated/<module>/*/manifest.json
    for any run whose _manifest.thesis_run_label == cell.thesis_run_label
    AND _manifest.run_seed == cell.run_seed.
    """
    module_dir = results_root / "federated" / cell.module
    if not module_dir.exists():
        return False
    for manifest_path in module_dir.glob("*/manifest.json"):
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if (m.get("thesis_run_label") == cell.thesis_run_label
                and m.get("run_seed") == cell.run_seed
                and m.get("ablation_dimension", "none") == cell.ablation_dimension
                and m.get("ablation_value", "") == cell.ablation_value):
            return True
    return False


def cell_run_config_string(cell: ThesisCell) -> str:
    """Build the --run-config string for scripts/run.py.

    Includes: run-seed + thesis-run-label + ablation-dimension + ablation-value
    + extra_run_config + wandb-run-name (D-21 naming).
    """
    parts = [f"run-seed={cell.run_seed}"]
    # D-22: pass thesis fields as run_config so server_app can mutate manifest
    parts.append(f'thesis-run-label="{cell.thesis_run_label}"')
    parts.append(f'ablation-dimension="{cell.ablation_dimension}"')
    parts.append(f'ablation-value="{cell.ablation_value}"')
    # D-21: W&B run name pattern (see naming-mapping section below)
    if cell.thesis_run_label == "main":
        wandb_name = f"thesis-main-{cell.module}-seed{cell.run_seed}"
    else:
        # cell.thesis_run_label == f"ablation_{dim}={val}"
        knob_eq = cell.thesis_run_label.removeprefix("ablation_")  # e.g., "fusion_type=add"
        wandb_name = f"thesis-ablation-{cell.module}-seed{cell.run_seed}-{knob_eq}"
    parts.append(f'wandb-run-name="{wandb_name}"')
    # Extra config (e.g., --alpha-method=multi_factor)
    for k, v in cell.extra_run_config.items():
        parts.append(f'{k}="{v}"')
    return " ".join(parts)


def execute_cell(cell: ThesisCell, repo_root: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Fire scripts/run.py for one cell. Return (success, stderr_excerpt).

    Captures stdout+stderr; on non-zero exit, appends to failed_cells.json.
    """
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run.py"),
        cell.module,
        cell.mode,
        "--run-config",
        cell_run_config_string(cell),
    ]
    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        return True, ""
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, proc.stderr[-2000:]  # last 2KB of stderr
    return True, ""
```

### Pattern 4: Aggregator Filter + Render (D-19)

**What:** Glob `results/federated/<module>/*/results.json`, filter manifests, group by (module, thesis_run_label) keys, compute mean ± std across seeds, render markdown + CSV.

**Example skeleton:**
```python
def collect_thesis_results(results_root: Path) -> List[Dict]:
    """Glob results.json files; return only those whose _manifest carries
    the Phase 7 thesis fields and run_seed in the canonical seed set.
    """
    THESIS_SEEDS = {42, 1337, 2026}
    all_results = []
    for module in ("baseline", "personalized", "adaptive", "pfedrec"):
        module_dir = results_root / "federated" / module
        if not module_dir.exists():
            continue
        for results_json in module_dir.glob("*/results.json"):
            try:
                with open(results_json) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            manifest = data.get("_manifest", {})
            label = manifest.get("thesis_run_label", "")
            seed = manifest.get("run_seed", -1)
            if label and seed in THESIS_SEEDS:
                all_results.append({
                    "module": module,
                    "thesis_run_label": label,
                    "ablation_dimension": manifest.get("ablation_dimension", "none"),
                    "ablation_value": manifest.get("ablation_value", ""),
                    "run_seed": seed,
                    "results_data": data,
                    "results_path": results_json,
                })
    return all_results


def extract_metric(data: Dict, metric_key: str) -> Optional[float]:
    """Read final_metrics['best'][metric_key].

    Critical: PFedRec uses slash-delimited evaluated_users keys (evaluated_users/sparse),
    while baseline/personalized/adaptive use underscore (evaluated_users_sparse).
    HR/NDCG keys use slash uniformly across all 4 modules (sampled_ndcg@10/sparse).
    """
    best = data.get("final_metrics", {}).get("best", {})
    val = best.get(metric_key)
    return float(val) if val is not None else None


def aggregate_by_seed(records: List[Dict], metric_key: str) -> Dict[Tuple, Tuple[float, float, int]]:
    """Group by (module, thesis_run_label); compute (mean, std, n_seeds_with_metric).

    Returns: {(module, label): (mean, std, n)}
    """
    groups: Dict[Tuple, List[float]] = {}
    for rec in records:
        key = (rec["module"], rec["thesis_run_label"])
        val = extract_metric(rec["results_data"], metric_key)
        if val is not None:
            groups.setdefault(key, []).append(val)
    out = {}
    for key, vals in groups.items():
        if vals:
            out[key] = (float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals))
    return out


def expected_main_cells() -> Set[Tuple[str, str, int]]:
    """D-20 expected cell set for main_comparison: 4 modules × 3 seeds × {main}.

    Returns: set of (module, thesis_run_label, seed) tuples.
    """
    return {
        (module, "main", seed)
        for module in ("baseline", "personalized", "adaptive", "pfedrec")
        for seed in (42, 1337, 2026)
    }


def expected_ablation_cells() -> Set[Tuple[str, str, int]]:
    """D-20 expected cell set for ablations: 7 ablation cells × 3 seeds.
    Always module='adaptive' per D-13.
    """
    ablation_labels = [
        "ablation_alpha_method=multi_factor",
        "ablation_alpha_method=data_quantity",
        "ablation_per_user_alpha=true",
        "ablation_item_perturbation=true",
        "ablation_contrastive_lambda=0.1",
        "ablation_fusion_type=add",
        "ablation_fusion_type=gate",
    ]
    return {
        ("adaptive", label, seed)
        for label in ablation_labels
        for seed in (42, 1337, 2026)
    }


def find_missing_cells(records: List[Dict], expected: Set[Tuple]) -> List[Tuple]:
    """D-20 hard-fail set: subtract observed (module, label, seed) from expected."""
    observed = {(r["module"], r["thesis_run_label"], r["run_seed"]) for r in records}
    return sorted(expected - observed)
```

### Pattern 5: Markdown Table Rendering with Bold-the-Winner (D-24 + Discretion)

**What:** Render `0.4123 ± 0.0089` cells; bold the row that wins under D-11 criterion.

**Example:**
```python
def fmt_cell(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "—"
    return f"{mean:.4f} ± {std:.4f}"


def is_winner(my_mean: float, my_std: float, others: List[Tuple[float, float]]) -> bool:
    """D-11 win criterion: my (mean - std) strictly > every other's (mean + std)."""
    my_lower = my_mean - my_std
    for other_mean, other_std in others:
        if my_lower <= other_mean + other_std:
            return False
    return True


def render_main_md(rows: List[Dict], path: Path) -> None:
    """Write main_comparison.md with bold-the-winner."""
    # rows = [{"module": "baseline", "ndcg10_mean": 0.4123, "ndcg10_std": 0.0089, ...}, ...]
    headers = ["Module", "NDCG@10", "HR@10", "Sparse NDCG@10", "Sparse HR@10"]
    # Identify winner per metric (D-11 applied to baseline/personalized/adaptive ONLY;
    # PFedRec excluded from win comparison per D-05).
    comparable_rows = [r for r in rows if r["module"] in ("baseline", "personalized", "adaptive")]
    pfedrec_rows = [r for r in rows if r["module"] == "pfedrec"]

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in comparable_rows + pfedrec_rows:  # PFedRec last (footnoted)
        cells = [row["module"]]
        for metric in ("ndcg10", "hr10", "ndcg10_sparse", "hr10_sparse"):
            cell = fmt_cell(row.get(f"{metric}_mean"), row.get(f"{metric}_std"))
            # Bold if winner
            if row in comparable_rows:
                others = [(r[f"{metric}_mean"], r[f"{metric}_std"])
                          for r in comparable_rows if r is not row
                          and r.get(f"{metric}_mean") is not None]
                if (row.get(f"{metric}_mean") is not None
                        and is_winner(row[f"{metric}_mean"], row[f"{metric}_std"], others)):
                    cell = f"**{cell}**"
            if row["module"] == "pfedrec":
                cell = f"{cell} †"  # footnote marker
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    # D-08 footnote for PFedRec
    lines.append("")
    lines.append("† PFedRec (paper-faithful) — `dim=32, SGD lr=0.1, BCE, fraction-train=1.0; "
                 "matches IJCAI-23 reference within ±2 points`. Not counted toward "
                 "\"adaptive beats baselines\" claim per Phase 7 D-05.")
    atomic_write_text(str(path), "\n".join(lines) + "\n")
```

### Pattern 6: Atomic Markdown Write (extend `atomic.py`)

**What:** Add `atomic_write_text` companion to `atomic.py` for markdown output. Mirrors `atomic_write_json` but skips the JSON serialization step.

**Example:**
```python
# Source: scripts/foundation/fedrec_foundation/atomic.py:16-48 (atomic_write_json pattern)

def atomic_write_text(path: str, content: str) -> None:
    """Write a text string atomically via tempfile + ``os.replace``.

    Mirrors :func:`atomic_write_json` for plain-text payloads (markdown,
    CSV, etc.).

    Parameters
    ----------
    path : str
        Destination path. Parent directories are created if absent.
    content : str
        UTF-8 text payload.

    Returns
    -------
    None
    """
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
```

### Anti-Patterns to Avoid

- **Storing the matrix as YAML/JSON config:** D-18 is explicit — Python script with matrix-as-data. Avoids parsing layer + lets the planner add Python logic (e.g., conditional ablation skipping if a module hasn't run yet).
- **Running cells in parallel by default:** GPU contention risk (CONTEXT discretion section recommends serial). Don't add `multiprocessing.Pool` parallelism to v1.
- **Re-aggregating after every cell:** Aggregator is a separate script for a reason — it's idempotent and fast (reads JSONs, writes tables). Running it 33 times during the sweep adds nothing. Emit `_progress.json` (lightweight) per cell instead.
- **Reading from `eval_metrics_history`:** The canonical metric path is `final_metrics["best"]["sampled_ndcg@10"]`. Phase 6 D-06 explicitly says reading from `eval_metrics_history[best_round_num]` is forbidden — those are stale in-loop sufficient stats, not restored-state metrics.
- **Including `cross_silo_legacy` runs in the thesis filter:** The `_manifest.thesis_run_label == ""` filter handles this naturally (empty label → not a thesis run). Do NOT add explicit mode filtering — `paper_compat_pfedrec` runs ARE thesis runs (PFedRec footnoted row). The filter is on `thesis_run_label`, not on `mode`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-run-dir results write path | Custom path resolution from cwd | `module_run_results_dir(module, run_id)` from `fedrec_foundation.paths` | Already validated in Phase 6; whitelist enforces typo guard; handles cwd-independence; per-run dir auto-created |
| Manifest read/write | Custom JSON munging | `RunManifest` dataclass + `embed_manifest_in_result` + `write_manifest_sibling` from `fedrec_foundation.manifest` | Schema enforced; `dataclasses.replace` preserves type safety; D-15 double-write semantics already proven |
| ModeProfile resolution | If/elif on mode string | `resolve_mode_defaults("thesis_crossdevice_main")` | Closed-enum whitelist already enforces typo guard; raises `ValueError` on unknown mode |
| Atomic file writes | `open(..., "w")` + `f.write` | `atomic_write_json` (existing) + `atomic_write_text` (new companion) | Same-FS rename semantics; never partial files; uniform across foundation |
| Subprocess invocation of `flwr run` | Direct `subprocess.run(["flwr", "run", ...])` | `subprocess.run([sys.executable, scripts/run.py, module, mode, --run-config, ...])` via the established launcher | `scripts/run.py` knows mode → num-supernodes mapping (federation-level, can't set from app) and TOML-quotes string values for `flwr run --run-config`'s parser |
| Per-run-id generation | `time.time()` strings | `generate_run_id()` from `fedrec_foundation.manifest` | Sortable + 6-hex tail disambiguates simultaneous launches; format pinned by tests |

**Key insight:** Phase 7 is mostly *consumption* of the foundation contract Phases 1-6 built. Hand-rolling any of the above replicates code that's already test-pinned, and creates parallel paths that drift over time.

## Runtime State Inventory

> Phase 7 adds a new mode profile + new manifest fields + new orchestrator/aggregator scripts. The thesis run is largely a fresh execution — but the new mode profile registration, the schema bump, and the new manifest fields all interact with existing on-disk state and code paths. Inventory below is mandatory.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| **Stored data** | The `_manifest.thesis_run_label` field is a NEW key in `results/federated/<module>/<run_id>/manifest.json`. Pre-Phase-7 manifests don't have it. The aggregator's `manifest.get("thesis_run_label", "")` filter naturally treats absent → "" → non-thesis run, so legacy runs are correctly excluded from the thesis tables. **Verified safe.** | None — pre-Phase-7 manifests stay readable; aggregator filter handles absence. |
| **Live service config** | None — no external services are involved. W&B is the only "live service" and routing change is just a different run name pattern (D-21); the project string `federated-cf-cross-device` is unchanged. | None. |
| **OS-registered state** | None — Phase 7 has no Windows Task Scheduler / pm2 / systemd / launchd registrations. | None. |
| **Secrets/env vars** | `WANDB_API_KEY` (existing). No new env vars introduced. | None. |
| **Build artifacts / installed packages** | `fedrec-foundation` is editable-installed in all 4 federated modules (per Phase 1 Plan 06). Adding a new mode profile + manifest schema bump does NOT require re-install (editable install picks up source changes immediately). However: **the foundation tests must be re-run after the mode/manifest changes land** to confirm no regression. | Re-run `pytest scripts/foundation/tests/` after `mode.py` + `manifest.py` edits land. |

**Nothing found in category** "OS-registered state" and "Live service config": **verified by grep + project structure inspection — Phase 7 introduces no new external integrations**.

**The canonical question:** *After every file in the repo is updated, what runtime systems still have old state?* Answer: **None** that affect Phase 7 correctness. Pre-Phase-7 result manifests remain on disk but are filtered out by `thesis_run_label == ""`.

## Common Pitfalls

### Pitfall 1: Per-group key delimiter divergence (slash vs underscore for `evaluated_users`)
**What goes wrong:** Aggregator silently miscounts sparse-user evaluations because it reads `evaluated_users_sparse` from PFedRec results (which actually emit `evaluated_users/sparse`) — or vice versa.
**Why it happens:** Phase 6 confirmed (06-CONTEXT.md §"Per-user-group key delimiter divergence" + verified in `pfedrec/strategy.py:121` and `baseline/strategy.py:99-101` during this research):
- **HR/NDCG keys**: All 4 modules use **slash** (`sampled_hr@10/sparse`, `sampled_ndcg@10/sparse`). UNIFORM.
- **`evaluated_users` keys**: Baseline / personalized / adaptive use **underscore** (`evaluated_users_sparse`). PFedRec uses **slash** (`evaluated_users/sparse`). DIVERGES.
**How to avoid:** The aggregator only needs `sampled_ndcg@10`, `sampled_hr@10`, plus `/sparse`, `/medium`, `/dense` variants for THE THESIS TABLE. It does NOT need `evaluated_users_*` for the canonical thesis row metrics — those are diagnostic exposure counts, not table cells. Solution: read only the slash-delimited HR/NDCG keys (uniform across modules); ignore `evaluated_users_*` for table rendering.
**Warning signs:** Aggregator outputs unexpectedly high or low std (e.g., 0.0 from "missing key → 0.0 fallback"); silent zero-fill instead of hard fail.

### Pitfall 2: D-22 fields not flowing into manifest from run_config
**What goes wrong:** Orchestrator passes `--run-config "thesis-run-label=main ablation-dimension=none ..."` but each `server_app.py` doesn't know to mutate the manifest with these values. Manifests come out empty, aggregator filter `thesis_run_label != ""` → all runs filtered out → "Missing 33 cells" hard fail.
**Why it happens:** `RunManifest` is built once per run via `build_run_manifest(...)` near line 442 (varies per server_app). The 3 new fields default to `""` / `"none"` / `""`. Without explicit mutation, they stay at defaults.
**How to avoid:** Each `server_app.py` MUST read the 3 thesis fields from `context.run_config` and either (a) pass to `build_run_manifest` if extended, or (b) post-build mutate via `dataclasses.replace`. Pattern:
```python
# After build_run_manifest(...)
manifest = dataclass_replace(
    manifest,
    thesis_run_label=str(context.run_config.get("thesis-run-label", "")),
    ablation_dimension=str(context.run_config.get("ablation-dimension", "none")),
    ablation_value=str(context.run_config.get("ablation-value", "")),
)
```
This pattern matches Phase 6's D-07 mutation: `manifest = dataclass_replace(manifest, final_eval_round_index=N, metrics=...)` before `embed_manifest_in_result`.
**Warning signs:** First sweep run completes, but `manifest.json` has `"thesis_run_label": ""`. Aggregator finds 0 thesis runs.

### Pitfall 3: New mode `thesis_crossdevice_main` not in `mode in (...)` tuples → wrong W&B project + wrong write path
**What goes wrong:** Each `server_app.py` has TWO `mode in ("benchmark_cross_device", "paper_compat_pfedrec")` gates:
1. **W&B project routing** (e.g., `baseline/server_app.py:292-296`): falls back to module-historical project (`"federated-cf"` / `"federated-personalized-cf"` / `"federated-pfedrec"` / `"federated-adaptive-personalized-cf"`) if mode not in the tuple.
2. **`module_run_results_dir` results path** (e.g., `baseline/server_app.py:988`): falls back to legacy flat path if mode not in the tuple.

If we add the new mode profile but DON'T update these gates, thesis runs go to the WRONG W&B project and the WRONG file path — and the aggregator's `results/federated/<module>/<run_id>/results.json` glob finds nothing.
**Why it happens:** The mode profile registry is one place; the per-module code that branches on mode literals is four other places.
**How to avoid:** Update both `mode in (...)` tuples in all 4 server_apps:
```python
# BEFORE (every server_app.py, ~2 sites):
if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):

# AFTER:
if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec"):
```
That's 8 line edits (4 server_apps × 2 sites). Add a regression test that asserts `thesis_crossdevice_main` triggers the per-run-dir code path (e.g., a mock test in each module's existing `test_server_integration.py`).
**Warning signs:** Thesis runs land at `results/federated/<module>_results.json` (legacy flat) instead of `results/federated/<module>/<run_id>/results.json`. W&B runs appear in `federated-adaptive-personalized-cf` instead of `federated-cf-cross-device`.

### Pitfall 4: Per-module `pyproject.toml` doesn't accept the new `mode` value
**What goes wrong:** Each module's `pyproject.toml` declares `mode = "<default>"` and the launcher passes `mode=thesis_crossdevice_main` via `--run-config`. The Flower TOML parser accepts any string for `mode`, but the foundation `resolve_mode_defaults` raises `ValueError("Unknown mode")` if the registry doesn't know it.
**Why it happens:** The mode value is a typo guard at `resolve_mode_defaults` (mode.py:219-222). Adding `_THESIS_CROSSDEVICE_MAIN` to `_REGISTRY` is what makes the launcher's run-config valid.
**How to avoid:** Confirm `_REGISTRY["thesis_crossdevice_main"] = _THESIS_CROSSDEVICE_MAIN` is the FIRST edit. Test with `python scripts/run.py adaptive thesis_crossdevice_main --dry-run` BEFORE running any real cell.
**Warning signs:** Launcher dies on first cell with `ValueError: Unknown mode 'thesis_crossdevice_main'`.

### Pitfall 5: `scripts/run.py:64-68` MODE_NUM_SUPERNODES dict missing the new mode
**What goes wrong:** Launcher's `argparse choices=` clamp is `sorted(MODE_NUM_SUPERNODES.keys())`. Without adding `"thesis_crossdevice_main": 6040` to the dict, the launcher rejects `python scripts/run.py adaptive thesis_crossdevice_main` with `argparse.ArgumentTypeError` BEFORE the registry-resolution code even runs.
**How to avoid:** Add to `scripts/run.py`:
```python
MODE_NUM_SUPERNODES = {
    "benchmark_cross_device": 6040,
    "thesis_crossdevice_main": 6040,  # NEW
    "paper_compat_pfedrec": 6040,
    "cross_silo_legacy": 5,
}
```
Co-edit with the mode.py change in the same wave.
**Warning signs:** `argparse: error: argument mode: invalid choice: 'thesis_crossdevice_main' (choose from 'benchmark_cross_device', 'cross_silo_legacy', 'paper_compat_pfedrec')`.

### Pitfall 6: Missing-cell hard-fail (D-20) accidentally ignores PFedRec at `paper_compat_pfedrec`
**What goes wrong:** Aggregator's expected-cell set assumes mode=`thesis_crossdevice_main`. PFedRec runs at `paper_compat_pfedrec` (D-06). If the expected-cell set filters by mode == thesis_crossdevice_main, PFedRec is wrongly flagged as missing → hard fail even after a successful sweep.
**How to avoid:** Aggregator's expected-cell set is keyed on `(module, thesis_run_label, run_seed)` — NOT on `mode`. PFedRec's manifests carry `thesis_run_label="main"` (orchestrator passes this regardless of mode). Mode is irrelevant to the table-cell-presence question.
**Warning signs:** Aggregator hard-fails with "Missing 3 cells: [(pfedrec, main, 42), ...]" even though those runs visibly completed.

### Pitfall 7: Manifest schema v3 backward-compat broken
**What goes wrong:** Adding required (non-default) fields to `RunManifest` would break `test_run_manifest_backward_compat_v1` (test_manifest.py:181-221), which constructs a manifest with only the v1 field set.
**How to avoid:** All 3 new fields MUST have safe defaults (D-22 already specifies). Verify by extending the existing backward-compat test rather than writing a new one:
```python
# Update test_run_manifest_backward_compat_v1 (or add v2-backward-compat sibling test):
# Same construction without the 3 new fields; assert they default correctly.
manifest = RunManifest(...)  # No thesis_run_label kwarg
assert manifest.thesis_run_label == "", "Default for thesis_run_label is empty string"
assert manifest.ablation_dimension == "none"
assert manifest.ablation_value == ""
```
**Warning signs:** Pytest fails with `TypeError: RunManifest.__init__() missing 3 required positional arguments`.

### Pitfall 8: Orchestrator skip-on-existing logic doesn't disambiguate cells
**What goes wrong:** Two cells with the same `(module, run_seed)` but different `(thesis_run_label, ablation_dimension, ablation_value)` collide → orchestrator skips one because the other already wrote a `manifest.json`.
**Why it happens:** A naive "is there any manifest with `module=adaptive seed=42`?" check is wrong — adaptive runs at seed=42 happen 8 times (1 main + 7 ablations).
**How to avoid:** `cell_already_done` MUST match on the full tuple `(module, thesis_run_label, run_seed, ablation_dimension, ablation_value)` — that's the cell identity per D-22.
**Warning signs:** First sweep run completes 12/33 cells, second sweep run reports "all cells done" even though 21 ablations are missing.

### Pitfall 9: `ablation_dimension` field naming inconsistency
**What goes wrong:** D-22 uses underscore-snake-case `ablation_dimension` (Python field). D-21 W&B run names use mix of dash and equals: `thesis-ablation-adaptive-seed42-fusion=add`. Aggregator's `is_ablation_label_match` regex must handle both forms.
**Why it happens:** Two namespaces:
- **Manifest field name (snake_case)**: `ablation_dimension="fusion_type"`, `ablation_value="add"`.
- **W&B run name (kebab + equals)**: `thesis-ablation-adaptive-seed42-fusion=add`.
- **Manifest `thesis_run_label`**: `"ablation_fusion_type=add"` (snake-case for the dimension, equals for the value).
**How to avoid:** Pin the canonical mapping in code documentation:
```
ablation_dimension="fusion_type" + ablation_value="add"
  → thesis_run_label="ablation_fusion_type=add"  (manifest)
  → wandb-run-name="thesis-ablation-adaptive-seed42-fusion=add"  (run name)
```
For the run name, the orchestrator can use a shortened form (`fusion=add` instead of `fusion_type=add`) — D-21 example uses the short form. Pin a `_ABLATION_NAME_FOR_WANDB` mapping if needed:
```python
_ABLATION_NAME_FOR_WANDB = {
    "alpha_method": "alpha",
    "per_user_alpha": "pua",
    "item_perturbation": "ip",
    "contrastive_lambda": "cl",
    "fusion_type": "fusion",
}
```
**Warning signs:** W&B run names look right but aggregator can't match them to manifest fields (irrelevant — aggregator reads manifest fields, not W&B names). The bug surface is the orchestrator's name-construction code.

### Pitfall 10: `sampled_hr@10/sparse` etc. are absent when no sparse users were evaluated in a run
**What goes wrong:** Some seeds may produce zero evaluable sparse interactions (CONTEXT discretion section flags this as "extremely rare given 6040 users but theoretically possible"). When this happens, the strategy's `_sufficient_stats_to_thesis_metrics` returns `0.0` for the per-group ratio (zero-divide-safe). But std across seeds with one seed = 0 produces inflated noise.
**How to avoid:** Aggregator should detect zero `evaluated_users_sparse` (or `evaluated_users/sparse` for PFedRec — Pitfall 1) for a given seed and treat that seed's value as "unavailable" rather than 0.0. Footnote `n_seeds_with_sparse=2/3` per CONTEXT discretion.
**Warning signs:** Sparse NDCG@10 std for some module is suspiciously high (e.g., `0.0521` when overall NDCG std is `0.0089`).

## Code Examples

Verified patterns from official sources / existing codebase. All paths are absolute.

### Example 1: ModeProfile registration (existing pattern)
```python
# Source: /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/fedrec_foundation/mode.py:118-186

_BENCHMARK_CROSS_DEVICE = ModeProfile(
    mode="benchmark_cross_device",
    num_supernodes=6040,
    partition_mode="natural",
    weight_policy="num_positives",
    primary_evaluator="sampled_loo_99",
    fraction_train=0.1,
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=64,
    optimizer="adam",
    lr=0.001,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",
    assert_one_user_per_client=True,
)


_REGISTRY: Dict[str, ModeProfile] = {
    "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
    "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
    "cross_silo_legacy": _CROSS_SILO_LEGACY,
}


MODE_NAMES = tuple(_REGISTRY.keys())
```

### Example 2: Manifest schema bump pattern (Phase 6 precedent)
```python
# Source: /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/fedrec_foundation/manifest.py:28-29 + 86-103

# Bump when the manifest schema gains/loses a field or changes semantics.
RUN_MANIFEST_SCHEMA_VERSION: int = 2  # Phase 6: adds final_eval_round_index + metrics fields

# At the end of the RunManifest dataclass:
    # Phase 6 additions (both with safe defaults so v1 fixtures still construct
    # without TypeError — Pitfall 3 from RESEARCH.md):
    final_eval_round_index: int = 0
    """Index of the post-restore extra-eval-round broadcast (D-06)."""
    metrics: Dict[str, Any] = field(default_factory=dict)
    """Mirror of ``results_data["final_metrics"]`` block (D-07)."""
```

### Example 3: Subprocess test pattern (existing in test_baseline_subprocess_determinism.py)
```python
# Source: /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/tests/test_baseline_subprocess_determinism.py:94-105

cmd = [
    sys.executable,
    str(_RUN_PY),
    "baseline",
    "benchmark_cross_device",
    "--run-config",
    "run-seed=42 num-server-rounds=2 fraction-train=0.001 wandb-enabled=false",
]
proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
assert proc.returncode == 0, proc.stderr
```

### Example 4: Run config TOML quoting (from `scripts/run.py`)
```python
# Source: /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/run.py:73-95

def _quote_value_for_flwr(v: str) -> str:
    """Bare-word values like 'benchmark_cross_device' need quoting because
    Flower's parse_config_args rebuilds the --run-config string into TOML
    and parses with tomli. String values must be TOML-quoted."""
    if not v:
        return '""'
    if (v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'"):
        return v
    try:
        float(v)
        return v
    except ValueError:
        pass
    if v.lower() in ("true", "false"):
        return v.lower()
    return f'"{v}"'
```

### Example 5: How baseline currently writes per-run results (D-02 / D-04)
```python
# Source: /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf/federated_baseline_cf/server_app.py (Phase 6 wired)

if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):
    run_dir = module_run_results_dir(_MODULE, run_id)
    results_filename = run_dir / "results.json"  # D-04 clean filename
    atomic_write_json(str(results_filename), results_data)
    # ... write_manifest_sibling with sibling_name="manifest.json" ...
```

This is the gate that needs `"thesis_crossdevice_main"` added (Pitfall 3).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `results/federated/<run_id>_results.json` (legacy flat) | `results/federated/<module>/<run_id>/{results.json,manifest.json}` (per-run-dir) | Phase 6 (2026-04-29) | Aggregator globs the per-run-dir layout; legacy flat files are filtered by absence of `_manifest.thesis_run_label` |
| `final_metrics["sampled_ndcg@10"]` (flat) | `final_metrics["best"]["sampled_ndcg@10"]` (nested) | Phase 6 D-07 | Aggregator MUST traverse the nested structure |
| `wandb.run.summary["final/sampled_ndcg@10"]` | `wandb.run.summary["best/sampled_ndcg@10"]` | Phase 6 (Pitfall 7) | Phase 7 W&B summary keys inherit; aggregator does not read from W&B (only from results.json) |
| W&B sweeps via `sweep.yaml` (Bayesian + Hyperband) | Standalone matrix orchestrator (Phase 7 D-18) | Phase 7 (NEW) | Deterministic, idempotent, no Bayesian search |
| `random.seed(seed)` inline | Three-tier RNG factories (`py_rng`, `np_rng`, `torch_gen`) per FND-06 | Phase 1 (2026-04-19) | All seeds derive deterministically from `(run_seed, user_idx, round_num, purpose)` |

**Deprecated/outdated:**
- `compare_all_results.py` (`scripts/compare_all_results.py`) — pre-Phase-6 aggregator that reads from the legacy flat `results/federated/*_results.json` layout. Phase 7 D-19 explicitly says new script, not extension. The planner MAY inspect it for parsing-convention ideas but should not extend it.
- `eval_metrics_history[best_round_num]` lookups — Phase 6 D-06 forbidden. Use `final_metrics["best"]` (post-restore evaluation) instead.

## Open Questions

1. **Should `thesis_run_label`, `ablation_dimension`, `ablation_value` be passed via run_config or computed server-side from mode?**
   - What we know: D-22 says they're manifest fields; CONTEXT doesn't specify HOW server_app populates them.
   - What's unclear: Whether all 4 server_apps need a manifest-mutation patch (similar to Phase 6's `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)`), or whether the orchestrator can post-process manifest.json after the run completes (e.g., open + mutate + atomic-rewrite).
   - Recommendation: **Patch each server_app** (consistent with Phase 6 D-07 pattern). Pass via `--run-config "thesis-run-label=main ablation-dimension=none ablation-value="`; each server_app reads with `context.run_config.get(...)` and mutates manifest before `embed_manifest_in_result`. 4 patch sites, ~5 lines each = ~20 lines total. Cleaner than orchestrator post-processing because it preserves the D-15 atomic double-write invariant.

2. **Where does the orchestrator live regarding cwd?**
   - What we know: `scripts/run.py` accepts `cwd=PROJECT_ROOT` from `subprocess.run`. Phase 7 orchestrator should mirror.
   - What's unclear: Whether `scripts/thesis/run_thesis_sweep.py` should compute `repo_root` via `fedrec_foundation.paths.repo_root()` (consistent with foundation tests) or via `Path(__file__).parent.parent.parent` (file-relative).
   - Recommendation: Use `fedrec_foundation.paths.repo_root()`. Same pattern as `test_baseline_subprocess_determinism.py:43`. Single source of truth.

3. **Does the win criterion (D-11) apply to HR@10 too, or NDCG@10 only?**
   - What we know: REQUIREMENTS THS-03/THS-04 specifically say "NDCG@10". CONTEXT D-11 says "adaptive's mean NDCG@10 strictly greater AND non-overlapping ±1σ intervals vs every baseline".
   - What's unclear: Whether the markdown table should bold the winner per metric (NDCG-only bolding + HR informational) or per row (winner of one metric → row bolded).
   - Recommendation: **Bold the winner cell per metric (independent per column).** The thesis claim is NDCG-specific; bolding only the NDCG cells (overall + sparse) keeps the visual signal aligned with the claim. HR cells stay un-bolded as informational. Document this in the markdown body. This is Claude's discretion per CONTEXT and matches "bold the cell whose row 'beats' all comparable rows under the D-11 win criterion."

4. **What happens if a partial PFedRec reproduction fails (HR@10 outside ±2 points)?**
   - What we know: PFR-08 requires HR@10 ≈ 0.729, NDCG@10 ≈ 0.441, ±2 points; Phase 5 reproduction passed once (`pfr08_verification.passed=true`); Phase 6 UAT item #3 is pending the full 100-round run.
   - What's unclear: Whether Phase 7 should hard-fail if PFedRec doesn't reproduce, or whether it footnotes the row with a divergence warning.
   - Recommendation: **Footnote with divergence warning in the markdown body**, not a hard fail. PFedRec is a calibration reference (D-05), not a primary thesis claim. If reproduction drifts post-Phase-5, the failure is documented in the table footer; the orchestrator does NOT halt the rest of the matrix. The aggregator can compare the PFedRec row's mean values against `(0.729, 0.441)` thresholds and emit the divergence note.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7+ (already installed in foundation `[project.optional-dependencies] dev`) |
| Config file | `scripts/foundation/pyproject.toml` `[tool.pytest.ini_options]` (testpaths=["tests"], addopts="-ra") |
| Quick run command | `pytest scripts/foundation/tests/test_thesis_orchestrator.py scripts/foundation/tests/test_thesis_aggregator.py scripts/foundation/tests/test_mode.py scripts/foundation/tests/test_manifest.py -x -v` |
| Full suite command | `cd scripts/foundation && pytest -ra` (runs all 100+ existing foundation tests + new thesis tests) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| THS-01 | `resolve_mode_defaults("thesis_crossdevice_main")` returns the expected ModeProfile with embedding_dim=64, optimizer="adam", lr=0.001 | unit | `pytest scripts/foundation/tests/test_mode.py::test_thesis_crossdevice_main_profile -v` | ❌ Wave 0 |
| THS-01 | `MODE_NAMES` contains the new mode | unit | `pytest scripts/foundation/tests/test_mode.py::test_all_four_modes_registered -v` | ❌ Wave 0 (extends existing `test_all_three_modes_registered`) |
| THS-01 | `scripts/run.py adaptive thesis_crossdevice_main --dry-run` succeeds | smoke | `pytest scripts/foundation/tests/test_launcher.py::test_thesis_mode_dry_run -v` | ❌ Wave 0 |
| THS-02 | Orchestrator builds 12 main cells (3 modules × 3 seeds + pfedrec × 3 seeds) | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_main_matrix_size -v` | ❌ Wave 0 |
| THS-02 | Orchestrator builds 21 ablation cells (7 ablations × 3 seeds) | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_ablation_matrix_size -v` | ❌ Wave 0 |
| THS-02 | `cell_already_done` matches on (module, label, seed, dim, value) tuple | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_skip_on_existing_full_tuple -v` | ❌ Wave 0 |
| THS-02 | `cell_run_config_string` produces valid `--run-config` strings (TOML-quoted) | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_run_config_quoting -v` | ❌ Wave 0 |
| THS-02 | `--dry-run` prints commands without executing subprocess | smoke | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_dry_run_no_subprocess -v` | ❌ Wave 0 |
| THS-03/04 | Aggregator reads `final_metrics["best"]["sampled_ndcg@10"]` correctly from synthetic results.json fixture | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_extract_overall_ndcg10 -v` | ❌ Wave 0 |
| THS-03/04 | Aggregator reads `sampled_ndcg@10/sparse` (slash) for ALL 4 modules | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_extract_sparse_ndcg10_uniform_slash -v` | ❌ Wave 0 |
| THS-03 | Win-criterion D-11 detection: adaptive (0.42, 0.005) beats personalized (0.40, 0.005) under non-overlap rule | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_d11_win_criterion -v` | ❌ Wave 0 |
| THS-03 | Win-criterion D-11 NEGATIVE: overlapping intervals correctly NOT flagged as winner | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_d11_overlap_no_winner -v` | ❌ Wave 0 |
| THS-04 | Sparse-slice rendering with `n_seeds_with_sparse=2/3` footnote (one seed missing sparse data) | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_sparse_partial_seeds -v` | ❌ Wave 0 |
| THS-05 | Aggregator filter recognizes `ablation_<dim>=<val>` labels and groups them | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_ablation_label_grouping -v` | ❌ Wave 0 |
| THS-05 | Hard-fail on missing cells lists exact missing tuples per D-20 | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_d20_hard_fail_missing -v` | ❌ Wave 0 |
| THS-06 | Aggregator preserves per-group keys (medium, dense) in CSV output | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_csv_per_group_columns -v` | ❌ Wave 0 |
| THS-07 | Aggregator emits 6 files: main_comparison.{md,csv}, ablations.{md,csv}, sparse_slice.{md,csv} | smoke | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_six_output_files -v` | ❌ Wave 0 |
| THS-07 | Atomic write: no `.tmp-*` leftovers after aggregator run | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_atomic_write_no_tmp -v` | ❌ Wave 0 |
| THS-07 | Cell format `0.4123 ± 0.0089` (4 decimals) matches D-24 | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_cell_format -v` | ❌ Wave 0 |
| THS-22 (manifest) | Schema bump to v3 + 3 new fields with defaults | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_schema_version_3 -v` | ❌ Wave 0 (extends existing v2 test) |
| THS-22 (manifest) | v1 backward-compat: pre-v3 manifests load with default values for 3 new fields | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_backward_compat_v2 -v` | ❌ Wave 0 |
| THS-22 (manifest) | Post-build mutation via `dataclasses.replace` populates 3 thesis fields | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_carries_thesis_fields -v` | ❌ Wave 0 |
| THS-22 (server_app) | Each server_app reads thesis-run-label from run_config and mutates manifest | integration | per-module: `pytest federated-<module>-cf/tests/test_server_integration.py::test_thesis_label_in_manifest -v` | ❌ Wave 0 (4 module tests) |
| THS-validation | atomic_write_text leaves no .tmp-* leftovers and writes content correctly | unit | `pytest scripts/foundation/tests/test_atomic.py::test_atomic_write_text -v` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** Run `pytest scripts/foundation/tests/test_thesis_orchestrator.py scripts/foundation/tests/test_thesis_aggregator.py -x -v` (the new Phase-7 test files; ~30s).
- **Per wave merge:** Run full foundation suite via `cd scripts/foundation && pytest -ra` plus per-module `pytest federated-<module>-cf/tests/ -ra`.
- **Phase gate:** Full suite green AND a 1-cell smoke run completes (e.g., `python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --thesis-run-label=main`) AND a re-run of the same cell hits the skip-on-existing path AND aggregator hard-fails with "Missing 32 cells" (proves D-20 detection works on the partial state) BEFORE running the full ~50hr matrix.
- **Pre-aggregation gate:** Full matrix complete (33 manifests on disk with thesis fields populated, verifiable by `find results/federated -name manifest.json -exec grep -l 'thesis_run_label' {} \; | wc -l → 33`) BEFORE running aggregator.

### Edge Cases

- **Missing seed in some module** (D-20 hard-fail surface): Test with synthetic fixture having only seeds {42, 1337} for adaptive; expect aggregator to fail with explicit list of (adaptive, main, 2026) missing.
- **schema-v2 manifest** (legacy run from Phase 6 still on disk): Test that aggregator filter `thesis_run_label != ""` correctly excludes it (returns "" via dict.get default).
- **Empty results.json** (corrupt mid-write): Test that aggregator catches `json.JSONDecodeError` per-file and continues to the next results.json — does NOT propagate corruption.
- **One seed has zero sparse evaluations** (Pitfall 10): Test sparse-slice rendering emits `n_seeds_with_sparse=2/3` footnote when `evaluated_users_sparse=0` for one seed.
- **PFedRec mode collision**: Test that a `paper_compat_pfedrec` run with `thesis_run_label="main"` is correctly counted in the main_comparison table (not filtered by mode).
- **Two cells writing simultaneously** (race): Test orchestrator's atomic skip — `cell_already_done` reads manifest once; doesn't double-skip on concurrent reads (single-writer is enforced by serial execution per CONTEXT default).

### Wave 0 Gaps

- [ ] `scripts/foundation/fedrec_foundation/atomic.py` — extend with `atomic_write_text` (Pattern 6)
- [ ] `scripts/foundation/fedrec_foundation/mode.py` — add `_THESIS_CROSSDEVICE_MAIN` + register
- [ ] `scripts/foundation/fedrec_foundation/manifest.py` — bump schema 2→3 + 3 new fields
- [ ] `scripts/run.py` — add `"thesis_crossdevice_main": 6040` to MODE_NUM_SUPERNODES
- [ ] All 4 `server_app.py` files — add `"thesis_crossdevice_main"` to BOTH `mode in (...)` tuples + read+mutate manifest with thesis fields
- [ ] `scripts/foundation/tests/test_atomic.py` — covers `atomic_write_text`
- [ ] `scripts/foundation/tests/test_mode.py` — extend with thesis mode resolution test
- [ ] `scripts/foundation/tests/test_manifest.py` — extend with v3 schema + thesis fields tests
- [ ] `scripts/foundation/tests/test_thesis_orchestrator.py` — NEW, covers matrix builders + skip + dry-run + run-config quoting
- [ ] `scripts/foundation/tests/test_thesis_aggregator.py` — NEW, covers result extraction + win criterion + sparse handling + missing-cell hard-fail + 6-file emission + atomic write
- [ ] Per-module `tests/test_server_integration.py` — extend with thesis-label-flows-into-manifest test (4 modules × ~10 lines each)
- [ ] `scripts/thesis/__init__.py` — empty file (D-18)
- [ ] `scripts/thesis/run_thesis_sweep.py` — orchestrator (D-18)
- [ ] `scripts/thesis/aggregate_results.py` — aggregator (D-19)
- [ ] No framework install needed — pytest already in foundation dev deps.

## Sources

### Primary (HIGH confidence)
- `scripts/foundation/fedrec_foundation/mode.py` (verified 2026-04-29) — `_BENCHMARK_CROSS_DEVICE` ModeProfile to clone; existing `_REGISTRY` pattern.
- `scripts/foundation/fedrec_foundation/manifest.py` (verified 2026-04-29) — `RunManifest` dataclass; `RUN_MANIFEST_SCHEMA_VERSION = 2`; D-15 double-write helpers; Phase 6 schema-bump precedent at lines 86-103.
- `scripts/foundation/fedrec_foundation/paths.py` (verified 2026-04-29) — `module_run_results_dir` + `_ALLOWED_MODULES` whitelist; `repo_root()` walk-up.
- `scripts/foundation/fedrec_foundation/atomic.py` (verified 2026-04-29) — `atomic_write_json` pattern to extend.
- `scripts/run.py` (verified 2026-04-29) — Mode → num-supernodes launcher, TOML quoting helper, argparse choices clamp.
- `federated-baseline-cf/federated_baseline_cf/strategy.py` (verified 2026-04-29) — Per-group HR/NDCG keys (slash) + evaluated_users keys (underscore for non-pfedrec).
- `federated-pfedrec/federated_pfedrec/strategy.py` (verified 2026-04-29) — `evaluated_users/sparse` (slash for pfedrec) confirmation at line 121.
- `federated-baseline-cf/federated_baseline_cf/server_app.py:291-295, 988-991` (verified 2026-04-29) — `mode in (...)` tuple sites needing `thesis_crossdevice_main` addition.
- `federated-personalized-cf/federated_personalized_cf/server_app.py:380-382, 988-991` (verified 2026-04-29) — Same pattern.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:487-491, 1292-1295` (verified 2026-04-29) — Same pattern.
- `federated-pfedrec/federated_pfedrec/server_app.py:512-515, 1156-1159` (verified 2026-04-29) — Same pattern.
- `scripts/foundation/tests/test_baseline_subprocess_determinism.py:94-105` (verified 2026-04-29) — Subprocess test pattern for orchestrator validation.
- `scripts/foundation/tests/test_manifest.py:181-221` (verified 2026-04-29) — `test_run_manifest_backward_compat_v1` precedent for v3 backward-compat test.
- `results/federated/baseline/20260429-082522-984e98/results.json` (verified 2026-04-29) — Real on-disk schema confirmation: `final_metrics.best.sampled_ndcg@10/sparse` slash delimiter, `_manifest.schema_version=2`.
- `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py` (verified 2026-04-29) — Pattern reference for orchestrator.
- `.planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md` (verified 2026-04-29) — D-01..D-09 schema consumed by aggregator.
- `.planning/REQUIREMENTS.md` (verified 2026-04-29) — THS-01..THS-07 requirements.
- `.planning/PROJECT.md` (verified 2026-04-29) — Core Value statement, constraints.
- `.planning/STATE.md` (verified 2026-04-29) — Phase 6 closeout state, 278 GREEN tests baseline.

### Secondary (MEDIUM confidence)
- pandas 2.3.3 (installed) — version verified by direct import; CSV write API stable.
- numpy 2.2.6 (installed) — `np.std(arr, ddof=0)` semantics for population std (matches D-24 "± std" convention with `ddof=0`; if `ddof=1` is preferred per Bessel's correction, document choice in aggregator).

### Tertiary (LOW confidence)
- None — every claim in this research is verified against existing on-disk code or schema.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library is pre-installed and validated through Phases 1-6.
- Architecture: HIGH — patterns mirror existing precedents (Phase 1 mode profile, Phase 6 manifest schema bump, Phase 4 subprocess determinism test).
- Pitfalls: HIGH — Pitfalls 1, 3, 4, 5, 7 surface from concrete code reading; the rest are derived from existing test invariants.

**Research date:** 2026-04-29
**Valid until:** 2026-05-29 (30 days; Phase 7 is mostly orchestration on a stable foundation, low drift risk)
