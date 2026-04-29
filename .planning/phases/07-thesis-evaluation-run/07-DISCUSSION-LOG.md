# Phase 7: Thesis Evaluation Run - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-29
**Phase:** 07-thesis-evaluation-run
**Areas discussed:** Standardized config (THS-01), PFedRec role, Seeds & statistical comparison, Ablation scope, Export pipeline, Operational details

---

## Area 1: Standardized config (THS-01)

### Q1.1: Use existing `benchmark_cross_device` profile, or override?

| Option | Description | Selected |
|--------|-------------|----------|
| Use benchmark_cross_device as-is | dim=64, Adam lr=0.001, 100 rounds, 1 local epoch, fraction_train=0.1, num_train_negatives=4. Validated through Phases 2-6. Zero churn. | ✓ |
| Override num_server_rounds to 50 | Faster runs, ~half compute. Risk: under-convergence. | |
| Override fraction_train upward (0.2) | More clients/round, fewer rounds, similar wallclock. | |
| Override local_epochs (5) | More local training; risk of drift from baseline 1-epoch protocol. | |

**User's choice:** Use benchmark_cross_device as-is (Recommended).
**Notes:** Zero-churn principle; preserves Phases 2-6 validation work.

### Q1.2: Which adaptive next-gen features are ON in the main comparison config?

| Option | Description | Selected |
|--------|-------------|----------|
| model-type=dual + alpha-method=hierarchical_conditional | The thesis contribution. Non-negotiable for the main table. | ✓ |
| enable-per-user-alpha=true | Per-user learned alpha. ADP-02 fix landed Phase 4. | |
| enable-item-perturbation=true | Local item perturbation. Adds parameters; complicates attribution. | |
| contrastive-lambda=0.1 | InfoNCE auxiliary loss. Helps train per-user alpha. | |

**User's choice:** Only `dual + hierarchical_conditional` — others are ablation knobs.
**Notes:** Clean attribution. "Adaptive method" = dual + HC-alpha mechanism. Other knobs ablated separately.

### Q1.3: FedAvg or FedProx for the main comparison?

| Option | Description | Selected |
|--------|-------------|----------|
| FedAvg only | Cleanest: thesis is about personalization, not aggregation strategy. | ✓ |
| FedProx mu=0.01 | More stable under non-IID; adds a hyperparameter to defend. | |
| Run both per-module | Comprehensive but doubles compute. | |

**User's choice:** FedAvg only.

### Q1.4: Where does the canonical thesis config live?

| Option | Description | Selected |
|--------|-------------|----------|
| Add `thesis_crossdevice_main` mode profile to mode.py | Same pattern as benchmark_cross_device. One source of truth. | ✓ |
| Reuse benchmark_cross_device unchanged | Zero new code; "thesis config" is a label not a mode. | |
| YAML config under .planning/phases/07/configs/ | Explicit but introduces new config-load path. | |

**User's choice:** Add new mode profile `thesis_crossdevice_main` to mode.py.

---

## Area 2: PFedRec's role in the main table

### Q2.1: Does "adaptive beats all three baselines" include PFedRec?

| Option | Description | Selected |
|--------|-------------|----------|
| PFedRec is calibration reference, NOT counted | Apples-to-apples = same config; PFedRec at paper-compat is footnoted. | ✓ |
| PFedRec counted as a baseline | Stricter claim; risk: PFedRec-tuned-for-ML-1M may genuinely beat dim=64 Adam BPR. | |
| Two separate tables | Table 1 comparable methods; Table 2 published SOTA reference. | |

**User's choice:** PFedRec is calibration reference, NOT counted toward "beats baselines".
**Notes:** Adaptive must beat baseline + personalized only on NDCG@10.

### Q2.2: Run PFedRec at thesis_crossdevice_main config as an extra row?

| Option | Description | Selected |
|--------|-------------|----------|
| No — keeps Phase 5 D-05 clean | "PFedRec at non-PFedRec hyperparams is incoherent." | ✓ |
| Yes, third row in main table | Closes apples-to-apples gap; risk: muddy table. | |
| Yes, in appendix only | Compromise; available if reviewers ask. | |

**User's choice:** No.

### Q2.3: PFedRec seed budget?

| Option | Description | Selected |
|--------|-------------|----------|
| 3 seeds (~10 hours) | Minimum for THS-02. | ✓ |
| 1 run for PFedRec | Treat as one-shot reproduction; saves ~7 hours. | |
| 5 seeds (~15 hours) | Symmetric across modules. | |

**User's choice:** 3 seeds (~10 hours).

### Q2.4: Where does PFedRec's row live in the export?

| Option | Description | Selected |
|--------|-------------|----------|
| Same `_thesis/` markdown table, separate row, footnoted | One table, distinct row, distinct config note. | ✓ |
| Separate appendix table | Avoids muddying main comparison. | |
| Both: main + appendix | Repetition. | |

**User's choice:** Same table, separate row, footnoted.

---

## Area 3: Seeds & statistical comparison (THS-02, THS-03, THS-04)

### Q3.1: Number of seeds for the main comparison?

| Option | Description | Selected |
|--------|-------------|----------|
| 3 seeds | Meets THS-02 minimum. ~10.5 hr main runs. | ✓ |
| 5 seeds | Tighter CIs; ~17.5 hr main. | |
| 5 for adaptive only, 3 for others | Asymmetric pragmatic. | |

**User's choice:** 3 seeds.

### Q3.2: Which seeds?

| Option | Description | Selected |
|--------|-------------|----------|
| {42, 1337, 2026} | XKCD / leet / year. Documentable. | ✓ |
| {42, 1337, 2026, 7777, 12345} (if 5 seeds) | Fixed extension. | |
| Random per session | Manifest records them but reviewers may ask. | |

**User's choice:** `{42, 1337, 2026}`.

### Q3.3: Win criterion for "adaptive beats baselines"?

| Option | Description | Selected |
|--------|-------------|----------|
| Mean strictly greater AND non-overlapping ±1σ | Defensible; visually obvious in tables. | ✓ |
| Paired t-test, p < 0.05 | Low power at 3-5 seeds; risk of false negatives. | |
| Wilcoxon signed-rank, p < 0.05 | Non-parametric; same low-power issue. | |
| Mean strictly greater (no σ check) | Tiny margins don't credibly support thesis. | |

**User's choice:** Mean strictly greater AND non-overlapping ±1σ.

### Q3.4: Contingency if adaptive doesn't win?

| Option | Description | Selected |
|--------|-------------|----------|
| Document negative result + ablations as recovery | Honest; saves thesis work via variant exploration. | ✓ |
| Wider hyperparameter sweep on adaptive | Maybe undertrained; risk: more compute, deadline pressure. | |
| Lower bar to sparse-only (THS-04 only) | Thesis pivots to "especially helps sparse". | |
| Stop and replan thesis | If even ablation recovery fails. | |

**User's choice:** Document negative result + ablations as recovery.
**Notes:** Per PROJECT.md "Methodological correctness is non-negotiable." Escalate to thesis-level replanning only if ablation recovery also fails.

---

## Area 4: Ablation scope (THS-05, THS-06)

### Q4.1: Ablation strategy?

| Option | Description | Selected |
|--------|-------------|----------|
| One-factor-at-a-time from main config | ~7-12 cells × 3 seeds = ~21-36 runs (~30-54 hr). Direct attribution. | ✓ |
| Two-stage: pick best alpha, then ablate others | ~81 runs (~120 hr). Captures interactions in post-alpha cube. | |
| Full Cartesian (216 runs, ~13 days) | Maximally defensible; blows deadline. | |
| Hand-curated subset (~6 cells) | Smallest defensible set. | |

**User's choice:** One-factor-at-a-time from main config.

### Q4.2: Seeds per ablation cell?

| Option | Description | Selected |
|--------|-------------|----------|
| 3 seeds for all ablation cells | Consistency with main; cleaner narrative. | ✓ |
| 1 seed for ablations, 3 for main | Saves 2/3 of ablation compute. | |
| 2 seeds for ablations | Compromise. | |

**User's choice:** 3 seeds for all ablation cells.

### Q4.3: Per-group columns in ablation table?

| Option | Description | Selected |
|--------|-------------|----------|
| Overall + Sparse only | Thesis hinges on overall + sparse. Cleaner narrative. | ✓ |
| All four (Overall, sparse, medium, dense) | Full transparency; 4× wider table. | |
| Two tables: main + detailed | Best of both; same data different views. | |

**User's choice:** Overall + Sparse only. Medium / dense available in per-run JSON.

### Q4.4: Run sequence — main vs ablation?

| Option | Description | Selected |
|--------|-------------|----------|
| Main first, ablations after | Ablation interpretation depends on main result. | ✓ |
| Parallel | Max GPU utilization; risk of wasted compute on doomed cells. | |
| Interleaved per-module | Cache-friendly; worse for incremental review. | |

**User's choice:** Main first, ablations after.

---

## Area 5: Export pipeline (THS-07)

### Q5.1: Export formats?

| Option | Description | Selected |
|--------|-------------|----------|
| Markdown | Mandatory per THS-07. Multi-platform. | ✓ |
| CSV | Spreadsheet-friendly. | ✓ |
| LaTeX tabular | Direct paste into thesis. | |
| JSON aggregate | Raw machine-readable. | |

**User's choice:** Markdown + CSV.

### Q5.2: Orchestrator structure?

| Option | Description | Selected |
|--------|-------------|----------|
| Python script `scripts/thesis/run_thesis_sweep.py` | Mirrors run_wandb_sweep.py. Re-runnable to fill gaps. | ✓ |
| Bash script | Mirrors existing scripts/run_baseline_sweep_loo.sh; worse retry. | |
| W&B Sweeps via sweep.yaml | Best for Bayesian; less ideal for deterministic matrix. | |
| Manual flwr run | No orchestration; high human-error risk. | |

**User's choice:** Python script.

### Q5.3: Aggregator location?

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone Python `scripts/thesis/aggregate_results.py` | Idempotent: rebuilds from disk anytime. | ✓ |
| Inline in orchestrator | Tighter coupling; loses anytime-rebuild property. | |
| Jupyter notebook | Better exploration; worse reproducibility. | |

**User's choice:** Standalone Python script.

### Q5.4: Missing-cell handling?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard fail with explicit list | No partial tables; safety net for forgotten retries. | ✓ |
| Soft warn + emit table with MISSING markers | Risk of accidentally shipping broken tables. | |
| Auto-rerun missing cells | Tighter pipeline but couples concerns. | |

**User's choice:** Hard fail with explicit list.

---

## Area 6: Operational details

### Q6.1: W&B project for thesis runs?

| Option | Description | Selected |
|--------|-------------|----------|
| Same `federated-cf-cross-device`, distinguishable by run name | Phase 6 D-05 zero-churn pattern. | ✓ |
| New `federated-cf-thesis` project | Cleaner separation; one extra project. | |
| Two projects: thesis-main + thesis-ablations | Maximally separated; overkill. | |

**User's choice:** Same project; encode cell in run name.

### Q6.2: Manifest fields for aggregator filtering?

| Option | Description | Selected |
|--------|-------------|----------|
| thesis_run_label ∈ {main, ablation_<knob>=<value>} | Single canonical label per cell. | ✓ |
| ablation_dimension ∈ {none, alpha_method, ...} | Lets aggregator filter by dimension. | ✓ |
| ablation_value | Specific value of ablated knob. | |
| thesis_seeds_set | Documents which seed set this run is from. | |

**User's choice:** thesis_run_label + ablation_dimension. (ablation_value implicitly captured in thesis_run_label string; CONTEXT.md added it explicitly as D-22 third field for aggregator simplicity.)

### Q6.3: Cell-failure handling during sweep?

| Option | Description | Selected |
|--------|-------------|----------|
| Skip + log to failed_cells.json + retry at end | Transparent; sweep doesn't block on transient crashes. | ✓ |
| Auto-retry with exponential backoff (3x) | Tighter recovery; wastes compute on deterministic bugs. | |
| Stop sweep on first failure | Conservative; worst budget-wise on transient crashes. | |

**User's choice:** Skip + log + retry at end.

### Q6.4: Markdown table cell format?

| Option | Description | Selected |
|--------|-------------|----------|
| `0.4123 ± 0.0089` | Mean and std on one line. ML-paper standard. | ✓ |
| `0.4123 (0.0089)` mean (std) | Compact; less explicit. | |
| Two columns: mean | std | Easier to sort; wider. | |
| `0.4123 [0.404, 0.421]` mean [min, max] | Range; non-standard. | |

**User's choice:** `0.4123 ± 0.0089`.

---

## Claude's Discretion

Areas where the user did not lock a specific decision; Claude is free to decide at planning time within reasonable principles:

- Bold-the-winner styling in markdown tables
- Sparse-user slice fill behavior on zero-evaluable-sparse seeds
- Wandb-summary key naming for thesis runs (top-level `thesis/run_label` field suggested)
- Intermediate result review checkpoints (suggested: `_thesis/_progress.json` per cell)
- Significance markers (asterisks, color)
- `_thesis/` directory creation handling (atomic write)
- Compute parallelism within / between modules (default: serial)
- `--retry-failed` semantics (default: filter by disk presence)

## Deferred Ideas

- Two-stage ablation (pick best alpha first, then ablate)
- Full Cartesian ablation matrix (216 runs)
- PFedRec at non-PFedRec hyperparams as an extra row
- 5 seeds (vs 3)
- LaTeX export format
- JSON aggregate export
- W&B Sweeps via sweep.yaml
- Auto-retry on cell failure with exponential backoff
- Stop-the-sweep-on-first-failure
- Per-user-group medium / dense columns in main ablation table
- DP / privacy quantification (PROJECT.md out-of-scope)
