# Phase 7: Thesis Evaluation Run - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the standardized cross-device thesis comparison + adaptive-method ablations across all four federated modules and export the thesis tables (main comparison, ablations, sparse-user slice) to `results/federated/_thesis/`.

This is the experiment phase that produces the actual thesis-contribution headline numbers. Phases 1–6 were the methodological scaffolding (cross-device migration + correct evaluation harness). Phase 7 is the experiment.

**In scope:**
- A new mode profile `thesis_crossdevice_main` for the main comparison config.
- Run orchestration (`scripts/thesis/run_thesis_sweep.py`) firing `flwr run` per cell.
- Result aggregation (`scripts/thesis/aggregate_results.py`) producing markdown + CSV from per-run JSON.
- Manifest schema extension to carry thesis-run metadata for aggregator filtering.
- One-factor ablations from the main config across the adaptive method's knobs.

**Out of scope (other phases / future work):**
- Re-tuning per-module hyperparameters (use the validated `benchmark_cross_device` profile as-is).
- Running PFedRec at non-paper-compat hyperparams (Phase 5 D-05 forbids this).
- DP / privacy quantification (PROJECT.md "Out of Scope").
- Datasets other than ML-1M.
- Two-stage or full-Cartesian ablations (one-factor only this cycle).

</domain>

<decisions>
## Implementation Decisions

### Standardized comparison config (THS-01)

- **D-01:** Use the existing `benchmark_cross_device` profile values verbatim for the thesis main comparison: `embedding_dim=64`, `optimizer="adam"`, `lr=0.001`, `local_epochs=1`, `num_server_rounds=100`, `fraction_train=0.1`, `num_train_negatives=4`, `weight_policy="num_positives"`, `primary_evaluator="sampled_loo_99"`. Zero churn — these have been validated through Phases 2–6 on baseline / personalized / adaptive.

- **D-02:** Adaptive's main-comparison config = `model-type=dual` + `alpha-method=hierarchical_conditional` ONLY. The "next-gen" knobs (`enable-per-user-alpha`, `enable-item-perturbation`, `contrastive-lambda`) are **OFF** in the main table. They are ablation knobs only. This makes "the adaptive method" specifically the dual-level + HC-alpha mechanism in the headline number — clean attribution.

- **D-03:** FedAvg only for the main comparison (no FedProx). The thesis claim is about personalization mechanism, not aggregation strategy. FedProx is an ablation knob if the planner wants to surface it (currently not in the THS-05 ablation list).

- **D-04:** Add a new mode profile `thesis_crossdevice_main` to `scripts/foundation/fedrec_foundation/mode.py`. Values clone `benchmark_cross_device` verbatim. Provenance tag: runs labeled `thesis_crossdevice_main` are the thesis comparison; runs labeled `benchmark_cross_device` are exploratory / non-thesis. Implementation pattern mirrors the existing `_BENCHMARK_CROSS_DEVICE` / `_PAPER_COMPAT_PFEDREC` `ModeProfile` instances.

### PFedRec role in the comparison (THS-03, THS-04)

- **D-05:** PFedRec is a **calibration reference**, NOT counted toward "adaptive beats baselines". The thesis claim (THS-03 overall NDCG@10, THS-04 sparse NDCG@10) requires adaptive to beat **baseline + personalized only** — both of which run at `thesis_crossdevice_main`. PFedRec runs at its paper-faithful config (`paper_compat_pfedrec`) for IJCAI-23 reproduction. Apples-to-apples = same config.

- **D-06:** PFedRec runs **only** at `paper_compat_pfedrec`. Do not run PFedRec at `thesis_crossdevice_main`. Phase 5 D-05 ("PFedRec at non-PFedRec hyperparams is philosophically incoherent") is honored. No extra appendix row.

- **D-07:** PFedRec uses 3 seeds for THS-02 minimum (~10 hours wallclock at ~3 hr/run on RTX 5090 per the IJCAI-23 reference logged time). Same seed set as the other modules (`{42, 1337, 2026}`).

- **D-08:** PFedRec is reported as a separate **footnoted row** in the same `_thesis/main_comparison.{md,csv}` table. Footnote text: "`PFedRec (paper-faithful)†` — `dim=32, SGD lr=0.1, BCE, fraction-train=1.0; matches IJCAI-23 reference within ±2 points`". One table, distinct row, distinct config note.

### Seeds & statistical comparison (THS-02, THS-03, THS-04)

- **D-09:** **3 seeds** for the main comparison across all four modules. Per-module wallclock estimate at fraction_train=0.1 / 100 rounds: baseline ≈ 1 hr/run, personalized ≈ 1 hr/run, adaptive ≈ 1.5 hr/run. Main: 3 modules × 3 seeds × ~1.2 hr ≈ **~10.5 hr**. Plus PFedRec 3 seeds × ~3 hr ≈ **~9 hr**. Main + PFedRec ≈ ~19.5 hr.

- **D-10:** **Seeds = `{42, 1337, 2026}`**. Fixed canonical set used by all four modules. Same seeds across modules so the LOO splits and 99-negative pools are byte-identical seed-by-seed (modulo the per-purpose RNG derivation from Phase 1 D-14). Documentable in the thesis appendix.

- **D-11:** **Win criterion** = adaptive's mean NDCG@10 strictly greater AND non-overlapping ±1σ intervals vs every baseline (baseline + personalized). Specifically: `adaptive.mean - adaptive.std > baseline.mean + baseline.std` AND same vs personalized. Defensible and visually obvious in tables. Survives the "lucky seed" criticism without invoking formal hypothesis testing (which has low power at 3 seeds anyway).

- **D-12:** **Contingency on failure to win:** Document the negative result honestly + run the ablations (D-13..D-16) as **recovery runs** to find which adaptive variant DOES beat baselines. If a variant wins, restructure the thesis claim around that variant. If still no win after ablation recovery, escalate to thesis-level replanning per PROJECT.md core value ("If the adaptive method does not win under the corrected protocol, the thesis contribution has to be rethought"). Methodological correctness is non-negotiable.

### Ablation scope (THS-05, THS-06)

- **D-13:** **One-factor-at-a-time ablation from main config.** Main = adaptive at `thesis_crossdevice_main` with `model-type=dual` + `alpha-method=hierarchical_conditional` + all next-gen knobs OFF + `fusion-type=concat`. Ablation cells flip exactly one knob:
    - Alpha method: `multi_factor`, `data_quantity` (2 cells)
    - Per-user alpha: `enable-per-user-alpha=true` (1 cell)
    - Item perturbation: `enable-item-perturbation=true` (1 cell, with `item-perturbation-reg=0.01` default)
    - Contrastive λ: `contrastive-lambda=0.1` (with `contrastive-tau=0.1` default) (1 cell)
    - Fusion type: `add`, `gate` (2 cells)
  Total: **7 ablation cells** (the main config itself is the reference / 8th implicit row, shared with the main comparison run).

- **D-14:** **3 seeds for all ablation cells** (same seed set `{42, 1337, 2026}`). 7 cells × 3 seeds ≈ 21 ablation runs at ~1.5 hr/run ≈ **~31.5 hr**.

- **D-15:** **Ablation table columns: Overall NDCG@10 + Sparse NDCG@10 only** (plus matching HR@10 columns) in `_thesis/ablations.{md,csv}`. Medium and dense per-group metrics are available in the per-run JSON artifacts for any reviewer who wants to inspect; not in the main ablation table to avoid 4× width and hiding the sparse story (which is the thesis claim per THS-04).

- **D-16:** **Run sequence:** main runs first; ablations after main pass. Reasoning: (a) if adaptive wins the main comparison cleanly, ablations are a "where does the win come from" analysis; (b) if adaptive loses, ablations become recovery runs and the planned matrix gives clean answers without re-thinking the cell list. Either way, main-first is the correct ordering.

### Export pipeline (THS-07)

- **D-17:** **Export formats: Markdown + CSV.** No LaTeX, no aggregate JSON. Output paths under `results/federated/_thesis/`:
    - `main_comparison.md` + `main_comparison.csv` — baseline / personalized / adaptive rows + footnoted PFedRec row.
    - `ablations.md` + `ablations.csv` — one row per ablation cell + main config row at top as reference.
    - `sparse_slice.md` + `sparse_slice.csv` — sparse-user-only NDCG@10 / HR@10 across all rows from the main comparison + ablations (the THS-04 thesis-claim view).

- **D-18:** **Orchestrator: Python script `scripts/thesis/run_thesis_sweep.py`.** Defines the matrix as data, fires `flwr run` per cell with the right `--run-config` flags, captures stdout/stderr, logs which cells succeeded vs failed. Re-runnable to fill gaps (skips cells whose result.json already exists for the (module, seed, knobs) tuple). Mirrors the structural pattern of `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py`.

- **D-19:** **Aggregator: standalone Python script `scripts/thesis/aggregate_results.py`.** Reads `results/federated/<module>/<run_id>/results.json` for every run that matches the thesis filter (`_manifest.thesis_run_label` is set AND `_manifest.run_seed ∈ {42, 1337, 2026}`). Produces all four output files (main_comparison.{md,csv}, ablations.{md,csv}, sparse_slice.{md,csv}) from one source. Idempotent: re-running rebuilds tables from disk without re-running any flwr cell.

- **D-20:** **Aggregator missing-cell handling: hard fail with explicit list.** Aggregator computes the expected cell set ({modules} × {seeds} × {knobs}) and verifies one result.json per expected (module, seed, thesis_run_label) tuple. On any miss: error with `"Missing N cells: [list]. Run them then re-aggregate."` — no partial tables.

### Operational details

- **D-21:** **W&B project = `federated-cf-cross-device`** (same project as main). Distinguishable by run name pattern:
    - Main runs: `thesis-main-<module>-seed<N>` (e.g., `thesis-main-adaptive-seed42`).
    - Ablation runs: `thesis-ablation-<module>-seed<N>-<knob>=<value>` (e.g., `thesis-ablation-adaptive-seed42-fusion=add`).
    - PFedRec runs: `thesis-main-pfedrec-seed<N>` (uses paper_compat_pfedrec mode but tagged `thesis_run_label=main`).
  Phase 6 D-05 zero-churn pattern preserved.

- **D-22:** **Manifest schema extension.** Bump `RUN_MANIFEST_SCHEMA_VERSION` 2→3 in `scripts/foundation/fedrec_foundation/manifest.py`. Add fields to `RunManifest`:
    - `thesis_run_label: str = ""` — `"main"` for main-comparison runs, `"ablation_<knob>=<value>"` for ablation runs (e.g., `"ablation_fusion=add"`), or `""` for non-thesis runs.
    - `ablation_dimension: str = "none"` — one of `{"none", "alpha_method", "per_user_alpha", "item_perturbation", "contrastive_lambda", "fusion_type"}`. `"none"` for main runs.
    - `ablation_value: str = ""` — specific value of the ablated knob (e.g., `"add"` when `ablation_dimension="fusion_type"`). Empty for main runs.
  These fields are aggregator-filterable and become an audit trail of which cell each run filled. Default values keep v1/v2 manifests backward-readable.

- **D-23:** **Cell failure handling: skip + log + retry at end.** When a single `flwr run` cell crashes (CUDA OOM, network hiccup, anything), the orchestrator:
    1. Catches the failure, logs the (module, seed, cell-knobs) tuple to `results/federated/_thesis/failed_cells.json` with stderr excerpt.
    2. Continues to the next cell (sweep doesn't block).
    3. At the end, emits a summary listing all failed cells and prints `python scripts/thesis/run_thesis_sweep.py --retry-failed` as the recovery command.
  Aggregator's hard-fail-on-missing (D-20) is the safety net if the user forgets to retry.

- **D-24:** **Table cell format: `0.4123 ± 0.0089`** (mean and std, one line, 4 decimal places for both). Markdown cells: `| 0.4123 ± 0.0089 |`. CSV cells: two columns per metric (`ndcg10_mean`, `ndcg10_std`). The bold-the-winner / asterisk-the-significant decoration is left to Claude's Discretion at planning time.

### Claude's Discretion

The following were not explicitly discussed; planner may decide at planning time within reasonable principles:

- **Bold-the-winner styling in markdown tables.** Reasonable default: bold the cell whose row "beats" all comparable rows under the D-11 win criterion. Per-module note in the markdown body if "beats" needs disambiguation.

- **Sparse-user slice fill behavior** when a seed produces zero evaluable sparse interactions (extremely rare given 6040 users but theoretically possible). Reasonable default: emit the sparse-row from the seeds that DID have evaluable sparse interactions, with a count footnote `n_seeds_with_sparse=2/3`.

- **Wandb-summary key naming for thesis runs.** Phase 6 already wired `best/*` and `last/*` namespaces. Recommended: keep them; add a top-level `thesis/run_label` summary field mirroring the manifest's `thesis_run_label` for dashboard filtering.

- **Intermediate result review checkpoints.** Whether the orchestrator emits the partial markdown table after every cell completes (so a long sweep can be reviewed mid-run) vs only at the end. Reasonable default: emit progress JSON to `_thesis/_progress.json` every cell; full tables only at the end via the aggregator.

- **Significance markers (asterisks, color).** Default: no special markers in markdown — bold-the-winner is enough. Add asterisks only if reviewers ask.

- **`_thesis/` directory creation handling.** First sweep run creates the directory; subsequent runs no-op. Atomic write per file via existing `atomic_write_json` pattern (extended to `atomic_write_text` for markdown if not yet present).

- **Compute parallelism.** Each `flwr run .` is one process; the GPU is shared. Whether to run cells serially (safer) vs interleaved (faster but risks GPU contention) is a planner call. Default: serial within a module, between-module serial too (one giant queue).

- **Retry semantics for `--retry-failed` flag.** Whether `--retry-failed` re-runs ALL cells in failed_cells.json or just the ones whose result.json still doesn't exist on disk. Default: filter by disk presence (idempotent).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap

- `.planning/REQUIREMENTS.md` §THS-01..THS-07 — Seven requirements this phase delivers.
- `.planning/ROADMAP.md` §"Phase 7: Thesis Evaluation Run" — Goal, dependencies (Phase 6), success criteria.
- `.planning/PROJECT.md` §"Core Value" — The thesis claim that THS-03/04 must validate. §"Active" requirements list confirms which Phase-1-through-6 deliverables this phase consumes.

### Foundation contract (consumed + extended)

- `scripts/foundation/fedrec_foundation/mode.py` — `_BENCHMARK_CROSS_DEVICE` and `_PAPER_COMPAT_PFEDREC` profiles to clone for the new `thesis_crossdevice_main` profile (D-04). Resolution: `resolve_mode_defaults("thesis_crossdevice_main")` returns the locked profile.
- `scripts/foundation/fedrec_foundation/manifest.py` — `RunManifest` dataclass to extend (D-22 schema bump v2→v3 with thesis-run metadata fields).
- `scripts/foundation/fedrec_foundation/paths.py` — `module_run_results_dir(module, run_id)` already used by every server_app per Phase 6 D-02; aggregator reads from this layout.

### Phase 6 outputs (consumed by aggregator)

- `.planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md` §D-01..D-09 — Per-run-dir layout, nested `final_metrics = {best, last, best_round, last_round, final_eval_round_index}`, per-group keys (sparse/medium/dense), exposure counts. Aggregator filters by `_manifest.thesis_run_label` and reads `final_metrics["best"]["sampled_ndcg@10"]` plus per-group variants.
- `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-{03,04,05,06}-SUMMARY.md` — Per-module wiring details if the planner needs to confirm key formats (note: PFedRec uses slash delimiter `evaluated_users/sparse`; others use underscore — documented in Plan 06 SUMMARY).

### Phase 5 PFedRec calibration

- `.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md` §D-05 — PFedRec ships only at paper_compat_pfedrec; Phase 7 honors this (D-06).
- `.planning/phases/05-pfedrec-migration-reproduction/` SUMMARY files — Reproduction targets HR@10 ≈ 0.729 ± 0.02, NDCG@10 ≈ 0.441 ± 0.02 (D-08 footnote).

### Adaptive module reference

- `federated-adaptive-personalized-cf/claude.md` — Documents `model-type`, `alpha-method`, `enable-per-user-alpha`, `enable-item-perturbation`, `contrastive-lambda`, `fusion-type` flags and their defaults. Confirms what's ablated.
- `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py` — Pattern reference for the Phase-7 orchestrator script (D-18).
- `federated-adaptive-personalized-cf/sweep.yaml` — W&B sweep config (Pitfall-7 closed in Phase 6 — `metric.name = best/sampled_ndcg@10`). Phase 7 orchestrator does NOT use W&B sweeps (uses targeted matrix execution per D-18).

### Project root scripts (pattern references)

- `scripts/run_baseline_sweep_loo.sh` — Existing bash sweep pattern (NOT used by Phase 7 — D-18 says Python).
- `scripts/run_all_baselines.sh` — Existing orchestration pattern (NOT used by Phase 7).
- `scripts/compare_all_results.py` — Existing aggregation pattern (planner may inspect for parsing conventions, but D-19 is a fresh script, not an extension).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`module_run_results_dir(module, run_id)`** (Phase 6 D-02) — Aggregator reads from `<repo>/results/federated/<module>/<run_id>/results.json`. No new path resolution code needed.
- **`RunManifest` + `embed_manifest_in_result` + `write_manifest_sibling`** — Manifest already double-written per Phase 6 D-02. Schema bump from v2→v3 adds three fields; existing readers must handle v2 manifests (default values).
- **`atomic_write_json`** at `scripts/foundation/fedrec_foundation/atomic.py` — Aggregator uses for safe writes. May need `atomic_write_text` companion for markdown (planner adds if not present).
- **`final_metrics["best"]`** schema (Phase 6 D-07) — Already carries `sampled_ndcg@10`, `sampled_hr@10`, plus per-group variants `sampled_ndcg@10/sparse`, `sampled_ndcg@10/medium`, `sampled_ndcg@10/dense` (or `_` underscore-delimited for the 3 non-pfedrec modules). Aggregator reads from this nested block.
- **`run_wandb_sweep.py` structural pattern** — Matrix-as-dict, `subprocess.run(["flwr", "run", ".", "--run-config", ...])` invocation, stdout capture. Phase 7 orchestrator mirrors structure but is matrix-driven (deterministic cell list), not Bayesian-driven.

### Established Patterns

- **Mode profile addition** — Pattern: define `_THESIS_CROSSDEVICE_MAIN = ModeProfile(...)`, register in `_REGISTRY`. Mirrors `_BENCHMARK_CROSS_DEVICE` and `_PAPER_COMPAT_PFEDREC`. Tests: extend `scripts/foundation/tests/test_mode.py` with a new mode-resolution test.
- **Manifest schema bump** — Pattern from Phase 6 D-07: add fields with safe defaults to `RunManifest`, increment `RUN_MANIFEST_SCHEMA_VERSION`, ensure backward-readable. Tests: extend `scripts/foundation/tests/test_manifest.py` with v3 invariants.
- **Per-group key delimiter divergence** — Phase 6 confirmed: baseline / personalized / adaptive use underscore (`evaluated_users_sparse`); pfedrec uses slash (`evaluated_users/sparse`). Aggregator must handle both.
- **D-09 exposure-count fields** (Phase 6) — `evaluated_users`, `evaluated_users_sparse`, `evaluated_users_medium`, `evaluated_users_dense` per round AND in canonical block. Aggregator reads canonical for the table; per-round version is for diagnostic checks.

### Integration Points

- **`scripts/thesis/`** — NEW directory. Contains `run_thesis_sweep.py` + `aggregate_results.py`. Add an `__init__.py` if any cross-script imports are needed.
- **`results/federated/_thesis/`** — NEW directory. Created by aggregator on first run. Contains 6 files (3 markdown + 3 CSV) + optional `_progress.json` + `failed_cells.json`.
- **`scripts/foundation/fedrec_foundation/mode.py`** — Extend `_REGISTRY` with `"thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN`.
- **`scripts/foundation/fedrec_foundation/manifest.py`** — Extend `RunManifest` with three thesis fields. Bump `RUN_MANIFEST_SCHEMA_VERSION = 3`.
- **Per-module `pyproject.toml`** — Each module needs to know that `mode = "thesis_crossdevice_main"` is a valid choice. The mode-string value flows into the existing `mode` config key; no new code path in any `server_app.py` (the mode profile machinery is shared via `resolve_mode_defaults`).

</code_context>

<specifics>
## Specific Ideas

- **"Mean ± std" cell format** is the de facto thesis convention in the FedRec literature (echoed in PFedRec's own IJCAI-23 reporting). Stick to that.
- **Seed set `{42, 1337, 2026}`** is chosen for documentability (XKCD / leet / year). All four modules use the same set so a reviewer can mentally pair runs across modules.
- **"Adaptive method" in the main comparison table = `dual + hierarchical_conditional` only** — clean attribution. The maximalist variant (with per-user alpha + perturbation + contrastive) is a specific ablation cell, not the headline number. This is the cleanest framing of the thesis claim: "the dual-level statistical-personalization mechanism beats the baselines; the additional knobs help further (or not) per the ablation table."
- **PFedRec footnote text:** "`† dim=32, SGD lr=0.1, BCE, fraction-train=1.0; matches IJCAI-23 reference within ±2 points.`"
- **Naming convention for runs:** `thesis-main-<module>-seed<N>` and `thesis-ablation-<module>-seed<N>-<knob>=<value>`. The dash-delimited form is W&B-friendly and human-readable. The `<knob>=<value>` segment is the same string used in the manifest's `thesis_run_label` field, modulo the `ablation_` prefix.

</specifics>

<deferred>
## Deferred Ideas

- **Two-stage ablation** (pick best alpha first, then ablate other knobs against the winner) — captured during discussion as a richer alternative to one-factor; deferred to a future ablation phase if Phase 7 results suggest interactions worth exploring.
- **Full Cartesian ablation matrix (72 cells × 3 seeds = 216 runs)** — captured; deferred. Out of scope for thesis deadline. Would belong to a "Phase 7.1: Deep Ablation" or post-thesis publication round.
- **PFedRec at non-PFedRec hyperparams as an extra row** — explicitly declined per Phase 5 D-05. Captured here so future phases know it was considered and rejected.
- **5 seeds (vs 3)** — captured as an option; declined for thesis budget. Could be a Phase 7.1 follow-up if reviewers ask for tighter confidence intervals.
- **LaTeX export format** — captured as an option; declined. Markdown + CSV are sufficient; thesis writer manually reformats markdown to LaTeX if needed.
- **JSON aggregate export (`_thesis/aggregated.json`)** — captured as an option; declined. Per-run `results.json` files ARE the aggregate JSON; aggregator reads them directly.
- **W&B Sweeps via `sweep.yaml`** — captured as an option; declined. `sweep.yaml` is for Bayesian hyperparameter exploration, not deterministic matrix execution.
- **Auto-retry on cell failure with exponential backoff** — captured; declined. Skip + log + manual retry-at-end is more transparent for thesis-pipeline debugging.
- **Stop-the-sweep-on-first-failure** — captured; declined. Worst budget-wise on transient crashes.
- **Per-user-group medium / dense columns in main ablation table** — captured; declined for table width reasons. Available in per-run JSON.
- **DP / privacy quantification** — out of scope per PROJECT.md. Future-work footnote in the thesis.

### Reviewed Todos (not folded)

None — `gsd-tools todo match-phase 7` returned 0 matches.

</deferred>

---

*Phase: 07-thesis-evaluation-run*
*Context gathered: 2026-04-29*
