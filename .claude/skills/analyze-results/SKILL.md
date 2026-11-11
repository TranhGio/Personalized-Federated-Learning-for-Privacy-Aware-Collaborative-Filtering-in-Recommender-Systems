---
name: analyze-results
description: "Read, recompute and tabulate thesis experiment results: results.json / manifest.json / EVAL_VALIDITY.json of federated runs, offline re-scores from scripts/thesis/reeval.py (reeval_*.json, per-user CSVs in three layouts), paired statistics from scripts/thesis/gate_stats.py (stats_wilcoxon.json), and the rules for which numbers may be cited; also how to fill RESULT.md / DECISION.md templates and check pre-registered predictions. Use when results arrive (git pull after a run), when comparing modules or runs, when auditing numbers in a document, or when the user says phân tích kết quả, điền RESULT, so sánh các run / các module (theo nhóm, có kiểm định theo cặp), kiểm tra số. Not for launching runs (train-remote) or W&B sweeps (sweep)."
user-invocable: true
argument-hint: "[run-dir | document-to-audit | latest]"
allowed-tools: Bash, Read, Glob, Grep, Write, Edit
---

# Analyzing results in this repository

Results are only as good as the file they come from and the validity flags around them. This skill is the map of those files and the rules for using them.

## 1. Run-level files (`results/federated/<module>/<run_id>/`)

| File | Use | Keys that matter |
|---|---|---|
| `results.json` | the run's own numbers | `final_metrics.best.{sampled_ndcg@10, sampled_hr@10, evaluated_users, d06_cache_misses}` and `/sparse`, `/medium`, `/dense` suffixes; `final_metrics.last` when a round-100 block exists; `checkpoint.best_round`; `_manifest.*` |
| `manifest.json` | provenance | `run_id`, `git_commit`, `run_seed`, `fraction_train`, `checkpoint_rule`, `thesis_run_label`, `split_hash`, `exclusion_sha256`, `overrides` (effective values live in `overrides`, not top-level — the "manifest effective-value trap") |
| `EVAL_VALIDITY.json` | citability | `status` ∈ {`valid`, `invalid_cold_eval`, `diagnostics_only`}, `reason`, `effective_checkpoint_rule`; regenerate all with `python3 scripts/thesis/eval_validity.py` |
| `global_state_best.pt` / `_last.pt`, `.embedding_cache/<run_id>/` | heavy inputs | training machine only; not on the laptop |

Citable claim-table runs as of 2026-11: baseline `20260504-203751-1bf513` (BPR-MF), `20260725-093704-e81518` (has round-100), personalized `20260610-064423-f18e64`, pfedrec `20260608-071106-ef41ab` (no calibration, 0.3352) and `20260726-083121-cf72e0` (final calibration, 0.3447), adaptive `20260613-152226-869730`. Always re-read `EVAL_VALIDITY.json` instead of trusting this list.

## 2. Citation rules (hard)
1. `EVAL_VALIDITY.status == "valid"`.
2. `evaluated_users == 6040` (809 / 2322 / 2909 by group); a `--limit-users` re-score is a probe, never a number.
3. `d06_cache_misses == 0` for modules with per-user state (personalized, adaptive, pfedrec).
4. Never put an in-loop best next to a full-population number; never compare `best` of one run with `last` of another without saying so.
5. Write the source path + key next to every number you hand to a writer.

## 3. Offline re-score (`scripts/thesis/reeval.py`)
```
python scripts/thesis/reeval.py --run-dir <run> --module {personalized|pfedrec|adaptive} \
  --globals {best|last} --local-variant <v>... [--device cpu] [--limit-users N] [--no-json] [--no-per-user] [--out-dir DIR]
python scripts/thesis/reeval.py --self-test
```
- Variants (all modules): `cached` (the run's own heads; fail-closed on cache miss), `random`, `shuffled`, `zero` (not pfedrec), `calibrated`.
- PFedRec extras: `global_head`, `global_head_weighted`, `global_head_cold`, `alpha_grid` (21 α), `alpha_oracle_group`, `alpha_oracle_user`, `alpha_policy_hc`, `qu_grid` (private item table blend grid; withdrawn experiment), `qu_diag` (α = 0, clean vs leaky pools = the post-tuning re-score).
- Trust gate: `cached` must equal the run's `final_metrics.best` field by field (|Δ| < 1e-6, exact user counts) or the tool exits 3 and writes no variant. `--limit-users` makes the gate tolerant (probe mode). `qu_*` add their own `trust_checks` (finite tables, untouched rows bit-exact, d_u invariant, n_hit per pool, leaky monotonicity, α = 1 parity) and also exit 3 on failure.
- Outputs: `reeval_<variant>_<globals>.json` (`metrics`, per-group keys, `metrics_by_cell` for grids, `trust_gate`, `trust_checks`, `globals_sha256`) and `reeval_<variant>_<globals>_per_user.csv`. `qu_*` write to `--out-dir`; everything else to the run dir.
- Adapters exist for personalized and pfedrec; adaptive is a stub; baseline has none (its per-user numbers are not available offline yet).

## 4. Per-user CSV layouts
| Layout | Columns | Produced by |
|---|---|---|
| standard (9) | `pid,user_group,n_interactions,staleness_at_best,alpha_used,sampled_ndcg@10,sampled_hr@10,eval_loss,evaluated` | cached, random, shuffled, calibrated, global_head*, alpha_policy_hc |
| wide α grid | `pid,user_group,n_interactions,staleness_at_best,ndcg@a0.00…ndcg@a1.00,hr@a…,loss@a…` | alpha_grid (`a1.00` column equals cached per user) |
| Q_u long (15) | `pid,user_group,n_interactions,staleness_at_best,alpha_table,epochs,pool,n_touched_rows,n_touched_candidates,test_row_touched,mean_row_shift,sampled_ndcg@10,sampled_hr@10,sampled_mrr,eval_loss` | qu_grid, qu_diag (one row per user per cell) |

Means over the CSV must match the JSON `metrics` to ~1e-15; a mismatch means the wrong file pair.

## 5. Paired statistics (`scripts/thesis/gate_stats.py`, needs scipy)
```
python scripts/thesis/gate_stats.py --grid <long_csv> --baseline <run>/reeval_cached_best_per_user.csv \
  --grid-json <reeval_qu_*_best.json> [--contrast CELL_A CELL_B]... --out <stats_wilcoxon.json>
```
Per cell and per group: `mean_delta_ndcg`, `ci95_ndcg` (bootstrap 10 000, seed 42), `n_nonzero`, `p_one_sided` (cell > baseline, wilcox zeros, approx), `p_two_sided`, `p_pratt_one_sided`, `p_holm_one_sided` (over tested cells), `rank_biserial_r`; `contrasts.<A>__minus__<B>` with the same block; `verdict` applies only to the withdrawn gate and is ignored elsewhere; `reproduction_ok` must be true. For other paired comparisons, reuse `wilcoxon_block` / `paired_bootstrap_ci` from the module rather than re-implementing.

## 6. Filling a template (RESULT.md / DECISION.md)
1. Header from JSON: run date (log), code commit (`git log` of the results commit), README version, `globals_sha256`, `trust_gate.status` + `max_abs_delta`, `trust_checks` counters, `elapsed_sec`, aborted runs.
2. Tables: 4-decimal metrics as `NDCG / HR / MRR`, overall + three groups; baseline row from the cached CSV/JSON.
3. Statistics table straight from `stats_wilcoxon.json`; p < 1e-3 in scientific notation; `n≠0` column.
4. Prediction check: apply the registered rule literally; verdict ∈ {Đúng, Sai, Không phân biệt}; show the numbers that decided it.
5. Limitation sentence: tick the registered branch; never write a new one.
6. Keep the Python used for every table in `_workspace/<track>/NN_results-analyst_tables.md` so the audit can rerun it.

## 7. Comparison across modules (placement study)
Per-user dumps exist for pfedrec (cf72e0: cached + all head-blend variants, qu_diag) and personalized (f18e64 via reeval); baseline and adaptive need adapters first (tooling task). Until then, the cross-module table uses `final_metrics.best` with EVAL_VALIDITY, labelled "best-round, no paired test".

## 8. Aggregator
`python scripts/thesis/aggregate_results.py` builds `results/federated/_thesis/{main_comparison,ablations,sparse_slice}.{md,csv}` from thesis-tagged runs (`thesis_run_label != ""`, seeds {42, 1337, 2026}); hard-fails on missing cells with a list — read the list, do not patch around it.
