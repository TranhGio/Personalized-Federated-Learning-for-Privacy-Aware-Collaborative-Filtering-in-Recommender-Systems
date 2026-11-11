---
name: thesis-tooling
description: "Map and conventions of the thesis analysis tooling for code changes: scripts/thesis/reeval.py (adapters, variants, trust gate, out-dir, self-test), gate_stats.py, eval_validity.py, aggregate_results.py, scripts/foundation tests and the failures that are pre-existing on the laptop, launch wrappers in .bug-investigation/ and the guards of scripts/remote/start_run.sh and finish_run.sh. Use when implementing or extending a reeval variant or adapter, a statistics script, instrumentation (communication bytes, attacks), tests, or a wrapper for the training machine; also when a test fails and you must decide whether the change caused it. Not for launching runs (train-remote) or for model/training-protocol changes."
---

# Thesis tooling — where things are and how to change them safely

Numbers in the thesis come from these scripts; a change that silently alters a replay invalidates every table that cites it. Hence the trust gates, the bit-exact tests and the rule "stop before writing a cell".

## 1. `scripts/thesis/reeval.py` (≈ 3 400 lines)
- Loads `global_state_{best,last}.pt` + `.embedding_cache/<run_id>/partition_{pid}.pt`, rebuilds the dataset snapshot once, replays the D-06 sampled eval with `round_num = final_eval_round_index` so candidate lists are byte-identical to the run.
- `ReevalConfig` (dataclass, `__post_init__` validation) carries run dir, seed, eval round, negatives, batch size, lr, lr_eta.
- Adapters: `PersonalizedAdapter`, `PfedrecAdapter` (`build_model`, `set_globals`, `set_local`, `load_cached_local`, `resolve_cache_dir` — must run first, `evaluate_user`). Adaptive = contract stub; Baseline = none. Adding a module = adding an adapter with the same interface + a subprocess-free test.
- Variants: `VARIANTS` (cached, random, shuffled, zero, calibrated) and `PFEDREC_VARIANTS` (+ head-blend suite, `qu_grid`, `qu_diag`). `run_reeval(adapter, cfg, run_data, global_state, variant, ...)` returns `(metrics, rows)`; `run_pfedrec_suite` dispatches pfedrec-only variants.
- Trust gate `trust_gate(base_metrics, best_block, limit_users, globals_kind, ...)`: `cached` vs `final_metrics.best`, |Δ| < 1e-6 and exact counts; FAIL → `SystemExit(3)`; tolerant when `--limit-users` is set or `--globals last`.
- Q_u section: `QU_*` constants, `qu_cell_key`, `finetune_item_table_one_epoch` (freezes head, SGD on item rows, seeded DataLoader), `blend_item_table` (masked), `run_qu_variant` (per-user loop, snapshots at E ∈ {1,3}, trust counters incl. `leaky_monotonicity_violations`), `_run_qu_suite_variant` (cached replay + gate, writes to `--out-dir`, exit 3 on check failure with a JSON that has `trust_checks` only).
- Output naming: `reeval_<variant>_<globals>.json` / `_per_user.csv`; `dump_json(..., out_dir=)`.
- Seeds: `fedrec_foundation.rng.np_rng / torch_gen(run_seed, pid, round, tag)` **inside** the function that samples; Q_u epochs use `round = 200 + e`.
- `--self-test` runs a synthetic pfedrec pipeline (adapter, gate, blends, grid, oracles, policy); keep it green.

## 2. Statistics: `scripts/thesis/gate_stats.py`
`GateStatsConfig` (grid CSV, baseline CSV, out path, optional grid JSON, `contrasts`), `load_baseline`, `load_grid` (suffixes the pool when >1 pool), `wilcoxon_block`, `paired_bootstrap_ci` (RNG created inside), `holm_adjust`, `select_winner`, `contrast_cells`, `compute_gate_stats`, `write_json_atomic`. New paired analyses import these; do not re-implement Wilcoxon. Needs `scipy` (optional extra `stats` in `scripts/foundation/pyproject.toml`).

## 3. Other scripts
- `eval_validity.py` — rewrites every `EVAL_VALIDITY.json`; run it after any run is recorded; statuses `valid` / `invalid_cold_eval` / `diagnostics_only`.
- `aggregate_results.py` — thesis tables from tagged runs; hard-fails on missing cells (by design).
- `run_thesis_sweep.py` — matrix launcher (`--dry-run`, `--phase`, `--retry-failed`); laptop dry-run only, real launches go through wrappers.
- `scripts/run.py <module> <mode> --run-config '...'` — the single launcher; `--dry-run` prints the `flwr run` line.

## 4. Tests (`scripts/foundation/tests/`)
```
cd scripts/foundation && python -m pytest tests/ -q                 # full
python -m pytest tests/test_reeval_qu.py tests/test_gate_stats.py -q  # Q_u + stats (21)
python -m pytest tests/test_reeval_pfedrec.py -q                       # adapter + suite
python scripts/thesis/reeval.py --self-test                            # from repo root
```
Pre-existing failures on the laptop (also fail on a clean `master` worktree; environment, not code): `test_integration.py::test_cross_module_imports[*]` (packages not `pip install -e`), `test_rng.py::test_derive_rng_stable_across_processes`, `test_pfedrec_subprocess_determinism.py::test_pfedrec_determinism_subprocess_byte_identical` (subprocess result JSON lacks `selected_clients_per_round`). Before blaming a change: `git worktree add <tmp> master && (cd <tmp>/scripts/foundation && python -m pytest <test> -q)`; remove the worktree after.
Synthetic fixtures: `test_reeval_qu.py::_build_run` (24 users, 60 items, dim 8) builds a run dir + cache without heavy files; reuse it for new variants/adapters. `FEDREC_SKIP_SLOW=1` skips slow suites.

## 5. Launch wrappers and remote scripts
- Wrapper = `.bug-investigation/run_<name>.sh`, committed, portable: `cd "$(git rev-parse --show-toplevel)"`, `PY=/home/bes/miniconda3/bin/python3; [ -x "$PY" ] || PY=python3`, `set -uo pipefail`, header comment with what it mirrors and the pre-flight date; aborts on any pre-flight or trust failure; never overwrites existing registered outputs; commits only small outputs (JSON/CSV/MD) when the run writes outside the run dir. Exemplar: `.bug-investigation/run_pfedrec_posttune_a.sh`.
- `scripts/remote/start_run.sh <wrapper> <tmux>`: `git pull --ff-only` (the branch must track its remote — set `--set-upstream-to` once), wrapper tracked, no uncommitted code (results/ and the IJCAI dir are tolerated), `MIN_FREE_GB` (default 40; `MIN_FREE_GB=5` for CPU-only re-scores), no live federation, free tmux name; logs to `.bug-investigation/<tmux>.log` (ignored).
- `scripts/remote/finish_run.sh <module> <run_id> [--archive-cache]`: eval_validity + commits only `results.json`, `manifest.json`, `EVAL_VALIDITY.json`, `per_user_metrics.json`, `reeval_*.json` in the run dir — files in sub-folders (e.g. `posttune/`) must be committed by the wrapper.
- Never edit code on the training machine; never put a running federation's cache on the HDD archive.

## 6. Conventions (beyond CLAUDE.md)
- `IJCAI-23-PFedRec/` is a gitlink (@ abc23809) and must not be modified; the compat patch used to run it lives at `.bug-investigation/ijcai_pfedrec_compat.patch`.
- Config via dataclasses with `__post_init__`; `typing.Dict/List/Optional`; NumPy docstrings; branch prefixes `feat/ fix/ chore/ docs/`; commit/push only when asked; `_workspace/` is ignored by git.
- When a change alters any output file name, column set or JSON key, update the `analyze-results` skill table in the same change.
