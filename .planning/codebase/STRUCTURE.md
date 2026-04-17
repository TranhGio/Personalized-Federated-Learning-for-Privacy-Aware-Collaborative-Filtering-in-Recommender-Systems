# Codebase Structure

**Analysis Date:** 2026-04-17

## Directory Layout

```
movie-recommendation-system/
├── CLAUDE.md                           # Root project instructions (auto-loaded)
├── README.md                           # Project overview
├── README_PAPER_KB.md                  # Paper knowledge base usage
├── RESULTS_FORMAT_EXAMPLE.md           # Results JSON schema reference
├── requirements.txt                    # Top-level Python deps (lightweight)
├── setup_paper_kb.sh                   # Bootstrap Papers/ knowledge base
├── centralize_baseline_ncf.py          # Centralized NCF baseline (Keras/TF)
├── centralized_baseline_svd.ipynb      # Centralized SVD baseline (Surprise)
│
├── data/                               # Raw datasets (git-ignored by convention)
│   └── ml-1m/                          # MovieLens 1M (users.dat, movies.dat, ratings.dat, README)
│
├── federated-baseline-cf/              # Module 1 — All-global lower bound (FedAvg/FedProx)
├── federated-pfedrec/                  # Module 2 — PFedRec IJCAI-23 calibration baseline
├── federated-personalized-cf/          # Module 3 — Split learning (local user embeddings)
├── federated-adaptive-personalized-cf/ # Module 4 — Thesis contribution (adaptive alpha + dual-level)
├── IJCAI-23-PFedRec/                   # Upstream PFedRec paper code (unmodified reference)
│
├── Papers/                             # Paper knowledge base
│   ├── raw/                            # Original PDFs (DualPersonalized-Zhang-2023.pdf, etc.)
│   └── digested/                       # Markdown digests + _INDEX.md + _EXAMPLE_*.md
│
├── docs/                               # Long-form docs
│   └── superpowers/plans/              # Planning docs for larger features
│
├── scripts/                            # Cross-module orchestration scripts
│   ├── compare_all_results.py
│   ├── run_all_baselines.sh
│   └── run_baseline_sweep_loo.sh
│
├── results/                            # Experiment outputs (JSON + markdown comparisons)
│   ├── centralized/                    # NCF, SVD, BPR-MF centralized results
│   ├── federated/                      # Baseline-cf JSONs at top, subdirs for other modules
│   │   ├── personalized/               # federated-personalized-cf outputs
│   │   ├── pfedrec/                    # federated-pfedrec outputs
│   │   └── sweeps/                     # wandb sweep result bundles
│   ├── comparison_latest.md            # Cross-module comparison
│   └── comparison_loo_<timestamp>.md   # Timestamped comparison snapshots
│
├── figures/                            # Project-level figures (cf_methods_comparison.png, etc.)
│
└── .planning/                          # GSD planning artefacts
    └── codebase/                       # This directory — codebase maps
```

**Per-module layout (identical skeleton across all four):**

```
federated-<name>-cf/
├── pyproject.toml                      # Flower app config + deps + federations
├── claude.md                           # Module-specific CLAUDE instructions
├── README.md                           # Module-specific docs
├── test_dataset.py                     # Standalone dataset tests (baseline, personalized)
├── test_models.py                      # Standalone model tests (baseline, personalized)
├── visualize_partitions.py             # Dirichlet partition diagnostics
├── scripts/                            # Module-specific sweep scripts
│   ├── analyze_sweep_results.py
│   └── run_fedprox_sweep.sh
├── figures/                            # Module-level partition visualizations
├── wandb/                              # wandb run artefacts (git-ignored)
│   ├── debug-cli.*.log
│   ├── debug-internal.log
│   ├── debug.log
│   ├── latest-run -> run-<timestamp>/  # Symlink to most recent
│   └── run-<timestamp>-<runid>/
├── .embedding_cache/                   # LOCAL param persistence (personalized/pfedrec/adaptive only)
│   └── partition_<id>/
│       ├── user_embeddings.pt          # personalized + adaptive
│       └── user_<uid>/affine_output.pt # PFedRec only (one per user)
└── federated_<name>_cf/                # Python package (snake_case)
    ├── __init__.py
    ├── dataset.py                      # Data loading + partitioning
    ├── task.py                         # Training, evaluation, metrics
    ├── client_app.py                   # Flower ClientApp hooks
    ├── server_app.py                   # Flower ServerApp main loop
    ├── strategy.py                     # (pfedrec/personalized/adaptive only) Split strategies
    ├── early_stopping.py               # EarlyStopping + EarlyStoppingState dataclass
    ├── models/
    │   ├── __init__.py
    │   ├── basic_mf.py                 # (not in pfedrec)
    │   ├── bpr_mf.py                   # (not in pfedrec)
    │   ├── losses.py                   # MSELoss, BPRLoss, BCELoss, InfoNCEContrastiveLoss
    │   ├── pfedrec_mlp.py              # (pfedrec only)
    │   ├── dual_personalized_bpr_mf.py # (adaptive only)
    │   └── adaptive_alpha.py           # (adaptive only) Alpha factory + configs
    └── evaluation/                     # (adaptive only)
        ├── __init__.py
        ├── alpha_analysis.py           # AlphaAnalyzer, AlphaStatistics
        └── user_groups.py              # UserGroupConfig + classify helpers

# Note: federated-adaptive-personalized-cf additionally has:
#   sweep.yaml                          # W&B Bayesian + Hyperband sweep config
```

## Directory Purposes

**`data/ml-1m/`:**
- Purpose: Raw MovieLens 1M dataset.
- Contains: `ratings.dat` (1M user-movie-rating-timestamp rows, `::` separated), `movies.dat` (3706 movies with `|`-separated genres), `users.dat` (6040 user demographics), `README`.
- Downloaded on-demand by `dataset.download_movielens_1m()` on first run — not required to pre-populate.
- Path resolved via `Path(__file__).parent.parent.parent / "data"` from any module's `dataset.py`.

**`federated-baseline-cf/`:**
- Purpose: Lower-bound federated baseline. All params GLOBAL via standard FedAvg/FedProx.
- Key files: `federated_baseline_cf/server_app.py` (uses `flwr.server.strategy.FedAvg`/`FedProx` directly — no custom strategy), `federated_baseline_cf/models/bpr_mf.py` (standard BPRMF without split methods).
- Distinctive: Includes a centralized evaluation phase in `server_app.main()` after training (since server has the full model).

**`federated-pfedrec/`:**
- Purpose: PFedRec (IJCAI-23) calibration baseline. Per-user `affine_output` instead of user embeddings.
- Key files: `federated_pfedrec/models/pfedrec_mlp.py` (no user embedding layer — just `Embedding(num_items) → Linear(dim, 1) → Sigmoid`), `federated_pfedrec/strategy.py` (global key is `embedding_item.weight` only).
- Distinctive: Training loops over users INSIDE a partition, each with its own model instance; item embeddings averaged across users before returning to server.

**`federated-personalized-cf/`:**
- Purpose: Middle-progression module — split learning with local user embeddings.
- Key files: `federated_personalized_cf/strategy.py` (`SplitFedAvg`/`SplitFedProx` with `GLOBAL_PARAM_KEYS = {'item_embeddings.weight', 'item_bias.weight', 'global_bias'}`), `federated_personalized_cf/models/bpr_mf.py` (has `get_/set_global_parameters()`, `get_/set_local_parameters()` methods).

**`federated-adaptive-personalized-cf/`:**
- Purpose: Thesis contribution — hierarchical conditional alpha + dual-level personalization + global prototype.
- Key files: `federated_adaptive_personalized_cf/models/adaptive_alpha.py` (three alpha classes + factory), `federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py` (DualPersonalizedBPRMF), `federated_adaptive_personalized_cf/strategy.py` (overrides `aggregate_fit` to EMA-aggregate user prototypes), `federated_adaptive_personalized_cf/evaluation/` (AlphaAnalyzer + user groups).
- Distinctive: Only module with `evaluation/` subpackage, `sweep.yaml`, and a `model-type = "dual"` option.

**`IJCAI-23-PFedRec/`:**
- Purpose: Upstream reference implementation kept unmodified for calibration.
- Contains: `train.py` (entry), `engine.py`, `mlp.py`, `data.py`, `metrics.py`, `utils.py`, plus `data/ml-1m/`, `data/ml-100k/`, `data/lastfm-2k/`, `data/amazon/`, `log/`, `sh_result/`, `architecture comparison.png`, `LICENSE`, `README.md`.
- Not a Python package (no `__init__.py`); run via `python train.py` from its own directory.

**`Papers/digested/`:**
- Purpose: Markdown digests of related papers, one file per paper using the `<firstauthor>_<year>_<tag>.md` convention.
- Files present: `arivazhagan_2019_fedper.md`, `bayer_2016_implicit.md`, `he_2017_ncf.md`, `he_2020_lightgcn.md`, `he_2024_cofedrec.md`, `hu_2025_p2fedrec.md`, `karimireddy_2020_scaffold.md`, `li_2020_fedprox.md`, `reddi_2021_adaptive_fedopt.md`, `yin_2025_devicers_survey.md`, `zhang_2022_lightfr.md`, `zhang_2023_pfedrec.md`, `zhang_2024_fedca.md`, `zhang_2025_survey_personalized_fedrec.md`, plus `_INDEX.md` (overview) and `_EXAMPLE_mcmahan_2017_fedavg.md` (template example).
- Populated via `/digest-paper` or `/digest-survey` slash commands operating on `Papers/raw/*.pdf`.

**`results/`:**
- Purpose: All experiment outputs. JSON-per-run convention.
- `results/centralized/`: `ncf_baseline_results.json`, `svd_baseline_results.json`, `bpr_mf_centralized_results.json`, plus the trained `ncf_model.keras`.
- `results/federated/`: Baseline-cf JSONs named `{model_type}_mf_{strategy}_mu{mu}_r{rounds}_f{frac}_results.json`. Also holds `comparison_latest.md` and timestamped comparison markdowns.
- `results/federated/personalized/`: Split-learning runs named `{model_type}_mf_split_{strategy}_mu{mu}_r{rounds}_f{frac}[_extras]_results.json`. Contains a `BENCHMARK_COMPARISON.md`.
- `results/federated/pfedrec/`: PFedRec runs named `pfedrec_mlp_{strategy}_dim{dim}_lr{lr}_eta{eta}_r{rounds}_f{frac}_results.json`.
- `results/federated/sweeps/`: Sweep outputs in directories like `fedprox_sweep_20251228_222711/`.

**`figures/` (project root):**
- Purpose: Cross-module / centralized figures used in thesis.
- Files: `bpr_mf_ranking_metrics.png`, `cf_methods_comparison.png`, `ncf_training_history.png`, `svd_prediction_analysis.png`.

**`federated-<name>-cf/figures/`:**
- Purpose: Per-module partition diagnostics from `visualize_partitions.py`.
- Typical contents: `partition_sizes_alpha_{0.1,0.5,1.0}.png`, `genre_distribution_alpha_*.png`, `rating_distribution_alpha_*.png`, `user_activity_alpha_*.png`, `partition_summary_alpha_*.csv`.
- The adaptive module also has `alpha_comparison.png`.

**`federated-<name>-cf/wandb/`:**
- Purpose: W&B SDK local artefacts (git-ignored).
- Contains: `debug-cli.*.log`, `debug-internal.log`, `debug.log`, `latest-run` (symlink), `run-<YYYYMMDD>_<HHMMSS>-<runid>/` dirs.

**`federated-<name>-cf/.embedding_cache/`:**
- Purpose: Client-side LOCAL parameter persistence between federated rounds. Created at runtime.
- Layout for personalized/adaptive: `.embedding_cache/partition_{id}/user_embeddings.pt` (one `.pt` per partition, contains user_embeddings.weight + user_bias.weight + optional `_round`, `_timestamp` metadata).
- Layout for pfedrec: `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt` (one `.pt` per user in that partition, since PFedRec keeps per-user score functions).
- Delete this directory to reset LOCAL state: `rm -rf .embedding_cache/`.

**`docs/superpowers/plans/`:**
- Purpose: Long-form planning/design docs for larger features.

**`scripts/` (project root):**
- Purpose: Cross-module orchestration (run all baselines, compare all results, LOO sweeps).

**`.planning/codebase/`:**
- Purpose: GSD codebase maps (this directory). Consumed by `/gsd:plan-phase` and `/gsd:execute-phase`.

## Key File Locations

**Entry Points:**
- `federated-baseline-cf/federated_baseline_cf/server_app.py`: Baseline Flower server (`@app.main()` at line ~180).
- `federated-baseline-cf/federated_baseline_cf/client_app.py`: Baseline Flower client.
- `federated-pfedrec/federated_pfedrec/server_app.py`: PFedRec server.
- `federated-pfedrec/federated_pfedrec/client_app.py`: PFedRec client (per-user alternating optimization loop).
- `federated-personalized-cf/federated_personalized_cf/server_app.py`: Split-learning server.
- `federated-personalized-cf/federated_personalized_cf/client_app.py`: Split-learning client.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py`: Adaptive server (`@app.main()` at line 207).
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py`: Adaptive client (`@app.train()` at line 215).
- `centralize_baseline_ncf.py`: Centralized NCF entry (standalone `python` script).
- `centralized_baseline_svd.ipynb`: Centralized SVD notebook.
- `IJCAI-23-PFedRec/train.py`: Upstream PFedRec reference entry.

**Configuration:**
- `federated-baseline-cf/pyproject.toml`: Flower app config + run-config defaults (`num-server-rounds=10`, `local-epochs=5`, `strategy="fedavg"`, `model-type="basic"`, `partition-mode="natural"`).
- `federated-pfedrec/pyproject.toml`: PFedRec config (`num-server-rounds=100`, `local-epochs=1`, `latent-dim=32`, `lr=0.1`, `lr-eta=80`).
- `federated-personalized-cf/pyproject.toml`: Personalized config (`num-server-rounds=10`, `local-epochs=5`, `strategy="fedavg"`).
- `federated-adaptive-personalized-cf/pyproject.toml`: Adaptive config (`num-server-rounds=50`, `local-epochs=10`, `strategy="fedprox"`, `model-type="dual"`, `alpha-method="multi_factor"`, full alpha/HC/per-user/perturbation/contrastive knobs, lines 41-199).
- `federated-adaptive-personalized-cf/sweep.yaml`: W&B Bayesian + Hyperband sweep definition.
- `requirements.txt` (root): lightweight top-level deps (used by centralized baselines).

**Core Logic:**
- `federated-*-cf/federated_*_cf/dataset.py`: `download_movielens_1m()`, `load_movielens_1m()`, `dirichlet_partition_users()`, `natural_partition_users()`, `create_leave_one_out_split()`, `create_global_mappings()`, `load_partition_data()`, `load_full_data()`.
- `federated-*-cf/federated_*_cf/task.py`: `get_model()`, `train()`, `test()`, `evaluate_ranking()`, `evaluate_ranking_sampled()`. Adaptive adds `compute_client_alpha()`, `compute_per_user_alpha()`, `get_user_stats()`. PFedRec has `train_pfedrec_single_user()`, `evaluate_pfedrec_sampled()`, `test_pfedrec()`, `prepare_user_train_data()`.
- `federated-pfedrec/federated_pfedrec/strategy.py`: `SplitFedAvg`, `SplitFedProx`, `GLOBAL_PARAM_KEYS = {'embedding_item.weight'}`.
- `federated-personalized-cf/federated_personalized_cf/strategy.py`: `SplitFedAvg`, `SplitFedProx`, `GLOBAL_PARAM_KEYS = {'item_embeddings.weight', 'item_bias.weight', 'global_bias'}`, `extract_global_params()`, `extract_local_params()`.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py`: `SplitFedAvg`/`SplitFedProx` with prototype EMA, `USER_PROTOTYPE_KEY = 'user_prototype'`.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py`: `AlphaConfig`, `HierarchicalConditionalAlphaConfig`, `DataQuantityAlpha`, `MultiFactorAlpha`, `HierarchicalConditionalAlpha`, `create_alpha_computer()`.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py`: `DualPersonalizedBPRMF` (with `PersonalMLP`, `FusionLayer`).

**Testing:**
- `federated-baseline-cf/test_dataset.py`, `federated-baseline-cf/test_models.py` (standalone scripts, `python test_dataset.py`).
- `federated-personalized-cf/test_dataset.py`, `federated-personalized-cf/test_models.py`.
- No pytest framework configured — these are plain executable Python scripts.
- `federated-pfedrec/` and `federated-adaptive-personalized-cf/` do not ship with test files.

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `client_app.py`, `bpr_mf.py`, `adaptive_alpha.py`, `early_stopping.py`, `pfedrec_mlp.py`).
- Markdown docs at root: `UPPERCASE.md` (e.g., `CLAUDE.md`, `README.md`, `RESULTS_FORMAT_EXAMPLE.md`).
- Per-module CLAUDE instructions: `claude.md` (lowercase inside module directories — note the inconsistency with root `CLAUDE.md`).
- Shell scripts: `snake_case.sh` (e.g., `run_fedprox_sweep.sh`, `run_all_baselines.sh`).
- Paper digests: `<firstauthor>_<year>_<tag>.md` (e.g., `zhang_2023_pfedrec.md`, `yin_2025_devicers_survey.md`). Reserved prefixes: `_INDEX.md`, `_EXAMPLE_*.md`.
- Results JSON: `{model_type}_mf_{split_}{strategy}_mu{mu}_r{rounds}_f{frac}[_extras]_results.json`. Extras for adaptive: `_{fusion}_mlp{dims}_{pua,ip,cl}{val}` (e.g., `dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_pua_ip0.01_results.json`).
- wandb run dirs: `run-<YYYYMMDD>_<HHMMSS>-<runid>/`.

**Directories:**
- Top-level module dirs: `kebab-case` (`federated-baseline-cf`, `federated-pfedrec`, `federated-personalized-cf`, `federated-adaptive-personalized-cf`).
- Python package dirs (inside each module): `snake_case` matching the project name (`federated_baseline_cf/`, `federated_pfedrec/`, `federated_personalized_cf/`, `federated_adaptive_personalized_cf/`).
- Result subdirs: `snake_case` (`personalized/`, `pfedrec/`, `sweeps/`).
- Partition cache dirs: `partition_<id>` (integer id), `user_<uid>` (integer user index).

**Classes:**
- `PascalCase` throughout: `MovieLensDataset`, `BasicMF`, `BPRMF`, `PFedRecMLP`, `DualPersonalizedBPRMF`, `SplitFedAvg`, `SplitFedProx`, `AlphaConfig`, `HierarchicalConditionalAlphaConfig`, `DataQuantityAlpha`, `MultiFactorAlpha`, `HierarchicalConditionalAlpha`, `EarlyStopping`, `EarlyStoppingState`, `AlphaAnalyzer`, `AlphaStatistics`, `UserGroupConfig`, `DummyClientProxy`, `MSELoss`, `BPRLoss`, `BCELoss`, `InfoNCEContrastiveLoss`.

**Functions / variables:**
- `snake_case` throughout, including tensor names (`user_embeddings`, `item_embeddings`, `global_bias`, `affine_output`).
- Module-level caches use leading underscore: `_partition_cache`, `_dataset_cache`, `_device_cache`, `_user_stats_cache`, `_item_popularity_cache`.
- Private class attributes: `_GLOBAL_PARAMS`, `_LOCAL_PARAMS` (tuples at class level).

**Configuration keys:**
- `kebab-case` in `pyproject.toml` `[tool.flwr.app.config]` (e.g., `num-server-rounds`, `model-type`, `alpha-method`, `enable-per-user-alpha`).
- Accessed via `context.run_config.get("kebab-case-key", default)`; converted mentally to `snake_case` Python variables.

## Where to Add New Code

**New federated variant (new algorithm):**
- Primary code: create `federated-<new-name>-cf/federated_<new_name>_cf/` mirroring the existing layout (`dataset.py`, `task.py`, `client_app.py`, `server_app.py`, `strategy.py`, `models/`, optional `evaluation/`).
- `pyproject.toml`: copy from the closest existing module, rename `[project].name` and `[tool.flwr.app.components]` paths.
- Tests (optional): `test_dataset.py` / `test_models.py` at module root.
- Results go to: `results/federated/<new_name>/`.

**New model within an existing module:**
- Implementation: `federated_<module>_cf/models/<new_model>.py`. Add `_GLOBAL_PARAMS` / `_LOCAL_PARAMS` tuples and `get_/set_global_parameters()` / `get_/set_local_parameters()` methods if using split learning.
- Register in `federated_<module>_cf/models/__init__.py` (add import + `__all__` entry).
- Wire into `task.get_model()` factory with a new `model_type` branch.
- Add `model-type = "<new>"` option in the module's `pyproject.toml` config comments.

**New alpha method (adaptive module only):**
- Implementation: extend `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py`. Add new class implementing `compute_from_stats(user_stats) -> float`.
- Update `create_alpha_computer(config, hc_config)` factory at bottom of that file.
- Update `AlphaConfig.__post_init__()` validation set (line ~85).
- Document in module's `claude.md` and `pyproject.toml` comments.

**New metric:**
- Implementation: add function in `federated_<module>_cf/task.py` (pattern: `compute_<metric>()`). Integrate into `evaluate_ranking()` or `evaluate_ranking_sampled()`.
- The metric will auto-flow through `client_app.evaluate()` → `weighted_average_metrics()` → `eval_metrics_history` → `wandb.log`.
- Add to `print_evaluation_metrics()` in `server_app.py` if it should appear in stdout.

**Shared utilities:**
- No project-wide `utils/` module exists. Put cross-module helpers in each module's own package, or — for orchestration — in root `scripts/`.
- Cross-module shared helpers are duplicated per-module (e.g., `download_movielens_1m`, `get_device`, `weighted_average_metrics`, `DummyClientProxy`). Maintain this duplication convention; it keeps each `flwr run .` module self-contained.

**New paper digest:**
- Add `Papers/raw/<filename>.pdf` (raw source).
- Run `/digest-paper Papers/raw/<filename>.pdf` (single method) or `/digest-survey Papers/raw/<filename>.pdf` (survey). Writes `Papers/digested/<firstauthor>_<year>_<tag>.md`.
- Update `Papers/digested/_INDEX.md` (usually done by the slash command).

**New centralized baseline:**
- Add `centralize_baseline_<name>.py` at project root (matching existing `centralize_baseline_ncf.py` pattern) or a Jupyter notebook (`centralized_baseline_<name>.ipynb`).
- Output to `results/centralized/<name>_baseline_results.json`.

## Special Directories

**`.embedding_cache/`:**
- Purpose: Per-client LOCAL parameter persistence (user embeddings / per-user affine_output / perturbation / logit_alpha).
- Generated: Yes (created at runtime by `get_cache_dir()` / `get_user_cache_dir()` helpers in `client_app.py`).
- Committed: No (should be in `.gitignore`).
- Reset with: `rm -rf federated-<module>-cf/.embedding_cache/` before running a fresh experiment.
- Note: personalized and adaptive modules place it at module root (`federated-personalized-cf/.embedding_cache/`); pfedrec places it at module root (`federated-pfedrec/.embedding_cache/`) with per-user subdirs.

**`wandb/`:**
- Purpose: Weights & Biases SDK local artefacts (logs + run metadata).
- Generated: Yes (via `wandb.init()` in `server_app.py`).
- Committed: No (should be in `.gitignore`).
- `latest-run` is a symlink to the most recent run.

**`data/`:**
- Purpose: Raw datasets. Auto-populated by `download_movielens_1m()` via HTTP on first run.
- Generated: Yes (first-run download of `ml-1m.zip`).
- Committed: Typically no; inspection-only contents (README files from GroupLens).
- Path resolution: `Path(__file__).parent.parent.parent / "data"` from any module's `dataset.py`.

**`IJCAI-23-PFedRec/`:**
- Purpose: Upstream reference implementation, not part of the thesis packages.
- Generated: No — cloned manually from upstream.
- Committed: Yes (vendored).
- Not a pip-installable package; run via `python train.py` from its own directory.

**`Papers/raw/`:**
- Purpose: Original PDFs of referenced papers.
- Generated: No — user-curated.
- Committed: Practice varies; check `.gitignore` locally.

**`.planning/`:**
- Purpose: GSD workflow artefacts (codebase maps here under `codebase/`, plus potential future `phases/`, `tasks/`).
- Generated: By `/gsd:map-codebase` and related commands.
- Committed: Typically yes — acts as documentation for future Claude instances.

---

*Structure analysis: 2026-04-17*
