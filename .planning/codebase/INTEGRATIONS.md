# External Integrations

**Analysis Date:** 2026-04-17

## APIs & External Services

**Experiment tracking:**
- Weights & Biases (wandb) - Logs federated training metrics, hyperparameters, and per-round diagnostics (alpha distribution, prototype norms, NDCG/HR per user group).
  - SDK/Client: `wandb>=0.16.0` (bumped to `>=0.19.0` in `federated-adaptive-personalized-cf/pyproject.toml`)
  - Initialized in each module's server: `federated-baseline-cf/federated_baseline_cf/server_app.py:236`, `federated-pfedrec/federated_pfedrec/server_app.py:185`, `federated-personalized-cf/federated_personalized_cf/server_app.py:240`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:284`
  - Auth: standard wandb flow - local `wandb login` (stores token in `~/.netrc`) or `WANDB_API_KEY` environment variable. No in-repo auth config.
  - Projects (from `[tool.flwr.app.config].wandb-project`):
    - `federated-cf` - `federated-baseline-cf/pyproject.toml:82`
    - `federated-pfedrec` - `federated-pfedrec/pyproject.toml:76`
    - `federated-personalized-cf` - `federated-personalized-cf/pyproject.toml:82`
    - `federated-adaptive-personalized-cf` - `federated-adaptive-personalized-cf/pyproject.toml:197`
  - Entity: hard-coded in sweep config as `vinh-federated-learning` (`federated-adaptive-personalized-cf/sweep.yaml:13`); runtime `wandb-entity` defaults to `""` in each `pyproject.toml` (falls back to user's default entity).
  - Sweeps: `federated-adaptive-personalized-cf/sweep.yaml` uses Bayesian method (`method: bayes`) with Hyperband early termination; launched via `wandb sweep sweep.yaml` then `wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>`. Orchestrated by `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py` which reads hyperparameters from the `WANDB_CONFIG` env var injected by the agent.

**Federated orchestration:**
- Flower SuperLink (optional, unused) - Each `pyproject.toml` declares a `[tool.flwr.federations.remote-federation]` block with placeholder `<SUPERLINK-ADDRESS>:<PORT>` and `insecure = true`. Not configured for real use; all current runs target the in-process `local-simulation` or `local-sim-gpu` federations.

**GitHub integrations:**
- Anthropic Claude Code action - `.github/workflows/claude.yml` invokes `anthropics/claude-code-action@v1` on PR events. Uses repo secret `CLAUDE_CODE_OAUTH_TOKEN`. Not a runtime integration for the ML stack.

**Centralized-baseline integrations:**
- Surprise dataset registry - `centralize_baseline_ncf.py:53` and `centralized_baseline_svd.ipynb` call `surprise.Dataset.load_builtin('ml-1m')`, which downloads MovieLens 1M to the Surprise cache (`~/.surprise_data/ml-1m/`) and reads `ml-1m.ratings`.
- Optuna (local, no hosted service) - Notebook uses `optuna.create_study(...)` with an in-memory backend (no SQLite / RDB tracking).

## Data Storage

**Databases:**
- None. No relational/NoSQL database is used.

**File Storage:**
- Local filesystem only. All experiment artefacts and dataset files are stored under the project tree (see Result Sinks below).

**Caching:**
- In-memory Python dicts:
  - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py:19` - `_partition_cache: Dict[str, dict]` keyed by partition config to avoid re-partitioning per client.
  - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py:23-29` - `_dataset_cache`, `_item_popularity_cache`, `_user_stats_cache` module-level dicts.
  - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:37` - `_device_cache` (CUDA detection result).
- On-disk checkpoint cache:
  - `.embedding_cache/partition_{id}/user_embeddings.pt` - Persists LOCAL user embeddings between rounds for split-learning clients. Used by `federated-personalized-cf` and `federated-adaptive-personalized-cf`. Default path: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:60` (`_MODULE_DIR.parent / ".embedding_cache"`), cleared via `clear_embedding_cache()` at `client_app.py:158`. The sweep script `scripts/run_baseline_sweep_loo.sh:97` removes this directory before each split-learning experiment.
  - `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt` - Per-user PFedRec score function cache (one Linear layer per user). Described in `federated-pfedrec/claude.md` under "Gotchas". Can be reset with `rm -rf .embedding_cache/`.
  - Additional LOCAL parameters (auto-cached for `federated-adaptive-personalized-cf` when enabled): per-user alpha (`logit_alpha`), item perturbation, PersonalMLP weights - all routed through `get_local_parameters()` / `set_local_parameters()` (see `federated-adaptive-personalized-cf/claude.md` "Parameter Classification").

## Authentication & Identity

**Auth Provider:**
- None in application code. The federated system is a local simulation with no user authentication surface.

**External service auth:**
- Weights & Biases: token in `~/.netrc` (set via `wandb login`) or `WANDB_API_KEY` env var. Not checked into the repo.
- Anthropic Claude Code GitHub Action: GitHub secret `CLAUDE_CODE_OAUTH_TOKEN` (`.github/workflows/claude.yml:34`).

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Rollbar, etc.). Errors surface via standard Python tracebacks in the Flower CLI output.

**Metrics / Logs:**
- Weights & Biases: primary sink for per-round metrics (loss, NDCG@K, HR@K, alpha stats, prototype norms).
- Console: `print(...)` statements in each `server_app.py` (e.g., `print_evaluation_metrics()` at `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:118`) emit Markdown-style tables per round.
- Local wandb run directories: `federated-adaptive-personalized-cf/wandb/run-<timestamp>-<id>/` contain per-run logs (`debug.log`, `debug-internal.log`, `output.log`). Gitignored via pattern, but `federated-adaptive-personalized-cf/wandb/` is currently present on disk.
- Flower's internal logger: `from flwr.common.logger import log` used in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:27`.

## CI/CD & Deployment

**Hosting:**
- None. Runs locally as Flower simulations (`local-simulation` CPU federation or `local-sim-gpu` with fractional GPU allocation).

**CI Pipeline:**
- GitHub Actions workflow at `.github/workflows/claude.yml` - triggers Claude Code on PR open / review comments. Not a test or deployment pipeline; no unit/integration tests are executed in CI.

**Sweep / experiment orchestration:**
- Bash orchestrators:
  - `scripts/run_baseline_sweep_loo.sh` - Runs 6 leave-one-out baseline experiments (BPR/Basic x FedAvg/FedProx x baseline/split).
  - `scripts/run_all_baselines.sh` - Full baseline suite.
  - `scripts/compare_all_results.py` - Aggregates JSON results into Markdown table `results/comparison_loo_<timestamp>.md`.
  - `federated-*/scripts/run_fedprox_sweep.sh` - Per-module FedProx hyperparameter sweeps.
  - `federated-adaptive-personalized-cf/scripts/run_ablation.sh` - Alpha method / fusion-type ablations.
  - `federated-adaptive-personalized-cf/scripts/sweep_commands.sh` - Canned `flwr run . --run-config ...` incantations.
- W&B sweep: `federated-adaptive-personalized-cf/sweep.yaml` + `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py` (reads `WANDB_CONFIG` env var, translates to `flwr run` config string).

## Environment Configuration

**Required env vars:**
- `WANDB_API_KEY` (optional, alternative to `wandb login`) - Authenticates wandb SDK.
- `WANDB_CONFIG` (sweep-only) - JSON-encoded hyperparameters injected by `wandb agent`; consumed by `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py:97`.
- `CLAUDE_CODE_OAUTH_TOKEN` (GitHub secret, CI-only) - `.github/workflows/claude.yml:34`.

**Optional:**
- `CUDA_VISIBLE_DEVICES` - Standard PyTorch GPU selection; relevant because `local-sim-gpu` federation requests fractional GPUs.

**Secrets location:**
- No in-repo secrets. `.env` files are not used; `.gitignore` does not specifically list `.env` (it only excludes `.vscode/`, `.idea`, Python build artefacts, `*.pt|*.pth|*.pkl`, and `.claude/`).
- wandb tokens live in `~/.netrc` per user; GitHub Actions tokens are managed via GitHub repo secrets.

## Dataset Sources

**MovieLens 1M (primary dataset):**
- Source: GroupLens / University of Minnesota.
- URL: `https://files.grouplens.org/datasets/movielens/ml-1m.zip` (hard-coded at `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py:79` and mirrored in the other three federated `dataset.py` files).
- Download: `urlretrieve(url, zip_path)` at `dataset.py:83`, extracted with `zipfile.ZipFile(...).extractall(...)` at `dataset.py:87`.
- Target path: `data/ml-1m/` at project root (computed from `_MODULE_DIR.parent.parent / "data"` at `dataset.py:16`).
- File format: `::`-separated text files parsed via `pd.read_csv(sep="::", engine="python", encoding="latin-1")` (`dataset.py:112-137`).
  - `data/ml-1m/ratings.dat` - 1,000,209 rows (user_id, movie_id, rating, timestamp)
  - `data/ml-1m/movies.dat` - movie_id, title, genres (pipe-separated genre list)
  - `data/ml-1m/users.dat` - user_id, gender, age, occupation, zip_code
  - `data/ml-1m/README` - GroupLens license (non-commercial research use; no redistribution)
- Currently present on disk under `data/ml-1m/`.
- Centralized baselines use a parallel path: `surprise.Dataset.load_builtin('ml-1m')` downloads the dataset into `~/.surprise_data/ml-1m/` (see `centralize_baseline_ncf.py:53`, `centralized_baseline_svd.ipynb`).

**License:**
- MovieLens 1M is restricted to non-commercial research use. Citation required: "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5, 4 (2015)." (per `data/ml-1m/README`).

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server or webhook listener is exposed.

**Outgoing:**
- HTTPS GET to `https://files.grouplens.org/datasets/movielens/ml-1m.zip` on first dataset load (one-shot via `urlretrieve`).
- HTTPS to W&B API (`api.wandb.ai`) when `wandb-enabled=true` - initialization, per-step metric uploads, artefact uploads handled by the wandb SDK.

## Result Sinks

All artefacts land under the project tree (not cloud storage). `*.pt`, `*.pth`, `*.pkl` files are gitignored but `results/**/*.json` files are explicitly whitelisted (`.gitignore:57`).

**Centralized baselines:**
- `results/centralized/bpr_mf_centralized_results.json` - Optuna-tuned BPR-MF metrics.
- `results/centralized/ncf_baseline_results.json` - NCF metrics (`centralize_baseline_ncf.py:391`).
- `results/centralized/svd_baseline_results.json` - Surprise SVD metrics.
- `results/centralized/ncf_model.keras` - Saved Keras NCF model (`centralize_baseline_ncf.py:386`).

**Federated baseline (`federated-baseline-cf`):**
- Directory: `results/federated/` (written from `federated-baseline-cf/federated_baseline_cf/server_app.py:575`).
- Filename pattern: `{model_type}_mf_{strategy}_mu{mu}_r{rounds}_f{fraction}_results.json` (e.g., `bpr_mf_fedprox_mu0.01_r50_f1.0_results.json`).

**Federated personalized (`federated-personalized-cf`):**
- Directory: `results/federated/personalized/` (`federated-personalized-cf/federated_personalized_cf/server_app.py:528-531`).
- Filename pattern: `{model_type}_mf_split_{strategy}_mu{mu}_r{rounds}_f{fraction}_results.json`.

**Federated PFedRec (`federated-pfedrec`):**
- Directory: `results/federated/pfedrec/` (`federated-pfedrec/federated_pfedrec/server_app.py:453`).

**Federated adaptive (`federated-adaptive-personalized-cf`):**
- Directory: `results/federated/personalized/` (shared with personalized module; see `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:691`).
- JSON includes full run config, per-round metrics, alpha statistics, and user-group breakdowns.

**Sweep logs:**
- `results/federated/sweeps/fedprox_sweep_<timestamp>/sweep_log.txt` - Plain-text logs from `run_fedprox_sweep.sh` invocations.

**Comparison tables:**
- `results/comparison_latest.md` and `results/comparison_loo_<timestamp>.md` - Markdown tables produced by `scripts/compare_all_results.py`.

**Figures:**
- `figures/` at project root - PNGs from the centralized notebook (training history, param importance, slice plots).
- `federated-adaptive-personalized-cf/figures/` - Alpha distribution and correlation plots from `federated_adaptive_personalized_cf/evaluation/alpha_analysis.py`.

**Local wandb run metadata:**
- `federated-adaptive-personalized-cf/wandb/run-<timestamp>-<run_id>/` - wandb SDK working directory (logs + sqlite metadata). Expected to be wiped periodically.

---

*Integration audit: 2026-04-17*
