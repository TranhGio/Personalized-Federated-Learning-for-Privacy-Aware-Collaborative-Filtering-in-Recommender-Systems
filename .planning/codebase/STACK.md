# Technology Stack

**Analysis Date:** 2026-04-17

## Languages

**Primary:**
- Python 3.9+ - All federated modules, centralized baselines, helper scripts (per `CLAUDE.md` "All Python code should be compatible with Python 3.9+")

**Secondary:**
- Bash - Experiment sweep/orchestration scripts (`scripts/run_baseline_sweep_loo.sh`, `scripts/run_all_baselines.sh`, `federated-*/scripts/run_fedprox_sweep.sh`, `federated-adaptive-personalized-cf/scripts/sweep_commands.sh`, `federated-adaptive-personalized-cf/scripts/run_ablation.sh`)
- Jupyter Notebook - Centralized SVD/BPR-MF baseline (`centralized_baseline_svd.ipynb`)
- YAML - Weights & Biases sweep config (`federated-adaptive-personalized-cf/sweep.yaml`), CI workflows (`.github/workflows/claude.yml`)
- TOML - Flower app configuration and dependencies (`federated-*/pyproject.toml`)

## Runtime

**Environment:**
- Python 3.9+ (target baseline)
- CPython interpreter (no Cython/PyPy-specific features observed)
- CUDA-capable GPU used for development (notebook comment in `centralized_baseline_svd.ipynb` mentions "NVIDIA GeForce RTX 5090"); CPU-only simulation also supported via `local-simulation` federation

**Package Manager:**
- `pip` (editable installs via `pip install -e .` per root `CLAUDE.md`)
- Build backend: `hatchling` (declared in every `federated-*/pyproject.toml` `[build-system]` block)
- Lockfiles: none detected. Two parallel dependency declarations exist:
  - Top-level `requirements.txt` (centralized baselines only; 6 packages, no pins)
  - Per-module `pyproject.toml` `[project].dependencies` (federated modules; minimum-version pins)

## Frameworks

**Core:**
- `flwr[simulation]>=1.22.0` - Flower federated learning framework; provides `ServerApp`, `ClientApp`, `Grid` message-passing API, and `FedAvg`/`FedProx` strategies. Imports seen in `federated-*/federated_*/server_app.py`, `client_app.py`, `strategy.py` (`from flwr.serverapp import Grid, ServerApp`, `from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx`).
- `flwr-datasets[vision]>=0.5.0` - Listed in all four federated `pyproject.toml`. Currently not imported by code; MovieLens is loaded directly via `urlretrieve` + pandas in `federated-*/federated_*/dataset.py`.
- `torch>=2.7.1` - PyTorch for all federated models (BPRMF, BasicMF, DualPersonalizedBPRMF, PFedRec MLP). Centralized notebook reports PyTorch 2.10.0.dev.
- `torchvision>=0.22.1` - Required transitively by `flwr-datasets[vision]`; not imported by project code.
- TensorFlow / Keras - Used ONLY by centralized NCF baseline (`centralize_baseline_ncf.py`: `from tensorflow import keras`, `from tensorflow.keras import layers, Model`). Declared in root `requirements.txt` (unpinned).
- `scikit-surprise` - Used ONLY by centralized baselines to fetch MovieLens via `surprise.Dataset.load_builtin('ml-1m')` (see `centralize_baseline_ncf.py` line 53 and `centralized_baseline_svd.ipynb`).

**Testing:**
- No formal testing framework detected (no `pytest`, `unittest`, `jest`, etc. in dependencies or CI). `CLAUDE.md` references `python test_dataset.py` and `python test_models.py` as ad-hoc test scripts; these are not present in the current tree as top-level files.

**Build/Dev:**
- `hatchling` - PEP 517 build backend declared in each module's `[build-system]`.
- `wandb>=0.16.0` (0.19.0+ for `federated-adaptive-personalized-cf`) - Experiment tracking SDK used from `server_app.py` of each module.

## Key Dependencies

**Critical:**
- `flwr[simulation]>=1.22.0` - Federated orchestration; every `server_app.py` calls `@app.main()` on a `ServerApp`, every `client_app.py` decorates handlers on a `ClientApp`.
- `torch>=2.7.1` - All federated model definitions in `federated-*/federated_*/models/*.py` subclass `torch.nn.Module`; optimizers are `torch.optim.Adam` / `SGD`.
- `numpy>=1.24.0` - Alpha computation, prototype aggregation, evaluation metric calculation throughout `task.py` and `evaluation/alpha_analysis.py`.
- `pandas>=2.0.0` - MovieLens `ratings.dat` / `movies.dat` / `users.dat` parsing with `::` separator in `federated-*/federated_*/dataset.py::load_movielens_1m()`.
- `scikit-learn>=1.3.0` - `sklearn.model_selection.train_test_split` in `centralize_baseline_ncf.py` and train/test splitting inside federated `task.py`.
- `wandb>=0.16.0` - Logged from every federated `server_app.py` (e.g., `wandb.init(project=..., config=...)` at `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:284`).

**Infrastructure:**
- `hatchling` - Build backend (declared `requires = ["hatchling"]` in each `pyproject.toml`).
- `flwr-datasets[vision]>=0.5.0` - Declared but not imported; likely retained from Flower template.
- `torchvision>=0.22.1` - Declared but not imported; transitive requirement of `flwr-datasets[vision]`.

**Centralized-only (root `requirements.txt`):**
- `scikit-surprise` - SVD baseline (`surprise.SVD`, pickled output), plus ML-1M download for NCF.
- `tensorflow` - NCF (Keras API) baseline.
- `numpy`, `pandas`, `matplotlib`, `seaborn` - General analysis/plotting.
- `optuna` (notebook-only; not in `requirements.txt`) - Bayesian hyperparameter optimization in `centralized_baseline_svd.ipynb` (imports `optuna`, `TPESampler`, `MedianPruner`, `plot_optimization_history`).

**Reference implementation (`IJCAI-23-PFedRec/requirements.txt`):**
- Pinned to older stack (`torch==1.8.0+cu111`, `pandas==1.3.5`, `scikit-learn==1.0.2`, `tensorboardX`, `matplotlib==3.5.3`). Independent of the Flower re-implementation.

## Configuration

**Environment:**
- Per-module Flower run config: `[tool.flwr.app.config]` block in each module's `pyproject.toml` (e.g., `federated-adaptive-personalized-cf/pyproject.toml:39`). Accessed at runtime via `context.run_config["<key>"]` inside `server_app.py` / `client_app.py`.
- Override at CLI with `flwr run . --run-config "key=value key2=value2"` (documented in each `claude.md` and used extensively in `scripts/run_baseline_sweep_loo.sh`).
- Sweep-driven config: `wandb agent` injects hyperparameters via `WANDB_CONFIG` env var, parsed by `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py::get_config_from_env()`.
- `.env` files: not detected. No dotenv library used.
- W&B authentication: relies on `wandb login` (local `~/.netrc`) or `WANDB_API_KEY` env var (managed outside the repo).

**Build:**
- `federated-baseline-cf/pyproject.toml` - Flower app + deps
- `federated-pfedrec/pyproject.toml` - Flower app + deps (100 rounds, latent-dim 32, dual LR)
- `federated-personalized-cf/pyproject.toml` - Flower app + deps (split learning)
- `federated-adaptive-personalized-cf/pyproject.toml` - Flower app + deps (50 rounds, dual model, HC alpha, next-gen flags)
- Root `requirements.txt` - Centralized baseline deps only
- `IJCAI-23-PFedRec/requirements.txt` - Legacy reference (conda-exported)

## Platform Requirements

**Development:**
- Linux (host observed: `Linux 6.17.0-19-generic`, zsh shell).
- CUDA runtime for GPU federation (`local-sim-gpu` declared in each `pyproject.toml`: `options.backend.client-resources.num-gpus = 0.2`, `num-cpus = 6` or `12`).
- ~1 GB disk for MovieLens 1M (auto-downloaded from `https://files.grouplens.org/datasets/movielens/ml-1m.zip` on first run).
- Node-less; pure Python toolchain.

**Production:**
- No production deployment target. This is a research/thesis codebase run as local Flower simulations (`[tool.flwr.federations.local-simulation]` / `local-sim-gpu`).
- `remote-federation` stub exists in each `pyproject.toml` pointing at an unfilled `<SUPERLINK-ADDRESS>:<PORT>` (not active).

## Federated Run Config Quick Reference

Key runtime knobs (shared across modules unless noted):

| Key | Default (adaptive) | Default (baseline) | Purpose |
|-----|--------------------|--------------------|---------|
| `num-server-rounds` | 50 | 10 | FL rounds (`R`) |
| `local-epochs` | 10 | 5 | Local steps per round (`K`) |
| `strategy` | `fedprox` | `fedavg` | Aggregation strategy |
| `proximal-mu` | 0.01 | 0.01 | FedProx proximal term |
| `model-type` | `dual` | `basic` / `bpr` | MF variant |
| `embedding-dim` | 128 | 128 | Latent factor dim |
| `alpha` | 0.5 | 0.5 | Dirichlet concentration |
| `partition-mode` | `natural` | `natural` | `natural` = cross-device (1 user = 1 client); `dirichlet` = cross-silo |
| `alpha-method` | `hierarchical_conditional` | n/a | Personalization heuristic |
| `wandb-enabled` | true | true | W&B logging toggle |
| `early-stopping-enabled` | true | false | Per-round metric early stop |

GPU federation: `flwr run . local-sim-gpu` (5 supernodes, 0.2 GPU each).

---

*Stack analysis: 2026-04-17*
