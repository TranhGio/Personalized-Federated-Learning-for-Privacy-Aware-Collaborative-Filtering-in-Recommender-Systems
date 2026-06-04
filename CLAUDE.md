# Movie Recommendation System - Federated Learning

Master thesis: **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**
Author: Dang Vinh | Dataset: MovieLens 1M | Framework: Flower (flwr) + PyTorch

## Project Structure

Four federated implementations (progression: baseline -> PFedRec -> personalized -> adaptive):

| Module | Approach | Key Difference |
|--------|----------|----------------|
| `federated-baseline-cf/` | All params global (FedAvg/FedProx) | Lower-bound baseline |
| `federated-pfedrec/` | PFedRec (IJCAI-23): local score function, global item embeddings | Calibration baseline |
| `federated-personalized-cf/` | Split learning (local user embeddings) | Privacy + personalization |
| `federated-adaptive-personalized-cf/` | Hierarchical conditional alpha + dual-level | Thesis contribution |

Centralized baselines: `centralized_baseline_svd.ipynb`, `centralize_baseline_ncf.py`
Reference implementations: `IJCAI-23-PFedRec/` (original PFedRec code)
Results: `results/centralized/` and `results/federated/`

## Commands

```bash
# Run federated experiments (from each subdirectory)
cd federated-adaptive-personalized-cf && flwr run .
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "model-type=dual fusion-type=concat alpha-method=hierarchical_conditional"

# Run PFedRec calibration baseline
cd federated-pfedrec && flwr run .
flwr run . --run-config "latent-dim=64 lr=0.05 lr-eta=40"

# Install dependencies
pip install -e .

# Tests
python test_dataset.py
python test_models.py

# W&B sweep
wandb sweep sweep.yaml
wandb agent <ENTITY>/federated-adaptive-personalized-cf/<SWEEP_ID>

# Visualize data partitions
python visualize_partitions.py
```

## Tech Stack

- **FL Framework**: Flower (flwr) v1.22.0+
- **ML**: PyTorch 2.7.1+, BPR-MF (ranking), NCF, SVD baselines
- **Tracking**: Weights & Biases (wandb)
- **Data**: MovieLens 1M (6,040 users, 3,706 movies, 1M ratings)
- **Partitioning**: Dirichlet distribution (alpha=0.5 for non-IID)

## Key Conventions

- Primary metric: NDCG@10 (ranking quality). BPR models have high RMSE by design - this is expected.
- Evaluation protocol: Leave-one-out with 99 negative samples (NCF protocol)
- User groups: sparse (0-30 interactions), medium (30-100), dense (100+)
- Results saved as JSON in `results/` with full experiment metadata
- Branch naming: `feat/`, `fix/`, `chore/` prefixes
- All Python code should be compatible with Python 3.9+

## Architecture Notes

- **Split learning**: User embeddings LOCAL (never sent to server), item embeddings GLOBAL (aggregated)
- **Adaptive alpha**: Per-client personalization level computed from user stats (quantity, diversity, coverage, consistency)
- **Hierarchical conditional alpha**: Resolves quantity-coverage redundancy (geometric mean) and diversity-consistency contradiction (harmonic mean), plus conditional rules for edge cases
- **Dual-level personalization**: Level 1 = alpha-blended embeddings, Level 2 = client-specific PersonalMLP
- **Global prototype**: EMA-based server-side user prototype for sparse user support

## Subdirectory Docs

Detailed architecture docs per module:
- @federated-baseline-cf/claude.md
- @federated-pfedrec/claude.md
- @federated-personalized-cf/claude.md
- @federated-adaptive-personalized-cf/claude.md

## MCP Usage

- Use **Context7 MCP** for library/API documentation and code examples

## Compaction Rules

When compacting, always preserve:
- List of all modified files in this session
- Current experiment configuration and parameters being tested
- Any running W&B sweep IDs or experiment tracking info
- The current branch name and recent commits
- Key metric values discussed (NDCG@10, Hit Rate@K, etc.)


## Paper Knowledge Base
When you need to understand prior work, reference implementations, or architectural decisions from related papers:
1. Read `Papers/digested/_INDEX.md` for an overview of all digested papers
2. Read the specific `Papers/digested/<paper_id>.md` for implementation details
3. Only read raw PDFs in `Papers/raw/` when you need to verify exact details not captured in the digest

Slash commands for paper management:
- `/digest-paper Papers/raw/<filename>.pdf` — digest a single method/technique paper
- `/digest-survey Papers/raw/<filename>.pdf` — digest a survey/review paper (different structure: extracts taxonomies, comparative tables, research gaps)
- `/batch-digest` — process all undigested PDFs in Papers/raw/

How to decide: If the paper proposes ONE new method → `/digest-paper`. If it reviews/surveys MANY methods → `/digest-survey`.

## Notation Convention (use consistently across ALL code and docs)
- `w` or `global_model` — server global model parameters
- `theta_i` or `personal_model_i` — client i's personalized model
- `D_i` — client i's local dataset
- `K` — number of local training steps per round
- `R` — total number of FL communication rounds
- `N` — total number of clients
- `C` — client sampling fraction per round

## Code Standards
- Type hints on all function signatures
- Docstrings on all public functions (NumPy style)
- Config via dataclasses, not loose dicts
- Experiments reproducible via seed + config file
- Log all metrics to CSV + console per round

## Project

**Federated Movie Recommendation — Cross-Device Migration & Thesis Evaluation**

Master's thesis project on **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**, implemented on MovieLens 1M with Flower (flwr) + PyTorch. The repository contains four parallel federated implementations — a lower-bound baseline, the PFedRec (IJCAI-23) calibration baseline, a split-learning personalized baseline, and the thesis contribution (adaptive/hierarchical-conditional alpha with dual-level personalization).

This planning cycle migrates the entire comparative study from its current cross-silo setup (`num-supernodes=5`) to the methodologically defensible cross-device setup used by every published FedRec paper (**1 user = 1 client, N=6040**), then re-runs the thesis evaluation under that corrected protocol.

**Core Value:** Under a correct cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method must beat all three baselines on NDCG@10 — including on sparse users — while PFedRec reproduces the published reference (HR@10 ≈ 0.70, NDCG@10 ≈ 0.38, within ±2 points).

If the adaptive method does not win under the corrected protocol, the thesis contribution has to be rethought. Methodological correctness is non-negotiable.

### Constraints

- **Tech stack**: Flower (flwr) ≥ 1.22.0, PyTorch ≥ 2.7.1, Python ≥ 3.9 — Fixed by existing code; changing is out of scope for this cycle.
- **Dataset**: MovieLens 1M only (6,040 users / 3,706 items / ~1M ratings) — Thesis scope; generalization deferred.
- **Evaluation protocol**: Leave-one-out + 99 negative samples (NCF protocol), NDCG@10 as the primary metric — Convention in the FedRec literature; required for apples-to-apples comparison.
- **Timeline**: Soft thesis deadline — Prioritize reproduction + thesis-contribution evaluation over nice-to-haves like shared-code refactoring.
- **Hardware**: Single-machine Flower simulation with 6,040 virtual clients — Blocks any design that assumes real distributed edge devices.
- **Backwards compatibility**: Cross-silo configs must continue to run (as an explicit opt-in) so existing W&B runs remain reproducible and appendix results can be regenerated if needed — We override defaults, we do not delete the code paths.
- **Tracking**: A new W&B project is used for cross-device runs to keep the run list clean and to avoid accidentally mixing cross-silo and cross-device numbers in comparison plots.

## Technology Stack

## Languages
- Python 3.9+ - All federated modules, centralized baselines, helper scripts (per `CLAUDE.md` "All Python code should be compatible with Python 3.9+")
- Bash - Experiment sweep/orchestration scripts (`scripts/run_baseline_sweep_loo.sh`, `scripts/run_all_baselines.sh`, `federated-*/scripts/run_fedprox_sweep.sh`, `federated-adaptive-personalized-cf/scripts/sweep_commands.sh`, `federated-adaptive-personalized-cf/scripts/run_ablation.sh`)
- Jupyter Notebook - Centralized SVD/BPR-MF baseline (`centralized_baseline_svd.ipynb`)
- YAML - Weights & Biases sweep config (`federated-adaptive-personalized-cf/sweep.yaml`), CI workflows (`.github/workflows/claude.yml`)
- TOML - Flower app configuration and dependencies (`federated-*/pyproject.toml`)
## Runtime
- Python 3.9+ (target baseline)
- CPython interpreter (no Cython/PyPy-specific features observed)
- CUDA-capable GPU used for development (notebook comment in `centralized_baseline_svd.ipynb` mentions "NVIDIA GeForce RTX 5090"); CPU-only simulation also supported via `local-simulation` federation
- `pip` (editable installs via `pip install -e .` per root `CLAUDE.md`)
- Build backend: `hatchling` (declared in every `federated-*/pyproject.toml` `[build-system]` block)
- Lockfiles: none detected. Two parallel dependency declarations exist:
## Frameworks
- `flwr[simulation]>=1.22.0` - Flower federated learning framework; provides `ServerApp`, `ClientApp`, `Grid` message-passing API, and `FedAvg`/`FedProx` strategies. Imports seen in `federated-*/federated_*/server_app.py`, `client_app.py`, `strategy.py` (`from flwr.serverapp import Grid, ServerApp`, `from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx`).
- `flwr-datasets[vision]>=0.5.0` - Listed in all four federated `pyproject.toml`. Currently not imported by code; MovieLens is loaded directly via `urlretrieve` + pandas in `federated-*/federated_*/dataset.py`.
- `torch>=2.7.1` - PyTorch for all federated models (BPRMF, BasicMF, DualPersonalizedBPRMF, PFedRec MLP). Centralized notebook reports PyTorch 2.10.0.dev.
- `torchvision>=0.22.1` - Required transitively by `flwr-datasets[vision]`; not imported by project code.
- TensorFlow / Keras - Used ONLY by centralized NCF baseline (`centralize_baseline_ncf.py`: `from tensorflow import keras`, `from tensorflow.keras import layers, Model`). Declared in root `requirements.txt` (unpinned).
- `scikit-surprise` - Used ONLY by centralized baselines to fetch MovieLens via `surprise.Dataset.load_builtin('ml-1m')` (see `centralize_baseline_ncf.py` line 53 and `centralized_baseline_svd.ipynb`).
- No formal testing framework detected (no `pytest`, `unittest`, `jest`, etc. in dependencies or CI). `CLAUDE.md` references `python test_dataset.py` and `python test_models.py` as ad-hoc test scripts; these are not present in the current tree as top-level files.
- `hatchling` - PEP 517 build backend declared in each module's `[build-system]`.
- `wandb>=0.16.0` (0.19.0+ for `federated-adaptive-personalized-cf`) - Experiment tracking SDK used from `server_app.py` of each module.
## Key Dependencies
- `flwr[simulation]>=1.22.0` - Federated orchestration; every `server_app.py` calls `@app.main()` on a `ServerApp`, every `client_app.py` decorates handlers on a `ClientApp`.
- `torch>=2.7.1` - All federated model definitions in `federated-*/federated_*/models/*.py` subclass `torch.nn.Module`; optimizers are `torch.optim.Adam` / `SGD`.
- `numpy>=1.24.0` - Alpha computation, prototype aggregation, evaluation metric calculation throughout `task.py` and `evaluation/alpha_analysis.py`.
- `pandas>=2.0.0` - MovieLens `ratings.dat` / `movies.dat` / `users.dat` parsing with `::` separator in `federated-*/federated_*/dataset.py::load_movielens_1m()`.
- `scikit-learn>=1.3.0` - `sklearn.model_selection.train_test_split` in `centralize_baseline_ncf.py` and train/test splitting inside federated `task.py`.
- `wandb>=0.16.0` - Logged from every federated `server_app.py` (e.g., `wandb.init(project=..., config=...)` at `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:284`).
- `hatchling` - Build backend (declared `requires = ["hatchling"]` in each `pyproject.toml`).
- `flwr-datasets[vision]>=0.5.0` - Declared but not imported; likely retained from Flower template.
- `torchvision>=0.22.1` - Declared but not imported; transitive requirement of `flwr-datasets[vision]`.
- `scikit-surprise` - SVD baseline (`surprise.SVD`, pickled output), plus ML-1M download for NCF.
- `tensorflow` - NCF (Keras API) baseline.
- `numpy`, `pandas`, `matplotlib`, `seaborn` - General analysis/plotting.
- `optuna` (notebook-only; not in `requirements.txt`) - Bayesian hyperparameter optimization in `centralized_baseline_svd.ipynb` (imports `optuna`, `TPESampler`, `MedianPruner`, `plot_optimization_history`).
- Pinned to older stack (`torch==1.8.0+cu111`, `pandas==1.3.5`, `scikit-learn==1.0.2`, `tensorboardX`, `matplotlib==3.5.3`). Independent of the Flower re-implementation.
## Configuration
- Per-module Flower run config: `[tool.flwr.app.config]` block in each module's `pyproject.toml` (e.g., `federated-adaptive-personalized-cf/pyproject.toml:39`). Accessed at runtime via `context.run_config["<key>"]` inside `server_app.py` / `client_app.py`.
- Override at CLI with `flwr run . --run-config "key=value key2=value2"` (documented in each `claude.md` and used extensively in `scripts/run_baseline_sweep_loo.sh`).
- Sweep-driven config: `wandb agent` injects hyperparameters via `WANDB_CONFIG` env var, parsed by `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py::get_config_from_env()`.
- `.env` files: not detected. No dotenv library used.
- W&B authentication: relies on `wandb login` (local `~/.netrc`) or `WANDB_API_KEY` env var (managed outside the repo).
- `federated-baseline-cf/pyproject.toml` - Flower app + deps
- `federated-pfedrec/pyproject.toml` - Flower app + deps (100 rounds, latent-dim 32, dual LR)
- `federated-personalized-cf/pyproject.toml` - Flower app + deps (split learning)
- `federated-adaptive-personalized-cf/pyproject.toml` - Flower app + deps (50 rounds, dual model, HC alpha, next-gen flags)
- Root `requirements.txt` - Centralized baseline deps only
- `IJCAI-23-PFedRec/requirements.txt` - Legacy reference (conda-exported)
## Platform Requirements
- Linux (host observed: `Linux 6.17.0-19-generic`, zsh shell).
- CUDA runtime for GPU federation (`local-sim-gpu` declared in each `pyproject.toml`: `options.backend.client-resources.num-gpus = 0.2`, `num-cpus = 6` or `12`).
- ~1 GB disk for MovieLens 1M (auto-downloaded from `https://files.grouplens.org/datasets/movielens/ml-1m.zip` on first run).
- Node-less; pure Python toolchain.
- No production deployment target. This is a research/thesis codebase run as local Flower simulations (`[tool.flwr.federations.local-simulation]` / `local-sim-gpu`).
- `remote-federation` stub exists in each `pyproject.toml` pointing at an unfilled `<SUPERLINK-ADDRESS>:<PORT>` (not active).
## Federated Run Config Quick Reference
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

## Conventions

## Python Target
- **Language version:** Python 3.9+ (`CLAUDE.md` > "Code Standards")
- **Typing style:** `typing.Dict`, `typing.List`, `typing.Tuple`, `typing.Optional`, `typing.Union` (pre-3.10 syntax is used throughout, e.g. `federated-baseline-cf/federated_baseline_cf/task.py:6`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:17`). Do NOT introduce `X | Y` or `list[int]` — match the existing `typing` module syntax for consistency.
## Naming Patterns
- Package dirs use `snake_case` matching the Flower app name (dashes replaced with underscores): `federated-baseline-cf/` -> `federated_baseline_cf/`.
- Python files: lowercase `snake_case.py` (e.g., `server_app.py`, `client_app.py`, `dataset.py`, `task.py`, `strategy.py`, `adaptive_alpha.py`, `dual_personalized_bpr_mf.py`, `early_stopping.py`).
- Test scripts: top-level per-module `test_dataset.py`, `test_models.py` (run directly as `python test_dataset.py` — not collected by pytest).
- Shell scripts: `run_*.sh` (sweeps), `sweep_commands.sh`.
- `snake_case` for functions and methods: `load_partition_data`, `compute_client_alpha`, `get_local_parameters`, `set_global_parameters`, `dirichlet_partition_users`, `evaluate_ranking_sampled`.
- Private / module-internal helpers: single leading underscore (`_compute_quantity_factor`, `_dataset_cache`, `_device_cache`, `_MODULE_DIR`).
- `PascalCase`: `BasicMF`, `BPRMF`, `MovieLensDataset`, `AlphaConfig`, `HierarchicalConditionalAlpha`, `DualPersonalizedBPRMF`, `SplitFedAvg`, `SplitFedProx`, `EarlyStopping`, `EarlyStoppingState`, `AlphaAnalyzer`, `UserGroupConfig`.
- Acronyms stay uppercase inside the name (`MF`, `BPR`, `MLP`, `MSE`, `CF`).
- `snake_case` everywhere (`num_users`, `num_items`, `embedding_dim`, `global_model`, `local_params`, `ratings_df`, `user2idx`, `item2idx`, `partition_id`, `proximal_mu`).
- DataFrames end with `_df`; PyTorch tensors have no suffix; lookup dicts use `a2b` form (`user2idx`, `item2idx`).
- Module-level constants: `UPPER_SNAKE_CASE` (`_DEFAULT_DATA_DIR`, `_MODULE_DIR` — leading underscore because they are module-private).
- Frozenset key registries: `GLOBAL_PARAM_KEYS`, `LOCAL_PARAM_KEYS`, `USER_PROTOTYPE_KEY` in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:34-46` and `federated-pfedrec/federated_pfedrec/strategy.py:15-19`.
## Notation Convention (Thesis / Algorithm)
| Symbol | Meaning | Typical code name |
|--------|---------|-------------------|
| `w` | Server global model parameters | `global_model`, `global_params`, `w_server` |
| `theta_i` | Client i's personalized model | `personal_model_i`, `local_params` |
| `D_i` | Client i's local dataset | `trainloader` / `testloader` on client i |
| `K` | Number of local training steps per round | `local-epochs` config key |
| `R` | Total number of FL communication rounds | `num-server-rounds` config key |
| `N` | Total number of clients | `num-supernodes`, `num_partitions` |
| `C` | Client sampling fraction per round | `fraction-train` / `fraction_fit` |
| `α` (alpha) | Personalization level (0 = global, 1 = local) | `alpha`, `alpha_i`, `p_effective = α·p_local + (1-α)·p_global` |
| `μ` (mu) | FedProx proximal term strength | `proximal-mu` / `proximal_mu` |
## Docstring Style
- **NumPy-style (preferred for public API):** `federated-baseline-cf/federated_baseline_cf/task.py:30-55` uses `Parameters\n----------` and `Returns\n-------` sections.
- **Google-style (common in models and older code):** `federated-baseline-cf/federated_baseline_cf/models/bpr_mf.py:42-55` and `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py:32-46` use `Args:` / `Returns:` / `Attributes:` sections.
- Required on **all public functions**.
- Parameter/return types are duplicated in both the type hint and the docstring prose.
## Type Hints
- Prefer `Optional[T]` over `Union[T, None]`.
- `Dict[str, float]`, `List[Tuple[int, int]]` — fully qualified generics, not bare `dict`/`list`.
- Union aliases are declared once at module bottom (e.g., `AlphaComputer = Union[DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha]` in `adaptive_alpha.py:601`).
## Config via Dataclasses (not loose dicts)
- `HierarchicalConditionalAlphaConfig` — `adaptive_alpha.py:99-174`
- `UserGroupConfig` — `evaluation/user_groups.py:18-37`
- `AlphaStatistics`, `HierarchicalAlphaStatistics` — `evaluation/alpha_analysis.py:13`, `:256`
- `EarlyStoppingState` — `early_stopping.py:12-21`
- Every dataclass with non-trivial invariants implements `__post_init__` and raises `ValueError` with a descriptive message on invalid input.
- Mutable defaults use `field(default_factory=...)`.
## Factory Pattern for Pluggable Strategies
## Split-Learning Parameter Protocol
- `get_global_parameters() -> OrderedDict`
- `set_global_parameters(global_state_dict: Dict[str, torch.Tensor]) -> None`
- `get_local_parameters() -> OrderedDict`
- `set_local_parameters(local_state_dict, strict=False) -> Tuple[List[str], List[str]]` (returns `(loaded_keys, missing_keys)`)
- `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:342-450`
- `federated-personalized-cf/federated_personalized_cf/models/basic_mf.py:221-330`
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py` (extends the pattern for `personal_mlp.*`, `fusion_*`, `logit_alpha`, `item_perturbation`).
## Import Organization
- **Absolute imports only** — no `from .task import ...`. All internal imports use the full package name.
- No star imports.
## Error Handling
- Raise `ValueError` for invalid config / input with a message containing the bad value and the expected range. Example: `adaptive_alpha.py:73-95`.
- Raise `ValueError` from factory functions on unknown enum strings (see `create_alpha_computer`, `get_model`).
- `try/except` is reserved for I/O and environment probing (CUDA availability check, atomic file save, cache load with shape mismatch). Example: `federated-baseline-cf/federated_baseline_cf/client_app.py:19-37` (safe CUDA detection); `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:96-120` (atomic embedding save via `tempfile.mkstemp` + rename).
- When loading cached state with possibly-mismatched shapes, use `strict=False` and return `(loaded_keys, missing_keys)` so the caller can log what was recovered.
## Logging and Metric Reporting
## Configuration Surface
- **Keys use kebab-case** in `pyproject.toml` (`num-server-rounds`, `model-type`, `alpha-method`, `enable-per-user-alpha`).
- **Variables use snake_case** in Python (`num_server_rounds`, `model_type`, `alpha_method`, `enable_per_user_alpha`).
- Always pass a default to `.get()` so a missing key never raises.
- Runtime overrides go via `flwr run . --run-config "key1=value1 key2=value2"`.
- No `.env` files, no `python-dotenv`, no `argparse` for core experiments.
- W&B sweep configs live in `federated-adaptive-personalized-cf/sweep.yaml` and use `snake_case` keys (Flower strips the dashes when `wandb` agents call the sweep runner).
## Reproducibility (Seed + Config)
- **Fixed default seed is 42** across all partitioning / negative-sampling code paths. Passed as a `seed: int = 42` parameter (e.g., `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py:285,424,452`; `.../task.py:925`).
- **Per-use reseeding**: `np.random.seed(seed)` and `random.seed(seed)` are called *inside* functions that sample (e.g., `dataset.py:304`, `task.py:953`), not once at process start. Match this pattern when adding new stochastic code paths.
- Full run config (all hyperparameters) is serialized into the `results/.../*.json` file and into the `wandb.config` dict so a run can be reproduced from either artifact.
- `centralize_baseline_ncf.py:18-19` sets `np.random.seed(42)` + `tf.random.set_seed(42)` at module import (centralized baseline only).
## Caching Patterns
- **Module-level in-memory caches:** `_dataset_cache = {}`, `_partition_cache = {}`, `_device_cache = None`, `_item_popularity_cache = {}` used to avoid re-partitioning per client and re-probing CUDA. See `federated-baseline-cf/federated_baseline_cf/task.py:14-17` and `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:37`.
- **On-disk embedding cache:** `.embedding_cache/partition_{id}/user_embeddings.pt` (split-learning modules) and `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt` (PFedRec). Writes use atomic tempfile + rename. Directory layout in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:47-66`.
- **Reset commands:** `rm -rf .embedding_cache/` is the official way to start fresh; documented in every split-learning `claude.md`.
## Comments
- Use `# ===...===` banner blocks to separate logical sections inside long functions (see the two-stage hierarchical alpha in `adaptive_alpha.py:436-456`, config sections in `pyproject.toml`, and sweep sections in `sweep.yaml`).
- Inline comments use `#` followed by a single space; Unicode math (`α`, `σ`, `μ`, `²`) is allowed and frequent in docstrings/comments because the research domain is math-heavy.
- TODO/FIXME usage is minimal; if added, prefix with the author's initials or reference the relevant paper/issue.
## Module Design
- **Each Flower app is its own package** (`federated_baseline_cf/`, `federated_personalized_cf/`, `federated_pfedrec/`, `federated_adaptive_personalized_cf/`) with the same canonical layout: `dataset.py`, `task.py`, `client_app.py`, `server_app.py`, `strategy.py` (where split-learning applies), `models/`, `evaluation/`.
- **Sub-packages expose a flat API via `__init__.py`** (e.g., `from federated_baseline_cf.models import BasicMF, BPRMF, MSELoss, BPRLoss`).
- **No barrel re-exports at the top package level** — callers import from the submodule.
- Packaging is declared in each module's `pyproject.toml` under `[tool.hatch.build.targets.wheel] packages = ["."]` using the Hatchling build backend.
- Install locally with `pip install -e .` from inside the module directory.
## Branch Naming and Commits
- `feat/<short-name>` — new features / experiments.
- `fix/<short-name>` — bug fixes.
- `chore/<short-name>` — formatting, refactors, housekeeping.
## Tooling (What Is NOT Configured)
- No formatter is configured (no `.prettierrc`, no `[tool.black]`, no `[tool.ruff]`, no `[tool.isort]` section in any `pyproject.toml`).
- No linter (no `.flake8`, no `.ruff.toml`, no `pylint` config).
- No `.pre-commit-config.yaml`.
- No `setup.cfg`.
- Existing code is nonetheless consistent with 4-space indents, ~100 column line length, and PEP 8 naming — match that by eye when editing.
- The only CI workflow is `.github/workflows/claude.yml` (Claude Code PR assistant); no test/lint CI pipeline.

## Architecture

## Pattern Overview
- **Flower Grid messaging API** — `grid.send_and_receive(messages)` drives a synchronous round-by-round loop in each `server_app.py`. Clients respond to `@app.train()` and `@app.evaluate()` hooks via `flwr.clientapp.ClientApp`.
- **Parallel comparative design** — Four sibling directories (`federated-baseline-cf/`, `federated-pfedrec/`, `federated-personalized-cf/`, `federated-adaptive-personalized-cf/`) implement the same MovieLens 1M collaborative-filtering task with varying global/local parameter boundaries. The **personalization boundary is the primary architectural differentiator**, not the framework code.
- **Client-side local caching** — All personalized variants persist LOCAL parameters to disk (`.embedding_cache/partition_{id}/`) between rounds, so the same partition resumes with its private state even though Flower simulation may recycle client processes.
- **BPR-MF is the canonical model** — All three BPR-capable modules share a nearly identical `BPRMF` class (file: `models/bpr_mf.py`); the personalized and adaptive variants extend it with `get_global_parameters()` / `set_global_parameters()` / `get_local_parameters()` / `set_local_parameters()` methods that classify weights as `_GLOBAL_PARAMS` or `_LOCAL_PARAMS`.
- **PFedRec deviates from MF** — `federated-pfedrec/` uses `PFedRecMLP` (file: `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py`) which has NO user embedding; personalization is a per-user Linear `affine_output` layer cached at `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt`.
- **Reference implementation preserved** — `IJCAI-23-PFedRec/` contains the unmodified upstream PFedRec code (`train.py`, `engine.py`, `mlp.py`, `data.py`) used as a calibration reference for `federated-pfedrec/`.
## Layers
- Purpose: Download ML-1M, build `user2idx`/`item2idx` mappings, partition into clients, build `MovieLensDataset` + `DataLoader` per partition.
- Location: `federated-baseline-cf/federated_baseline_cf/dataset.py`, `federated-pfedrec/federated_pfedrec/dataset.py`, `federated-personalized-cf/federated_personalized_cf/dataset.py`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py`
- Contains: `load_movielens_1m()`, `dirichlet_partition_users()`, `natural_partition_users()` (cross-device 1-user-per-client), `create_leave_one_out_split()`, `create_global_mappings()`, `load_partition_data()`, `load_full_data()`
- Module-level `_partition_cache` keyed by `f"{num_partitions}_{alpha}_{split_mode}_{partition_mode}"` prevents re-partitioning per client in single-process simulation.
- The adaptive variant additionally computes `user_stats` (n_interactions, genre_entropy, n_unique_items, rating_std) used for alpha.
- Depends on: `torch.utils.data.Dataset`, `pandas`, `numpy`, raw CSVs in `data/ml-1m/`
- Used by: `task.py` via `load_partition_data()` and `load_full_data()`
- Purpose: Model factory + train/test/eval_ranking functions. Module-level `_dataset_cache` remembers `num_users` / `num_items` after the first `load_data` call.
- Location: `federated-baseline-cf/federated_baseline_cf/task.py`, `federated-pfedrec/federated_pfedrec/task.py`, `federated-personalized-cf/federated_personalized_cf/task.py`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py`
- Contains: `load_data()`, `get_model()`, `train()` (dispatch on `model-type`), `train_basic_mf()`, `train_bpr_mf()`, `test()`, `evaluate_ranking()` (all-items ranking), `evaluate_ranking_sampled()` (leave-one-out + 99 negatives, NCF protocol).
- PFedRec variant instead has `prepare_user_train_data()`, `train_pfedrec_single_user()`, `evaluate_pfedrec_sampled()`, `test_pfedrec()` — trained per-user inside a partition.
- Adaptive variant adds `compute_client_alpha()`, `compute_per_user_alpha()`, `get_user_stats()`.
- Depends on: `models/`, `dataset.py`
- Used by: `client_app.py` (both train and evaluate), `server_app.py` (centralized eval in baseline only)
- Purpose: Implements `@app.train()` and `@app.evaluate()` message handlers. Drives the per-round split-learning protocol on each client.
- Location: one per module, e.g., `federated-personalized-cf/federated_personalized_cf/client_app.py`
- Responsibilities diverge per module:
- All variants include a `get_device()` helper with CUDA compatibility fallback (handles RTX 5090 vs old PyTorch).
- All personalized variants define disk caching helpers: `get_cache_dir()`, `save_local_user_embeddings()` / `save_user_local_params()`, `load_local_user_embeddings()` / `load_user_local_params()`, `clear_embedding_cache()`.
- Purpose: Owns the federated training loop, manages wandb, writes results JSON, runs early stopping, handles centralized evaluation (baseline only).
- Location: one per module, e.g., `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py`
- Uses `flwr.serverapp.Grid` to create/send/receive messages. Defines a `DummyClientProxy` shim (`ClientProxy` subclass returning `None` from all methods) so it can feed `FitRes` tuples into Flower's strategy aggregation functions.
- Round structure (consistent across all four modules):
- Writes final JSON to `../results/federated/` (baseline), `../results/federated/personalized/`, `../results/federated/pfedrec/`, `../results/federated/` depending on module.
- Purpose: Subclass `flwr.server.strategy.FedAvg` / `FedProx` to encode which param keys are global.
- Files: `federated-pfedrec/federated_pfedrec/strategy.py`, `federated-personalized-cf/federated_personalized_cf/strategy.py`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py`
- All three define module-level frozensets `GLOBAL_PARAM_KEYS` and `LOCAL_PARAM_KEYS` (contents differ per module — see boundary table below).
- The `SplitFedAvg` / `SplitFedProx` classes are thin wrappers; **the split actually happens in `client_app.py`** which only sends global params. They exist mainly to carry `_is_split_learning = True` and (for adaptive) to implement prototype aggregation.
- The adaptive variant's `SplitFedAvg.aggregate_fit()` overrides the base, calling `super().aggregate_fit()` then `self._aggregate_prototypes(results)` — it extracts `user_prototype` lists from client metrics, does weighted mean, and applies EMA: `p_global = momentum * p_old + (1 - momentum) * p_new`.
- Baseline module uses `FedAvg` / `FedProx` directly from `flwr.server.strategy` (no custom strategy file).
- Purpose: `nn.Module` subclasses plus loss functions. Split-aware models expose `_GLOBAL_PARAMS` / `_LOCAL_PARAMS` tuples and matching `get_/set_global_parameters()` / `get_/set_local_parameters()` methods.
- Layout per module:
- Location: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/`
- Files: `alpha_analysis.py` (`AlphaAnalyzer`, `AlphaStatistics`), `user_groups.py` (`UserGroupConfig` with sparse/medium/dense buckets, `classify_user_group()`, `aggregate_metrics_by_group()`).
## Personalization Boundary Matrix (the core architectural axis)
| Module | `GLOBAL_PARAM_KEYS` (shared) | `LOCAL_PARAM_KEYS` (private, cached) | Comm savings |
|--------|------------------------------|--------------------------------------|--------------|
| `federated-baseline-cf` | **all** state_dict keys (user + item embeddings, all biases) | (none) | baseline ~874K params |
| `federated-pfedrec` | `embedding_item.weight` | `affine_output.weight`, `affine_output.bias` (per-user, one file per user) | ~118K params/round |
| `federated-personalized-cf` | `item_embeddings.weight`, `item_bias.weight`, `global_bias` | `user_embeddings.weight`, `user_bias.weight` | ~485K params/round (-44%) |
| `federated-adaptive-personalized-cf` | same as personalized + prototype EMA state on server | same as personalized + `personal_mlp.*`, `fusion_gate/layer`, `logit_alpha` (per-user), `item_perturbation` | ~38% transmitted |
## Data Flow
- **Server state**: strategy object holds aggregation state (`self._global_prototype` EMA in adaptive module); `arrays: ArrayRecord` is rebuilt each round by mapping `parameters_to_ndarrays(aggregated_params)` back onto `list(arrays.to_torch_state_dict().keys())`.
- **Client state**: `_device_cache`, `_partition_cache`, `_dataset_cache`, `_user_stats_cache`, `_item_popularity_cache` — module-level globals that survive across same-process Flower simulation clients. Disk-side state: `.embedding_cache/partition_{id}/user_embeddings.pt` (or per-user `affine_output.pt` for PFedRec) loaded/saved via atomic temp-file-plus-rename pattern.
## Key Abstractions
- Each `nn.Module` subclass in the personalized/adaptive/pfedrec variants owns class-level tuples `_GLOBAL_PARAMS_*` and `_LOCAL_PARAMS_*` (e.g., `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:41-46`).
- Instance properties `_GLOBAL_PARAMS` / `_LOCAL_PARAMS` are resolved in `__init__` based on `use_bias` flag.
- Methods `get_global_parameters()`, `set_global_parameters(dict)`, `get_local_parameters()`, `set_local_parameters(dict, strict=False)`, `get_global_parameter_names()` are a uniform interface (see `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:342-466`). `set_local_parameters` has shape-mismatch handling: when saved tensor has fewer users than current, it partially loads `current_state[name][:num_saved] = saved_tensor`.
- `SplitFedAvg`/`SplitFedProx` subclasses in `strategy.py` are thin: the actual split happens in `client_app.py` where `model.get_global_parameters()` returns only the keys the server expects, and `ArrayRecord(global_params)` is sent back.
- Factory `create_alpha_computer(config: AlphaConfig, hc_config: HierarchicalConditionalAlphaConfig)` returns one of three classes:
- Per-client alpha clipped to `[min_alpha=0.1, max_alpha=0.95]`.
- `compute_client_alpha()` in `task.py` takes aggregated `user_stats` and returns a scalar; `compute_per_user_alpha()` returns a per-user tensor (used when `enable-per-user-alpha=true`).
- `DualPersonalizedBPRMF` holds both a standard MF pathway and a local `PersonalMLP` pathway.
- Level 1 (statistical): `p_effective = alpha * p_local + (1-alpha) * p_global` (called via `model.set_alpha(...)` and `model.set_global_prototype(...)` before `forward()`).
- Level 2 (neural): `PersonalMLP` scores element-wise product `p_effective ⊙ q_item`.
- Fusion (`fusion_type` = `add` | `gate` | `concat`): `add` = simple sum, `gate` = learnable sigmoid gate, `concat` = `Linear([score_cf; score_mlp])`.
- `PersonalMLP` weights (`personal_mlp.*`) and `fusion_gate/layer` are LOCAL — never transmitted.
- Server state in `SplitFedAvg._global_prototype` (numpy array).
- Each client computes `model.compute_user_prototype()` after training and serializes as a list into `FitRes.metrics[USER_PROTOTYPE_KEY]` (key = `"user_prototype"`, defined `strategy.py:46`).
- Server's `_aggregate_prototypes(results)`: `new_prototype = Σ(prototype * num_examples) / total_weight`, then `p_global = momentum * p_old + (1 - momentum) * new_prototype` with `momentum=0.9`. Missing on first round → `p_global = new_prototype` directly.
- Next round, server attaches `train_config_dict["global_prototype"] = global_prototype.tolist()` to the message; client converts back to tensor via `torch.tensor(global_prototype_list, dtype=torch.float32)` and calls `model.set_global_prototype(tensor)`.
- Stored as `logit_alpha` Embedding (LOCAL) inside `DualPersonalizedBPRMF`.
- Initialized from heuristic via `torch.logit(heuristic_alpha)`, refined by BPR gradient descent. Cached/restored through `get_local_parameters()`.
- `_item_perturbation = nn.Embedding(num_items, embedding_dim)` zero-initialized.
- `q_effective[i] = q_global[i] + perturbation[i]`, L2-regularized in loss: `reg * ||perturbation||^2`.
- LOCAL — never sent to server.
- `InfoNCEContrastiveLoss` (in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/losses.py`).
- `L_total = L_BPR + lambda * L_contrastive + reg * ||perturbation||^2`.
- Positive pair: `(p_local[u], p_effective[u])`; negatives: other users in batch.
## Entry Points
- Configured in each `pyproject.toml` under `[tool.flwr.app.components]`:
- Federation defined in `[tool.flwr.federations.local-simulation]` (or `local-sim-gpu`) with `options.num-supernodes = 5`.
- Runtime overrides: `flwr run . --run-config "key=value other-key=value"` reads into `context.run_config`.
- `@app.main()` in `server_app.py` is the server entrypoint; `@app.train()` and `@app.evaluate()` in `client_app.py` are the client entrypoints.
- `centralize_baseline_ncf.py` at project root — Keras/TensorFlow NCF baseline that writes to `results/centralized/ncf_baseline_results.json`.
- `centralized_baseline_svd.ipynb` — Surprise-SVD notebook writing to `results/centralized/svd_baseline_results.json`.
- `scripts/run_all_baselines.sh`, `scripts/run_baseline_sweep_loo.sh`, `scripts/compare_all_results.py` — orchestrate multiple `flwr run` invocations.
- Per-module `scripts/run_fedprox_sweep.sh` and `scripts/analyze_sweep_results.py`.
- `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py` — W&B sweep agent wired via `sweep.yaml` (Bayesian + Hyperband).
- Per-module `visualize_partitions.py`, `test_dataset.py`, `test_models.py` — standalone validation scripts at each module root.
## Error Handling
- `client_app.py`: every `@app.train()` / `@app.evaluate()` handler wraps the full train/eval path; exceptions propagate as Flower `Message.has_error() == True`.
- `server_app.py`: for each response, `if response.has_error(): print(warning); continue` (see `federated-baseline-cf/federated_baseline_cf/server_app.py:325-327`). Failed clients are excluded from `fit_results` and metrics aggregation.
- CUDA fallback: `get_device()` in every `client_app.py` tests `torch.zeros(1).cuda()` and falls back to CPU on `RuntimeError` (handles RTX 5090 + old PyTorch mismatch).
- Atomic cache saves: `save_local_user_embeddings()` uses `tempfile.mkstemp` + `torch.save` + `os.replace` pattern; on exception, temp file is `os.unlink`-ed.
- Shape mismatch in local params: `set_local_parameters(..., strict=False)` does partial load when saved tensor is smaller than current (see `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:436-455`).
- Early stopping shielding: `if early_stopper is not None and round_eval_metrics:` — never triggers on empty eval rounds.
## Cross-Cutting Concerns
- Per-round stdout with `print(f"{'='*50}\nRound {round_num}/{num_rounds}\n{'='*50}")` headers.
- W&B integrated in each `server_app.py`: `wandb.init(project=..., entity=..., config=...)` at start, `wandb.log(round_metrics, step=round_num)` each round, `wandb.run.summary[f"final/{key}"] = value` at end.
- wandb artefacts in `federated-<module>/wandb/` per module; most recent in `latest-run` symlink.
- `AlphaConfig.__post_init__()` validates `0 <= min_alpha < max_alpha <= 1`, positive thresholds, method in allowed set, and (for multi-factor) `factor_weights` summing to 1.0 ± 0.01.
- `UserGroupConfig.__post_init__()` validates adjacency `sparse[1] == medium[0]` and `medium[1] == dense[0]`.
- Primary metric: `sampled_ndcg@10` (NCF protocol — 1 positive + 99 negatives).
- Secondary: `hit_rate@{5,10,20}`, `mrr`, `coverage@K`, `novelty@K`.
- Rating prediction (`rmse`, `mae`) reported but not optimized under BPR.
- Results JSON structure: `{model_name, dataset, federated_config, early_stopping, timestamp, final_metrics, training_rounds}`.
