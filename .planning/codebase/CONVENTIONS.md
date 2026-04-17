# Coding Conventions

**Analysis Date:** 2026-04-17

This codebase is a Python 3.9+ research codebase built on Flower (flwr) + PyTorch. Conventions below are extracted from `CLAUDE.md`, the four per-module `claude.md` files, and the actual source style in the four `federated-*-cf/` module packages.

## Python Target

- **Language version:** Python 3.9+ (`CLAUDE.md` > "Code Standards")
- **Typing style:** `typing.Dict`, `typing.List`, `typing.Tuple`, `typing.Optional`, `typing.Union` (pre-3.10 syntax is used throughout, e.g. `federated-baseline-cf/federated_baseline_cf/task.py:6`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:17`). Do NOT introduce `X | Y` or `list[int]` — match the existing `typing` module syntax for consistency.

## Naming Patterns

**Files and modules:**
- Package dirs use `snake_case` matching the Flower app name (dashes replaced with underscores): `federated-baseline-cf/` -> `federated_baseline_cf/`.
- Python files: lowercase `snake_case.py` (e.g., `server_app.py`, `client_app.py`, `dataset.py`, `task.py`, `strategy.py`, `adaptive_alpha.py`, `dual_personalized_bpr_mf.py`, `early_stopping.py`).
- Test scripts: top-level per-module `test_dataset.py`, `test_models.py` (run directly as `python test_dataset.py` — not collected by pytest).
- Shell scripts: `run_*.sh` (sweeps), `sweep_commands.sh`.

**Functions / methods:**
- `snake_case` for functions and methods: `load_partition_data`, `compute_client_alpha`, `get_local_parameters`, `set_global_parameters`, `dirichlet_partition_users`, `evaluate_ranking_sampled`.
- Private / module-internal helpers: single leading underscore (`_compute_quantity_factor`, `_dataset_cache`, `_device_cache`, `_MODULE_DIR`).

**Classes:**
- `PascalCase`: `BasicMF`, `BPRMF`, `MovieLensDataset`, `AlphaConfig`, `HierarchicalConditionalAlpha`, `DualPersonalizedBPRMF`, `SplitFedAvg`, `SplitFedProx`, `EarlyStopping`, `EarlyStoppingState`, `AlphaAnalyzer`, `UserGroupConfig`.
- Acronyms stay uppercase inside the name (`MF`, `BPR`, `MLP`, `MSE`, `CF`).

**Variables:**
- `snake_case` everywhere (`num_users`, `num_items`, `embedding_dim`, `global_model`, `local_params`, `ratings_df`, `user2idx`, `item2idx`, `partition_id`, `proximal_mu`).
- DataFrames end with `_df`; PyTorch tensors have no suffix; lookup dicts use `a2b` form (`user2idx`, `item2idx`).

**Constants / keys:**
- Module-level constants: `UPPER_SNAKE_CASE` (`_DEFAULT_DATA_DIR`, `_MODULE_DIR` — leading underscore because they are module-private).
- Frozenset key registries: `GLOBAL_PARAM_KEYS`, `LOCAL_PARAM_KEYS`, `USER_PROTOTYPE_KEY` in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:34-46` and `federated-pfedrec/federated_pfedrec/strategy.py:15-19`.

## Notation Convention (Thesis / Algorithm)

From `CLAUDE.md` > "Notation Convention (use consistently across ALL code and docs)". Use these symbols in docstrings, comments, and paper-facing text:

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

Two styles coexist; both are acceptable but **NumPy style is preferred** per `CLAUDE.md`:

- **NumPy-style (preferred for public API):** `federated-baseline-cf/federated_baseline_cf/task.py:30-55` uses `Parameters\n----------` and `Returns\n-------` sections.
- **Google-style (common in models and older code):** `federated-baseline-cf/federated_baseline_cf/models/bpr_mf.py:42-55` and `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py:32-46` use `Args:` / `Returns:` / `Attributes:` sections.

Module-level docstrings are triple-quoted summaries, often followed by a structured block describing architecture, param classification, or algorithm stages (e.g. `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:1-14`, `.../models/adaptive_alpha.py:1-23`).

Docstring requirements (from `CLAUDE.md` > "Code Standards"):
- Required on **all public functions**.
- Parameter/return types are duplicated in both the type hint and the docstring prose.

## Type Hints

Required on **all function signatures** (`CLAUDE.md` > "Code Standards"). Observed patterns:

```python
# federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py
def create_alpha_computer(
    config: Optional[AlphaConfig] = None,
    hc_config: Optional[HierarchicalConditionalAlphaConfig] = None,
) -> AlphaComputer: ...

# federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py
def set_local_parameters(self, local_state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]: ...
```

- Prefer `Optional[T]` over `Union[T, None]`.
- `Dict[str, float]`, `List[Tuple[int, int]]` — fully qualified generics, not bare `dict`/`list`.
- Union aliases are declared once at module bottom (e.g., `AlphaComputer = Union[DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha]` in `adaptive_alpha.py:601`).

## Config via Dataclasses (not loose dicts)

`CLAUDE.md` > "Code Standards" mandates dataclass configs. Follow the existing pattern:

```python
# federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py:30-96
@dataclass
class AlphaConfig:
    method: str = "data_quantity"
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    factor_weights: Dict[str, float] = field(default_factory=lambda: {...})

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.min_alpha < self.max_alpha <= 1:
            raise ValueError(...)
```

Other dataclass configs:
- `HierarchicalConditionalAlphaConfig` — `adaptive_alpha.py:99-174`
- `UserGroupConfig` — `evaluation/user_groups.py:18-37`
- `AlphaStatistics`, `HierarchicalAlphaStatistics` — `evaluation/alpha_analysis.py:13`, `:256`
- `EarlyStoppingState` — `early_stopping.py:12-21`

Rules:
- Every dataclass with non-trivial invariants implements `__post_init__` and raises `ValueError` with a descriptive message on invalid input.
- Mutable defaults use `field(default_factory=...)`.

## Factory Pattern for Pluggable Strategies

Use a top-level factory function to select between interchangeable implementations, keyed by a string config value:

```python
# federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py:604-641
def create_alpha_computer(
    config: Optional[AlphaConfig] = None,
    hc_config: Optional[HierarchicalConditionalAlphaConfig] = None,
) -> AlphaComputer:
    if config is None:
        config = AlphaConfig()
    method = config.method.lower()
    if method == "data_quantity":
        return DataQuantityAlpha(config)
    elif method == "multi_factor":
        return MultiFactorAlpha(config)
    elif method == "hierarchical_conditional":
        return HierarchicalConditionalAlpha(hc_config or HierarchicalConditionalAlphaConfig(...))
    else:
        raise ValueError(f"Unknown alpha method: {method}")
```

The same shape is used for model selection (`get_model(model_type=...)` in every module's `task.py`, e.g. `federated-baseline-cf/federated_baseline_cf/task.py:76-120`) and strategy selection in `server_app.py` (`if strategy_name == "fedprox": SplitFedProx(...) else: SplitFedAvg(...)`).

## Split-Learning Parameter Protocol

Models that participate in split learning MUST expose four methods:
- `get_global_parameters() -> OrderedDict`
- `set_global_parameters(global_state_dict: Dict[str, torch.Tensor]) -> None`
- `get_local_parameters() -> OrderedDict`
- `set_local_parameters(local_state_dict, strict=False) -> Tuple[List[str], List[str]]` (returns `(loaded_keys, missing_keys)`)

Reference implementations:
- `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:342-450`
- `federated-personalized-cf/federated_personalized_cf/models/basic_mf.py:221-330`
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py` (extends the pattern for `personal_mlp.*`, `fusion_*`, `logit_alpha`, `item_perturbation`).

Each module's `strategy.py` defines module-level `frozenset`s that enumerate GLOBAL vs LOCAL keys — any new parameter must be classified explicitly:

```python
# federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:34-46
GLOBAL_PARAM_KEYS = frozenset(['item_embeddings.weight', 'item_bias.weight', 'global_bias'])
LOCAL_PARAM_KEYS = frozenset(['user_embeddings.weight', 'user_bias.weight'])
USER_PROTOTYPE_KEY = 'user_prototype'
```

## Import Organization

Standard 3-group order (PEP 8), separated by blank lines:

```python
# 1. stdlib
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

# 2. third-party
import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# 3. local / first-party (absolute, fully-qualified)
from federated_adaptive_personalized_cf.task import get_model, load_data
from federated_adaptive_personalized_cf.strategy import USER_PROTOTYPE_KEY
```

Example: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:13-31`.

- **Absolute imports only** — no `from .task import ...`. All internal imports use the full package name.
- No star imports.

## Error Handling

- Raise `ValueError` for invalid config / input with a message containing the bad value and the expected range. Example: `adaptive_alpha.py:73-95`.
- Raise `ValueError` from factory functions on unknown enum strings (see `create_alpha_computer`, `get_model`).
- `try/except` is reserved for I/O and environment probing (CUDA availability check, atomic file save, cache load with shape mismatch). Example: `federated-baseline-cf/federated_baseline_cf/client_app.py:19-37` (safe CUDA detection); `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:96-120` (atomic embedding save via `tempfile.mkstemp` + rename).
- When loading cached state with possibly-mismatched shapes, use `strict=False` and return `(loaded_keys, missing_keys)` so the caller can log what was recovered.

## Logging and Metric Reporting

This codebase does **NOT** use the `logging` module for application logs. Observed channels:

1. **`print()` statements** — used heavily (100+ occurrences per module) for per-round progress, config dumps, and milestones. Section banners use `"=" * 80` or `"=" * 60`.
2. **Weights & Biases (wandb)** — primary metric tracking. Pattern in `server_app.py`:
   ```python
   wandb_enabled = context.run_config.get("wandb-enabled", False)
   if wandb_enabled:
       wandb.init(project=..., entity=..., name=..., config=wandb_config)
       ...
       wandb.log(round_metrics, step=round_num)
       ...
       wandb.finish()
   ```
   See `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:250-290` and `:503-520`.
3. **Flower's logger** — used inside strategies only, imported as `from flwr.common.logger import log` and `from logging import WARNING` (e.g., `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py:18,27`).
4. **Results JSON files** — every run dumps a full experiment record to `results/federated/<module>/...json` via `json.dump(results_data, f, indent=4)`:
   - `federated-baseline-cf/federated_baseline_cf/server_app.py:580`
   - `federated-personalized-cf/federated_personalized_cf/server_app.py:533`
   - `federated-pfedrec/federated_pfedrec/server_app.py:461`
   - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:716`

**Note on `CLAUDE.md` vs reality:** `CLAUDE.md` > "Code Standards" states "Log all metrics to CSV + console per round". The actual implementation uses **JSON + console + W&B**. Prefer matching the existing JSON pattern when adding new metric persistence; if CSV is truly required, add it alongside the JSON dump rather than replacing it.

## Configuration Surface

All tunable hyperparameters live in `pyproject.toml` under `[tool.flwr.app.config]` and are fetched via `context.run_config.get("<key>", <default>)`:

```python
# federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py (45 usages)
num_server_rounds = context.run_config.get("num-server-rounds", 50)
strategy_name = context.run_config.get("strategy", "fedprox")
model_type = context.run_config.get("model-type", "dual")
```

Rules:
- **Keys use kebab-case** in `pyproject.toml` (`num-server-rounds`, `model-type`, `alpha-method`, `enable-per-user-alpha`).
- **Variables use snake_case** in Python (`num_server_rounds`, `model_type`, `alpha_method`, `enable_per_user_alpha`).
- Always pass a default to `.get()` so a missing key never raises.
- Runtime overrides go via `flwr run . --run-config "key1=value1 key2=value2"`.
- No `.env` files, no `python-dotenv`, no `argparse` for core experiments.
- W&B sweep configs live in `federated-adaptive-personalized-cf/sweep.yaml` and use `snake_case` keys (Flower strips the dashes when `wandb` agents call the sweep runner).

## Reproducibility (Seed + Config)

`CLAUDE.md` > "Code Standards": "Experiments reproducible via seed + config file".

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

From `CLAUDE.md` > "Key Conventions":
- `feat/<short-name>` — new features / experiments.
- `fix/<short-name>` — bug fixes.
- `chore/<short-name>` — formatting, refactors, housekeeping.

Recent examples (from `git log`): `feat/try_to_run_the_baseline`, commits like `feat: result of run_baseline_sweep_loo.sh`, `chore: format .json files`.

## Tooling (What Is NOT Configured)

- No formatter is configured (no `.prettierrc`, no `[tool.black]`, no `[tool.ruff]`, no `[tool.isort]` section in any `pyproject.toml`).
- No linter (no `.flake8`, no `.ruff.toml`, no `pylint` config).
- No `.pre-commit-config.yaml`.
- No `setup.cfg`.
- Existing code is nonetheless consistent with 4-space indents, ~100 column line length, and PEP 8 naming — match that by eye when editing.
- The only CI workflow is `.github/workflows/claude.yml` (Claude Code PR assistant); no test/lint CI pipeline.

---

*Convention analysis: 2026-04-17*
