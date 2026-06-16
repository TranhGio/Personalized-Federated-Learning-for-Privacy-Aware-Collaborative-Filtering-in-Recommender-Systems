# Movie Recommendation System — Federated Learning

Master's thesis: **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**
Author: Dang Vinh | Dataset: MovieLens 1M | Framework: Flower (flwr) + PyTorch | Python 3.9+

## Thesis Claim (the bar every run must clear)

Under cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method must beat all three baselines on **NDCG@10** — including on sparse users — while PFedRec reproduces the published reference (HR@10 ≈ 0.70, NDCG@10 ≈ 0.38, within ±2 points).

If the adaptive method does not win under this protocol, the thesis contribution has to be rethought. Methodological correctness is non-negotiable — that's the whole reason for migrating off the old cross-silo `num-supernodes=5` setup.

## Four Federated Modules (the comparison)

| Module | Approach | Role in Thesis |
|---|---|---|
| `federated-baseline-cf/` | All params global (FedAvg/FedProx) | Lower bound |
| `federated-pfedrec/` | Local score fn + global item embeddings (IJCAI-23) | Calibration baseline — must reproduce paper |
| `federated-personalized-cf/` | Split learning (local user embeddings) | Privacy + personalization step |
| `federated-adaptive-personalized-cf/` | Hierarchical conditional α + dual-level | **Thesis contribution** |

Centralized comparators: `centralized_baseline_svd.ipynb`, `centralize_baseline_ncf.py`.
Reference: `IJCAI-23-PFedRec/` — upstream PFedRec code. **Do not modify.**

Per-module details live in each sub-`claude.md` (see end of this file).

## Run Experiments

```bash
# Install once per module
cd <module> && pip install -e .

# Run a federation
cd federated-adaptive-personalized-cf && flwr run .
flwr run . --run-config 'strategy="fedprox" proximal-mu=0.01'
flwr run . --run-config 'model-type="dual" fusion-type="concat" alpha-method="hierarchical_conditional"'

# W&B sweep (adaptive only)
wandb sweep sweep.yaml && wandb agent <ENTITY>/federated-adaptive-personalized-cf/<SWEEP_ID>

# Centralized comparators
python centralize_baseline_ncf.py
```

All run-config knobs live in each module's `pyproject.toml` under `[tool.flwr.app.config]` — read that file when uncertain about defaults rather than guessing.

**Config quoting rule:** `flwr --run-config` parses values as TOML. **String values must be double-quoted** inside the outer string: use `'key="string-val" int-key=42'` (single quotes outside, double inside). Numbers and booleans don't need quotes. Unquoted strings fail with `TOMLDecodeError: Invalid value`.

## Evaluation Protocol (locked)

- **Partition mode**: `natural` (cross-device, 1 user = 1 client, N=6040). `dirichlet` (cross-silo) remains as an opt-in for reproducing old runs only.
- **Eval**: leave-one-out + 99 negative samples (NCF protocol).
- **Primary metric**: `sampled_ndcg@10`. Also report `hit_rate@10`, `mrr`.
- **User groups**: sparse (0-30 interactions), medium (30-100), dense (100+) — sparse-user NDCG is what the thesis hinges on.
- BPR-MF has high RMSE by design (~2.2). Not a bug. Don't optimize for it.
- Cross-device runs use a **separate W&B project** from cross-silo so plots don't get contaminated.

## Notation (use everywhere in code and docs)

| Symbol | Meaning | Code name |
|---|---|---|
| `w` | server global model | `global_model`, `global_params` |
| `theta_i` | client i's personalized model | `personal_model_i`, `local_params` |
| `D_i` | client i's local dataset | `trainloader` / `testloader` |
| `K` | local steps per round | `local-epochs` |
| `R` | total FL rounds | `num-server-rounds` |
| `N` | total clients | `num-supernodes` |
| `C` | client sampling fraction | `fraction-train` / `fraction_fit` |
| `α` | personalization level (0=global, 1=local) | `alpha`, `alpha_i` |
| `μ` | FedProx proximal strength | `proximal-mu` |

## Code Standards (non-derivable)

- Type hints with **pre-3.10 syntax** (`typing.Dict/List/Optional`, not `X | Y` / `list[int]`) — Python 3.9 compatibility is a hard requirement.
- NumPy-style docstrings on public functions.
- Config via `@dataclass` with `__post_init__` validation, **not loose dicts**.
- Fixed seed = **42**; call `np.random.seed(seed)` + `random.seed(seed)` **inside** the sampling function (not once at process start), so each call is reproducible regardless of caller.
- Log per-round metrics to **CSV + console + W&B**. Full run config gets serialized into the results JSON for reproducibility.
- Branch prefixes: `feat/`, `fix/`, `chore/`.

## Paper Knowledge Base

When you need to understand prior work or check a reference implementation:
1. Read `Papers/digested/_INDEX.md` for the catalog.
2. Read the specific `Papers/digested/<paper_id>.md` digest.
3. Only fall back to `Papers/raw/<file>.pdf` if the digest misses a detail.

Slash commands: `/digest-paper` (one method), `/digest-survey` (review/taxonomy), `/batch-digest` (all undigested).

## MCP

- **Context7** for library/API docs and code examples (PyTorch, Flower, scikit-learn).

## Compaction Rules

When compacting, always preserve:
- Files modified this session
- Current experiment config + parameters being tested
- Any running W&B sweep IDs
- Current branch name + recent commits
- Key metric values discussed (NDCG@10, HR@K)

## Module Docs

- @federated-baseline-cf/claude.md
- @federated-pfedrec/claude.md
- @federated-personalized-cf/claude.md
- @federated-adaptive-personalized-cf/claude.md
