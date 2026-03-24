# Movie Recommendation System - Federated Learning

Master thesis: **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**
Author: Dang Vinh | Dataset: MovieLens 1M | Framework: Flower (flwr) + PyTorch

## Project Structure

Three federated implementations (progression: baseline -> personalized -> adaptive):

| Module | Approach | Key Difference |
|--------|----------|----------------|
| `federated-baseline-cf/` | All params global (FedAvg/FedProx) | Lower-bound baseline |
| `federated-personalized-cf/` | Split learning (local user embeddings) | Privacy + personalization |
| `federated-adaptive-personalized-cf/` | Hierarchical conditional alpha + dual-level | Thesis contribution |

Centralized baselines: `centralized_baseline_svd.ipynb`, `centralize_baseline_ncf.py`
Results: `results/centralized/` and `results/federated/`

## Commands

```bash
# Run federated experiments (from each subdirectory)
cd federated-adaptive-personalized-cf && flwr run .
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "model-type=dual fusion-type=concat alpha-method=hierarchical_conditional"

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
- @federated-personalized-cf/claude.md
- @federated-adaptive-personalized-cf/claude.md

## MCP Usage

- Use **Context7 MCP** for library/API documentation and code examples
- Use **Pal MCP** for deep thinking, planning, code reviews, and multi-model consensus

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
