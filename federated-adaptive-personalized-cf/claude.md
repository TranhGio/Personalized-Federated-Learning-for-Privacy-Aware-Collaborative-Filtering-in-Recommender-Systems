# Federated Adaptive Personalized Collaborative Filtering

> Thesis contribution: Hierarchical Conditional Alpha + Dual-Level Personalization
> MovieLens 1M | Split Learning | FedAvg/FedProx | BPR-MF | Global Prototype

## Directory Structure

```
federated_adaptive_personalized_cf/
  dataset.py                       - MovieLens loading, Dirichlet partitioning
  task.py                          - Training, evaluation, alpha computation
  client_app.py                    - Split learning client + adaptive alpha
  server_app.py                    - Server with wandb + alpha analysis
  strategy.py                      - SplitFedAvg/FedProx with prototype aggregation
  models/
    adaptive_alpha.py              - DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha
    dual_personalized_bpr_mf.py    - DualPersonalizedBPRMF (novel architecture)
    bpr_mf.py                      - Standard BPRMF with split learning
    basic_mf.py                    - BasicMF
    losses.py                      - Loss implementations
  evaluation/
    alpha_analysis.py              - Alpha distribution & correlation analysis
    user_groups.py                 - Per-group metrics (sparse/medium/dense)
```

## Key Innovations

### 1. Hierarchical Conditional Alpha (recommended, `alpha-method=hierarchical_conditional`)

Two-stage alpha computation addressing multi-factor conflicts:

**Stage 1 - Hierarchical Aggregation:**
- `data_volume = sqrt(f_quantity * f_coverage)` (geometric mean - resolves quantity-coverage redundancy, corr 0.8-1.0)
- `preference_quality = harmonic_mean(f_diversity, f_consistency)` (resolves diversity-consistency contradiction, corr -0.3 to -0.5)
- `base_alpha = 0.55 * data_volume + 0.45 * preference_quality`

**Stage 2 - Conditional Rules:**
- Sparse users (n < 20): penalty up to 50%
- Niche specialists (low diversity, high quantity): +0.15 bonus
- Inconsistent raters (f_s < 0.3): 30% penalty
- Completionists (high coverage, low diversity): +0.1 bonus

### 2. Multi-Factor Alpha (alternative, `alpha-method=multi_factor`)

`alpha = 0.40*f_quantity + 0.25*f_diversity + 0.20*f_coverage + 0.15*f_consistency`

Known issues: quantity-coverage redundancy, diversity-consistency contradiction (why hierarchical conditional was created).

### 3. Dual-Level Personalization (`model-type=dual`)

- **Level 1 (Statistical)**: `p_effective = alpha * p_local + (1-alpha) * p_global`
- **Level 2 (Neural)**: PersonalMLP scores element-wise product of embeddings
- **Fusion**: `add`, `gate` (learnable sigma), or `concat` (Linear([cf; mlp]))

### 4. Global Prototype

Server-side EMA prototype: `p_global = 0.9 * p_old + 0.1 * weighted_avg(client_protos)`
Helps sparse users by providing population-average user representation.

### 5. Per-User Learned Alpha (`enable-per-user-alpha=true`)

Replaces single client-level alpha scalar with per-user learnable alpha.
- Initialized from hierarchical conditional heuristic via logit transform
- Refined by BPR gradient descent: `logit_alpha[u] -> sigmoid -> alpha[u]`
- `p_effective[u] = alpha[u] * p_local[u] + (1-alpha[u]) * p_global`
- LOCAL parameter (never sent to server)

### 6. Dual-Side Item Perturbation (`enable-item-perturbation=true`)

Local item embedding adjustments: `q_effective[i] = q_global[i] + perturbation[i]`
- Zero-initialized (no effect initially), refined by gradient descent
- L2 regularized: `reg * ||perturbation||^2` added to loss
- LOCAL parameter (never sent to server)

### 7. Contrastive Local-Global Alignment (`contrastive-lambda > 0`)

InfoNCE auxiliary loss contrasting local vs blended user embeddings.
- `L_total = L_BPR + lambda * L_contrastive + reg * ||perturbation||^2`
- Positive: (p_local[u], p_effective[u]), Negatives: other users in batch
- Provides gradient signal to per-user alpha through effective embeddings

## Parameter Classification

| Parameter | Type | Privacy |
|-----------|------|---------|
| `user_embeddings`, `user_bias` | Local | Private (never sent) |
| `personal_mlp.*`, `fusion_gate/layer` | Local | Private (never sent) |
| `logit_alpha` (per-user alpha) | Local | Private (never sent) |
| `item_perturbation` | Local | Private (never sent) |
| `item_embeddings`, `item_bias`, `global_bias` | Global | Shared each round |

~38% of parameters transmitted per round (unchanged by new techniques).

## Commands

```bash
# Default: BPR-MF with hierarchical conditional alpha
flwr run .

# Dual-level with concat fusion
flwr run . --run-config "model-type=dual fusion-type=concat"

# FedProx
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"

# Alpha method comparison
flwr run . --run-config "alpha-method=hierarchical_conditional"
flwr run . --run-config "alpha-method=multi_factor"
flwr run . --run-config "alpha-method=data_quantity"

# W&B sweep
wandb sweep sweep.yaml
wandb agent <ENTITY>/federated-adaptive-personalized-cf/<SWEEP_ID>

# Early stopping
flwr run . --run-config "early-stopping-enabled=true early-stopping-patience=10"

# Next-Gen Personalization Techniques
flwr run . --run-config "enable-per-user-alpha=true"
flwr run . --run-config "enable-item-perturbation=true item-perturbation-reg=0.01"
flwr run . --run-config "contrastive-lambda=0.1 contrastive-tau=0.1"

# Full combo (all three techniques)
flwr run . --run-config "enable-per-user-alpha=true enable-item-perturbation=true contrastive-lambda=0.1"
```

## Key Config (pyproject.toml)

- `num-server-rounds`: 50, `local-epochs`: 12
- `model-type`: "bpr", "basic", or "dual"
- `alpha-method`: "hierarchical_conditional" (default), "multi_factor", "data_quantity"
- `mlp-hidden-dims`: "512,256,128", `fusion-type`: "concat"
- `prototype-momentum`: 0.9
- `early-stopping-metric`: "sampled_ndcg@10"
- `enable-per-user-alpha`: false, `enable-item-perturbation`: false
- `contrastive-lambda`: 0.0, `contrastive-tau`: 0.1
- User groups: sparse (0-30), medium (30-100), dense (100+)

## Factory Pattern

```python
from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    create_alpha_computer, AlphaConfig, HierarchicalConditionalAlphaConfig
)
config = AlphaConfig(method="hierarchical_conditional")
hc_config = HierarchicalConditionalAlphaConfig()
alpha_computer = create_alpha_computer(config, hc_config=hc_config)
alpha = alpha_computer.compute_from_stats(user_stats)
```

## Gotchas

- Alpha range clipped to [0.1, 0.95] - never fully local or fully global
- Hierarchical conditional alpha needs all 4 user stats: n_interactions, genre_entropy, n_unique_items, rating_std
- DualPersonalizedBPRMF has `set_alpha()` and `set_global_prototype()` methods - must be called before forward pass
- PersonalMLP is LOCAL (client-specific, never aggregated)
- Early stopping monitors `sampled_ndcg@10` by default
- `sweep.yaml` uses Bayesian optimization with Hyperband early termination
- Per-user alpha and item perturbation are LOCAL, auto-cached via `get/set_local_parameters()`
- Per-user alpha is initialized from heuristic only on first call; subsequent rounds load from cache
- All three next-gen techniques are disabled by default (backward-compatible)
