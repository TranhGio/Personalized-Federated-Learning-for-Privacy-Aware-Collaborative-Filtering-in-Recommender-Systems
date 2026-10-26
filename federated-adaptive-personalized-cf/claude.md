# Federated Adaptive Personalized — **Thesis Contribution**

> Hierarchical Conditional α + Dual-Level Personalization on top of split learning. EMA global prototype to help sparse users.

**Thesis bar (non-negotiable)**: this module must beat baseline, PFedRec, AND split-learning personalized on NDCG@10 — **especially on sparse users**. If it doesn't, the thesis story breaks.

## Personalization Boundary

| Scope | Params |
|---|---|
| **Local** | `user_embeddings`, `user_bias`, `personal_mlp.*`, `fusion_gate/layer`, `logit_alpha` (per-user), `item_perturbation` |
| **Global** | `item_embeddings`, `item_bias`, `global_bias` + server-side `_global_prototype` EMA |

~38% of params transmitted per round.

## The Innovations (in order of importance to the thesis)

### 1. Hierarchical Conditional α — `alpha-method=hierarchical_conditional` (default)

Two-stage; solves the multi-factor pitfalls of the naive `multi_factor` formula.

**Stage 1 — hierarchical aggregation:**
- `data_volume = sqrt(f_quantity * f_coverage)` — geometric mean resolves quantity-coverage redundancy (corr 0.8-1.0).
- `preference_quality = harmonic_mean(f_diversity, f_consistency)` — resolves diversity-consistency contradiction (corr -0.3 to -0.5).
- `base_alpha = 0.55 * data_volume + 0.45 * preference_quality`

**Stage 2 — conditional rules:**
- Sparse (n < 20): up to 50% penalty
- Niche specialists (low diversity, high quantity): +0.15
- Inconsistent raters (f_s < 0.3): 30% penalty
- Completionists (high coverage, low diversity): +0.10

α clipped to **[0.1, 0.95]** — never fully local or fully global.

Ablation comparators: `multi_factor` (naive weighted sum, has the conflicts), `data_quantity` (single-factor, trivial baseline).

### 2. Dual-Level Personalization — `model-type=dual`

- **Level 1 (statistical)**: `p_effective = α·p_local + (1-α)·p_global`
- **Level 2 (neural)**: `PersonalMLP` scores `p_effective ⊙ q_item`
- **Fusion**: `add` | `gate` | `concat` (concat is default)

`PersonalMLP` and fusion weights are **LOCAL** — never aggregated.

### 3. Global Prototype (server-side EMA)

`p_global = 0.9 · p_old + 0.1 · weighted_avg(client_protos)`. Sent down each round, used in α-blending. The mechanism by which sparse users borrow from the population.

### 4-6. Optional Next-Gen Techniques (off by default, backwards-compatible)

| Flag | Effect |
|---|---|
| `enable-per-user-alpha=true` | Replace scalar α with per-user learnable α (init from heuristic via `torch.logit`, refined by BPR gradient). Local. |
| `enable-item-perturbation=true` | `q_effective[i] = q_global[i] + perturbation[i]`. Zero-init, L2-regularized. Local. |
| `contrastive-lambda > 0` | InfoNCE on `(p_local, p_effective)` with batch users as negatives. Gives per-user α a stronger gradient signal. |

## Run

```bash
# Default: BPR-MF + hierarchical conditional α
flwr run .

# Thesis main configuration (dual + concat fusion)
flwr run . --run-config "model-type=dual fusion-type=concat"

# α-method ablation
flwr run . --run-config "alpha-method=hierarchical_conditional"   # default
flwr run . --run-config "alpha-method=multi_factor"               # ablation: known conflicts
flwr run . --run-config "alpha-method=data_quantity"              # naive baseline

# Next-gen techniques (toggle individually for ablation)
flwr run . --run-config "enable-per-user-alpha=true"
flwr run . --run-config "enable-item-perturbation=true item-perturbation-reg=0.01"
flwr run . --run-config "contrastive-lambda=0.1 contrastive-tau=0.1"

# W&B sweep (Bayesian + Hyperband)
wandb sweep sweep.yaml
wandb agent <ENTITY>/federated-adaptive-personalized-cf/<SWEEP_ID>
```

Defaults: `num-server-rounds=50`, `local-epochs=12`, `model-type=dual`, `alpha-method=hierarchical_conditional`, `fusion-type=concat`, `prototype-momentum=0.9`, `early-stopping-metric=sampled_ndcg@10`.

## Gotchas

- Hierarchical conditional α **requires all 4 user stats**: `n_interactions`, `genre_entropy`, `n_unique_items`, `rating_std`. Missing any → fall back to `data_quantity`.
- `DualPersonalizedBPRMF.set_alpha()` and `.set_global_prototype()` **MUST** be called before each `forward()`.
- Per-user α is initialized from the heuristic on first call; subsequent rounds load from cache. Don't re-initialize blindly.
- Next-gen techniques are LOCAL and auto-cached via `get_local_parameters() / set_local_parameters()`.
- Early stopping monitors `sampled_ndcg@10` by default — change only if you know what you're trading off.

## Factory (for sanity-checking α values)

```python
from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    create_alpha_computer, AlphaConfig, HierarchicalConditionalAlphaConfig,
)
alpha_computer = create_alpha_computer(
    AlphaConfig(method="hierarchical_conditional"),
    hc_config=HierarchicalConditionalAlphaConfig(),
)
alpha = alpha_computer.compute_from_stats(user_stats)
```

## What "Winning" Looks Like

Beat baseline, PFedRec, and split-learning personalized on `sampled_ndcg@10` — overall AND for the sparse user group (0-30 interactions). The sparse-user case is the load-bearing claim of the thesis.
