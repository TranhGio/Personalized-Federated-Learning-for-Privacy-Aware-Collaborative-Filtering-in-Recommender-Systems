# federated-adaptive-personalized-cf

**Role:** Thesis contribution.
**Approach:** Split learning (local user embeddings) + **hierarchical-conditional α** for per-client personalization strength + dual-level personalization (statistical blending + neural PersonalMLP) + EMA-based server-side global user prototype. Optional next-generation techniques: per-user learned α, dual-side item perturbation, contrastive local-global alignment.
**Model:** BPR-MF or `DualPersonalizedBPRMF` on MovieLens 1M.

See [`../README.md`](../README.md) for the four-way thesis comparison context and [`claude.md`](./claude.md) for the complete architecture tour (factory pattern, parameter classification, loss composition).

## Why This Module Exists

Under a correct cross-device protocol, this module must **beat all three baselines on NDCG@10 — especially on sparse users**. That is the falsifiable thesis claim defined in [`../.planning/PROJECT.md`](../.planning/PROJECT.md).

The contribution combines six ingredients, each individually optional via config:

1. **Hierarchical conditional α** (default) — resolves multi-factor conflicts.
2. **Dual-level personalization** — statistical blend + neural head.
3. **Global user prototype (EMA)** — server-side support for sparse users.
4. **Per-user learned α** (`enable-per-user-alpha`).
5. **Dual-side item perturbation** (`enable-item-perturbation`).
6. **Contrastive local-global alignment** (`contrastive-lambda`).

## The Core Ingredients

### 1. Hierarchical Conditional α (default: `alpha-method=hierarchical_conditional`)

Two-stage per-client personalization strength computation. Addresses the quantity-coverage redundancy (correlation 0.8–1.0) and diversity-consistency contradiction (correlation -0.3 to -0.5) that a flat multi-factor weighted sum cannot express.

**Stage 1 — hierarchical aggregation:**
- `data_volume = sqrt(f_quantity × f_coverage)` (geometric mean collapses the redundant pair)
- `preference_quality = harmonic_mean(f_diversity, f_consistency)` (handles the contradictory pair)
- `base_alpha = 0.55 × data_volume + 0.45 × preference_quality`

**Stage 2 — conditional rules:**
- Sparse users (`n < 20`): penalty up to 50 %
- Niche specialists (low diversity, high quantity): +0.15 bonus
- Inconsistent raters (`f_s < 0.3`): 30 % penalty
- Completionists (high coverage, low diversity): +0.10 bonus

Output clipped to `[0.1, 0.95]`. Alternative methods: `multi_factor` (flat weighted sum — the ingredient being improved on) and `data_quantity` (interaction count only — simplest baseline).

### 2. Dual-Level Personalization (`model-type=dual`)

- **Level 1 — statistical blending:** `p_effective = α · p_local + (1-α) · p_global`
- **Level 2 — neural:** `PersonalMLP` scores the element-wise product of effective user × item embeddings
- **Fusion:** `add` (sum), `gate` (learnable sigmoid), or `concat` (Linear over `[score_cf ; score_mlp]`)

### 3. Global User Prototype (EMA)

Server maintains `p_global = 0.9 · p_old + 0.1 · weighted_avg(client_prototypes)`. Clients compute their local prototype after training and return it in `FitRes.metrics`. Helps sparse users by providing a population-average backbone in the α blending. Best-round checkpoint restores this alongside model state (Phase 4 of migration).

### 4. Per-User Learned α (`enable-per-user-alpha=true`)

Replaces the single per-client α scalar with a per-user learnable `nn.Embedding` (`logit_alpha`). Initialized from the hierarchical-conditional heuristic (`torch.logit(heuristic_alpha)`), then refined by BPR gradient descent. Cached as a LOCAL parameter. Bug fix from Phase 1 research: `enable_per_user_alpha()` must be called BEFORE `load_local_user_embeddings()` for cached values to restore across rounds.

### 5. Dual-Side Item Perturbation (`enable-item-perturbation=true`)

Local item embedding adjustment: `q_effective[i] = q_global[i] + perturbation[i]`. Zero-initialized, refined by gradient descent, L2-regularized (`reg · ‖perturbation‖²`). LOCAL parameter — never sent to server.

### 6. Contrastive Local-Global Alignment (`contrastive-lambda > 0`)

InfoNCE auxiliary loss: `L_total = L_BPR + λ · L_contrastive + reg · ‖perturbation‖²`. Positive pair `(p_local[u], p_effective[u])`; negatives are the other users in the batch. Gives a direct gradient signal to the per-user α through the blended embedding.

## Parameter Classification

| Parameter | Where it lives | Aggregation |
|-----------|----------------|-------------|
| `user_embeddings`, `user_bias` | Local | Never transmitted |
| `personal_mlp.*`, `fusion_gate / fusion_layer` | Local | Never transmitted |
| `logit_alpha` (per-user α) | Local | Never transmitted |
| `item_perturbation` | Local | Never transmitted |
| `item_embeddings`, `item_bias`, `global_bias` | Global | FedAvg / FedProx |
| Server prototype EMA (`p_global`) | Server-only | Aggregated from client prototypes; broadcast in config each round |

≈ 38 % of parameters transmitted per round.

## Quick Start

```bash
pip install -e .

# Default: dual model + hierarchical-conditional α + prototype EMA
flwr run .

# Ingredient toggles
flwr run . --run-config "model-type=dual fusion-type=concat"
flwr run . --run-config "alpha-method=multi_factor"           # ablation vs hierarchical
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"

# Next-gen techniques
flwr run . --run-config "enable-per-user-alpha=true"
flwr run . --run-config "enable-item-perturbation=true item-perturbation-reg=0.01"
flwr run . --run-config "contrastive-lambda=0.1 contrastive-tau=0.1"
flwr run . --run-config "enable-per-user-alpha=true enable-item-perturbation=true contrastive-lambda=0.1"

# Mode selector
flwr run . --run-config "mode=cross_silo_legacy"              # pre-migration config

# W&B sweep (see sweep.yaml)
wandb sweep sweep.yaml
wandb agent <ENTITY>/federated-adaptive-personalized-cf/<SWEEP_ID>
```

## Configuration Surface (`pyproject.toml`)

Mode-locked by the top-level `mode` selector. Key overrides:

| Key | Default (benchmark) | Purpose |
|-----|---------------------|---------|
| `num-supernodes` | 6040 | Client universe (= `num_users`) |
| `partition-mode` | `natural` | 1 user / 1 client |
| `model-type` | `dual` | `bpr` / `basic` / `dual` |
| `alpha-method` | `hierarchical_conditional` | `hierarchical_conditional` / `multi_factor` / `data_quantity` |
| `fusion-type` | `concat` | `add` / `gate` / `concat` |
| `mlp-hidden-dims` | `512,256,128` | PersonalMLP hidden sizes |
| `prototype-momentum` | 0.9 | Server EMA momentum |
| `enable-per-user-alpha` | false | Turn on next-gen technique 1 |
| `enable-item-perturbation` | false | Turn on next-gen technique 2 |
| `item-perturbation-reg` | 0.01 | L2 on perturbation |
| `contrastive-lambda` | 0.0 | InfoNCE auxiliary weight |
| `contrastive-tau` | 0.1 | InfoNCE temperature |
| `early-stopping-metric` | `sampled_ndcg@10` | Primary monitoring metric |
| `ranking-k-values` | `5,10,20` | NDCG / HR cutoffs |

User-group bucket boundaries used for reporting: sparse (0–30), medium (30–100), dense (100+).

## Factory Pattern

```python
from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    create_alpha_computer, AlphaConfig, HierarchicalConditionalAlphaConfig
)
config = AlphaConfig(method="hierarchical_conditional")
hc_config = HierarchicalConditionalAlphaConfig()
alpha_computer = create_alpha_computer(config, hc_config=hc_config)
alpha = alpha_computer.compute_from_stats(user_stats)   # -> float in [0.1, 0.95]
```

## Gotchas

- **α range is clipped to `[0.1, 0.95]`** — never fully local or fully global.
- **Hierarchical-conditional α needs all four user stats**: `n_interactions`, `genre_entropy`, `n_unique_items`, `rating_std`. Phase 1 foundation precomputes and persists them in the split manifest.
- **`DualPersonalizedBPRMF` requires** `model.set_alpha(...)` AND `model.set_global_prototype(...)` to be called before `forward()` — check client wiring if you hit NaN.
- **PersonalMLP is LOCAL** (client-specific, never aggregated).
- **Per-user learned α load order bug (known, Phase 4 fix)**: `enable_per_user_alpha()` must be called BEFORE `load_local_user_embeddings()` so `_logit_alpha.weight` is in `_LOCAL_PARAMS` when the loader runs. Same for `enable_item_perturbation()`.
- **Prototype EMA restore at best-round checkpoint** (Phase 4 fix): the server's `p_global` state must be snapshot alongside model arrays so the final post-restore evaluation uses the best-round EMA, not the last-round EMA.
- **Early stopping** monitors `sampled_ndcg@10` by default; `sweep.yaml` uses Bayesian optimization with Hyperband early termination.
- **Contrastive `λ > 0`** only makes sense once `enable-per-user-alpha=true`, otherwise the auxiliary loss has no effective α to refine.

## Testing

```bash
python test_dataset.py
python test_models.py
```

## Results Location

`results/federated/adaptive/<run_id>/` with full protocol fingerprint manifest. Ablation tables for Phase 7 land under `results/federated/_thesis/`.

## References

- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI 2009.
- Singhal et al., "Federated Reconstruction: Partially Local Federated Learning," NeurIPS 2021.
- Zhang et al., "Dual Personalization on Federated Recommendation (PFedRec)," IJCAI 2023.
- Zhang et al., "GPFedRec," KDD 2024 (global prototype aggregation).
- Zhang et al., "FedCA — Beyond Similarity," 2024 (per-user-group reporting).
- Further reading: `../Papers/digested/_INDEX.md`.

## Citation

```bibtex
@mastersthesis{vinh2026personalizedfl,
  title={Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems},
  author={Dang Vinh},
  year={2026}
}
```
