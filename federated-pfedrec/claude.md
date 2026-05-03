# PFedRec: Personalized Federated Recommendation (IJCAI-23)

> Calibration baseline: Client-local score function + global item embeddings
> MovieLens 1M | SplitFedAvg/FedProx | BCE loss | Alternating Optimization

## Role in Thesis

Calibration baseline from published work. PFedRec uses a fundamentally different
personalization approach: instead of local user embeddings, it uses a client-local
Linear score function (affine_output) per user.

Compared against:
1. `federated-baseline-cf` - All params global (lower bound)
2. `federated-personalized-cf` - Split learning with local user embeddings
3. `federated-adaptive-personalized-cf` - Hierarchical alpha + dual-level (thesis)

## Architecture

**Model**: `Embedding(num_items, latent_dim) -> Linear(latent_dim, 1) -> Sigmoid`

NO explicit user embeddings. Each user has their own `affine_output` (Linear layer),
which serves as the user's personalized score function.

### Parameter Classification

| Parameter | Type | Privacy |
|-----------|------|---------|
| `embedding_item.weight` | Global | Shared each round |
| `affine_output.weight` | Local | Private (per-user) |
| `affine_output.bias` | Local | Private (per-user) |

Communication per round: `num_items * latent_dim` params (e.g., 3706 * 32 = 118K)

## Key Algorithm: Alternating Optimization

Per batch, TWO separate forward+backward passes:
1. Update `affine_output` only (SGD, lr=lr)
2. Update `embedding_item` only (SGD, lr=lr*num_items*lr_eta)

The huge item LR (e.g., 0.1 * 3706 * 80 = 29,648) compensates for sparse
item gradients across the large embedding table.

## Commands

```bash
flwr run .                                              # Default: 100 rounds, FedAvg
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "latent-dim=64 lr=0.05 lr-eta=40"
flwr run . --run-config "num-server-rounds=50"
flwr run . --run-config "wandb-enabled=false"
flwr run . --run-config "early-stopping-enabled=true early-stopping-patience=10"
rm -rf .embedding_cache/                                # Reset per-user cache
```

## Key Config (pyproject.toml)

- `num-server-rounds`: 100, `local-epochs`: 1
- `latent-dim`: 32, `lr`: 0.1, `lr-eta`: 80
- `num-negatives`: 4, `batch-size`: 256
- `strategy`: "fedavg" or "fedprox" (proximal-mu=0.01)
- `alpha`: 0.5 (Dirichlet)

## Gotchas

- **Per-user models**: Unlike other modules that have one model per partition, PFedRec
  maintains a SEPARATE affine_output per user. Cache: `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt`
- **Dual LR**: The item embedding LR is lr * num_items * lr_eta. This is intentional.
- **SGD only**: Paper uses SGD, not Adam. Don't change this.
- **BCE loss**: Binary cross-entropy on binarized implicit feedback (0/1).
- **Alternating optimization**: Two forward passes per batch, not joint optimization.
- **No user embedding**: The model's forward() takes only item_indices, not user_ids.
- Reference implementation: `IJCAI-23-PFedRec/` in project root

## Expected Performance (ML-1M, 100 rounds)

- HR@10: ~0.65-0.70
- NDCG@10: ~0.35-0.42
- BCE Loss: decreasing
