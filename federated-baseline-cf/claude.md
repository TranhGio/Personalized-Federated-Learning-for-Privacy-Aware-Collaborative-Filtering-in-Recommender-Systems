# Federated Collaborative Filtering Baseline

> Baseline for thesis comparison: All parameters GLOBAL (aggregated via FedAvg/FedProx)
> MovieLens 1M | Dirichlet Partitioning | BPR-MF

## Role in Thesis

Lower-bound baseline. All parameters (including user embeddings) are aggregated on server.
No personalization - user embeddings get averaged across clients.

Compared against:
1. `federated-personalized-cf` - Split learning (local user embeddings)
2. `federated-adaptive-personalized-cf` - Multi-factor alpha + dual-level personalization

## Directory Structure

```
federated_baseline_cf/
  dataset.py       - MovieLens loading, Dirichlet partitioning, DataLoaders
  task.py          - Training (BasicMF/BPRMF), testing, ranking evaluation
  client_app.py    - Flower client, local training/eval loops
  server_app.py    - Flower server, aggregation, centralized eval, W&B logging
  models/
    basic_mf.py    - BasicMF (MSE loss, rating prediction)
    bpr_mf.py      - BPRMF (BPR loss, ranking optimization) - RECOMMENDED
    losses.py      - MSE and BPR loss implementations
```

## Architecture

All parameters are GLOBAL (sent to server each round):
- `user_embeddings.weight` (6040 x 128) - GLOBAL (this is the key limitation)
- `item_embeddings.weight` (3706 x 128) - GLOBAL
- `user_bias`, `item_bias`, `global_bias` - GLOBAL
- Total: ~874K params transmitted each round

Prediction: `global_bias + user_bias[u] + item_bias[i] + dot(user_emb[u], item_emb[i])`

## Commands

```bash
flwr run .                                              # Default BPR-MF
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "model-type=basic"              # MSE loss instead
flwr run . --run-config "num-server-rounds=50 embedding-dim=256"
flwr run . --run-config "alpha=0.1"                     # More non-IID
flwr run . --run-config "wandb-enabled=false"
```

## Key Config (pyproject.toml)

- `num-server-rounds`: 10 (default), `local-epochs`: 5
- `model-type`: "bpr" (ranking) or "basic" (MSE)
- `strategy`: "fedavg" or "fedprox" (proximal-mu=0.01)
- `embedding-dim`: 128, `alpha`: 0.5 (Dirichlet)
- `enable-ranking-eval`: true, `ranking-k-values`: "5,10,20"

## Gotchas

- BPR-MF RMSE is high (~2.2) - this is EXPECTED. BPR optimizes ranking, not rating prediction. Focus on Hit Rate@K and NDCG@K.
- Xavier initialization is critical (50% performance variance with poor init per RecSys 2024)
- Negative sampling must exclude rated items
- Training loss oscillates with non-IID data - normal behavior
- FedProx proximal term: `L = L_task + (mu/2) * ||w - w_server||^2`

## Expected Performance (BPR-MF, 10 rounds)

- Hit Rate@10: 0.65-0.75
- NDCG@10: 0.15-0.25
- RMSE: ~2.2 (not optimized for this)
