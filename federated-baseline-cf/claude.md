# Federated Baseline (lower bound)

> All parameters GLOBAL — no personalization at all. User embeddings get averaged across clients each round.

**Role in thesis**: lower-bound baseline that the other three modules must beat on NDCG@10. If they don't, there's no story.

**Model**: BPR-MF (default) or BasicMF.
Prediction: `global_bias + user_bias[u] + item_bias[i] + dot(user_emb[u], item_emb[i])`.
~874K params transmitted per round (all of them).

## Run

```bash
flwr run .                                              # default: BPR-MF + FedAvg
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "model-type=basic"              # MSE loss instead of BPR
flwr run . --run-config "alpha=0.1"                     # more non-IID
flwr run . --run-config "wandb-enabled=false"
```

Defaults in `pyproject.toml` (`num-server-rounds=10`, `local-epochs=5`, `embedding-dim=128`, `alpha=0.5`).

## Gotchas

- BPR-MF RMSE ~2.2 is **expected** — BPR optimizes ranking, not rating. Look at HR@K / NDCG@K only.
- **Xavier init is critical** (RecSys 2024: 50% variance from poor init).
- Negative sampling must exclude rated items.
- Non-IID → training loss oscillates. Normal.
- FedProx here regularizes **all** params (`L = L_task + (μ/2)·||w - w_server||²`), unlike split-learning variants which only regularize globals.

## Expected (BPR-MF, 10 rounds, cross-device)

HR@10: 0.65-0.75 | NDCG@10: 0.15-0.25 | RMSE: ~2.2

If numbers fall outside these bands, suspect partition mode, negative sampling, or init — not the model.
