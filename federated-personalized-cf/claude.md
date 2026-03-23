# Federated Personalized Collaborative Filtering

> Split Learning: LOCAL user embeddings + GLOBAL item embeddings
> MovieLens 1M | SplitFedAvg/FedProx | BPR-MF

## Role in Thesis

Middle step in progression: baseline -> **personalized (this)** -> adaptive.
Key innovation: User embeddings stay private on clients, only item embeddings communicated.

## Directory Structure

```
federated_personalized_cf/
  dataset.py       - MovieLens loading, Dirichlet partitioning
  task.py          - Training, evaluation, ranking metrics
  client_app.py    - Split learning client, embedding caching, FedProx
  server_app.py    - Flower server, W&B logging, results export
  strategy.py      - SplitFedAvg & SplitFedProx custom strategies
  models/
    basic_mf.py    - BasicMF with get/set_local/global_parameters()
    bpr_mf.py      - BPRMF with split learning support
    losses.py      - MSE and BPR loss implementations
```

## Split Learning Architecture

**LOCAL parameters** (stay on client, cached in `.embedding_cache/`):
- `user_embeddings.weight` (6040 x 128) - PRIVATE
- `user_bias.weight` (6040 x 1) - PRIVATE

**GLOBAL parameters** (aggregated on server):
- `item_embeddings.weight` (3706 x 128)
- `item_bias.weight` (3706 x 1)
- `global_bias` (1,)

Communication savings: ~485K params vs 874K in baseline (44% reduction)

### Training Flow Per Round

1. Server sends GLOBAL params to client
2. Client loads LOCAL params from `.embedding_cache/partition_{id}/user_embeddings.pt`
3. Client trains on local data (all params updated)
4. FedProx proximal term ONLY on GLOBAL params (user embeddings NOT regularized)
5. Client saves LOCAL params to cache, sends ONLY GLOBAL params to server
6. Server aggregates GLOBAL params

## Key Differences from Baseline

| Aspect | Baseline | This Project |
|--------|----------|-------------|
| User Embeddings | Global (averaged) | **Local (private)** |
| Communication | 874K params | **485K params** |
| FedProx scope | All params | **Global only** |
| User history | Reset each round | **Accumulated** |

## Commands

```bash
flwr run .                                              # Default BPR-MF + SplitFedAvg
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "model-type=basic"
flwr run . --run-config "num-server-rounds=50"
rm -rf .embedding_cache/                                # Reset user embedding cache
```

## Key Config (pyproject.toml)

- `num-server-rounds`: 10, `local-epochs`: 5
- `model-type`: "bpr" or "basic"
- `strategy`: "fedavg" (SplitFedAvg) or "fedprox" (SplitFedProx)
- `embedding-dim`: 128, `alpha`: 0.5

## Gotchas

- `.embedding_cache/` is created at runtime - delete to start fresh
- Shape mismatch handling: partial loading when user population changes between rounds
- Models expose `get_global_parameters()`, `set_global_parameters()`, `get_local_parameters()`, `set_local_parameters()` - these are critical for split learning
- `strategy.py` defines `GLOBAL_PARAM_KEYS` and `LOCAL_PARAM_KEYS` frozensets

## Expected Performance (BPR-MF, 10 rounds)

- Hit Rate@10: 0.70-0.80 (+5-7% over baseline)
- NDCG@10: 0.18-0.28
- Communication: -44% vs baseline
