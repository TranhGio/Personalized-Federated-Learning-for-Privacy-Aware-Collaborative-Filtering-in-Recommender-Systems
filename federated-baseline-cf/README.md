# federated-baseline-cf

**Role:** Lower-bound baseline in the thesis comparison.
**Approach:** All model parameters (including user embeddings) are aggregated on the server via FedAvg / FedProx. No personalization — user embeddings are averaged across clients.
**Model:** BPR-MF (ranking, default) or BasicMF (MSE) on MovieLens 1M.

See [`../README.md`](../README.md) for the four-way thesis comparison context and [`claude.md`](./claude.md) for module-level architecture detail.

## What This Module Does

Under cross-device (`num-supernodes = 6040`, one user per client), each selected client receives the full model state, trains locally on its single user's data, and returns the full state to the server for aggregation. Since user embeddings are global, the server effectively averages personal representations — this is the lower bound against which the three personalized approaches are measured.

**Personalization boundary:**

| Parameter | Where it lives | Aggregation |
|-----------|----------------|-------------|
| User embeddings (6040 × d) | Global | FedAvg / FedProx |
| Item embeddings (3706 × d) | Global | FedAvg / FedProx |
| User / item / global biases | Global | FedAvg / FedProx |

Communication cost per round: all ~874K params (baseline to beat).

## Quick Start

```bash
pip install -e .

# Default cross-device benchmark (post-migration default)
flwr run .

# FedProx instead of FedAvg
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"

# MSE rating-prediction variant
flwr run . --run-config "model-type=basic"

# Reproduce a pre-migration cross-silo run
flwr run . --run-config "mode=cross_silo_legacy"

# Disable W&B logging
flwr run . --run-config "wandb-enabled=false"
```

## Configuration Surface (`pyproject.toml`)

Mode-locked by the top-level `mode` selector (`benchmark_cross_device` — default — or `cross_silo_legacy`). Key overrides:

| Key | Default (benchmark) | Purpose |
|-----|---------------------|---------|
| `num-supernodes` | 6040 | Client universe (= `num_users`) |
| `partition-mode` | `natural` | `natural` = 1 user / 1 client; `dirichlet` = cross-silo legacy |
| `fraction-train` | swept | Per-round client sampling fraction `C` |
| `weight-policy` | `num_positives` | Aggregation weighting |
| `strategy` | `fedavg` | `fedavg` / `fedprox` |
| `proximal-mu` | 0.01 | FedProx proximal strength |
| `model-type` | `bpr` | `bpr` (ranking, recommended) / `basic` (MSE) |
| `embedding-dim` | 128 | Latent factor dimensionality |
| `primary-evaluator` | `sampled_loo_99` | LOO + 99 negatives (NCF protocol) |
| `early-stopping-enabled` | true | Best-round checkpoint restore |

Full list: see `pyproject.toml` `[tool.flwr.app.config]`.

## Evaluation

- **Primary:** NDCG@10 under `sampled_loo_99` (leave-one-out + 99 negatives). Per-user-group slicing (sparse / medium / dense) reported first-class.
- **Secondary:** HR@{5,10,20}, MRR, Coverage@K, Novelty@K. `allrank_*` full-rank variants logged as diagnostics under a namespaced key.
- **Rating prediction** (RMSE, MAE) logged but **not optimized** under BPR — RMSE ≈ 2.2 is expected for BPR-MF and not a bug. Focus on ranking metrics.

## Gotchas

- **BPR RMSE is high by design.** BPR optimizes pairwise ranking, not rating prediction. Don't compare this number to SVD/NCF centralized baselines' RMSE.
- **Xavier init is critical** — RecSys 2024 work shows up to 50% performance variance from poor initialization.
- **Training-negative exclusion.** After Phase 1, training negatives exclude the held-out LOO test positive. Previously (pre-Phase-1) the held-out positive could leak into train as a "negative" — bug fixed across all four modules.
- **Training loss oscillates under non-IID.** Normal behavior; watch NDCG@10 not loss.
- **FedProx proximal term** is applied to all params in this baseline (not split-selective), since there are no local params: `L = L_task + (μ/2) ⋅ ‖w - w_server‖²`.

## Testing

```bash
python test_dataset.py   # Dataset loader + partitioning sanity
python test_models.py    # Model forward/backward sanity
```

## Results Location

`results/federated/baseline/<run_id>/` with full protocol fingerprint manifest (mode, num-supernodes, fraction-train, weight-policy, primary-evaluator, seeds, checkpoint rule, git-commit). Cross-silo legacy results stay under the pre-migration paths and are not overwritten.

## References

- Koren et al., "Matrix factorization techniques for recommender systems," 2009.
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI 2009.
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017 (FedAvg).
- Li et al., "Federated Optimization in Heterogeneous Networks (FedProx)," MLSys 2020.
