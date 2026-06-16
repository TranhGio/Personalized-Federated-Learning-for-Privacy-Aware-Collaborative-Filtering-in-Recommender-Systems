# PFedRec (IJCAI-23) — Calibration Baseline

> Per-user local `Linear` score function + global item embeddings. **No user embedding** in the model — `affine_output` IS the personalization.

**Role in thesis**: calibration baseline — a **cross-device C=0.1 reproduction**. Under the thesis protocol (natural partition N=6040, `fraction-train=0.1`, best_round_restore + warm full-pop D-06 eval), the bar is **full-pop NDCG@10 ≈ 0.335, HR@10 ≈ 0.59** (run `20260608-071106-ef41ab`, after the run_id eval fix `01d8b72`). The paper band (HR@10 0.65-0.70 / NDCG@10 0.35-0.42) applies ONLY to paper-compat full participation (`fraction=1.0`, ~6 days compute — infeasible; documented limitation, not a failed reproduction). Historical cross-silo runs reproduced the paper numbers (HR@10 ≈ 0.70, NDCG@10 ≈ 0.38). If we can't hit the bar, every downstream comparison is suspect.

Reference (do NOT modify): `IJCAI-23-PFedRec/` at project root.

**Model**: `Embedding(num_items, latent_dim) → Linear(latent_dim, 1) → Sigmoid`

| Param | Type | Notes |
|---|---|---|
| `embedding_item.weight` | Global | Aggregated each round |
| `affine_output.weight`, `affine_output.bias` | **Local, per-user** | Cached at `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt` |

Comm/round: `num_items * latent_dim` (~118K for ML-1M @ `latent_dim=32`).

## Alternating Optimization (do NOT change)

Per batch, **two** separate forward+backward passes:
1. Update `affine_output` only — SGD, `lr = lr`
2. Update `embedding_item` only — SGD, `lr = lr * num_items * lr_eta`

The huge item LR (e.g., `0.1 * 3706 * 80 ≈ 29,648`) compensates for sparse item gradients across the embedding table. **SGD only** — paper does not use Adam.

## Run

```bash
flwr run .                                              # default: 100 rounds, FedAvg
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "latent-dim=64 lr=0.05 lr-eta=40"
flwr run . --run-config "early-stopping-enabled=true early-stopping-patience=10"
flwr run . --run-config "final-calibration-enabled=true"  # 1 local epoch to ALL partitions after best-round restore — realigns stale affine_output heads before D-06
flwr run . --run-config "final-calibration-enabled=true final-calibration-freeze-items=false"  # ablation: let item-emb move during calibration
rm -rf .embedding_cache/                                # reset per-user cache
```

Defaults: `num-server-rounds=100`, `local-epochs=1`, `latent-dim=32`, `lr=0.1`, `lr-eta=80`, `num-negatives=4`, `batch-size=256`, `final-calibration-enabled=false`, `final-calibration-freeze-items=true` (lr_eta=0 during calibration so heads align to the exact restored item embeddings).

## Gotchas

- **Per-USER models** (not per-partition): each user has their own `affine_output`. Cache layout: `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt`.
- **BCE loss** on binarized implicit feedback (0/1), not BPR.
- `forward()` takes only `item_indices`, not `user_ids` (no user embedding in the model).
- **D-06 full-pop eval MUST stamp `run_id`/`reuse_cache`** (fix `01d8b72`) — without it every user is scored with a COLD `affine_output` from `.embedding_cache/default/` and full-pop craters ~5× (0.0711 vs 0.3478). If full-pop ever craters again, check the eval config stamps before blaming the model.
- If reproduction misses the bar, compare against `IJCAI-23-PFedRec/` line-by-line before blaming the framework.

## Expected (ML-1M, 100 rounds)

Cross-device C=0.1 (thesis protocol): full-pop NDCG@10 ≈ 0.335 | HR@10 ≈ 0.59
Paper-compat full participation (`fraction=1.0`): HR@10: 0.65-0.70 | NDCG@10: 0.35-0.42
