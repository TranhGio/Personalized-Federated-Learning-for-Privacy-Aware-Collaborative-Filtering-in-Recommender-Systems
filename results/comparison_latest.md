# Thesis Comparison: MovieLens 1M

Generated: 2026-03-23 12:33:35

## Centralized Baselines (Upper Bound - Full Data Access)

| Method | RMSE | MAE | HR@10 | NDCG@10 | MRR |
|--------|------|-----|-------|---------|-----|
| SVD (Centralized) | 0.8546 | 0.6731 | - | - | - |
| NCF (Centralized) | 0.8759 | 0.6904 | - | - | - |
| BPR-MF (Centralized) | - | - | 59.01% | 33.32% | 27.46% |

## Published Federated Baselines (from IJCAI'23)

| Method | HR@10 | NDCG@10 | Source |
|--------|-------|---------|--------|
| FedMF | 65.15% | 39.38% | IJCAI'23 |
| FedNCF | 60.62% | 33.25% | IJCAI'23 |
| FedRecon | 64.45% | 37.78% | TFF |
| PFedRec (SOTA) | **73.26%** | **44.36%** | IJCAI'23 |

*Note: Published baselines use leave-one-out + 99 negative samples (sampled evaluation).*

## Federated Methods - Full-Rank Evaluation

Rankings computed over ALL ~3,700 items (harder, more realistic).

| # | Method | Rounds | HR@10 | NDCG@10 | MRR | Category |
|---|--------|--------|-------|---------|-----|----------|
| 1 | Fed Basic-MF + FedAvg | 2 | 53.21% | 9.31% | 22.65% | Baseline |
| 2 | Fed Basic-MF + FedProx | 50 | 48.04% | 7.93% | 19.25% | Baseline |
| 3 | Fed BPR-MF + FedAvg | 10 | 64.91% | 12.54% | 28.62% | Baseline |
| 4 | Fed BPR-MF + FedProx | 10 | 57.08% | 10.06% | 24.25% | Baseline |
| 5 | Adaptive BPR-MF + FedAvg | 50 | 58.01% | 10.27% | 25.35% | **Proposed** |
| 6 | Adaptive BPR-MF + FedProx | 16 (ES@6) | 56.48% | 10.20% | 24.73% | **Proposed** |
| 7 | Dual-Level (add) | 50 | 62.19% | 11.14% | 26.28% | **Proposed** |
| 8 | Dual-Level (concat) | 100 | 60.28% | 10.93% | 26.02% | **Proposed** |
| 9 | Dual-Level (concat) + CL | 31 (ES@21) | 59.19% | 10.54% | 24.12% | **Proposed** |
| 10 | Dual-Level (concat) + IP | 29 (ES@19) | 58.72% | 10.64% | 24.68% | **Proposed** |
| 11 | Dual-Level (concat) + PUA | 38 (ES@28) | 58.38% | 10.19% | 22.71% | **Proposed** |
| 12 | Dual-Level (concat) + PUA+CL | 33 (ES@23) | 57.82% | 9.92% | 22.24% | **Proposed** |
| 13 | Dual-Level (concat) + PUA+IP | 24 (ES@14) | 57.31% | 9.92% | 22.38% | **Proposed** |
| 14 | Dual-Level (concat) + PUA+IP+CL | 41 (ES@31) | 59.26% | 10.51% | 24.06% | **Proposed** |
| 15 | Dual-Level (gate) | 50 | 61.13% | 10.78% | 24.94% | **Proposed** |

## Federated Methods - Sampled Evaluation (NCF Protocol)

Leave-one-out + 99 negative samples. **Directly comparable to published baselines.**

| # | Method | Rounds | s_HR@10 | s_NDCG@10 | s_MRR | vs PFedRec |
|---|--------|--------|---------|-----------|-------|------------|
| 1 | Adaptive BPR-MF + FedAvg | 50 | 57.67% | 35.75% | 30.76% | -8.61pp |
| 2 | Adaptive BPR-MF + FedProx | 16 (ES@6) | 58.55% | 35.62% | 30.34% | -8.74pp |
| 3 | Dual-Level (add) | 50 | 67.73% | 41.21% | 34.61% | -3.15pp |
| 4 | Dual-Level (concat) | 100 | 68.32% | 42.70% | 36.38% | -1.66pp |
| 5 | Dual-Level (concat) + CL | 31 (ES@21) | 66.04% | 41.18% | 35.13% | -3.18pp |
| 6 | Dual-Level (concat) + IP | 29 (ES@19) | 65.57% | 41.08% | 35.24% | -3.28pp |
| 7 | Dual-Level (concat) + PUA | 38 (ES@28) | 65.50% | 41.08% | 35.24% | -3.28pp |
| 8 | Dual-Level (concat) + PUA+CL | 33 (ES@23) | 64.88% | 39.99% | 34.08% | -4.37pp |
| 9 | Dual-Level (concat) + PUA+IP | 24 (ES@14) | 63.25% | 38.76% | 32.99% | -5.60pp |
| 10 | Dual-Level (concat) + PUA+IP+CL | 41 (ES@31) | 64.84% | 40.79% | 35.08% | -3.57pp |
| 11 | Dual-Level (gate) | 50 | 67.44% | 41.37% | 34.97% | -2.99pp |

## Summary

**Best full-rank NDCG@10**: 12.54% (Fed BPR-MF + FedAvg)
**Best sampled NDCG@10**: 42.70% (Dual-Level (concat))

### Improvement Over Best Baseline (full-rank)

| Metric | Best Baseline | Best Proposed | Improvement |
|--------|---------------|---------------|-------------|
| HR@10 | 64.91% | 62.19% | -4.2% |
| NDCG@10 | 12.54% | 11.14% | -11.1% |
| MRR | 28.62% | 26.28% | -8.2% |

## Result Files

- `results/centralized/svd_baseline_results.json` -> SVD (Centralized)
- `results/centralized/ncf_baseline_results.json` -> NCF (Centralized)
- `results/centralized/bpr_mf_centralized_results.json` -> BPR-MF (Centralized)
- `results/federated/basic_mf_fedavg_mu0.01_r2_f1.0_results.json` -> Fed Basic-MF + FedAvg
- `results/federated/basic_mf_fedprox_mu0.05_r50_f0.3_results.json` -> Fed Basic-MF + FedProx
- `results/federated/bpr_mf_federated_results.json` -> Fed BPR-MF + FedAvg
- `results/federated/bpr_mf_fedprox_mu0.01_r10_f1.0_results.json` -> Fed BPR-MF + FedProx
- `results/federated/personalized/bpr_mf_split_fedavg_mu0.01_r50_f1.0_results.json` -> Adaptive BPR-MF + FedAvg
- `results/federated/personalized/bpr_mf_split_fedprox_mu0.01_r50_f1.0_results.json` -> Adaptive BPR-MF + FedProx
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_results.json` -> Dual-Level (add)
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r100_f1.0_concat_mlp512-256-128_results.json` -> Dual-Level (concat)
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_cl0.1_t0.1_results.json` -> Dual-Level (concat) + CL
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_ip0.01_results.json` -> Dual-Level (concat) + IP
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_pua_results.json` -> Dual-Level (concat) + PUA
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_pua_cl0.1_t0.1_results.json` -> Dual-Level (concat) + PUA+CL
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_pua_ip0.01_results.json` -> Dual-Level (concat) + PUA+IP
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp512-256-128_results.json` -> Dual-Level (concat) + PUA+IP+CL
- `results/federated/personalized/dual_mf_split_fedprox_mu0.01_r50_f1.0_gate_mlp256-128-64_results.json` -> Dual-Level (gate)
