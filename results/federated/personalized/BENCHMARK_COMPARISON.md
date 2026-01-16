# MovieLens 1M Benchmark Comparison

## Your Current Results (Multi-Factor Adaptive Federated BPR-MF)

| Metric | @5 | @10 | @20 |
|--------|-----|------|------|
| **Hit Rate** | 38.94% | **56.83%** | 74.42% |
| **Precision** | 9.98% | 9.46% | 8.73% |
| **Recall** | 2.47% | 4.47% | 8.00% |
| **NDCG** | 10.24% | **10.15%** | 10.64% |
| **MAP** | 5.46% | 4.05% | 3.33% |
| **MRR** | - | **24.08%** | - |
| **Coverage** | 0.32% | 0.54% | 0.87% |

**Configuration:**
- Model: BPR_MF_Personalized_Split_FedProx_Adaptive
- Rounds: 10 | Clients: 5 | Embedding: 128
- Strategy: FedProx (μ=0.01)
- Alpha Method: Multi-Factor

---

## Federated Learning Benchmarks (FAIR COMPARISON)

Source: [IJCAI 2023 - Dual Personalization on Federated Recommendation](https://www.ijcai.org/proceedings/2023/0507.pdf)

### MovieLens-1M Results (Federated Methods)

| Method | HR@10 | NDCG@10 | Paper | Notes |
|--------|-------|---------|-------|-------|
| **FedMF** | 65.15% | 39.38% | IJCAI'23 | Federated Matrix Factorization |
| **FedNCF** | 60.62% | 33.25% | IJCAI'23 | Federated Neural CF |
| **FedRecon** | 64.45% | 37.78% | TFF | TensorFlow Federated baseline |
| **FedPerGNN** | ~62% | ~35% | WWW'22 | Graph-based federated |
| **PFedRec** | **73.26%** | **44.36%** | IJCAI'23 | Personalized FedRec (SOTA) |
| **Your Model** | 56.83% | 10.15% | - | **10 rounds only** |

### Key Federated Recommendation Papers

| Paper | Method | Venue | Approach |
|-------|--------|-------|----------|
| [FedMF](https://arxiv.org/abs/1906.05108) | Secure Federated MF | IEEE'20 | HE-encrypted gradients |
| [FCF](https://arxiv.org/abs/1901.09888) | Federated CF | - | First federated CF |
| [FedNCF](https://www.researchgate.net/publication/358631253) | Federated Neural CF | KBS'22 | NCF + FedAvg |
| [PFedRec](https://github.com/Zhangcx19/IJCAI-23-PFedRec) | Personalized FedRec | IJCAI'23 | Client-specific MLPs |
| [CoLR-FedRec](https://github.com/NNHieu/CoLR-FedRec) | Low-rank FedRec | arXiv'24 | Communication-efficient |

---

## Gap Analysis: Your Model vs Federated Baselines

| Metric | Your Model | FedMF | FedNCF | PFedRec (SOTA) |
|--------|------------|-------|--------|----------------|
| **HR@10** | 56.83% | 65.15% | 60.62% | **73.26%** |
| **NDCG@10** | 10.15% | 39.38% | 33.25% | **44.36%** |
| **Gap to FedMF** | -8.32% | - | - | - |
| **Gap to SOTA** | -16.43% | - | - | - |

---

## Evaluation Protocols

### Now Supporting Both Protocols!

Your implementation now supports **both** evaluation methods:

| Metric Prefix | Protocol | Ranking Scope | Use Case |
|---------------|----------|---------------|----------|
| `hit_rate@K`, `ndcg@K` | Full-rank | All ~3,700 items | Realistic performance |
| `sampled_hr@K`, `sampled_ndcg@K` | Leave-one-out + 99 negatives | 100 items | Fair baseline comparison |

### Protocol Comparison

| Aspect | IJCAI Papers | Full-Rank (your) | Sampled (your) |
|--------|--------------|------------------|----------------|
| **Negative Sampling** | 99 negatives | All items | 99 negatives ✓ |
| **Test Set** | Leave-one-out | Hold-out split | Leave-one-out ✓ |
| **Ranking Scope** | 100 items | ~3,700 items | 100 items ✓ |
| **Difficulty** | Easier | Harder | Easier ✓ |

### Why Two Protocols?

**Full-rank evaluation** (original metrics):
- More realistic: ranks among ALL items
- Harder task → lower scores
- Better reflects real-world performance

**Sampled evaluation** (`sampled_*` metrics):
- Follows NCF/FedMF/PFedRec protocol
- Enables direct comparison with published baselines
- Configure via `eval-num-negatives` in pyproject.toml

---

## Centralized SOTA (For Reference Only)

Source: [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN)

| Model | HR@10 | NDCG@10 | Type |
|-------|-------|---------|------|
| BPR | 71.99% | 24.01% | Centralized |
| LightGCN | 73.30% | 25.38% | Centralized + GNN |
| XSimGCL | **77.43%** | **27.50%** | Centralized + Contrastive |

**Note:** These are NOT comparable - centralized methods have full data access.

---

## Most Similar Papers to Your Approach

### 1. [FedMF (Chai et al., 2020)](https://arxiv.org/abs/1906.05108)
- **Similarity:** Matrix Factorization + FedAvg
- **Difference:** Uses homomorphic encryption, no split learning
- **Results:** HR@10 ~65% on ML-1M

### 2. [PFedRec (IJCAI 2023)](https://github.com/Zhangcx19/IJCAI-23-PFedRec)
- **Similarity:** Personalized federated + user/item split
- **Difference:** Uses client-specific MLPs instead of adaptive alpha
- **Results:** HR@10 73.26%, NDCG@10 44.36%

### 3. [FedRecon (TensorFlow Federated)](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization)
- **Similarity:** Split learning (user embeddings local)
- **Difference:** Uses reconstruction instead of aggregation
- **Results:** Similar to FedMF

### 4. [CoLR-FedRec (2024)](https://arxiv.org/html/2401.03748v1)
- **Similarity:** Efficient federated MF
- **Difference:** Focuses on communication reduction
- **Results:** Competitive with FedMF at 16x less communication

---

## Recommendations

### To Match FedMF (~65% HR@10):
1. **Increase rounds to 50-100**
2. **Use leave-one-out evaluation** for fair comparison
3. **Sample 99 negatives** for NDCG calculation

### To Match PFedRec (~73% HR@10):
1. All above +
2. **Add client-specific MLP layers**
3. **Fine-tune item embeddings per client**

### Novel Contributions of Your Approach:
- **Adaptive alpha** based on user statistics (not in existing papers)
- **Multi-factor personalization** (quantity + diversity + coverage + consistency)
- **Split learning + FedProx** combination

---

## References

### Federated Recommendation
- [Dual Personalization on Federated Recommendation (IJCAI 2023)](https://www.ijcai.org/proceedings/2023/0507.pdf)
- [Secure Federated Matrix Factorization](https://arxiv.org/abs/1906.05108)
- [Federated Collaborative Filtering](https://arxiv.org/abs/1901.09888)
- [CoLR-FedRec](https://arxiv.org/html/2401.03748v1)
- [GPFedRec](https://arxiv.org/html/2305.07866)

### Centralized (Reference Only)
- [RecBole-GNN Benchmarks](https://github.com/RUCAIBox/RecBole-GNN)
- [Papers with Code - MovieLens 1M](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-1m)
