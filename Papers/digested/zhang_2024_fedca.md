# Beyond Similarity: Personalized Federated Recommendation with Composite Aggregation

- **Authors**: Honglei Zhang, Haoxuan Li, Jundong Chen, Sen Cui, Kunda Yan, Abudukelimu Wuerkaixi, Xin Zhou, Zhiqi Shen, Yidong Li
- **Venue**: arXiv:2406.03933, June 2024 (Under review)
- **Paper ID**: zhang_2024_fedca
- **Tags**: #federated-learning, #recommender-system, #personalization, #aggregation, #collaborative-filtering, #implicit-feedback

---

## 1. Core Idea

Current federated recommendation methods borrow similarity-based aggregation from federated vision (FedAvg, clustering, attention). This is suboptimal because recommendation models use **one-to-one item embedding tables** (not structured CNN parameters), causing **embedding skew**: embeddings for interacted items improve during aggregation while non-interacted item embeddings stagnate/degrade. FedCA proposes **composite aggregation** from both **similar** clients (enhance trained embeddings) and **complementary** clients (update non-trained embeddings), formulated as a unified quadratic optimization solved on the server.

## 2. Problem & Motivation

- **Embedding skew**: Similarity-only aggregation causes growing train/test gap (Figure 2: as similar clients increase, train HR rises but test HR drops).
- **Architecture mismatch**: Federated vision uses structured CNNs where similarity works; FR uses flat embedding tables where different clients train different rows.
- **Heterogeneity decomposition**: $p(x,y) = p(x)p(y|x)$. Similarity aligns $p(y|x)$; complementarity addresses $p(x)$ differences.

## 3. Method

### 3.1 Objective Function

$$\min_{\{p_u, Q_u, w_u\}} \sum_u \left[ \mathcal{L}_u(p_u, Q_u; D_u) + \alpha \sum_v F_s(w_{uv}; Q_u, Q_v) + \beta \sum_v F_c(w_{uv}; D_u, D_v) \right]$$

$$\text{s.t.} \quad \mathbf{1}^T w_u = 1; \quad w_u > 0$$

Where:
- $\mathcal{L}_u$ — local BCE loss
- $F_s(w_{uv}) = (w_{uv} - \sigma(Q_u, Q_v))^2$ — **similarity loss** with $\sigma(Q_u, Q_v) = 1/(1 + \|Q_u - Q_v\|^2)$
- $F_c(w_{uv}) = -w_{uv} \cdot \cos(\phi(X_u, X_v))$ — **complementarity loss** where $\phi$ measures angle between SVD-reduced interacted item embeddings
- $\alpha, \beta$ — tuning coefficients
- $w_u$ — per-client simplex-constrained aggregation weights

**Server-side QP** (convex, solvable by standard solvers):

$$\min_{w_u} \sum_v \left[(w_{uv} - p_v)^2 + \alpha F_s + \beta F_c \right], \quad \text{s.t.} \quad \mathbf{1}^T w_u = 1, w_u > 0$$

**Local inference interpolation**: $Q_u^t = \rho Q_u^{t-1} + (1-\rho) Q_g^t$ ($\rho$=0.8-0.9).

### 3.2 Algorithm

**Per round:**
1. Server sends personalized aggregated embeddings $Q_u^t = \sum_v w_{uv} Q_v^t$ to each client
2. Client trains locally (BCE on implicit feedback), uploads $Q_u$
3. Server computes similarity $s_u$ from $Q_u, Q_v$
4. Server computes complementarity $c_u$ via SVD of interacted item embeddings (first $k$ singular vectors)
5. Server solves QP per client for $w_u$

**Key finding**: FedProx proximal term **hurts** FR performance (Table 2). FR needs stronger personalization.

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Similarity coefficient | $\alpha$ | Tuned per dataset | Weight of similarity loss |
| Complementarity coefficient | $\beta$ | Tuned per dataset | Weight of complementarity loss |
| Interpolation ratio | $\rho$ | 0.8 (ML-100K), 0.9 (ML-1M) | Local vs global item embed balance |
| SVD rank | $k$ | 4 | Singular vectors for complementarity |
| Negative samples | $N$ | 4 | Per positive |

**FedCA subsumes**: FedAvg ($\alpha=\beta=0$), pFedGraph ($\beta=0$), FedFast ($\alpha=0$).

### 3.4 Architectural Decisions

1. **Model-agnostic**: Plug-and-play server-side module. Tested with PMF and NCF.
2. **No proximal term**: FR needs full personalization, not convergence to global model.
3. **SVD for privacy**: Interacted item embeddings → SVD as proxy for $p(x)$.
4. **Per-client personalized weights**: Each client gets different weighted combination.

## 4. Implementation Notes

- **Code**: github.com/hongleizhang/FedCA
- **Evaluation**: Leave-one-out, N=4 negatives, HR@10, NDCG@10
- **Loss**: BCE (not BPR)
- **QP solver**: Standard convex optimization
- **5 repeated experiments**: Averaged results

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Type | Scale |
|---|---|---|
| ML-100K | Movie ratings → implicit | Small |
| Filmtrust | Movie ratings → implicit | Small |
| **ML-1M** | **Movie ratings → implicit** | **Medium (thesis dataset)** |
| Microlens-100K | Short video implicit | Medium |

### 5.2 Key Results (ML-1M, PMF backbone)

| Method | HR@10 | NDCG@10 |
|---|---|---|
| FedAvg | 0.4912 | 0.2751 |
| FedFast | 0.5061 | 0.2898 |
| pFedGraph | 0.7904 | 0.6347 |
| PFedRec | 0.8032 | 0.6519 |
| **FedCA** | **0.8348** | **0.7118** |

FedCA surpasses PFedRec by +3.2% HR, +9.2% NDCG on ML-1M.

### 5.3 Ablation Highlights

- **Proximal term hurts**: Without: 0.8348 HR. With: 0.8168 HR.
- **$\rho$**: Optimal 0.8-0.9 (local model dominates; 10-20% global contribution).
- **Loss ablation**: Both similarity and complementarity contribute; combining yields best.
- **Data sparsity**: FedCA especially strong at 40% training data.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Embedding skew awareness**: Global item embeddings for non-interacted items may degrade during FedAvg. Theoretical justification for adaptive alpha.
2. **Interpolation ($\rho$) ≈ alpha blending**: $Q_u^t = \rho Q_u^{t-1} + (1-\rho) Q_g^t$ is conceptually identical to $p_{eff} = \alpha p_{local} + (1-\alpha) p_{global}$. FedCA applies to items at inference; thesis applies to users.
3. **FedProx hurts FR**: If FedProx underperforms FedAvg, this paper provides theoretical backing.
4. **Complementarity concept**: Clients with different item sets help each other. Global prototype partially serves this role.

### 6.2 Potential Integration Points

1. **Composite aggregation for item embeddings**: Replace SplitFedAvg with FedCA-style aggregation in `strategy.py`. Server-side only, no client changes.
2. **Item embedding interpolation**: Add $\rho$-based blending complementary to user-embedding alpha.
3. **Per-client aggregation from user stats**: Combine thesis alpha (user embed blend) with FedCA weights (item embed aggregation).

### 6.3 Limitations & Gaps

1. **BCE, not BPR**: Loss adaptation needed.
2. **No Flower integration**: Standalone Python.
3. **SVD reveals interacted item sets**: Privacy concern.
4. **No Dirichlet partitioning**: Each user = one client.
5. **HP sensitivity**: $\alpha, \beta, \rho$ need per-dataset tuning.

## 7. Key References to Follow

- **FCF (Ammad-Ud-Din et al., 2019)** — Foundational federated CF
- **PFedRec (Zhang et al., IJCAI 2023)** — Calibration baseline
- **pFedGraph** — Graph-based similarity aggregation
- **FedFast (Muhammad et al., SIGKDD 2020)** — Clustering + dissimilarity
- **LightFR (Zhang et al., 2022)** — Same first author, binary codes for communication efficiency
