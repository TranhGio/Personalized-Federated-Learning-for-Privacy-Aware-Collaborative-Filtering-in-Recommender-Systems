# LightFR: Lightweight Federated Recommendation with Privacy-preserving Matrix Factorization

- **Authors**: Honglei Zhang, Fangyuan Luo, Jun Wu, Xiangnan He, Yidong Li
- **Venue**: ACM Transactions on Information Systems (TOIS), 2022
- **Paper ID**: zhang_2022_lightfr
- **Tags**: #federated-learning, #recommender-system, #communication-efficiency, #privacy, #matrix-factorization, #collaborative-filtering

---

## 1. Core Idea

LightFR replaces real-valued user/item embeddings in federated MF with compact **binary codes** (+1/-1) learned via **Federated Discrete Optimization** using Discrete Coordinate Descent (DCD). Similarity shifts from inner product to Hamming distance (XOR). This simultaneously achieves: (1) 8x communication reduction (1 bit vs 64-bit float per dimension), (2) orders-of-magnitude faster inference (XOR + popcount), and (3) inherent privacy (discrete sign is non-invertible — cannot reconstruct ratings from uploaded gradients).

## 2. Problem & Motivation

Traditional federated RS transmit real-valued item embedding matrices ($m \times f \times 64$ bits per round). At 10M items, f=128, this is 10.2 GB — infeasible for mobile. Additionally, real-valued gradients can be inverted to reconstruct private data (Zhu et al., 2019). LightFR solves storage, inference speed, and privacy simultaneously via learning-to-hash.

## 3. Method

### 3.1 Objective Function

$$\mathcal{L} = \sum_{u \in U} \frac{|I_u|}{N} \sum_{(i,r_{ui}) \in \Omega_u} (r_{ui} - \text{sim}(b_u, d_i))^2 + \lambda(\|\sum_u b_u\|^2 + \|\sum_i d_i\|^2)$$

$$\text{s.t.} \quad b_u \in \{+1,-1\}^f, \quad d_i \in \{+1,-1\}^f$$

Where:
- $\text{sim}(b_u, d_i) = \frac{1}{2} + \frac{1}{2f} b_u^T d_i$ — Hamming similarity [0,1]
- $b_u$ — user binary vector (LOCAL)
- $d_i$ — item binary vector (GLOBAL)
- $\lambda$ — balanced constraint (maximizes info entropy per bit, not L2)

### 3.2 Algorithm

**Local Discrete Optimization (client $u$):**
1. For each bit $k = 1..f$:
   - $b^*_{uk} = \sum_{i \in \Omega_u} \frac{1}{f}(r_{ui} - \frac{1}{2} - \frac{1}{2f} d_{i\neq k}^T b_{u\neq k}) d_{ik} - 2\lambda b_{uk} \sum_{k'} b_{uk'}$
   - Update: $b_{uk} = \text{sign}(b^*_{uk})$
2. Compute item gradients $\Delta D^u$ and upload to server

**Global Discrete Aggregation (server):**
- **Agg_grad** (preferred): $D = \text{sign}(\frac{1}{f} \sum_u \Delta D^u - 2\lambda D')$
- **Agg_para**: $D = \text{sign}(\sum_u D^u)$ (slightly worse)

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Binary code length | $f$ | 64 | Dimensionality (saturates at 64) |
| Balance trade-off | $\lambda$ | 0.6 (ML-1M) | Reconstruction vs balanced constraint |
| Client ratio | $p$ | 0.6 | Fraction sampled per round |
| Rounds | $T$ | 50 | Communication rounds |
| Local epochs | $E$ | 1 | Per round |

### 3.4 Architectural Decisions

1. **Binary {+1,-1} not {0,1}**: Enables Hamming similarity in [0,1].
2. **DCD over continuous relaxation**: Avoids quantization loss of two-stage approaches.
3. **Gradient aggregation > parameter aggregation**: Preserves more signal before sign.
4. **Balanced constraint replaces L2**: Binary codes have constant L2 norm.
5. **User LOCAL, item GLOBAL**: Same split-learning philosophy as thesis.
6. **No learning rate**: DCD produces closed-form bit updates.

## 4. Implementation Notes

- **Dimensions**: f=64 binary ≈ f=32 real-valued (storage-comparable)
- **Evaluation**: Leave-one-out, HR@10, NDCG@10 on ML-1M
- **Each user = one client**
- **Communication**: $m \times f$ bits per round (vs $m \times f \times 64$ bits for real-valued)
- **Inference**: XOR + popcount — hardware-accelerated

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Users | Items | Ratings | Density |
|---|---|---|---|---|
| MovieLens-1M | 6,040 | 3,952 | 1,000,209 | 4.19% |
| Filmtrust | 1,508 | 2,071 | 35,497 | 1.14% |
| Douban-Movie | 2,964 | 39,695 | 894,888 | 0.76% |
| Ciao | 7,375 | 105,096 | 282,619 | 0.04% |

### 5.2 Key Results (ML-1M)

| Method | Type | HR@10 | NDCG@10 |
|---|---|---|---|
| NCF | Centralized | 0.5342 | 0.2901 |
| PMF | Centralized | 0.5124 | 0.2768 |
| FCF | Federated | 0.4945 | 0.2625 |
| MetaMF | Federated | 0.4994 | 0.2691 |
| **LightFR** | **Federated** | **0.5014** | **0.2709** |

Best federated method. Competitive with centralized PMF. 8x communication reduction.

### 5.3 Ablation Highlights

- **Random codes**: 0.2419 HR. Two-stage binarize: 0.3110. End-to-end DCD: **0.5014** (+108% over random).
- **Code length**: Improves 8→64, saturates after.
- **$\lambda$**: Robust across 0.2-1.0. Best ~0.6.
- **Client ratio $p$**: Best 0.6-0.8.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Communication baseline**: Split learning reduces ~44%; LightFR achieves 8x. Orthogonal: split = which params; hashing = how large.
2. **Same evaluation protocol**: Leave-one-out, HR@10, NDCG@10 on ML-1M.
3. **User LOCAL / item GLOBAL**: Validates split-learning as consensus design.

### 6.2 Potential Integration Points

1. **Post-training binarization**: Binarize converged item embeddings for efficient deployment.
2. **Communication compression baseline**: Cite as orthogonal efficiency technique.
3. **Privacy argument**: Discrete gradients prevent rating reconstruction — strengthens privacy narrative.

### 6.3 Limitations & Gaps

1. **No BPR loss**: MSE-based squared loss only. DCD designed for pointwise, not pairwise.
2. **No personalization**: All users get same global item binary matrix.
3. **Limited accuracy**: HR@10=0.5014 < thesis baseline (0.65-0.75 with BPR).
4. **No non-IID handling**: No Dirichlet partitioning.
5. **f=64 binary vs f=32 real not apples-to-apples**: 2x dimension compensates info loss.
6. **E=1 only**: May not scale to 5-12 local epochs.

## 7. Key References to Follow

- **FCF (Ammad-Ud-Din et al., 2019)** — Foundational FRS with user-local/item-global split
- **FedMF (Chai et al., 2020)** — Secure federated MF with HE
- **MetaMF (Lin et al., 2020)** — Meta-learning for personalized item embeddings
- **Zhu et al., 2019 (Deep Leakage)** — Proves real-valued gradients leak data
- **Shen et al., 2015 (SDH)** — DCD algorithm source for hashing
