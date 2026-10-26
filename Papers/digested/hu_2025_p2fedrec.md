# P2FedRec: Towards Privacy-Preserving and Personalized Federated Recommendation via Relationship Awareness

- **Authors**: Chenfei Hu, Zihao Xu, Tong Wu, You Li, Chuan Zhang, Liehuang Zhu
- **Venue**: Proc. ACM Manag. Data (SIGMOD), Vol. 3, No. 6, Article 346, December 2025
- **Paper ID**: hu_2025_p2fedrec
- **Tags**: #federated-learning, #recommender-system, #privacy, #personalization, #collaborative-filtering, #differential-privacy, #graph-neural-network, #implicit-feedback, #homomorphic-encryption

---

## 1. Core Idea
P2FedRec achieves personalized federated recommendation with **multi-level privacy** (data-level + edge-level) by constructing user relationship graphs in a privacy-preserving way. It uses a two-server model with additive secret sharing (ASS) + homomorphic encryption (HE) for secure item embedding similarity computation, and local differential privacy (LDP) for edge-level privacy when uploading local relationship graphs. Users get **user-specific item embeddings** aggregated from neighbors with similar preferences, enabling personalization without exposing raw data or social connections.

## 2. Problem & Motivation
Prior federated recommendation methods have two key limitations:
1. **No personalization**: FedMF, FedNCF use FedAvg to produce a single global model, ignoring user heterogeneity.
2. **Incomplete privacy**: PFedRec [46] and GPFedRec [47] achieve personalization via relationship graphs but expose item embeddings (data-level leakage per [6]) and relationship graph structure (edge-level leakage). GPFedRec directly exposes item embeddings and user relationships to the untrusted server.

P2FedRec is the first framework addressing **both** data-level and edge-level privacy in relationship graph-enhanced federated recommendation.

## 3. Method

### 3.1 Objective Function

$$\min_{\{\theta_1,...,\theta_{|U|}\}} \sum_{i=1}^{|U|} \mathcal{L}_i(\theta_i) + \beta \mathcal{R}(\mathbf{g}_i^t, \hat{\mathbf{g}}_i^t)$$

Where:
- $\theta_i = \{p_i, g_i, f_i\}$ — model parameters for user $u_i$ (user embedding $p_i$, item embedding $g_i$, score function $f_i$)
- $\mathcal{L}_i(\theta_i) = -\sum_{m=1}^{|M_i|} \log \tilde{y}_{im} - \sum_{m=1}^{|M_i^-|} \log(1 - \tilde{y}_{im})$ — BCE loss over positive interactions $M_i$ and negative samples $M_i^-$
- $\mathcal{R}(\mathbf{g}_i^t, \hat{\mathbf{g}}_i^t)$ — MSE regularization aligning local item embedding $g_i^t$ with user-specific aggregated item embedding $\hat{g}_i^t$ from neighbors
- $\beta$ — regularization coefficient (best: 0.3)

### 3.2 Algorithm

The system has two phases:

**Phase I: Offline Initialization (one-time)**
1. Servers $\mathcal{P}_1$ and $\mathcal{P}_2$ generate Beaver triples via HE for secure multiplication
2. Each server selects random shares $\langle r^{(i)} \rangle$ and computes triples $(r^{(i)}, r^{(j)}, s^{(i,j)})$ where $s^{(i,j)} = r^{(i)} \cdot r^{(j)}$
3. This yields $|U|(|U|-1)/2$ triples shared between $\mathcal{P}_1$ and $\mathcal{P}_2$

**Phase II: Online Training (per round $t$)**

**Module 1 — Embedding-shared Local Graph Construction:**
1. Each user $u_i$ normalizes item embedding $g_i^t$, encodes via Q-bit fixed-point ($\lfloor g_i^t[h] \cdot 2^Q \rfloor$), splits into two shares via ASS, sends to $\mathcal{P}_1$ and $\mathcal{P}_2$
2. Servers compute pairwise cosine similarity shares $\langle z_{i,j}^t \rangle$ using Beaver triples:
   $$\langle z_{i,j}^t \rangle_1 = \sum_{h=1}^{H} (e_i^t[h] \cdot e_j^t[h] + e_i^t[h] \cdot \langle \bar{g}_j^t[h] \rangle_1 + e_j^t[h] \cdot \langle \bar{g}_i^t[h] \rangle_1 + \langle s_{(i,j)} \rangle_1)$$
3. Servers assign anonymous identities to decouple users from embeddings
4. Servers send similarity shares + average similarity $\bar{z}^t$ to users
5. Each user constructs local relationship graph: $\mathcal{L}_{ij} = 1$ if $z_{i,j}^t \geq \alpha \bar{z}^t$, else 0

**Module 2 — Noisy Global Graph-Guided Aggregation:**
1. **User-side noise**: Each user perturbs local graph with LDP:
   - Degree: Laplace noise $\tilde{d}_i = d_i + l_i$, $l_i \sim Laplace(0, 1/\epsilon_d)$
   - Adjacency: Random flipping with probability $1/(1+\exp(\epsilon_a))$
   - Privacy budget split: $\epsilon_d = \gamma\epsilon$, $\epsilon_a = (1-\gamma)\epsilon$
2. **Server-side denoising**: Servers use $\beta$-model + Bayesian estimation to reconstruct global graph:
   - Posterior probability: $\hat{p}_{ij} = q_{ij} p_{ij} / (q_{ij} p_{ij} + q'_{ij}(1-p_{ij}))$
   - Threshold: $\hat{\mathcal{L}}_{ij} = 1$ if $\hat{p}_{ij} > 0.5$
3. **User-specific embedding aggregation**: Average neighbor item embeddings:
   $$\langle \hat{g}_i^t \rangle = \frac{1}{|\mathcal{B}_i|} \sum_{j \in \mathcal{B}_i} \langle g_j^t \rangle$$
4. **Global embedding**: Average all user-specific embeddings:
   $$\langle \hat{g}_{average}^t \rangle = \frac{1}{|U|} \sum_{i=1}^{|U|} \langle \hat{g}_i^t \rangle$$

**Module 3 — Personalized Model Updating:**
1. User recovers: $\hat{g}_{average}^t$ (global) and $\hat{g}_i^t$ (user-specific) from shares
2. Updates item embedding: $g_i^t \leftarrow \hat{g}_{average}^t$ (global average replaces local)
3. Regularizes toward user-specific: $\mathcal{L}_i = \mathcal{L}(\theta_i; y_{im}, \tilde{y}_{im}) + \beta \mathcal{R}(g_i^t, \hat{g}_i^t)$
4. SGD update: $\theta_i^{t+1} = \theta_i^t - \eta \frac{\partial \mathcal{L}_i}{\partial \theta_i^t}$

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Neighbor threshold | $\alpha$ | 0.7 | Controls graph density (fraction of avg similarity to include neighbor) |
| Regularization coefficient | $\beta$ | 0.3 | Weight for user-specific embedding alignment |
| Privacy budget | $\epsilon$ | varies (0.1-4) | Total LDP privacy budget |
| Privacy allocation | $\gamma$ | 0.5 | Split between degree ($\gamma\epsilon$) and adjacency ($(1-\gamma)\epsilon$) privacy |
| Embedding dimension | $H$ | 32 | Optimal at 32, degrades at 16 and 64+ |
| Fixed-point precision | $Q$ | 15 | Bits for encoding float→int for ASS |
| Training rounds | $T$ | 100 | Total FL communication rounds |
| Negative samples | — | 4 | Per positive for training |
| Learning rate | $\eta$ | 0.1 | SGD learning rate |
| Batch size | — | 256 | Training batch size |
| Graph update interval | — | 10-20 | Rounds between graph reconstruction (for efficiency) |

### 3.4 Architectural Decisions

**Why two-server model?** Single-server requires HE or DP for item embedding privacy, both causing excessive overhead or accuracy loss. Two non-colluding servers enable ASS which is information-theoretically secure with no accuracy loss.

**Why ASS + HE (not pure HE)?** Pure HE is computationally expensive. ASS handles most operations cheaply; HE is only used offline for Beaver triple generation (one-time cost).

**Why LDP for edge privacy (not ASS)?** Local graphs are dense, so LDP noise dilutes naturally. The $\beta$-model + Bayesian denoising effectively recovers the true graph structure despite noise.

**Why regularization (not direct replacement)?** Simply replacing local item embeddings with user-specific ones would discard local training signal. The MSE regularization balances local knowledge with neighborhood information.

**Why embedding dim = 32?** Larger dims overfit (Fig 8 shows performance drops at 64 and 128). This is specific to their cross-device setting with small per-user datasets.

## 4. Implementation Notes
- **Framework**: Custom Python implementation (not Flower/PySyft)
- **Model architecture**: Neural recommendation model $\theta = \{p, g, f\}$ — user embedding $p$, item embedding $g$, score function $f$ (following GPFedRec [47] / PFedRec [46] architecture)
- **Training details**: SGD, lr=0.1, batch=256, 100 rounds, 4 negative samples per positive
- **Data preprocessing**: Implicit feedback (all interactions binarized to 1). Leave-one-out split: most recent → test, second most recent → validation, rest → training. Filter users with <5 interactions (for Lastfm-2K).
- **Evaluation**: Leave-one-out + 99 random negative samples → rank 100 items → HR@10, NDCG@10
- **Crypto details**: Ring $\mathbb{Z}_{2^{32}}$, Q=15 fixed-point precision, Beaver triples precomputed offline
- **Hardware**: Intel Xeon Gold 6430 @ 2.40GHz, RTX 4090 GPU
- **Partitioning**: **Each user = one client** (cross-device, natural partitioning, 6040 clients for ML-1M)
- **Tricks & gotchas**:
  - Graph update every 10-20 rounds (not every round) dramatically reduces overhead with minimal performance loss
  - Beaver triples are data-agnostic → precomputed offline once, reused forever
  - Anonymous identity reassignment every round prevents cross-round user tracking
  - Beauty dataset scaled test: users divided into 4 random groups, subgraphs built independently

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users | Items | Interactions | Density |
|---|---|---|---|---|
| Movie-100K | 943 | 1,682 | 100,000 | ~6.3% |
| **Movie-1M** | **6,040** | **3,706** | **1,000,209** | ~4.5% |
| HetRec2011 | 2,113 | 10,109 | 855,598 | ~4.0% |
| Lastfm-2K | 1,600 | 12,454 | 185,650 | ~0.9% |
| Douban | 2,509 | 39,576 | 893,575 | ~0.9% |
| Beauty | 22,363 | 12,101 | 198,502 | ~0.07% |

### 5.2 Key Results (Movie-1M)

| Method | HR@10 | NDCG@10 | Privacy Level |
|---|---|---|---|
| FedMF | 65.94 | 38.73 | Data-level (HE) |
| PFedRec_s ($\epsilon$=1) | 67.21 | 39.19 | Data-level (strong LDP) |
| PFedRec_w ($\epsilon$=5) | 68.11 | 40.25 | Data-level (weak LDP) |
| GPFedRec_s ($\epsilon$=1) | 66.57 | 38.86 | Data-level (strong LDP) |
| GPFedRec_w ($\epsilon$=5) | 67.81 | 40.09 | Data-level (weak LDP) |
| N-E (no privacy) | 68.36 | 42.66 | None |
| **P2FedRec** | **68.01** | **40.21** | **Data + Edge** |
| Li_P2FedRec | 67.59 | 40.02 | Data + Edge (lightweight) |

**Key finding**: P2FedRec achieves 68.01% HR@10 and 40.21% NDCG@10 on Movie-1M while providing **both** data-level and edge-level privacy. This is competitive with N-E (no privacy, 68.36/42.66) and better than PFedRec_s (67.21/39.19) which only provides data-level privacy.

### 5.3 Ablation Highlights
- **$\alpha$ (neighbor threshold)**: Best at 0.7. Too low (0.1) → graph too dense, user-specific embeddings collapse to global average. Too high (0.9) → too few neighbors, biased personalization.
- **$\beta$ (regularization)**: Best at 0.3. Too high → user deviates from own preferences toward neighbors.
- **$\gamma$ (privacy budget split)**: Impact depends on $\epsilon$. At low $\epsilon$ (strong privacy), allocate more budget to degree info. At high $\epsilon$, allocate more to adjacency list.
- **Embedding dim $H$**: Optimal at 32. Performance degrades at 64 and 128 (overfitting in cross-device setting with small per-user data).
- **Graph update interval**: Every 10-20 rounds is sufficient (vs every round), reducing overhead 5-10x.
- **Computational overhead per round**: 325.76s for Movie-1M (vs FedMF 16304.32s, PFedRec 151.36s, GPFedRec 204.24s). Offline Beaver triple prep is the expensive part but done once.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
1. **User-specific item embeddings via neighbor aggregation** (Eq. 18): The core idea of averaging item embeddings from similar users is directly analogous to the thesis's global prototype mechanism. But P2FedRec does this per-user (each user gets embeddings from THEIR neighbors), while the thesis uses a single EMA prototype for all users.
2. **Regularization toward personalized embeddings** (Eq. 21, $\beta$=0.3): Similar concept to the thesis's alpha-blended embedding interpolation. P2FedRec uses MSE regularization ($\beta \mathcal{R}(g_i, \hat{g}_i)$), thesis uses direct interpolation ($\alpha \cdot p_{local} + (1-\alpha) \cdot p_{global}$). Both achieve the same goal: balance local vs collaborative knowledge.
3. **Embedding dim = 32 is optimal in cross-device**: The thesis uses 128, which may be fine for cross-silo (more data per client) but is worth testing at 32 for fairer comparison with published baselines.

### 6.2 Potential Integration Points
1. **Privacy layer**: The thesis currently has NO formal privacy guarantees. Adding LDP to item embeddings before upload (like PFedRec/P2FedRec) would strengthen the privacy argument. Specifically, applying randomized response or Laplace noise to the global item embedding parameters before sending to server in `client_app.py`.
2. **Relationship-aware prototype**: Instead of a single global prototype (EMA average), compute per-user prototypes by clustering users based on item embedding similarity (a simplified version of P2FedRec's relationship graph, without the two-server crypto overhead). This could be implemented in `strategy.py`.
3. **Baseline implementation**: P2FedRec's FedMF/PFedRec/GPFedRec results on Movie-1M (Table 2) use **the same evaluation protocol** (leave-one-out + 99 negatives) but with **each user = 1 client**. These numbers confirm the comparison gap identified earlier — the thesis's cross-silo setting is fundamentally different.

### 6.3 Limitations & Gaps
1. **Cross-device only**: P2FedRec assumes each user = 1 client (6,040 clients for ML-1M). The thesis uses cross-silo (5-10 clients with grouped users). Results are NOT directly comparable.
2. **Two-server assumption**: Requires two non-colluding servers — a strong assumption not applicable to the thesis's single-server Flower setup.
3. **No BPR loss**: Uses BCE on implicit feedback, not BPR pairwise ranking. The thesis's BPR approach optimizes ranking directly.
4. **Small embedding dim**: Optimal at 32, while thesis uses 128. This reflects the cross-device vs cross-silo difference (more data per client → larger models feasible).
5. **High computation overhead**: Beaver triple generation and ASS-based similarity computation add significant overhead (325s/round for ML-1M). Not practical for the thesis's Flower simulation.
6. **No Dirichlet partitioning**: Natural user-level partition means no control over non-IID degree. Thesis's Dirichlet(0.5) creates controlled heterogeneity between organizational clients.

## 7. Key References to Follow
- **[46] Zhang et al., IJCAI 2023 (PFedRec)** — The main personalized FedRec baseline; uses client-specific score functions + LDP item perturbation. Already identified as key comparison target.
- **[47] Zhang et al., KDD 2024 (GPFedRec)** — Graph-guided personalization (P2FedRec's direct predecessor without privacy). Shows relationship graphs improve FedRec significantly.
- **[6] Chai et al., 2020 (FedMF)** — Secure federated MF with HE-encrypted gradients. Establishes that item embeddings leak user interaction data.
- **[34] Singhal et al., NeurIPS 2021 (FedRecon)** — Partially local federated learning via reconstruction. Alternative approach to split learning.
- **[13] Han et al., SIGIR 2025 (FedCIA)** — Federated collaborative information aggregation for privacy-preserving recommendation. Very recent work in the same space.
