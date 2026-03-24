# Co-clustering for Federated Recommender System

- **Authors**: Xinrui He, Shuo Liu, Jacky Wai Keung, Jingrui He
- **Venue**: WWW 2024 (The ACM Web Conference)
- **Paper ID**: he_2024_cofedrec
- **Tags**: #federated-learning, #recommender-system, #personalization, #collaborative-filtering, #implicit-feedback, #aggregation, #non-iid

---

## 1. Core Idea
CoFedRec groups clients via **item-category-based co-clustering** instead of traditional user clustering (K-Means on gradients/embeddings, which fails in high-dimensional sparse FedRec). For a randomly chosen item category each round, clients are split into "similar" and "dissimilar" groups using an elbow-point method on cosine similarity scores. Only the similar group receives an aggregated group model, while dissimilar clients keep their local models. A **supervised contrastive learning** term using server-provided item cluster membership is added to local training to encode global item relationships.

## 2. Problem & Motivation
K-Means clustering on client item embeddings/gradients fails catastrophically in FedRec due to: (1) **curse of dimensionality** — flattened item embedding matrices are very high-dimensional and sparse, making distances nearly uniform; (2) K-Means produces heavily imbalanced clusters (e.g., k=10 on ML-100K: 8/10 clusters contain just 1 user). CoFedRec sidesteps this by clustering on **per-item-category similarity** (lower-dimensional, denser signal) and using the elbow method instead of fixed k.

## 3. Method

### 3.1 Objective Function

$$L_u(V_u, \theta_u) = -\sum_{(u,i) \in D_u} \log \hat{r}_{ui} - \sum_{(u,i') \in D_u^-} \log(1 - \hat{r}_{ui'}) + \lambda L_{sup}$$

Where:
- $V_u$ — client $u$'s item embedding network (personalized via post-tuning)
- $\theta_u$ — client $u$'s score function parameters (LOCAL, never sent to server)
- $\hat{r}_{ui}$ — predicted score via score function $\theta_u$ applied to item embedding $V_u$
- $D_u$ — positive interactions, $D_u^-$ — sampled negative interactions
- $\lambda$ — weight for supervised contrastive learning term
- $L_{sup}$ — supervised contrastive loss using item cluster membership

**Supervised Contrastive Loss:**

$$L_{sup} = -\sum_{i \in I} \log \left\{ \frac{1}{|Z(i)|} \sum_{z \in Z(i)} \left( \frac{\exp(V_{u,i} \cdot V_{u,z} / \tau)}{\sum_{a \in I \setminus \{i\}} \exp(V_{u,i} \cdot V_{u,a} / \tau)} \right) \right\}$$

Where:
- $Z(i) \equiv \{z \in I \setminus \{i\} : \bar{y}_z = \bar{y}_i\}$ — items in the same cluster as item $i$
- $\tau$ — temperature parameter
- $V_{u,i}$ — embedding vector for item $i$ in user $u$'s item network

### 3.2 Algorithm

**Server-side (per round $t$):**
1. Receive updated item embeddings $V_i$ from all participant clients $P$
2. **Global aggregation**: $V_g \leftarrow \frac{1}{|P|} \sum_{u=1}^{|P|} V_u$
3. **Item clustering**: Apply K-Means on global item embeddings $\{V_{g,i}\}_{i=1}^{|I|}$ to get $K$ clusters → item membership vector $M \in \mathbb{R}^{1 \times |I|}$ where $M[j]$ = cluster index of item $j$
4. **User partitioning** (co-clustering):
   1. Randomly select a core client $c \in P$
   2. Randomly select an item category $k$ from item membership $M$
   3. Get items in category: $M_k = \{i \mid M[i] = k\}$
   4. Compute cosine similarity between core client $c$ and each client $u$ on category-$k$ items:
      $$s_u = \sum_{i \in M_k} \frac{V_{c,i} \cdot V_{u,i}}{|V_{c,i}| \cdot |V_{u,i}|}, \quad u \in P$$
   5. Sort similarity scores, find elbow point $e$ using orthogonal distance to line connecting first and last points:
      - Line: $L(x) = s_{1'} + x(s_{|P|'} - s_{1'})$
      - For each point $s_u$: compute projection $x_u$ and distance $d_u = |h_u - x_u(s_{|P|'} - s_{1'})|$
      - Elbow = point with maximum distance $d_e$
   6. Split: $D_s$ (similar, $d_u \geq d_e$) and $D_{dis}$ (dissimilar, $d_u < d_e$)
5. **Group aggregation**: $V_s \leftarrow \frac{1}{|D_s|} \sum_{u \in D_s} V_u$
6. Send $V_s$ and $M$ to similar-group clients; send only $M$ to dissimilar-group clients

**Client-side (per round):**
1. If in similar group: initialize $V_u \leftarrow V_s$ (group model). If dissimilar group or round 0: initialize $V_u \leftarrow V_0$ (keep local / use initial)
2. Download item membership $M$ from server
3. Sample negative instances $D_u^-$
4. For each local epoch $e$:
   1. For each batch $b$:
      1. Compute loss $L_u(V_u, \theta_u)$ with BCE + supervised contrastive (Eq. 9)
      2. Update score function $\theta_u$ via SGD
      3. Update item embedding $V_u$ via SGD (post-tuning, same as PFedRec)
5. Return $V_u$ to server

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | ML-100K | ML-1M | Role |
|---|---|---|---|---|
| Contrastive weight | $\lambda$ | 0.005 | 0.005 | Weight for supervised contrastive loss |
| Temperature | $\tau$ | 0.1 | 0.5 | Contrastive loss temperature |
| Item clusters | $K$ | 30 | 45 | Number of K-Means clusters for items |
| Best round | — | 93 | 70 | Round with best validation performance |
| Learning rate | — | 0.05 | 0.05 | SGD learning rate |
| Embedding dim | $d$ | 32 | 32 | Item embedding dimensionality |
| Batch size | $B$ | 256 | 256 | Training batch size |
| Training negatives | — | 4 | 4 | Negative samples per positive |
| Eval negatives | — | 99 | 99 | Leave-one-out evaluation |
| Training rounds | $T$ | 100 | 100 | Total communication rounds |

### 3.4 Architectural Decisions

**Why cluster items instead of users?** User clustering via K-Means on item embedding matrices fails due to curse of dimensionality — flattened embeddings are too high-dimensional and sparse. Item clustering on the globally aggregated embedding (lower-dimensional, denser) succeeds because the server has access to the full global item embedding matrix.

**Why co-cluster on one item category per round?** Users don't have identical preferences across ALL items, but may agree on specific categories. Clustering per-category captures fine-grained preference similarity. Over many rounds, different categories are sampled, covering the full preference space.

**Why elbow method instead of fixed k?** The number of similar/dissimilar users varies per item category. A fixed k would be arbitrary. The elbow method adaptively finds the natural split point in the similarity distribution.

**Why supervised contrastive loss?** Dissimilar-group clients miss the group aggregation. The item membership $M$ provides global structure information to ALL clients. The contrastive loss ensures local item representations respect global inter-item relationships even without group model updates.

**Backbone = PFedRec**: CoFedRec uses PFedRec's dual personalization (local score function + post-tuned item embedding) as its base architecture. The co-clustering and contrastive loss are added on top.

## 4. Implementation Notes
- **Framework**: PyTorch
- **Backbone model**: PFedRec (item embedding global + one-layer MLP score function local)
- **Training details**: SGD, lr=0.05 (CoFedRec), batch=256, 100 rounds, 4 negatives per positive
- **Data preprocessing**: Implicit feedback (binarized). Leave-one-out by timestamp: latest → test, second latest → validation, rest → training. Users with <5 interactions filtered (FilmTrust, LastFM-2K).
- **Evaluation**: Leave-one-out + 99 negative samples, K=10
- **Partitioning**: **Each user = one client** (cross-device, 6040 clients for ML-1M)
- **Repetitions**: 5 trials, report lowest value (conservative estimate)
- **Tricks & gotchas**:
  - Item clusters K is dataset-dependent: ML-100K=30, ML-1M=45, FilmTrust=30, LastFM-2K=500
  - The contrastive weight $\lambda$ is very small (0.005) — too large degrades performance (Table 8: $\lambda$=0.3 drops HR by ~15pp)
  - Temperature $\tau$ also dataset-dependent: 0.1 for ML-100K, 0.5 for ML-1M
  - Centralized MF and NCF use Adam optimizer; all federated methods use SGD
  - GPFedRec results cited from original paper (not re-run)

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users | Items | Interactions | Sparsity |
|---|---|---|---|---|
| MovieLens-100K | 943 | 1,682 | 100,000 | 93.70% |
| **MovieLens-1M** | **6,040** | **3,706** | **1,000,209** | **95.53%** |
| FilmTrust | 1,227 | 2,059 | 34,888 | 98.62% |
| LastFM-2K | 1,600 | 12,454 | 185,650 | 99.07% |

### 5.2 Key Results

**MovieLens-1M — Sampled evaluation (Table 1, HR@10 / NDCG@10):**

| Method | HR@10 | NDCG@10 | Type |
|---|---|---|---|
| MF (centralized) | 68.61 | 41.33 | Centralized |
| NCF (centralized) | 68.76 | 41.90 | Centralized |
| FedMF | 67.52 | 38.12 | Federated |
| FedNCF | 65.78 | 38.67 | Federated |
| FedRecon | 60.43 | 34.89 | Federated |
| MetaMF | 39.82 | 25.07 | Federated |
| PFedRec | 73.62 | 44.35 | Federated |
| GPFedRec | 72.17 | 43.61 | Federated |
| **CoFedRec** | **77.75** | **48.81** | **Federated** |

**CoFedRec beats PFedRec by +4.13pp HR@10 and +4.46pp NDCG@10 on ML-1M.**

**MovieLens-1M — Full-rank evaluation (Table 9):**

| Method | HR@10 | NDCG@10 |
|---|---|---|
| PFedRec | 11.02 | 5.04 |
| **CoFedRec** | **13.20** | **8.90** |

**DualPer (CoFedRec mechanism) integration (Table 3, ML-1M):**

| Base Method | HR@10 | NDCG@10 | +CoFedRec HR@10 | +CoFedRec NDCG@10 | Improvement |
|---|---|---|---|---|---|
| FedMF | 67.52 | 38.12 | 71.39 | 45.10 | +5.73% / +18.31% |
| FedNCF | 65.78 | 38.67 | 66.16 | 41.88 | +0.58% / +8.30% |
| PFedRec | 73.62 | 44.35 | 77.75 | 48.81 | +5.61% / +10.06% |

### 5.3 Ablation Highlights (Table 2, ML-1M)
| Variant | HR@10 | NDCG@10 |
|---|---|---|
| Origin (PFedRec, no clustering) | 73.62 | 44.35 |
| + User_P (co-clustering only) | 73.92 | 45.72 |
| + Item_S (item similarity loss only) | 73.66 | 44.67 |
| + Item_SC (supervised contrastive) | 74.09 | 44.45 |
| **CoFedRec (User_P + Item_SC)** | **77.75** | **48.81** |

Both components contribute, but the combination is synergistic — the full model gains much more than either component alone (+4.13pp over baseline vs +0.30pp and +0.47pp individually).

**Virtual rating privacy (Table 4)**: With noise ratio 0.4, CoFedRec (HR=72.43, NDCG=45.31) still outperforms PFedRec without noise (HR=71.05, NDCG=43.89) on ML-100K.

**Contrastive weight $\lambda$ sensitivity (Table 8, ML-1M)**: Best at $\lambda$=0.005. Performance degrades severely at $\lambda \geq 0.1$.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Co-clustering is a newer SOTA** that beats PFedRec. CoFedRec (77.75/48.81) >> PFedRec (73.26/44.36) on ML-1M. The thesis's comparison target should be updated to include CoFedRec.

2. **Item-category-based user grouping**: The thesis's hierarchical conditional alpha groups users by individual statistics (quantity, diversity, coverage, consistency). CoFedRec groups users by item-category preference similarity. These are complementary approaches — the thesis could combine both: use HC-alpha for within-client personalization level AND co-clustering for between-client aggregation grouping.

3. **Supervised contrastive loss on item embeddings**: Directly applicable. The thesis already has a contrastive loss (`contrastive-lambda`), but it contrasts local vs blended USER embeddings. CoFedRec's contrastive loss operates on ITEM embeddings using cluster membership labels. Both could coexist — user-side contrastive + item-side contrastive.

### 6.2 Potential Integration Points

1. **Implement co-clustering in `strategy.py`**: After global aggregation in `SplitFedAvg`/`SplitFedProx`, add item K-Means clustering on the aggregated item embedding. Send item membership vector $M$ alongside global parameters. In `client_app.py`, use $M$ to compute the supervised contrastive loss during local training. This requires:
   - Adding K-Means to server strategy (on item embedding tensor, K=45 for ML-1M)
   - Passing $M$ via Flower's `config` dict to clients
   - Adding $L_{sup}$ term to the BPR training loss in `task.py`

2. **Selective aggregation by item category**: Instead of aggregating all clients equally (FedAvg), group clients per item category and aggregate within groups. This could replace or augment the thesis's prototype mechanism. Implement in `strategy.py` by computing per-category cosine similarity and using elbow-point partitioning.

3. **Cross-silo adaptation**: CoFedRec assumes 1 user = 1 client. In the thesis's cross-silo setting (5-10 clients with ~600 users each), the co-clustering concept adapts differently — each client already has diverse users, so item-category clustering within each client's user population could be applied locally.

### 6.3 Limitations & Gaps

1. **Cross-device only**: Same as PFedRec — 1 user = 1 client. Not directly comparable to thesis's cross-silo setting.
2. **BCE loss, not BPR**: Uses pointwise BCE, not pairwise BPR ranking loss.
3. **Random core client and category selection**: Each round randomly picks one core client and one item category. Over many rounds this covers the space, but convergence may be slow. The thesis's deterministic alpha computation is more principled.
4. **No formal privacy guarantees**: Virtual rating experiments (Table 4) show noise robustness, but no DP analysis.
5. **Contrastive weight is fragile**: $\lambda$=0.005 is optimal; $\lambda$=0.1 already hurts significantly (Table 8). This narrow sweet spot makes tuning difficult.
6. **K-Means on items still has limitations**: While clustering items (32-dim) is more tractable than clustering users (flattened item matrices), the optimal K varies dramatically across datasets (30 to 500).

## 7. Key References to Follow
- **[66] Zhang et al., IJCAI 2023 (PFedRec)** — Backbone architecture for CoFedRec. Already digested as `zhang_2023_pfedrec`.
- **[65] Zhang et al., 2023 (GPFedRec)** — Graph-guided personalization; predecessor to P2FedRec. Builds user relationship graph from item embeddings.
- **[36] Luo et al., CIKM 2022 (PerFedRec jointly)** — Clusters users via user embeddings, then learns per-cluster models. Alternative clustering approach.
- **[23] Khosla et al., NeurIPS 2020 (SupCon)** — Supervised contrastive learning theory. Foundation for CoFedRec's contrastive loss.
- **[11] Frisch et al., ECML 2021** — Co-clustering for fair recommendation. Related co-clustering concept in centralized setting.
