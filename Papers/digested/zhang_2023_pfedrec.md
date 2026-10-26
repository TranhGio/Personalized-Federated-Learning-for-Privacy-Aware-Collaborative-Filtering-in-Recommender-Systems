# Dual Personalization on Federated Recommendation

- **Authors**: Chunxu Zhang, Guodong Long, Tianyi Zhou, Peng Yan, Zijian Zhang, Chengqi Zhang, Bo Yang
- **Venue**: IJCAI 2023 (International Joint Conference on Artificial Intelligence)
- **Paper ID**: zhang_2023_pfedrec
- **Tags**: #federated-learning, #personalization, #recommender-system, #collaborative-filtering, #implicit-feedback, #model-splitting

---

## 1. Core Idea
PFedRec achieves **dual personalization** in federated recommendation by keeping a personalized **score function** (e.g., one-layer MLP) entirely local on each client, and **post-tuning** the shared item embedding on each client's data to create user-specific item representations. Unlike prior methods that only personalize user embeddings, PFedRec removes user embeddings entirely and personalizes both the scoring logic and the item view, achieving SOTA on multiple benchmarks. The dual personalization mechanism is modular and can be plugged into any existing FedRec method for immediate improvement.

## 2. Problem & Motivation
Existing federated recommendation methods (FedMF, FedNCF, MetaMF, FedPerGNN) share identical item embeddings across all users and only personalize via user embeddings. This ignores that: (1) users have different **decision-making logic** (captured by score function), and (2) users perceive items differently — the same movie may be "action" to one user and "comedy" to another (captured by personalized item embeddings). PFedRec addresses both by deploying lightweight per-client models rather than a heavyweight server model, aligning with on-device intelligence architectures.

## 3. Method

### 3.1 Objective Function

The learning is formulated as a bi-level optimization:

$$\min_{\theta^m, \{\theta_i\}_{i=1}^{N}} \sum_{i=1}^{N} \alpha_i L_i(\theta_i; r, \hat{r})$$

$$\text{s.t.} \quad \theta_i := (\theta^m - \nabla_{\theta^m} L_i, \theta_i^s)$$

Where:
- $\theta^m$ — item embedding module parameters (GLOBAL, aggregated on server)
- $\theta_i^s$ — score function parameters for client $i$ (LOCAL, never sent to server)
- $\theta_i := (\theta_i^m, \theta_i^s)$ — full personalized model for client $i$
- $\alpha_i := |D_i| / \sum_{j=1}^{N} |D_j|$ — client weight proportional to data size
- $L_i$ — BCE loss on client $i$'s local data $D_i$

**Prediction model:**

$$\hat{r}_{ij} = S_i(E_i(e^j))$$

Where $S_i$ is client $i$'s personalized score function, $E_i$ is the (post-tuned) item embedding, and $e^j$ is the one-hot encoding of item $j$.

**Loss function (BCE on implicit feedback):**

$$L_i(\theta_i; r, \hat{r}) = -\sum_{(i,j) \in D_i} \log \hat{r}_{ij} - \sum_{(i,j') \in D_i^-} \log(1 - \hat{r}_{ij'})$$

Where $D_i^-$ is the negative instance set, constructed by uniformly sampling from uninteracted items $\mathcal{I}_i^- = \mathcal{I} \setminus \mathcal{I}_i$ with a predefined sampling ratio.

### 3.2 Algorithm

**Server-side:**
1. Initialize item embedding $\theta^m$ and score function $\theta^s$
2. For each round $t = 1, 2, ..., T$:
   1. Select client set $S_t$ of size $n$ randomly from all $N$ clients
   2. Distribute global item embedding $\theta^m$ to each client $i \in S_t$
   3. Receive post-tuned item embeddings $\theta_i^m$ from clients
   4. Aggregate: $\theta^m \leftarrow \frac{1}{n} \sum_{i=1}^{n} \theta_i^m$

**Client-side (ClientUpdate for client $i$):**
1. Initialize $\theta_i^m \leftarrow \theta^m$ (replace local item embedding with global)
2. Initialize $\theta_i^s$ with latest locally stored score function
3. Construct negative instance set: $\mathcal{I}_i^- = \mathcal{I} \setminus \mathcal{I}_i$, sample $D_i^-$ from $\mathcal{I}_i^-$
4. Create batches $\mathcal{B}$ from $D_i \cup D_i^-$ with batch size $B$
5. For $e = 1$ to $E$ local epochs:
   1. For each batch $b \in \mathcal{B}$:
      1. Compute loss $L_i(\theta_i; r, \hat{r})$ with Eq. (7) using current $\theta_i^m$ and $\theta_i^s$
      2. **Update score function** (keep item embedding fixed): $\theta_i^s \leftarrow \theta_i^s - \eta \nabla_{\theta^s} L_i$
      3. Recompute loss with the **updated** $\theta_i^s$ (this is the "post-tuning" step)
      4. **Update item embedding** (with personalized score function): $\theta_i^m \leftarrow \theta_i^m - \eta' \nabla_{\theta^m} L_i$
6. Return $\theta_i^m$ to server (score function $\theta_i^s$ stays local)

**Key insight — alternating optimization**: The item embedding is updated AFTER the score function within each batch. This means the item embedding gradient flows through the already-personalized score function, creating user-specific item representations. This "post-tuning" is the mechanism that produces personalized item embeddings.

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Embedding dimension | $H$ | 32 | User/item latent factor size |
| Batch size | $B$ | 256 | Training batch size |
| Training negatives | — | 4 per positive | Negative sampling ratio |
| Eval negatives | — | 99 | Leave-one-out evaluation |
| Score function LR | $\eta$ | 0.1 (SGD) | Learning rate for score function |
| Item embedding LR | $\eta'$ | $\eta \times |M| \times \eta_{scale}$ | Much higher LR for item embedding post-tuning |
| LR scale factor | $\eta_{scale}$ | 80 | Multiplier for item embedding LR (from code) |
| Communication rounds | $T$ | 100 | Total FL rounds |
| Local epochs | $E$ | 1 | Per-round local training epochs |
| Client sample ratio | — | 1.0 | All clients participate each round |
| Number of clients | $N$ | = num users | Each user is one client |

### 3.4 Architectural Decisions

**Why remove user embeddings?** In federated settings, the score function is already personalized per client, so it implicitly captures user preferences. Adding user embeddings is redundant — the score function IS the user representation. This also reduces model complexity on-device.

**Why personalize item embeddings?** Different users have different views of items. A shared global item embedding forces all users to see items the same way. Post-tuning allows each user to slightly adjust item representations to match their preferences, while still benefiting from collaborative knowledge via the global initialization.

**Why alternating optimization (not joint)?** Training the score function first, then post-tuning item embeddings ensures the item embedding update is guided by the personalized scoring logic. If trained jointly, the item embedding gradient would not benefit from the user-specific score function signal.

**Why one-layer MLP as score function?** Simplicity and efficiency for on-device deployment. The paper notes this can be replaced with deeper networks but one layer suffices. With $H=32$, the score function is just $\text{Linear}(32, 1) + \text{Sigmoid}$ = 33 parameters per client.

**Efficient on-device update**: Each client only maintains item embeddings for interacted items + sampled negatives, not all items. This dramatically reduces memory requirements for clients with few interactions.

## 4. Implementation Notes
- **Framework**: PyTorch
- **Code**: https://github.com/Zhangcx19/IJCAI-23-PFedRec
- **Model architecture**: Item embedding $\in \mathbb{R}^{|M| \times 32}$ (global) + Score function = $\text{Linear}(32, 1)$ (local per client)
- **Training details**: SGD optimizer, score function lr=0.1, item embedding lr = 0.1 * num_items * 80 (extremely high for aggressive post-tuning), batch=256, 1 local epoch, 100 rounds
- **Data preprocessing**: Implicit feedback (binarized). Leave-one-out: most recent interaction = test, second most recent = validation, rest = training (by timestamp). 4 negative samples per positive for training.
- **Evaluation**: Leave-one-out + 99 random negative samples, rank 100 items, HR@10 and NDCG@10
- **Partitioning**: **Each user = one client** (natural cross-device, 6040 clients for ML-1M)
- **Repetitions**: 5 trials, report mean ± std
- **Tricks & gotchas**:
  - The item embedding LR is orders of magnitude higher than the score function LR — this is intentional for aggressive post-tuning
  - `affine_output.weight` (score function) is explicitly excluded from server aggregation in code
  - Client sample ratio = 1.0 for reported results (all 6040 clients per round)

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users | Items | Interactions | Sparsity |
|---|---|---|---|---|
| MovieLens-100K | 943 | 1,682 | 100,000 | 93.70% |
| **MovieLens-1M** | **6,040** | **3,706** | **1,000,209** | **95.53%** |
| Lastfm-2K | 1,600 | 12,454 | 185,650 | 99.07% |
| Amazon-Video | 8,072 | 11,830 | 63,836 | 99.93% |

### 5.2 Key Results

**MovieLens-1M (Table 2):**

| Method | HR@10 | NDCG@10 | Type |
|---|---|---|---|
| NCF (centralized) | 64.17 ± 0.99 | 37.85 ± 0.68 | CenRec |
| MF (centralized) | 68.45 ± 0.34 | 41.37 ± 0.18 | CenRec |
| FedMF | 67.72 ± 0.14 | 40.90 ± 0.14 | FedRec |
| FedNCF | 60.54 ± 0.46 | 34.17 ± 0.40 | FedRec |
| FedRecon | 63.28 ± 0.15 | 36.59 ± 0.33 | FedRec |
| MetaMF | 45.61 ± 0.18 | 25.24 ± 0.35 | FedRec |
| FedPerGNN | 9.69 ± 0.23 | 4.37 ± 0.31 | FedRec |
| **PFedRec (Ours)** | **73.26 ± 0.20** | **44.36 ± 0.16** | **FedRec** |

PFedRec beats ALL baselines including centralized MF (68.45→73.26 HR@10). The federated personalized approach actually outperforms centralized training because per-user score functions capture individual preferences better than a shared model.

**DualPer integration into other methods (Table 3, ML-1M):**

| Method | HR@10 | NDCG@10 | Improvement |
|---|---|---|---|
| FedMF | 67.72 | 40.90 | — |
| FedMF + DualPer | 73.26 | 44.36 | +8.18% / +8.46% |
| FedNCF | 60.54 | 34.17 | — |
| FedNCF + DualPer | 68.17 | 39.56 | +12.60% / +15.77% |
| FedRecon | 63.28 | 36.59 | — |
| FedRecon + DualPer | 68.89 | 40.04 | +8.87% / +9.43% |

**LDP integration (Table 4, ML-1M):**

| Noise $\lambda$ | HR@10 | NDCG@10 |
|---|---|---|
| 0 | 73.26 | 44.36 |
| 0.1 | 73.13 | 44.16 |
| 0.2 | 73.05 | 44.25 |
| 0.3 | 73.18 | 44.23 |
| 0.4 | 73.08 | 44.18 |
| 0.5 | 73.08 | 44.18 |

LDP with moderate noise ($\lambda=0.3$) has negligible impact on performance.

### 5.3 Ablation Highlights
- **t-SNE visualization (Fig 2)**: PFedRec clearly separates positive (interacted) and negative items in embedding space, while baselines (FedNCF, FedRecon, FedMF) show mixed embeddings. This confirms personalized item embeddings learn user-specific item views.
- **Inference comparison (Fig 3)**: Using a client's "Own" post-tuned item embedding >> "Global" shared embedding >> "Random" user's embedding. This proves the personalization is meaningful and user-specific, not just noise.
- **DualPer is modular**: The mechanism provides consistent 8-18% improvement when plugged into FedMF, FedNCF, or FedRecon. The improvement is larger on MovieLens datasets (more interactions per user → better personalization signal).

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **PFedRec IS the primary SOTA baseline** for the thesis. Its exact architecture (item embedding global, score function local) and ML-1M results (HR@10=73.26, NDCG@10=44.36) are the target to beat or match.

2. **The "post-tuning" concept** is directly comparable to the thesis's alpha-blended embedding interpolation. PFedRec post-tunes item embeddings via gradient descent on each client. The thesis interpolates: $p_{effective} = \alpha \cdot p_{local} + (1-\alpha) \cdot p_{global}$. Both achieve "personalized view of shared parameters" but through different mechanisms.

3. **Score function as local personalization** maps to the thesis's PersonalMLP in Dual-Level architecture. PFedRec uses `Linear(32,1)+Sigmoid` (33 params). The thesis uses `MLP(512,256,128)` with concat fusion (much larger). The thesis's approach is more expressive but PFedRec's is more parameter-efficient.

4. **Embedding dim = 32** is critical. PFedRec uses 32, thesis uses 128. For fair comparison, experiments should be run at matching embedding dimensions, or both dimensions should be reported.

### 6.2 Potential Integration Points

1. **Implement PFedRec as a baseline in the thesis's Flower framework**: Create a `pfedrec_baseline/` module with the exact PFedRec architecture (item embedding global + one-layer MLP local + post-tuning). Run with the thesis's Dirichlet partitioning to get a fair cross-silo comparison. This is the **most important next step** for thesis credibility.

2. **Adapt post-tuning to BPR loss**: PFedRec uses BCE; the thesis uses BPR. The post-tuning mechanism (train score function first, then fine-tune item embedding) can be applied with BPR loss. In `task.py`, modify the training loop to alternate: (a) update user embedding + personal MLP with BPR, (b) post-tune item embedding with BPR using the updated user-side parameters.

3. **DualPer as a plug-in for the thesis approach**: Table 3 shows DualPer improves any FedRec method by 8-18%. The thesis could apply the post-tuning mechanism on top of its adaptive alpha approach — after alpha-blending the embeddings, additionally post-tune the item embedding with a separate gradient step.

### 6.3 Limitations & Gaps

1. **Cross-device only**: PFedRec assumes 1 user = 1 client (6040 clients). The thesis uses cross-silo (5-10 clients). **Results are NOT directly comparable** without reimplementing PFedRec in the thesis's setting.

2. **No user embeddings**: PFedRec removes user embeddings entirely. The thesis preserves them as local parameters. This is a fundamental design difference — the thesis argues user embeddings are valuable for privacy AND personalization.

3. **BCE loss, not BPR**: PFedRec uses pointwise BCE. The thesis uses pairwise BPR (ranking-optimized). Direct metric comparison is confounded by loss function choice.

4. **No adaptive personalization**: PFedRec applies the same personalization mechanism to all users regardless of data quantity/quality. The thesis's hierarchical conditional alpha adapts personalization level per user — a contribution PFedRec doesn't address.

5. **No formal privacy analysis**: PFedRec mentions LDP as an add-on (Table 4) but doesn't provide formal DP guarantees. The thesis could claim stronger privacy analysis as a contribution.

6. **Very high item embedding LR**: The code uses lr * num_items * 80 for item embedding, which is an unusual and potentially fragile hyperparameter. The thesis's approach of alpha-blending is more principled.

## 7. Key References to Follow
- **[Singhal et al., NeurIPS 2021] FedRecon** — Partially local federated learning; preserves local model + trains global collaboratively. Alternative to PFedRec's approach.
- **[Fallah et al., NeurIPS 2020] Per-FedAvg** — MAML-based personalized FL; PFedRec's bi-level formulation is inspired by this. Useful for understanding the optimization perspective.
- **[Lin et al., SIGIR 2020] MetaMF** — Meta-network for federated MF; generates rating prediction + private item embeddings. Different personalization philosophy.
- **[Wu et al., Nature Comms 2022] FedPerGNN** — Graph-based federated recommendation; poor results in PFedRec's experiments, but GNN approach is relevant to P2FedRec's relationship-awareness.
- **[Perifanis & Efraimidis, KBS 2022] FedNCF** — Federated Neural CF; the FedNCF baseline used in experiments. Relevant as another approach to federated neural recommendation.
