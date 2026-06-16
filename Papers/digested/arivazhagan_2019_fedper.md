# Federated Learning with Personalization Layers

- **Authors**: Manoj Ghuhan Arivazhagan (Adobe Research), Vinay Aggarwal (IIT Roorkee), Aaditya Kumar Singh (IIT Kharagpur), Sunav Choudhary (Adobe Research)
- **Venue**: arXiv:1912.00818v1, Preliminary work under review at AISTATS 2020
- **Paper ID**: arivazhagan_2019_fedper
- **Tags**: #federated-learning, #personalization, #model-splitting, #non-iid, #deep-learning

---

## 1. Core Idea

FedPer decomposes a deep neural network into **base layers** (lower layers, shared globally via FedAvg) and **personalization layers** (upper layers, kept entirely local on each client). Base layers learn shared feature representations collaboratively across all clients, while personalization layers capture user-specific decision boundaries. This simple structural split combats statistical heterogeneity without requiring any changes to the aggregation algorithm -- the server runs standard FedAvg on the base layers only, while personalization layers are trained purely with local SGD and never leave the device.

## 2. Problem & Motivation

Standard FedAvg trains a single global model replicated on every client. This fails for personalization tasks where the same input receives different labels from different users (e.g., image aesthetics). Individual users lack enough data to train in isolation, so collaborative learning is needed, but a one-size-fits-all global model cannot capture divergent preferences. FedPer draws from **multi-task learning**: base layers = shared layers (general features), personalization layers = task-specific heads (user-specific aspects). Smith et al. (2017) showed multi-task formulations address FL heterogeneity but were restricted to convex models. FedPer extends this to deep feedforward networks.

## 3. Method

### 3.1 Objective Function

The model for client $j$ is decomposed into base layers $W_B$ (shared) and personalization layers $W_{P_j}$ (local):

$$\hat{y} = f(x; W_B, W_{P_j})$$

The global objective minimizes the **average personalized population risk**:

$$\mathcal{L}^{PR}(W_B, W_{P_1}, \ldots, W_{P_N}) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{E}_{(x,y) \sim P_j} [\ell(y, f(x; W_B, W_{P_j}))]$$

Where:
- $W_B$ — base layer parameters (shared globally via FedAvg)
- $W_{P_j}$ — personalization layer parameters for client $j$ (local, never communicated)
- $P_j$ — data distribution at client $j$
- $N$ — total number of clients

In practice, empirical risk at client $j$:

$$\mathcal{L}_j^{ER}(W_B, W_P) = \frac{1}{n_j} \sum_{i=1}^{n_j} \ell(y_{j,i}, f(x_{j,i}; W_B, W_P))$$

### 3.2 Algorithm

**Client-side (FedPer-Client $j$):**
1. Initialize $W_{P_j}^{(0)}$ at random
2. Send $n_j$ (local dataset size) to server
3. For each global round $k = 1, 2, \ldots$:
   - Receive $W_B^{(k-1)}$ from server
   - Run SGD on local data, updating BOTH $(W_{B,j}^{(k)}, W_{P_j}^{(k)})$ jointly
   - Send ONLY $W_{B,j}^{(k)}$ to server (personalization layers never leave client)

**Server-side (FedPer-Server):**
1. Initialize $W_B^{(0)}$ at random
2. Compute aggregation weights $\gamma_j = n_j / \sum n_j$
3. For each global round $k = 1, 2, \ldots$:
   - Receive $W_{B,j}^{(k)}$ from each client $j$
   - Aggregate: $W_B^{(k)} = \sum_{j=1}^{N} \gamma_j W_{B,j}^{(k)}$ (weighted FedAvg)
   - Send $W_B^{(k)}$ to all clients

**Fine-tuning variant**: After receiving updated base layers, freeze base and fine-tune personalization for 1 epoch. Helps personalization layers adapt to new base parameters.

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Number of base layers | $K_B$ | Architecture-dependent (e.g., 14/16 blocks for ResNet-34) | Controls how many layers are shared globally |
| Number of personalization layers | $K_P$ | 1-4 basic blocks | Controls local layers; $K_P=0$ reduces to FedAvg |
| Global rounds | $R$ | 100 | Communication rounds |
| Local epochs per round | $e$ | 4 | SGD epochs between aggregations |
| Learning rate | $\eta$ | 0.01 | Constant across all rounds/clients |
| Batch size | $b$ | 128 (CIFAR) / 4 (FLICKR-AES) | Mini-batch size |
| Number of clients | $N$ | 10 (CIFAR) / 30 (FLICKR-AES) | Total devices |
| Non-IID degree | $k$ | {4, 8, 10} for CIFAR-10 | Lower $k$ = more non-IID |

### 3.4 Architectural Decisions

1. **Split point in "basic blocks"**: For CNNs, split is in units of basic blocks, not individual layers. $K_P=1$ = classifier only personalized. Architecture-agnostic specification.
2. **Personalization layers always include classifier**: Final FC layer is always local (minimum personalization).
3. **No change to FedAvg aggregation**: Server runs exactly FedAvg on base layers. Composable with FedProx, compression, etc.
4. **Joint local training**: Both layer types updated simultaneously during local SGD (not alternating).
5. **FedPer reduces to FedAvg when $K_P = 0$**: Clean special case.

## 4. Implementation Notes

- **Framework**: Custom implementation
- **Models tested**: ResNet-34 (16 basic blocks), MobileNet-v1 (11 basic blocks)
- **Training**: SGD, lr=0.01 constant, no LR schedule
- **Non-IID partitioning**: Restricts each client to at most $k$ classes, samples evenly distributed
- **Full participation**: All $N$ clients participate every round
- **Tricks & gotchas**: Communication savings modest for deep CNNs (most params in early layers). No dropout/DP/secure aggregation. Personalization layers initialized independently at each client.

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Type | Samples | Clients | Classes |
|---|---|---|---|---|
| CIFAR-10 | Image classification | 60K | 10 | 10 |
| CIFAR-100 | Image classification | 60K | 10 | 100 |
| FLICKR-AES | Personalized image aesthetics | 40K photos | 30 | Regression binarized |

### 5.2 Key Results

- **CIFAR-10 ($k=4$, highly non-IID)**: FedPer ~85-90% accuracy vs FedAvg ~60-70% on MobileNet-v1
- **CIFAR-10 ($k=10$, IID)**: FedPer converges toward FedAvg performance
- **FLICKR-AES**: FedAvg completely fails (~20% = random). FedPer achieves ~40-42%. Strongest evidence for personalization.
- FedPer produces **lower cross-client accuracy variance** (fairer outcomes)

### 5.3 Ablation Highlights

1. **$K_P$ choice**: No monotonic relationship. $K_P=2$ best for CIFAR-10, $K_P=1$ for CIFAR-100. At least $K_P \geq 1$ consistently helps.
2. **Non-IID degree**: FedPer's advantage most pronounced at high non-IID (low $k$). Converges to FedAvg at IID.
3. **Fine-tuning**: Improves on CIFAR-100, no effect on FLICKR-AES.
4. **Local-only training**: Significantly worse than FedPer, confirming value of collaborative base layers.
5. **Base layer ablation**: Replacing base with single linear layer drops performance significantly.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **The base + personalization split is exactly what the thesis implements for MF**: Item embeddings (GLOBAL, aggregated) = base layers; user embeddings (LOCAL, never sent) = personalization layers. FedPer provides theoretical/empirical foundation for this split.
2. **Thesis progression maps to FedPer**: `federated-baseline-cf/` (all params global) = FedPer with $K_P=0$. `federated-personalized-cf/` (local user embeddings) = FedPer with user embedding as personalization layer. Progression: $K_P=0 \to K_P=1$ (fixed) $\to K_P=1$ (adaptive alpha).
3. **Joint local training**: FedPer trains both layer types jointly with SGD, exactly like split-learning BPR-MF.
4. **Fine-tuning after receiving new base**: After receiving global item embeddings, optionally freeze items and fine-tune user embeddings for 1 epoch. Could help sparse users.

### 6.2 Potential Integration Points

1. **FedPer as cited framework**: $W_B = \{\text{item\_embeddings, item\_bias, global\_bias}\}$, $W_{P_j} = \{\text{user\_embeddings}_j, \text{user\_bias}_j\}$. Thesis contribution: "FedPer proposes binary split; we extend with adaptive alpha-blending."
2. **Adaptive alpha as soft generalization**: FedPer = hard split (100% global or 100% local). Alpha creates a continuum: $\alpha=1.0$ = fully local (FedPer), $\alpha=0.0$ = fully global (FedAvg). Strictly more general.
3. **Per-user $K_P$ analogy**: FedPer uses fixed $K_P$ for all clients. Per-user alpha gives each user different "degree of personalization." Sparse users benefit from more global knowledge.
4. **Dual-level extends FedPer**: Level 2 PersonalMLP = deeper personalization layer in FedPer's framework.

### 6.3 Limitations & Gaps

1. **Only tested on CNNs**: No recommendation or CF experiments. Must argue principle transfers to MF.
2. **Binary split, no gradient**: No mechanism for partial sharing/blending. Adaptive alpha fills this gap.
3. **No per-client adaptation**: Fixed $K_P$ for all clients. Per-user alpha addresses this.
4. **No convergence guarantees**: Purely empirical.
5. **No recommendation metrics**: No NDCG, Hit Rate, ranking evaluation.
6. **Full participation assumed**: No partial participation analysis.

## 7. Key References to Follow

- **McMahan et al., 2017 (FedAvg)** — Original federated averaging; FedPer's baseline
- **Smith et al., NeurIPS 2017 (MOCHA)** — Federated multi-task learning for convex models; theoretical inspiration
- **Zhao et al., 2018** — FedAvg failure modes under non-IID; contextualizes why FedPer is needed
- **Sahu et al., 2018 (FedProx)** — Complementary to FedPer (can use FedProx for base layer aggregation)
- **Vepakomma et al., 2018 (Split Learning)** — Related structural split for privacy in healthcare
