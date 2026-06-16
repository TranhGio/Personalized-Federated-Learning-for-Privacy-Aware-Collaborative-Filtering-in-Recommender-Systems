# Neural Collaborative Filtering

- **Authors**: Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua
- **Venue**: WWW (International World Wide Web Conference), 2017
- **Paper ID**: he_2017_ncf
- **Tags**: #recommender-system, #collaborative-filtering, #implicit-feedback, #matrix-factorization, #neural-network, #deep-learning

---

## 1. Core Idea

Replace the fixed inner product in Matrix Factorization (MF) with a learnable neural network to model user-item interactions. The paper proposes a general **Neural Collaborative Filtering (NCF)** framework with three instantiations: **GMF** (Generalized Matrix Factorization), **MLP** (Multi-Layer Perceptron), and **NeuMF** (Neural Matrix Factorization) which fuses GMF and MLP. The key insight is that the inner product is a linear interaction function that cannot capture complex, non-linear user-item interaction patterns in the low-dimensional latent space.

## 2. Problem & Motivation

- **Implicit feedback** (clicks, views, purchases) is abundant but noisy: observed entries indicate interest but not necessarily preference; unobserved entries are a mix of negative signal and missing data.
- MF uses a fixed inner product `y_hat_ui = p_u^T * q_i` which is a **linear model** of latent factors.
- The paper provides a geometric example (Figure 1) showing MF with inner product cannot faithfully recover complex user similarity structures in low-dimensional space.
- Prior deep learning work mostly used DNNs for auxiliary features but still relied on inner product for core user-item interaction.

## 3. Method

### 3.1 Objective Function

**Binary cross-entropy (log loss):**

$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} \left[ y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui}) \right]$$

Where:
- $y_{ui} = 1$ if interaction observed, $0$ otherwise
- $\hat{y}_{ui}$ — predicted score (sigmoid output)
- $\mathcal{Y}$ — observed interactions, $\mathcal{Y}^-$ — uniformly sampled negatives

### 3.2 Algorithm

#### GMF (Generalized Matrix Factorization)

$$\hat{y}_{ui} = a_{out}(\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i))$$

If $a_{out}$ = identity and $\mathbf{h}$ = uniform 1s, this **exactly recovers standard MF**.

#### MLP (Multi-Layer Perceptron)

$$\mathbf{z}_1 = [\mathbf{p}_u; \mathbf{q}_i], \quad \phi_l(\mathbf{z}_{l-1}) = a_l(\mathbf{W}_l^T \mathbf{z}_{l-1} + \mathbf{b}_l)$$

$$\hat{y}_{ui} = \sigma(\mathbf{h}^T \phi_L(\mathbf{z}_{L-1}))$$

Tower structure: each successive layer halves the size (e.g., 32→16→8).

#### NeuMF (Neural Matrix Factorization — Fusion)

GMF and MLP use **separate embeddings** (critical design decision):

$$\phi^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G, \quad \phi^{MLP} = a_L(\mathbf{W}_L^T(\ldots))$$

$$\hat{y}_{ui} = \sigma(\mathbf{h}^T [\phi^{GMF}; \phi^{MLP}])$$

**Pre-training**: Train GMF and MLP separately, initialize NeuMF with $\mathbf{h} \leftarrow [\alpha \cdot \mathbf{h}^{GMF}; (1-\alpha) \cdot \mathbf{h}^{MLP}]$ ($\alpha=0.5$). Fine-tune with vanilla SGD (not Adam).

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Predictive factors | $K$ | 8, 16, 32, 64 | Last hidden layer / embedding dim |
| MLP hidden layers | $L$ | 3 (tested 0-4) | Depth of non-linear interaction |
| MLP layer sizes | — | Tower: 4K→2K→K | Each layer halves |
| Learning rate | $\eta$ | {0.0001, 0.0005, 0.001, 0.005} | Adam for standalone; SGD for fine-tuning |
| Batch size | — | {128, 256, 512, 1024} | Mini-batch |
| Negative sampling ratio | — | 3-6 (optimal) | Negatives per positive |
| Pre-training alpha | $\alpha$ | 0.5 | GMF/MLP balance in NeuMF init |
| Activation | — | ReLU | MLP hidden layers |
| Eval negatives | — | 99 | Leave-one-out: 1 pos + 99 neg |

### 3.4 Architectural Decisions

1. **Separate embeddings for GMF/MLP in NeuMF**: Sharing forces same representation, limiting the fused model.
2. **Log loss over BPR**: Log loss with sampling ratio 3-6 significantly outperforms BPR (ratio=1). GMF with ratio=1 matches BPR.
3. **ReLU over sigmoid/tanh**: Avoids saturation, encourages sparse activations.
4. **Tower structure**: Halving forces abstraction. Deeper is better (MLP-4 > MLP-3 > ... >> MLP-0).
5. **Pre-training then SGD**: Adam for initial training, SGD for fine-tuning (avoids stale momentum).

## 4. Implementation Notes

- **Framework**: Keras (Theano backend). Code: github.com/hexiangnan/neural_collaborative_filtering
- **Evaluation**: Leave-one-out. Latest interaction = test. 1 positive + 99 random negatives. HR@10, NDCG@10.
- **Implicit feedback**: Ratings binarized to 0/1.
- **Data filtering**: Users with <20 interactions removed.
- **Negative sampling**: Uniform from unobserved, refreshed each epoch.
- **Init**: Gaussian(0, 0.01).
- **Convergence**: ~50 epochs; overfitting possible after ~15-20 for NeuMF.

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Interactions | Items | Users | Sparsity |
|---|---|---|---|---|
| MovieLens 1M | 1,000,209 | 3,706 | 6,040 | 95.53% |
| Pinterest | 1,500,809 | 9,916 | 55,187 | 99.73% |

### 5.2 Key Results (MovieLens 1M, factors=64)

| Method | HR@10 | NDCG@10 |
|---|---|---|
| ItemPop | ~0.45 | ~0.26 |
| ItemKNN | ~0.65 | ~0.35 |
| BPR | ~0.67 | ~0.36 |
| eALS | ~0.69 | ~0.38 |
| GMF | ~0.71 | ~0.42 |
| MLP (3 layers) | ~0.69 | ~0.41 |
| **NeuMF (pre-trained)** | **0.730** | **0.447** |

NeuMF over eALS: ~4.5% HR@10, ~4.9% NDCG@10 improvement.

### 5.3 Ablation Highlights

- **Pre-training**: Helps significantly at larger factors (2.2% improvement at K=64). Minimal effect at K=8.
- **MLP depth**: Each layer improves consistently. MLP-0 (~0.45 HR) ≈ ItemPop. Stacking linear layers (identity activation) much worse than ReLU.
- **Negative ratio**: Optimal 3-6. Ratio=1 matches BPR; higher ratios outperform BPR.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Evaluation protocol**: This paper establishes leave-one-out + 99 negatives — the canonical reference for the thesis's evaluation approach.
2. **BPR-MF centralized reference**: BPR HR@10≈0.67, NDCG@10≈0.36 on ML-1M (factors=64). Centralized upper-bound for federated BPR-MF.
3. **Negative sampling ratio 3-6**: Directly relevant to BPR training configuration.
4. **Log loss as BCE alternative**: PFedRec already uses BCE; NCF validates this choice.

### 6.2 Potential Integration Points

1. **NeuMF as centralized baseline**: `centralize_baseline_ncf.py` likely implements this. Pre-trained NeuMF gives strongest centralized upper bound.
2. **MLP pathway validates PersonalMLP**: NCF's finding that separate embeddings for linear+nonlinear pathways outperform shared embeddings validates the thesis's PersonalMLP as a separate local component.
3. **Tower structure for PersonalMLP**: `mlp-hidden-dims: "512,256,128"` already follows NCF's convention.
4. **Pre-training for warm-start**: Run a few rounds of simple BPR-MF before switching to dual-level model.

### 6.3 Limitations & Gaps

1. **Centralized only**: No FL, privacy, or communication cost consideration.
2. **No user embedding split**: NCF trains user embeddings centrally.
3. **Pointwise only**: Uses log loss, leaves pairwise NCF as future work. Thesis uses BPR.
4. **No personalization**: Single global model, no per-user adaptation.
5. **No sparse user analysis**: No per-group (sparse/medium/dense) evaluation.

## 7. Key References to Follow

- **Rendle et al., UAI 2009 (BPR)** — Core BPR loss used in thesis; foundational pairwise ranking
- **He et al., SIGIR 2016 (eALS)** — Advanced MF with non-uniform negative weighting
- **Koren, KDD 2008 (SVD++)** — Hybrid MF; context for SVD baseline
- **Wu et al., WSDM 2016 (CDAE)** — Two-pathway architecture similar to NeuMF
- **Cheng et al., 2016 (Wide & Deep)** — Google's linear+deep fusion concept
