# A Generic Coordinate Descent Framework for Learning from Implicit Feedback

- **Authors**: Immanuel Bayer, Xiangnan He, Bhargav Kanagal, Steffen Rendle
- **Venue**: arXiv:1611.04666v1 [cs.IR], November 2016
- **Paper ID**: bayer_2016_implicit
- **Tags**: #implicit-feedback, #matrix-factorization, #collaborative-filtering, #recommender-system, #optimization, #coordinate-descent

---

## 1. Core Idea

An efficient **implicit Coordinate Descent (iCD)** framework for implicit feedback recommendation. The key insight: implicit learning = explicit loss on observed data + an **implicit regularizer** $R(\Theta)$ that penalizes non-zero predictions (not parameters). For **k-separable** models (MF, FM, PARAFAC, Tucker), this regularizer decomposes so context-side and item-side computations are independent, reducing complexity from $O(|C||I|)$ to $O((|C|+|I|)k^2)$.

## 2. Problem & Motivation

Implicit feedback treats ALL context-item pairs as training data ($|S_{impl}| = |C| \times |I|$, ~13.6B for 200K users × 68K items). CD was limited to simple MF; complex models (FM, tensors) relied exclusively on BPR-style SGD with known convergence issues. Naive CD is $O(|C||I|)$ per update — computationally infeasible (4 orders of magnitude slower than iCD empirically).

## 3. Method

### 3.1 Objective Function

$$\mathcal{L}(\Theta | S_{impl}) = \sum_{(c,i,y,\alpha) \in S} \alpha ({\hat{y}(c,i) - y})^2 + \sum_{\theta \in \Theta} \lambda_\theta \theta^2$$

**Key reformulation (Lemma 1):**

$$\arg\min_\Theta \mathcal{L}(\Theta | S_{impl}) = \arg\min_\Theta \left( \mathcal{L}(\Theta | S') + \alpha_0 R(\Theta) \right)$$

Where:
- $S'$ — rescaled observed feedback
- $R(\Theta) = \sum_{c \in C} \sum_{i \in I} \hat{y}(c,i)^2$ — implicit regularizer (penalizes non-zero predictions)
- $\alpha_0$ — confidence for unobserved items

**k-Separability**: Model $\hat{y}(c,i) = \langle \phi(c), \psi(i) \rangle = \sum_{f=1}^k \phi_f(c) \psi_f(i)$

**Lemma 2 (Decomposition):** $R(\Theta) = \sum_f \sum_{f'} J_C(f,f') \cdot J_I(f,f')$ where $J_C, J_I$ are $k \times k$ Gram matrices computed independently.

### 3.2 Algorithm

**iCD for MF**: MF model $\hat{y}(c,i) = \langle \mathbf{w}_c, \mathbf{h}_i \rangle$ is trivially k-separable.

1. Initialize $\Theta$ from $\mathcal{N}(0, \sigma)$
2. Repeat until convergence:
   a. Compute $J_I(f^*, f) = \sum_{i \in I} h_{i,f^*} h_{i,f}$ ($O(|I|k)$)
   b. For each user $c$, each dimension $f^*$:
      - Compute explicit loss gradients $L', L''$ (over observed data only)
      - Compute implicit regularizer gradients $R', R''$ (using $J_I$ decomposition)
      - Newton update: $\theta \leftarrow \theta - \frac{L' + \alpha_0 R'}{L'' + \alpha_0 R''}$
   c. Symmetric steps for item-side

**Complexity**: $O((|I| + |C|)k^2 + |S|k)$ per epoch — linear in observed interactions.

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Embedding dimension | $k$ | 10-160 | Latent factor size |
| L2 regularization | $\lambda_\theta$ | Tuned per model | Standard L2 on parameters |
| Confidence (observed) | $\alpha$ | > $\alpha_0$ | Weight of observed interactions |
| Confidence (unobserved) | $\alpha_0$ | 1 | Implicit regularizer strength |
| Step size | $\eta$ | 1 (bilinear) | Newton step; $\eta=1$ safe for multilinear |

### 3.4 Architectural Decisions

1. **Pointwise loss, not pairwise**: Uses squared loss over all items, contrasting with BPR's pairwise ranking.
2. **No sampling required**: iCD considers ALL non-observed items through the implicit regularizer — exact optimization without sampling noise.
3. **Newton updates**: Second-order per coordinate, faster convergence than SGD.
4. **k-separability as design principle**: Covers MF, FM, PARAFAC, Tucker but NOT arbitrary neural networks.

## 4. Implementation Notes

- **Complexity for MF**: $O((|I|+|C|)k^2 + |S|k)$ per epoch
- **Runtime**: Minutes on YouTube (200K users, 68K items) vs. weeks for conventional CD
- **Full Newton step**: $\eta=1$ safe for all multilinear models
- **Traversal order**: By embedding dimension $f$, then by context/item index

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Users | Items | Domain |
|---|---|---|---|
| YouTube | 200,000 | 68,000 | Video watching |

### 5.2 Key Results

- **iCD-FM with all features** achieves best quality across all scenarios
- Quality improves consistently with embedding dimension up to $k=160$
- **Computational**: iCD is 4 orders of magnitude faster than conventional CD
- Cold-start: iCD-FM with user attributes achieves ~2x improvement over baselines

### 5.3 Ablation Highlights

- More context features consistently improve FM models
- MF saturates earlier (~k=40-80) while FM continues improving to k=160
- Paper explicitly avoids BPR vs CD comparison, noting different strengths

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Implicit regularizer concept**: Provides theoretical grounding for why BPR-MF works differently — BPR sidesteps the $O(|C||I|)$ problem via sampling, while iCD solves it via decomposition.
2. **k-separability of MF**: $\hat{y}(u,i) = \langle \mathbf{w}_u, \mathbf{h}_i \rangle$ is trivially k-separable. Confirms MF's bilinear structure enables efficient implicit learning.
3. **All non-observed items matter**: Formalizes that $S_{impl}$ includes ALL pairs. Leave-one-out + 99 negatives is an approximation.

### 6.2 Potential Integration Points

1. **iCD as alternative local optimizer**: Could replace BPR-SGD on each client. J_I computation requires all item embeddings (already available as global parameters).
2. **Implicit regularizer as federated regularizer**: Conceptually similar to FedProx proximal term — both constrain model updates.
3. **FM extensions**: User statistics (quantity, diversity, coverage) could be FM features rather than just alpha inputs.

### 6.3 Limitations & Gaps

1. **No federated setting**: Centralized data access assumed. J_C over all users would need per-partition computation.
2. **Pointwise loss**: Thesis prioritizes NDCG@10; BPR pairwise loss is better aligned with ranking.
3. **No neural models**: k-separability excludes PersonalMLP (DualPersonalizedBPRMF Level 2).
4. **No MovieLens experiments**: YouTube dataset only.
5. **No privacy considerations**.

## 7. Key References to Follow

- **Rendle et al., UAI 2009 (BPR)** — Core BPR loss used in thesis
- **Hu et al., ICDM 2008 (WMF)** — Original CD-MF for implicit feedback; precursor generalized here
- **Rendle, 2012 (libFM)** — FM model; relevant for feature-based extensions
- **Rendle & Freudenthaler, WSDM 2014** — Non-uniform sampling for BPR; directly relevant to improving BPR training
- **He & McAuley, AAAI 2016 (VBPR)** — BPR + visual features
