# SCAFFOLD: Stochastic Controlled Averaging for Federated Learning

- **Authors**: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh
- **Venue**: ICML 2020 (PMLR 119)
- **Paper ID**: karimireddy_2020_scaffold
- **Tags**: #federated-learning, #communication-efficiency, #non-iid, #optimization, #variance-reduction, #control-variates

---

## 1. Core Idea

SCAFFOLD introduces **control variates** to correct **client drift** in federated learning — where local updates on heterogeneous clients diverge from the global optimum. Each client and server maintain a control variate estimating the update direction; the difference corrects local SGD steps. SCAFFOLD is provably unaffected by data heterogeneity and robust to partial participation, requiring the same communication per round as FedAvg.

## 2. Problem & Motivation

FedAvg suffers from client drift under non-IID data: local updates drift toward local optima rather than the global optimum. The paper proves even with full-batch gradients and all clients, FedAvg with $K \geq 2$ local steps can be slower than centralized SGD. FedProx offers no theoretical advantage. SCAFFOLD achieves convergence **independent of heterogeneity**.

## 3. Method

### 3.1 Objective Function

$$\min_{x \in \mathbb{R}^d} f(x) := \frac{1}{N} \sum_{i=1}^{N} f_i(x)$$

**Bounded Gradient Dissimilarity (BGD)**: $\frac{1}{N} \sum \|\nabla f_i(x)\|^2 \leq G^2 + B^2 \|\nabla f(x)\|^2$

Where $G=0, B=1$ = IID; large $G$ = high heterogeneity.

### 3.2 Algorithm

**Server state**: model $x$, control variate $c$ (init 0), step-size $\eta_g$.

**Client state**: control variate $c_i$ (init 0, persisted across rounds), step-size $\eta_l$.

**Server (per round):**
1. Sample subset $S \subset \{1,...,N\}$
2. Broadcast $(x, c)$ to clients in $S$
3. Receive $(\Delta y_i, \Delta c_i)$ from each client
4. $x \leftarrow x + \eta_g \cdot \frac{1}{|S|} \sum_{i \in S} \Delta y_i$
5. $c \leftarrow c + \frac{|S|}{N} \cdot \frac{1}{|S|} \sum_{i \in S} \Delta c_i$

**Client $i$ (per round):**
1. $y_i \leftarrow x$
2. For $k = 1, \ldots, K$:
   - Compute mini-batch gradient $g_i(y_i)$
   - **Corrected update**: $y_i \leftarrow y_i - \eta_l (g_i(y_i) - c_i + c)$
3. Update control variate (Option II, cheap): $c_i^+ \leftarrow c_i - c + \frac{1}{K \eta_l}(x - y_i)$
4. Send $(\Delta y_i, \Delta c_i) = (y_i - x, c_i^+ - c_i)$
5. $c_i \leftarrow c_i^+$

**Key insight**: The correction $(g_i - c_i + c)$ replaces biased local gradient with approximately unbiased estimate of global gradient, preventing drift. Setting all $c_i = 0$ recovers FedAvg.

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Global step-size | $\eta_g$ | $\sqrt{S}$ or 1.0 | Scales aggregated updates; can be >1 |
| Local step-size | $\eta_l$ | $O(1/(\beta K))$ | Controls local update magnitude |
| Local steps | $K$ | 5 | SCAFFOLD benefits from larger $K$ unlike FedAvg |
| Client sampling | $|S|/N$ | 0.2 | SCAFFOLD is robust to small $|S|$ |
| CV option | — | Option II | Cheap: no extra gradient computation |

### 3.4 Architectural Decisions

1. **Stateful clients**: Must persist $c_i$ across rounds (unlike stateless FedAvg).
2. **Option II preferred**: No extra gradient pass. Option I (full gradient at server model) requires double computation.
3. **Two separate step-sizes**: $\eta_l$ for local, $\eta_g$ for server — critical for decoupling.
4. **Convergence rate**: $O(\frac{\sigma^2}{\mu K S \epsilon} + \frac{\beta}{\mu} + \frac{N}{S})$ — **no dependence on $G$** (heterogeneity vanishes).

## 4. Implementation Notes

- **Storage**: Each client stores $d$-dimensional $c_i$ (same size as model). For 6,040 users × 485K global params = ~11.5 GB total.
- **Communication**: Same as FedAvg if server reconstructs $\Delta c_i$ from $\Delta y_i$ (Option II).
- **SCAFFOLD + FedAvg**: Setting $c_i = 0$ permanently gives exactly FedAvg.
- **No momentum/Adam**: All experiments use vanilla SGD.

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Task | Model | Clients |
|---|---|---|---|
| Synthetic quadratic | Convex optimization | Quadratic | 2 |
| EMNIST | Classification | Logistic Regression / 2-layer FC | 100 |

### 5.2 Key Results

- **SCAFFOLD convergence is unaffected by $G$**: Identical performance at $G=1, 10, 100$.
- **More local steps $K$ → better convergence** (opposite of FedAvg under non-IID).
- **EMNIST (0% similarity, 5 epochs)**: SCAFFOLD 152 rounds vs FedAvg 428 rounds vs FedProx 1000+.
- **Client sampling**: SCAFFOLD more resilient — at 1% sampling: 790 rounds vs FedAvg 1000+.
- **Non-convex (2-layer FC)**: SCAFFOLD 0.801 accuracy vs FedAvg 0.787 vs SGD 0.766.

### 5.3 Ablation Highlights

1. **FedAvg**: More local steps = worse under heterogeneity. SCAFFOLD: more = better.
2. **Heterogeneity G**: FedAvg degrades linearly. SCAFFOLD: identical convergence.
3. **Hessian similarity (quadratic, Theorem IV)**: Optimal $K = \beta/\delta$. First result quantifying benefit of local steps.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Control variates for non-IID BPR-MF**: Dirichlet(0.5) partitioning creates significant heterogeneity. SCAFFOLD correction on global item embeddings could improve convergence.
2. **Stronger baseline than FedProx**: Paper proves FedProx ≡ FedAvg theoretically for client drift. SCAFFOLD > FedAvg ≥ FedProx.
3. **Safe to increase local epochs**: With 12 local epochs under Dirichlet(0.5), drift is significant. SCAFFOLD allows safely increasing further.

### 6.2 Potential Integration Points

1. **ScaffoldFedAvg strategy in Flower**: Client persists $c_i$ (like user embedding cache in `.embedding_cache/`). Apply control variates only to global parameters (item embeddings, item bias, global bias).
2. **SCAFFOLD + adaptive alpha**: Orthogonal. SCAFFOLD corrects optimization drift; alpha controls personalization. Corrected global embeddings provide better foundation for alpha blending.
3. **Per-group alignment**: Sparse users (high drift) benefit most from SCAFFOLD correction.

### 6.3 Limitations & Gaps

1. **Stateful clients**: 485K extra floats per client (~1.9 MB × 6040 users = 11.5 GB). Feasible for cross-silo.
2. **2x communication** if $\Delta c_i$ sent explicitly.
3. **No RecSys experiments**: Only classification. BPR-MF gradient sparsity not studied.
4. **No personalization**: Purely global model convergence.
5. **SGD only**: No adaptive optimizer interaction studied.
6. **BPR loss not covered**: Pairwise loss with negative sampling has different variance properties.

## 7. Key References to Follow

- **Li et al., 2020 (FedProx)** — Proximal-term alternative; same theoretical complexity as FedAvg
- **Reddi et al., ICLR 2021 (FedOpt)** — Adaptive server optimizers; extends SCAFFOLD ideas
- **Khaled et al., 2020** — Tightest FedAvg bounds pre-SCAFFOLD
- **Acar et al., 2021 (FedDyn)** — Dynamic regularization alternative to control variates
- **Woodworth et al., 2018** — Lower bounds SCAFFOLD matches
