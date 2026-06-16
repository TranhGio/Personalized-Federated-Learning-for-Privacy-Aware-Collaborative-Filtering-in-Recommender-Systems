# Federated Optimization in Heterogeneous Networks

- **Authors**: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith
- **Venue**: MLSys 2020
- **Paper ID**: li_2020_fedprox
- **Tags**: #federated-learning, #non-iid, #optimization, #regularization

---

## 1. Core Idea

FedProx generalizes FedAvg by adding a **proximal term** $(\\mu/2)\|w - w^t\|^2$ to each device's local objective, handling both **statistical heterogeneity** (non-IID data) and **systems heterogeneity** (variable compute). The proximal term keeps local updates close to the global model, preventing divergence. FedProx also tolerates **partial work** from stragglers rather than dropping them. Despite being a minimal modification, FedProx provides the first formal convergence guarantees for local-updating FL under non-IID data with partial participation.

## 2. Problem & Motivation

FedAvg suffers from: (1) statistical heterogeneity — non-IID local updates drift toward local optima, causing slow/unstable convergence with no theoretical guarantees; (2) systems heterogeneity — devices with different compute capabilities must complete the same epochs, and stragglers are dropped. FedProx addresses both with one simple modification.

## 3. Method

### 3.1 Objective Function

**FedProx local objective:**

$$\min_w h_k(w; w^t) = F_k(w) + \frac{\mu}{2}\|w - w^t\|^2$$

Where:
- $F_k(w)$ — client $k$'s local empirical risk
- $w^t$ — current global model at round $t$
- $\mu$ — proximal term weight (FedAvg is special case with $\mu=0$)

**Global objective**: $\min_w f(w) = \sum_{k=1}^{N} p_k F_k(w)$ with $p_k = n_k/n$.

### 3.2 Algorithm

**Server (per round):**
1. Select subset $S_t$ of $K$ devices
2. Send $w^t$ to all chosen devices
3. Receive $w_k^{t+1}$ from each (possibly partial solutions)
4. Aggregate: $w^{t+1} = \frac{1}{K} \sum_{k \in S_t} w_k^{t+1}$

**Client $k$:**
1. Receive $w^t$
2. Find $\gamma_k^t$-inexact minimizer of $h_k(w; w^t)$ via local SGD
3. Send $w_k^{t+1}$ to server

**$\gamma$-inexact**: $\|\nabla h(w^*; w_0)\| \leq \gamma \|\nabla h(w_0; w_0)\|$, $\gamma \in [0,1]$. Formalizes partial work.

**B-local dissimilarity**: $B(w) = \sqrt{E_k[\|\nabla F_k(w)\|^2] / \|\nabla f(w)\|^2}$ — measures heterogeneity ($B=1$ = IID).

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Proximal weight | $\mu$ | {0.001, 0.01, 0.1, 1} | Regularization toward global model |
| Local epochs | $E$ | 20 (synthetic) | Amount of local computation |
| Devices per round | $K$ | 10 | Client sampling |
| Learning rate | $\eta$ | Dataset-dependent | SGD step size |

**Adaptive $\mu$ heuristic**: Increase by 0.1 when loss increases; decrease by 0.1 when loss decreases for 5 consecutive rounds.

### 3.4 Architectural Decisions

1. **Proximal only, no gradient correction**: FedDane (gradient correction) is unstable on non-IID; proximal is simpler and more robust.
2. **Solver-agnostic**: Convergence holds with any local solver.
3. **Tolerates partial work**: Accepts incomplete solutions (stragglers contribute rather than being dropped).

## 4. Implementation Notes

- **Framework**: TensorFlow 1.10.1. Code: github.com/litian96/FedProx
- **Straggler simulation**: Each device completes $x$ epochs drawn uniformly from $[1, E]$. Settings: 0%, 50%, 90% stragglers.
- **Best $\mu$ found**: Synthetic=1, MNIST=1, FEMNIST=1, Shakespeare=0.001, Sent140=0.01.
- **Your codebase**: Already implemented as `strategy=fedprox proximal-mu=0.01`. Proximal term only on GLOBAL params (item embeddings) in split learning.

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Devices | Task | Model |
|---|---|---|---|
| Synthetic | 30 | Classification | Logistic regression |
| MNIST | 1,000 | Digit classification | Logistic regression |
| FEMNIST | 200 | Character classification | Logistic regression |
| Shakespeare | 143 | Next-char prediction | 2-layer LSTM |
| Sent140 | 772 | Sentiment | 2-layer LSTM |

### 5.2 Key Results

- **90% stragglers**: FedProx improves absolute test accuracy by **22% on average** over FedAvg.
- **Partial work tolerance**: Incorporating partial solutions ($\mu=0$) consistently better than dropping stragglers.
- **Adding $\mu > 0$**: Further stabilizes and accelerates convergence under high heterogeneity.
- **IID data**: FedProx provides minimal improvement — benefits are specifically for heterogeneous settings.

### 5.3 Ablation Highlights

1. **$\mu=0$ vs $\mu>0$**: Even $\mu=0$ (just partial work) helps. $\mu>0$ adds stability for high heterogeneity.
2. **Adaptive $\mu$**: Simple heuristic converges to effective values from adversarial initialization.
3. **FedProx vs FedDane**: FedDane unstable on non-IID even with many devices.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **Already implemented**: `strategy=fedprox proximal-mu=0.01` in baseline, personalized, and adaptive modules.
2. **Proximal scope in split learning**: Applied only to global parameters — principled choice since only aggregated params need drift prevention.
3. **$\mu$ tuning guidance**: Best varies by dataset (0.001-1.0). Include in W&B sweeps.
4. **Non-IID justification**: Dirichlet(0.5) creates exactly the heterogeneity FedProx targets.

### 6.2 Potential Integration Points

1. **Adaptive $\mu$ heuristic**: Implement loss-based $\mu$ adaptation in `server_app.py`.
2. **Dissimilarity metric**: Compute $B(w)$ each round as diagnostic for heterogeneity.
3. **$\mu$ + alpha interaction**: Per-client $\mu$ (stronger for sparse, weaker for dense) could complement adaptive alpha.
4. **Theoretical framing**: Cite FedProx convergence (Theorem 4, 6) to justify the FL approach; motivate why additional personalization beyond FedProx is needed.

### 6.3 Limitations & Gaps

1. **No recommendation tasks**: All classification/NLP. BPR-MF not validated.
2. **No split learning**: Assumes all parameters global.
3. **No personalization**: Single global model. Thesis fills this gap.
4. **Fixed $\mu$**: Global, not per-client or per-parameter.
5. **Communication not reduced**: Same as FedAvg per round.

## 7. Key References to Follow

- **McMahan et al., 2017 (FedAvg)** — Foundation; FedProx's baseline
- **Smith et al., NeurIPS 2017 (MOCHA)** — Federated multi-task learning
- **Li et al., 2020 (FedDANE)** — Gradient correction that FedProx outperforms
- **Bonawitz et al., MLSys 2019** — Production FL system design
- **Zhao et al., 2018** — Non-IID analysis; data-sharing strategies
