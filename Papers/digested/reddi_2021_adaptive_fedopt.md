# Adaptive Federated Optimization

- **Authors**: Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecny, Sanjiv Kumar, H. Brendan McMahan (Google Research)
- **Venue**: ICLR 2021
- **Paper ID**: reddi_2021_adaptive_fedopt
- **Tags**: #federated-learning, #optimization, #communication-efficiency, #non-iid

---

## 1. Core Idea

Standard FedAvg uses SGD on both clients and server (server LR=1), which is hard to tune and converges poorly under non-IID data. This paper proposes **FEDOPT** — a framework decoupling client-side and server-side optimizers. Using adaptive optimizers (Adagrad, Adam, Yogi) as the **server optimizer** with SGD on clients creates **FEDADAGRAD**, **FEDADAM**, and **FEDYOGI**. No additional client state or communication overhead. Dramatic improvements on sparse-gradient tasks (2.2x on Stack Overflow).

## 2. Problem & Motivation

Two issues: (1) **Client drift** from heterogeneous data — local models diverge. (2) **Lack of adaptivity** — FL settings often have sparse, heavy-tailed gradient distributions where adaptive methods excel. SCAFFOLD-style control variates require stateful clients, incompatible with cross-device FL.

## 3. Method

### 3.1 Objective Function

$$\min_{x \in \mathbb{R}^d} f(x) = \frac{1}{m} \sum_{i=1}^{m} F_i(x)$$

### 3.2 Algorithm

**FEDOPT (General Framework):**

**Client-side** (SGD):
1. Receive $x_t$ from server
2. For $k = 0, \ldots, K-1$: $x_{i,k+1}^t = x_{i,k}^t - \eta_l g_{i,k}^t$
3. Compute pseudo-gradient: $\Delta_i^t = x_{i,K}^t - x_t$

**Server-side** (adaptive):
1. Average: $\Delta_t = \frac{1}{|S|} \sum_{i \in S} \Delta_i^t$
2. Momentum: $m_t = \beta_1 m_{t-1} + (1-\beta_1) \Delta_t$
3. Second moment (**three variants**):
   - **FEDADAGRAD**: $v_t = v_{t-1} + \Delta_t^2$
   - **FEDYOGI**: $v_t = v_{t-1} - (1-\beta_2) \Delta_t^2 \cdot \text{sign}(v_{t-1} - \Delta_t^2)$
   - **FEDADAM**: $v_t = \beta_2 v_{t-1} + (1-\beta_2) \Delta_t^2$
4. Update: $x_{t+1} = x_t + \eta \frac{m_t}{\sqrt{v_t} + \tau}$

**FedAvg = FEDOPT with SGD on server ($\eta=1$, no momentum/adaptivity).**

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Client learning rate | $\eta_l$ | Task-dependent | Local SGD step size |
| Server learning rate | $\eta$ | Task-dependent | Server adaptive step size |
| Adaptivity parameter | $\tau$ | $10^{-3}$ (robust) | Degree of adaptivity; one less HP to tune |
| First moment decay | $\beta_1$ | 0.9 (Adam/Yogi); 0 (Adagrad) | Momentum |
| Second moment decay | $\beta_2$ | 0.99 (Adam/Yogi) | Accumulator decay |
| Local steps | $K$ | 1 epoch typical | Per-round local computation |
| Initial $v$ | $v_{-1}$ | $\tau^2$ | Must be $\geq \tau^2$ |

**$\eta_l$ and $\eta$ have inverse relationship**: increasing one allows decreasing the other.

### 3.4 Architectural Decisions

1. **Adaptive on server only**: No additional client state — critical for cross-device FL.
2. **Pseudo-gradients as inputs**: Model differences $\Delta_t$ (NOT true gradients) proven valid for adaptive optimizers.
3. **No bias correction**: $v_{-1} \geq \tau^2$ and $\tau$ handle cold-start.
4. **Communication identical to FedAvg**: Adaptive state lives entirely on server.
5. **Combinable with FedProx**: Proximal SGD on clients + Adam on server.

## 4. Implementation Notes

- **Framework**: TensorFlow Federated
- **$\tau = 10^{-3}$** works across nearly all tasks — effectively not a hyperparameter
- **Wider good HP regions**: FedAdam/FedYogi easier to tune than FedAvg
- **Flower integration**: Subclass `SplitFedAvg` strategy, add $m_t, v_t$ state in `aggregate_fit`
- **LR decay**: Decaying $\eta_l$ (not $\eta$) can improve empirical performance

## 5. Experimental Results

### 5.1 Datasets Used

| Dataset | Task | Model | Clients |
|---|---|---|---|
| CIFAR-10/100 | Image classification | ResNet-18 | Synthetic (Dirichlet) |
| EMNIST CR/AE | Character recognition/Autoencoder | CNN/AE | Natural (by writer) |
| Shakespeare | Next-char prediction | RNN | Natural (by role) |
| Stack Overflow LR/NWP | Tag prediction/Next-word | LR/RNN | Natural (342K users) |

### 5.2 Key Results

| Task | FedAvg | FedAdagrad | FedAdam | FedYogi |
|---|---|---|---|---|
| CIFAR-10 (acc%) | 72.8 | 77.4 | **78.0** | 77.4 |
| CIFAR-100 (acc%) | 44.7 | **52.5** | 52.4 | 52.4 |
| SO LR (Recall@5×100) | 30.0 | **67.1** | 65.8 | 65.9 |
| EMNIST AE (MSE×1000) | 6.47 | 4.20 | 1.01 | **0.98** |

**Sparse-gradient tasks**: Up to **2.2x improvement** (SO LR). SCAFFOLD performs comparably or worse in cross-device settings.

### 5.3 Ablation Highlights

1. **Ease of tuning**: Many good $(\eta_l, \eta)$ combinations for FedAdam/FedYogi vs narrow good region for FedAvg.
2. **$\tau$ robustness**: $10^{-3}$ near-optimal across all tasks.
3. **Convergence rate**: $O(1/\sqrt{mKT})$ — matches best known for nonconvex FL.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **FedAdam as server optimizer**: BPR-MF item embeddings produce **sparse gradients** (most items not interacted per user per round) — exactly where adaptive methods excel (SO LR: 2x).
2. **$\tau = 10^{-3}$ default**: One fewer HP to sweep.
3. **Wider HP regions**: Less risk of missing good configs in W&B sweeps.

### 6.2 Potential Integration Points

1. **SplitFedAdam strategy**: Apply adaptive server optimizer to global item embeddings only. Add $m_t, v_t$ state to `strategy.py`. Sparse item embedding gradients are ideal for Adagrad/Adam.
2. **Complementary to FedProx**: Use proximal SGD on clients + Adam on server (orthogonal improvements).
3. **Alpha heterogeneity**: Different alpha values produce heterogeneous pseudo-gradients — adaptive server handles per-coordinate.

### 6.3 Limitations & Gaps

1. **No RecSys experiments**: All classification/NLP. BPR-MF ranking not tested.
2. **E=1 epoch**: Thesis uses 5-12. More local steps increase drift; convergence requires $\eta_l \leq 1/(16LK)$.
3. **No personalization interaction**: How adaptive server optimization interacts with split learning/alpha blending unknown.
4. **No BPR analysis**: Pairwise loss is not standard ERM; convergence guarantees may not apply directly.
5. **Server state**: Must maintain $m_t, v_t$ (same size as model) — trivial for 485K params.

## 7. Key References to Follow

- **Zaheer et al., NeurIPS 2018 (Yogi)** — Why Adam fails, Yogi fixes; second-moment update
- **Karimireddy et al., ICML 2020 (SCAFFOLD)** — Control variates alternative; underperforms in cross-device
- **McMahan et al., AISTATS 2017 (FedAvg)** — Baseline generalized by FEDOPT
- **Li et al., MLSys 2020 (FedProx)** — Orthogonal, combinable with adaptive server
- **Hsu et al., 2019 (FedAvgM)** — Server momentum; simpler alternative
