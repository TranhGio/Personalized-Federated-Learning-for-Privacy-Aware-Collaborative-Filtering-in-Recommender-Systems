# Communication-Efficient Learning of Deep Networks from Decentralized Data

- **Authors**: McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B. A.
- **Venue**: AISTATS 2017
- **Paper ID**: mcmahan_2017_fedavg
- **Tags**: #federated-learning, #aggregation, #communication-efficiency

---

## 1. Core Idea
FedAvg combines local SGD with model averaging: each client trains for multiple local epochs before sending updates to the server, dramatically reducing communication rounds compared to FedSGD while maintaining — or improving — model quality.

## 2. Problem & Motivation
Training deep networks on decentralized data (mobile devices) requires communication efficiency because: (1) upload bandwidth is limited, (2) devices are intermittently available, (3) data is non-IID across devices. Prior distributed SGD methods require per-step communication, which is impractical at mobile scale.

## 3. Method

### 3.1 Objective Function

$$\min_{w} f(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

Where:
- $w$: global model parameters
- $K$: number of clients
- $n_k$: number of data points on client $k$
- $n$: total data points across all clients ($n = \sum_k n_k$)
- $F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)$: local objective for client $k$

### 3.2 Algorithm

**Server-side (each round $t$):**
1. Sample a fraction $C$ of clients → set $S_t$
2. Send current global model $w_t$ to each client in $S_t$
3. Receive updated models $\{w_{t+1}^k\}$ from participating clients
4. Aggregate: $w_{t+1} \leftarrow \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} w_{t+1}^k$

**Client-side (client $k$, each round):**
1. Receive $w_t$ from server
2. Set $w \leftarrow w_t$
3. For each local epoch $e = 1, \ldots, E$:
   - For each batch $b \in \mathcal{B}$ of local data:
     - $w \leftarrow w - \eta \nabla \ell(w; b)$
4. Send $w$ (or $w - w_t$) back to server

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Client fraction | $C$ | 0.1 | Fraction of clients sampled per round |
| Local epochs | $E$ | 5 | Number of full passes over local data per round |
| Batch size | $B$ | 10 | Mini-batch size for local SGD |
| Learning rate | $\eta$ | varies | Client-side SGD learning rate |

### 3.4 Architectural Decisions
- **Weighted averaging by data size**: Clients with more data have proportionally more influence on the global model. This is a natural choice for minimizing the global empirical loss, but can bias toward high-data clients.
- **Multiple local epochs (E>1)**: The key insight — running more local computation reduces communication rounds. But too many local epochs cause client drift (models diverge), especially with non-IID data.
- **Full model transmission**: Each client sends the complete model, not just gradients. This enables averaging at any point, not just at the same iterate.

## 4. Implementation Notes
- **Framework used in paper**: TensorFlow (custom simulation)
- **Model architecture**: 2-layer CNN for MNIST/CIFAR, LSTM for language model
- **Training details**: SGD with fixed learning rate, no momentum in primary experiments
- **Data preprocessing**: Non-IID split by sorting labels and distributing shards (2 shards per client for MNIST)
- **Tricks & gotchas**:
  - Learning rate decay is NOT used in base FedAvg — later papers add it
  - Client drift worsens with higher E and lower C — monitor convergence
  - When implementing with Flower: use `flwr.server.strategy.FedAvg` as base class

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users/Clients | Items/Classes | Interactions/Samples | Density |
|---|---|---|---|---|
| MNIST | 100 (simulated) | 10 | 60,000 | N/A |
| CIFAR-10 | 100 (simulated) | 10 | 50,000 | N/A |
| Shakespeare | 1,146 (roles) | 86 (chars) | 4.2M chars | N/A |

### 5.2 Key Results
| Method | MNIST (IID) Acc | MNIST (Non-IID) Acc | Rounds to 97% |
|---|---|---|---|
| FedSGD (C=1.0) | 97.5% | 97.0% | 1200 |
| FedAvg (C=0.1, E=5) | 97.6% | 96.2% | 300 |
| FedAvg (C=0.1, E=20) | 97.9% | 95.0% | 80 |

### 5.3 Ablation Highlights
- Increasing $E$ from 1→5 reduces rounds by ~4x with minimal accuracy loss on IID data
- On non-IID data, $E > 20$ causes divergence — accuracy drops significantly
- Increasing $C$ helps stabilize non-IID training but adds communication per round
- Larger batch sizes $B$ reduce computation but slow convergence in rounds

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
- FedAvg aggregation formula is the starting point for the thesis server-side logic
- Non-IID data splitting strategy (label-sorted shards) can be adapted to create non-IID MovieLens splits by user rating patterns
- Client sampling mechanism ($C$ fraction) maps directly to Flower's `min_fit_clients` / `fraction_fit` config

### 6.2 Potential Integration Points
- **BPR-MF integration**: Replace the CNN/LSTM with BPR-MF model. Client local training becomes BPR optimization on user's implicit feedback. Aggregation averages item embedding matrix across clients while keeping user embeddings local.
- **Flower implementation**: Use `flwr.server.strategy.FedAvg` as base strategy, override `aggregate_fit` if custom weighting is needed. Client implementation extends `flwr.client.NumPyClient` with BPR training loop.

### 6.3 Limitations & Gaps
- **No personalization**: FedAvg learns a single global model — all users get the same item embeddings and recommendation logic. The thesis needs per-user adaptation.
- **No privacy guarantees**: FedAvg does not incorporate DP or secure aggregation. Raw model updates are shared.
- **Not tested on RecSys tasks**: All experiments are on classification/language modeling. BPR ranking loss + implicit feedback is structurally different.
- **Client drift on non-IID**: With highly heterogeneous user preferences (natural in RecSys), FedAvg may underperform significantly.

## 7. Key References to Follow
- [Konečný et al., 2016] — Federated optimization precursor, gradient compression techniques
- [Smith et al., 2017 — MOCHA] — Multi-task FL, personalization via task relationships
- [Li et al., 2020 — FedProx] — Addresses client drift with proximal term regularization
- [Bonawitz et al., 2019] — Secure aggregation protocol for production FL
- [Rendle et al., 2009 — BPR] — The BPR-MF method that serves as the thesis baseline
