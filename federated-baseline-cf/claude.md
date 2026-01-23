# Federated Collaborative Filtering Baseline

> **Baseline for Master Thesis: Personalized Federated Learning for Privacy-Aware Collaborative Filtering**
> MovieLens 1M | Dirichlet Partitioning | FedAvg/FedProx | BPR-MF | All Parameters Global (Aggregated)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Research Context](#-research-context)
3. [Architecture](#-architecture)
4. [Key Components](#-key-components)
5. [Data Pipeline](#-data-pipeline)
6. [Model Architecture](#-model-architecture)
7. [Federated Learning Setup](#-federated-learning-setup)
8. [Evaluation Metrics](#-evaluation-metrics)
9. [Usage Guide](#-usage-guide)
10. [Configuration](#-configuration)
11. [Performance Analysis](#-performance-analysis)
12. [Technical Details](#-technical-details)
13. [Future Work](#-future-work)
14. [References](#-references)

---

## 🎯 Project Overview

### Purpose

This project implements a **federated learning baseline** for collaborative filtering recommendation systems using **Matrix Factorization (MF)** on the MovieLens 1M dataset. It serves as the foundation for comparison with the thesis's personalized approaches:

- **Baseline Characteristic**: All parameters are **GLOBAL** (aggregated via FedAvg/FedProx)
- **No Personalization**: User embeddings are aggregated (not kept local)
- **Standard FL**: Traditional federated averaging without adaptive mechanisms

### Role in Thesis

This baseline establishes the **lower bound** for federated recommendation performance:

| Aspect | Baseline (This Project) | Adaptive (Thesis Contribution) |
|--------|-------------------------|-------------------------------|
| User Embeddings | Global (aggregated) | **Local** (private) |
| Personalization | None | **Adaptive α** + PersonalMLP |
| Communication | All parameters | Only item embeddings |
| Privacy | Limited | **Enhanced** |

### Why This Matters

Traditional recommendation systems centralize user data, raising **privacy concerns**. Federated Learning enables:

- ✅ **Privacy preservation**: User data stays on local devices
- ✅ **Scalability**: Distributes computation across clients
- ✅ **Regulatory compliance**: GDPR, CCPA compatibility
- ⚠️ **Limited personalization**: Users with different preferences get same treatment

---

## 🔬 Research Context

### Thesis Comparison Framework

This baseline is compared against:

1. **federated-personalized-cf**: Split learning (local user embeddings)
2. **federated-adaptive-personalized-cf**: Multi-factor α + Dual-level personalization

### RecSys 2024 Best Practices

This implementation follows **RecSys 2024 best practices**:

- **BPR-MF** (Bayesian Personalized Ranking) is state-of-the-art for ranking
- Proper implementation critical: **50% performance variance** with incorrect setup
- **Xavier initialization** for embeddings
- **Negative sampling** done correctly (exclude rated items)

### Standard Benchmark

- **MovieLens 1M**: 1 million ratings, 6,040 users, 3,883 movies
- **Non-IID partitioning**: Dirichlet distribution based on genre preferences
- **Metrics**: RMSE, MAE, Hit Rate@K, Precision@K, Recall@K, NDCG@K

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Flower Server                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              ALL Parameters are GLOBAL (Aggregated)            │  │
│  │                                                                 │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │  │
│  │  │ User Embeddings │  │ Item Embeddings │  │    Biases     │  │  │
│  │  │  (6040 × 128)   │  │  (3706 × 128)   │  │  user + item  │  │  │
│  │  │   GLOBAL ⚠️     │  │     GLOBAL      │  │    + global   │  │  │
│  │  └─────────────────┘  └─────────────────┘  └───────────────┘  │  │
│  │                                                                 │  │
│  │                    FedAvg / FedProx                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
  │   Client 0    │        │   Client 1    │        │   Client N    │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ Receives ALL  │        │ Receives ALL  │        │ Receives ALL  │
  │ parameters    │        │ parameters    │        │ parameters    │
  │ from server   │        │ from server   │        │ from server   │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ Local Data    │        │ Local Data    │        │ Local Data    │
  │ (partition)   │        │ (partition)   │        │ (partition)   │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ Train → Send  │        │ Train → Send  │        │ Train → Send  │
  │ ALL params    │        │ ALL params    │        │ ALL params    │
  └───────────────┘        └───────────────┘        └───────────────┘
```

### Key Limitation (Why Personalized FL is Needed)

```
⚠️ BASELINE LIMITATION:
─────────────────────────────────────────────────────────────────────

User embeddings are AGGREGATED across all clients:
  • Client A's user preferences → averaged with all others
  • Client B's user preferences → averaged with all others

Result: User embeddings lose personalization!
  • Action movie fans get averaged with Romance fans
  • Heavy users' embeddings diluted by sparse users
  • All clients receive the SAME global user embeddings

This is why the thesis proposes:
  • Split Learning: Keep user embeddings LOCAL
  • Adaptive α: Personalize based on user characteristics
```

### Parameter Classification

| Parameter | Dimensions | Type | Sent to Server |
|-----------|------------|------|----------------|
| `user_embeddings.weight` | (6040, 128) | **Global** | ✅ Yes |
| `user_bias.weight` | (6040, 1) | **Global** | ✅ Yes |
| `item_embeddings.weight` | (3706, 128) | **Global** | ✅ Yes |
| `item_bias.weight` | (3706, 1) | **Global** | ✅ Yes |
| `global_bias` | (1,) | **Global** | ✅ Yes |

**Total Parameters**: ~874K (all transmitted each round)

---

## 🔑 Key Components

### Directory Structure

```
federated-baseline-cf/
├── federated_baseline_cf/
│   ├── __init__.py
│   ├── dataset.py              # Data loading & Dirichlet partitioning
│   ├── task.py                 # Training, evaluation, ranking metrics
│   ├── client_app.py           # Flower client implementation
│   ├── server_app.py           # Flower server with W&B logging
│   └── models/
│       ├── __init__.py
│       ├── basic_mf.py         # Basic Matrix Factorization (MSE loss)
│       ├── bpr_mf.py           # BPR Matrix Factorization (ranking loss)
│       └── losses.py           # MSELoss, BPRLoss implementations
├── data/                       # MovieLens 1M dataset (auto-downloaded)
│   └── ml-1m/
├── figures/                    # Generated visualizations
├── scripts/                    # Analysis scripts
├── test_dataset.py             # Dataset loading tests
├── test_models.py              # Model tests
├── visualize_partitions.py     # Data partition visualization
└── pyproject.toml              # Configuration & dependencies
```

### File Purposes

| File | Lines | Purpose |
|------|-------|---------|
| `dataset.py` | ~450 | MovieLens loading, Dirichlet partitioning, DataLoaders |
| `task.py` | ~700 | Training (BasicMF/BPRMF), testing, ranking evaluation |
| `client_app.py` | ~180 | Flower client, local training/evaluation loops |
| `server_app.py` | ~370 | Flower server, aggregation, centralized eval, W&B logging |
| `basic_mf.py` | ~210 | BasicMF model with MSE loss baseline |
| `bpr_mf.py` | ~315 | BPRMF model with negative sampling & ranking loss |
| `losses.py` | ~130 | MSE and BPR loss function implementations |

---

## 📊 Data Pipeline

### Dataset: MovieLens 1M

**File**: `dataset.py`

- **1,000,209 ratings** from **6,040 users** on **3,883 movies**
- Rating scale: 1-5 stars
- 18 movie genres
- Auto-download from GroupLens on first run

### Dirichlet Partitioning

Creates **non-IID data distribution** based on genre preferences:

```python
# Algorithm Overview

# Step 1: Compute user genre preferences
user_genre_dist = compute_user_preferences(ratings_df, movies_df)
# Each user: vector of 18 genre probabilities

# Step 2: Sample client genre distributions
client_genre_dist = np.random.dirichlet([alpha] * num_genres, num_clients)
# α=0.5: High heterogeneity (realistic FL scenario)
# α=10.0: Nearly IID

# Step 3: Assign users to clients via KL divergence
for user in users:
    best_client = argmin(KL(user_pref || client_pref))
    assign(user → best_client)
```

**Partition Statistics** (α=0.5, 10 clients):

```
Client   Users   Ratings   Genre Focus
   0        8       990    Drama/Thriller (smallest)
   1      108     8,722    Comedy/Romance
   2    3,427   619,815    Diverse (largest)
   3       44     3,897    Action/Adventure
   ...
   9       60     5,899    Sci-Fi/Fantasy

Coefficient of Variation: 2.03 (high heterogeneity)
```

### Data Loading

```python
from federated_baseline_cf.task import load_data

# Client-side data loading
trainloader, testloader = load_data(
    partition_id=0,        # Which client (0-9)
    num_partitions=10,     # Total clients
    alpha=0.5,             # Dirichlet concentration
    test_ratio=0.2,        # 80% train, 20% test
    batch_size=256,        # DataLoader batch size
)

# Returns:
# - trainloader: PyTorch DataLoader with batches {user, item, rating}
# - testloader: Separate test set for evaluation
```

---

## 🧠 Model Architecture

### Overview

Both models use **embedding-based matrix factorization**:

```
Prediction = global_bias + user_bias[u] + item_bias[i] + dot(user_emb[u], item_emb[i])
```

### 1. BasicMF (MSE Loss)

**File**: `models/basic_mf.py`

**Purpose**: Traditional rating prediction (optimize RMSE/MAE)

```python
class BasicMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, dropout=0.1):
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_ids, item_ids):
        user_emb = self.dropout(self.user_embeddings(user_ids))
        item_emb = self.dropout(self.item_embeddings(item_ids))
        interaction = torch.sum(user_emb * item_emb, dim=1)
        prediction = self.global_bias + user_bias + item_bias + interaction
        return prediction  # Predicted rating
```

**Loss Function**:
```python
loss = MSELoss(predictions, ratings)
```

**Typical Performance** (federated):
- RMSE: 0.85-0.95
- MAE: 0.65-0.75
- Good for: Rating prediction, explicit feedback

### 2. BPRMF (BPR Loss) - Recommended

**File**: `models/bpr_mf.py`

**Purpose**: Ranking optimization (optimize top-K recommendations)

**Key Difference**: Optimizes **pairwise ranking** instead of absolute ratings.

**BPR Assumption**:
> "User prefers observed items over unobserved items"

```python
class BPRMF(nn.Module):
    def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
        pos_scores = self._compute_score(user_ids, pos_item_ids)

        if neg_item_ids is None:
            return pos_scores  # Prediction mode

        neg_scores = self._compute_score(user_ids, neg_item_ids)
        return pos_scores, neg_scores  # Training mode

    def sample_negatives(self, user_ids, pos_item_ids, user_rated_items):
        """Critical for BPR performance! Sample unrated items."""
        neg_items = []
        for user_id, pos_item in zip(user_ids, pos_item_ids):
            rated = user_rated_items[user_id]
            while True:
                neg_item = random.randint(0, num_items - 1)
                if neg_item not in rated:
                    neg_items.append(neg_item)
                    break
        return neg_items
```

**Loss Function**:
```python
# BPR Loss: Maximize score difference
diff = pos_scores - neg_scores
loss = -mean(log(sigmoid(diff)))
```

**Typical Performance** (federated):
- RMSE: 2.0-2.5 (not optimized for this!)
- Hit Rate@10: 0.65-0.75
- Precision@10: 0.08-0.12
- Good for: Top-K recommendations, implicit feedback

### Model Initialization

**Best Practices** (critical for performance):

```python
def _init_weights(self):
    # Xavier/Glorot initialization
    init.xavier_uniform_(self.user_embeddings.weight)
    init.xavier_uniform_(self.item_embeddings.weight)

    # Small random biases
    init.normal_(self.user_bias.weight, mean=0.0, std=0.01)
    init.normal_(self.item_bias.weight, mean=0.0, std=0.01)

    # Zero global bias (learns mean rating)
    init.zeros_(self.global_bias)
```

**Why This Matters**: 50% performance variance with poor initialization (RecSys 2024)

---

## 🌐 Federated Learning Setup

### Client App

**File**: `client_app.py`

```python
from flwr.client import ClientApp

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # 1. Load model with global weights from server
    model = get_model(model_type, embedding_dim)
    model.load_state_dict(msg.content["arrays"])

    # 2. Load local data partition
    partition_id = context.node_config["partition-id"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # 3. Optional: FedProx proximal term
    if proximal_mu > 0:
        global_params = [p.clone() for p in model.parameters()]

    # 4. Train locally
    train_loss = train_fn(model, trainloader, epochs, lr,
                          proximal_mu=proximal_mu,
                          global_params=global_params)

    # 5. Return ALL updated weights + metrics
    return Message(content={
        "arrays": model.state_dict(),  # ALL parameters sent!
        "metrics": {"train_loss": train_loss, "num_examples": len(trainloader)}
    })
```

### Server App

**File**: `server_app.py`

```python
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg, FedProx

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context):
    # 1. Initialize global model
    global_model = get_model(model_type, embedding_dim)

    # 2. Select strategy
    if strategy_name == "fedprox":
        strategy = FedProx(fraction_fit, proximal_mu=proximal_mu)
    else:
        strategy = FedAvg(fraction_fit)

    # 3. Run federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=global_model.state_dict(),
        num_rounds=num_rounds,
    )

    # 4. Post-training centralized evaluation
    final_model.load_state_dict(result.arrays)
    eval_metrics = evaluate_on_full_testset(final_model)

    # 5. Log to W&B and save results
    wandb.log({"final/rmse": eval_metrics['rmse'], ...})
    save_results_json(eval_metrics)
```

### FedAvg Algorithm

```python
# Weighted averaging of client models
global_weights = sum(
    num_examples_i / total_examples * client_weights_i
    for i in selected_clients
)

# All parameters aggregated (including user embeddings!)
# This is the KEY DIFFERENCE from personalized FL
```

### FedProx Support

```python
# Proximal term prevents client drift
L_total = L_task + (μ/2) × ||w - w_server||²

# Where:
# - L_task: MSE or BPR loss
# - μ: proximal_mu parameter (default: 0.01)
# - w: local model weights
# - w_server: initial global weights received from server
```

---

## 📈 Evaluation Metrics

### Rating Prediction Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

### Ranking Metrics (Primary Focus for BPR)

**File**: `task.py` - `evaluate_ranking()`

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Hit Rate@K** | hits / num_users | % users with ≥1 relevant item in top-K |
| **Precision@K** | hits / K | % of top-K that are relevant |
| **Recall@K** | hits / num_relevant | % of relevant items retrieved |
| **F1@K** | 2 × P × R / (P + R) | Harmonic mean |
| **NDCG@K** | DCG / IDCG | Position-weighted ranking quality |
| **MAP@K** | mean(AP@K) | Average precision over positions |
| **MRR** | mean(1 / rank_first_hit) | Position of first relevant item |
| **Coverage@K** | unique_items / catalog_size | Catalog diversity |
| **Novelty@K** | mean(-log2(popularity)) | Recommendation surprise |

### Evaluation Configuration

```toml
# In pyproject.toml
enable-ranking-eval = true
ranking-k-values = "5,10,20"
```

---

## 📖 Usage Guide

### Installation

```bash
cd federated-baseline-cf
pip install -e .
```

**Dependencies** (from `pyproject.toml`):
- `flwr[simulation]>=1.22.0`: Flower federated learning
- `torch>=2.7.1`: PyTorch deep learning
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing
- `scikit-learn>=1.3.0`: Machine learning utilities
- `wandb>=0.19.0`: Experiment tracking

### Basic Usage

**1. Run Federated Training** (default BPR-MF):
```bash
flwr run .
```

**2. Run with FedProx**:
```bash
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
```

**3. Run with BasicMF (MSE)**:
```bash
flwr run . --run-config "model-type=basic"
```

**4. Custom Configuration**:
```bash
# 50 rounds, larger embeddings
flwr run . --run-config "num-server-rounds=50 embedding-dim=256"

# More non-IID data
flwr run . --run-config "alpha=0.1"

# Disable W&B
flwr run . --run-config "wandb-enabled=false"
```

### Visualize Data Partitions

```bash
python visualize_partitions.py
```

Generates:
- `figures/partition_sizes_alpha_*.png`: Ratings per client
- `figures/genre_distribution_alpha_*.png`: Genre heatmaps
- `figures/rating_distribution_alpha_*.png`: Rating histograms
- `results/partition_analysis_alpha_*.csv`: Statistical summaries

### Programmatic Usage

```python
from federated_baseline_cf.task import get_model, load_data, train, test

# Load data for client 0
trainloader, testloader = load_data(
    partition_id=0,
    num_partitions=10,
    alpha=0.5
)

# Initialize model
model = get_model(model_type="bpr", embedding_dim=128)

# Train locally
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loss = train(model, trainloader, epochs=5, lr=0.005,
                   device=device, model_type="bpr")

# Evaluate
eval_loss, metrics = test(model, testloader, device, model_type="bpr")
print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

# Generate recommendations
user_id = 42
top_items, top_scores = model.recommend(user_id, top_k=10)
```

---

## ⚙️ Configuration

### pyproject.toml

```toml
[tool.flwr.app.config]

# =============================================================================
# Federated Learning Parameters
# =============================================================================
num-server-rounds = 10         # FL rounds (3-100)
fraction-train = 1.0           # Fraction of clients per round (0.1-1.0)
local-epochs = 5               # Local training epochs (1-10)

# =============================================================================
# Strategy Selection
# =============================================================================
strategy = "fedavg"            # "fedavg" or "fedprox"
proximal-mu = 0.01             # FedProx proximal term (0 = FedAvg)

# =============================================================================
# Model Parameters
# =============================================================================
model-type = "bpr"             # "basic" (MSE) or "bpr" (ranking)
embedding-dim = 128            # Latent dimensions (32, 64, 128, 256)
dropout = 0.1                  # Regularization (0.0-0.5)

# =============================================================================
# Training Parameters
# =============================================================================
lr = 0.005                     # Learning rate (1e-4 to 1e-2)
weight-decay = 1e-5            # L2 regularization (1e-6 to 1e-3)
num-negatives = 1              # Negatives per positive for BPR (1-5)

# =============================================================================
# Data Partitioning
# =============================================================================
alpha = 0.5                    # Dirichlet concentration (lower = more non-IID)

# =============================================================================
# Evaluation
# =============================================================================
enable-ranking-eval = true
ranking-k-values = "5,10,20"

# =============================================================================
# Experiment Tracking
# =============================================================================
wandb-enabled = true
wandb-project = "federated-cf"
```

### Parameter Guidelines

| Parameter | Recommended | Purpose | Trade-offs |
|-----------|-------------|---------|------------|
| **num-server-rounds** | 10-50 | FL convergence | More rounds = better but slower |
| **fraction-train** | 0.3-1.0 | Client sampling | Higher = faster convergence |
| **local-epochs** | 3-5 | Local training | More = better local fit, risk overfitting |
| **model-type** | bpr | Ranking optimization | Better for recommendations |
| **embedding-dim** | 64-128 | Model capacity | Higher = more expressive |
| **alpha** | 0.5 | Data heterogeneity | Lower = more non-IID |
| **strategy** | fedavg | Aggregation | fedprox helps with heterogeneity |

### Federation Configuration

```toml
# Local simulation (default)
[tool.flwr.federations.local-simulation]
options.num-supernodes = 5

# GPU-enabled simulation
[tool.flwr.federations.local-sim-gpu]
options.num-supernodes = 5
options.backend.client-resources.num-cpus = 6
options.backend.client-resources.num-gpus = 0.2
```

---

## 📊 Performance Analysis

### Expected Results

#### BPR-MF (Ranking Optimization) - Recommended

**Rating Metrics** (not optimized for):
| Round | Train Loss | RMSE | MAE |
|-------|------------|------|-----|
| 1 | 0.55 | 2.81 | 2.58 |
| 5 | 0.30 | 2.33 | 2.06 |
| 10 | 0.25 | 2.23 | 1.96 |

**Ranking Metrics** (optimized for):
- Hit Rate@10: **0.65-0.75**
- Precision@10: **0.08-0.12**
- NDCG@10: **0.15-0.25**

#### BasicMF (Rating Prediction)

**Rating Metrics** (optimized for):
| Round | Train Loss | RMSE | MAE |
|-------|------------|------|-----|
| 1 | 2.50 | 1.45 | 1.12 |
| 5 | 0.85 | 0.95 | 0.75 |
| 10 | 0.65 | 0.90 | 0.70 |

**Ranking Metrics** (not optimized for):
- Hit Rate@10: 0.50-0.60

### Key Insights

1. **BPR-MF RMSE is high (~2.2)** - This is **expected**!
   - BPR optimizes ranking, not rating prediction
   - Focus on Hit Rate@K and NDCG@K instead

2. **Training loss oscillates** - Normal for non-IID data
   - Different clients have different genre preferences
   - FedAvg smooths out conflicting updates

3. **Baseline vs Personalized FL**:
   - This baseline: ~65-75% Hit Rate@10
   - Personalized (split learning): Expected +10-15% improvement
   - Adaptive (thesis): Expected +15-20% improvement

---

## 🔬 Technical Details

### Why Dirichlet Partitioning?

**Random partitioning** creates **IID data** (unrealistic):
```
Each client gets random 10% of users
→ All clients have similar genre distributions
→ Doesn't reflect real-world heterogeneity
```

**Dirichlet partitioning** creates **non-IID data** (realistic):
```
Cluster users by preferences
→ Client A: Mostly action/sci-fi fans
→ Client B: Romance/comedy enthusiasts
→ Reflects real user demographics
```

**Impact on FL**:
- IID: Fast convergence, 10-20 rounds sufficient
- Non-IID (α=0.5): Slower convergence, 50-100 rounds needed

### Matrix Factorization Math

**Model**:
```
R ≈ U @ V.T

Where:
- R: Rating matrix (users × items)
- U: User embeddings (users × d)
- V: Item embeddings (items × d)
- d: Embedding dimension
```

**Prediction**:
```
r̂_ui = μ + b_u + b_i + u_u^T × v_i

Where:
- μ: Global mean rating
- b_u: User bias
- b_i: Item bias
- u_u: User embedding
- v_i: Item embedding
```

**BPR Objective**:
```
max  Σ log σ(x̂_uij)
    (u,i,j)

Where:
- x̂_uij = r̂_ui - r̂_uj (score difference)
- i: Observed item (positive)
- j: Unobserved item (negative)
- σ: Sigmoid function
```

### Safe Device Detection

```python
def get_device():
    """Safe CUDA detection (handles incompatible architectures)."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Test CUDA with a small tensor
        torch.zeros(1).cuda()
        return "cuda"
    except Exception:
        return "cpu"  # Fallback if CUDA fails
```

---

## 🚀 Future Work

### 1. Personalized Federated Learning (Thesis Contribution)

**Current (Baseline)**: All parameters global

**Proposed (See federated-personalized-cf)**:
```python
# Local parameters (NOT aggregated):
- User embeddings: Stay on client devices
- User biases: Client-specific

# Global parameters (aggregated):
- Item embeddings: Shared
- Item biases: Shared
```

### 2. Adaptive Personalization (Thesis Contribution)

**See federated-adaptive-personalized-cf**:
- Multi-factor adaptive α
- Dual-level personalization
- Global prototype aggregation

### 3. Additional Enhancements

- **Differential Privacy**: Add noise for stronger privacy
- **Neural CF**: MLP-based models (NCF, LightGCN)
- **Cross-silo deployment**: Real distributed clients
- **Continual learning**: Handle temporal dynamics

---

## 📚 References

### Papers

1. **BPR: Bayesian Personalized Ranking from Implicit Feedback**
   - Rendle et al., UAI 2009
   - Foundation for ranking-based CF

2. **Communication-Efficient Learning of Deep Networks from Decentralized Data**
   - McMahan et al., AISTATS 2017
   - Introduced FedAvg algorithm

3. **Federated Optimization in Heterogeneous Networks (FedProx)**
   - Li et al., MLSys 2020
   - Proximal term for non-IID data

4. **Revisiting BPR: A Replicability Study**
   - RecSys 2024
   - 50% performance variance with improper implementation

### Datasets

- **MovieLens 1M**: https://grouplens.org/datasets/movielens/1m/
  - 1 million ratings, 6,040 users, 3,883 movies

### Frameworks

- **Flower**: https://flower.ai - Federated learning framework
- **PyTorch**: https://pytorch.org - Deep learning framework
- **Weights & Biases**: https://wandb.ai - Experiment tracking

---

## 🤝 MCP Usage

- Always use **Context7 MCP** when needing library/API documentation, code generation, setup or configuration steps
- Use **Pal MCP** for ultrathink about improvements, planning, brainstorming, code reviews

---

## 📝 License

Apache License 2.0

---

**Last Updated**: 2026-01-22
**Project Status**: ✅ Baseline Complete, Serves as comparison for Personalized FL
**Role**: Baseline for thesis comparison
**Thesis Author**: Dang Vinh
**Thesis Title**: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
