# Federated Personalized Collaborative Filtering

> **Split Learning for Privacy-Preserving Personalized Recommendations**
> MovieLens 1M | Dirichlet Partitioning | SplitFedAvg/FedProx | BPR-MF | Local User Embeddings | Global Item Embeddings

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Research Context](#-research-context)
3. [Split Learning Architecture](#-split-learning-architecture)
4. [Key Components](#-key-components)
5. [Data Pipeline](#-data-pipeline)
6. [Model Architecture](#-model-architecture)
7. [Federated Learning Setup](#-federated-learning-setup)
8. [User Embedding Persistence](#-user-embedding-persistence)
9. [Custom Strategies](#-custom-strategies)
10. [Evaluation Metrics](#-evaluation-metrics)
11. [Usage Guide](#-usage-guide)
12. [Configuration](#-configuration)
13. [Performance Analysis](#-performance-analysis)
14. [Technical Details](#-technical-details)
15. [Comparison with Baseline](#-comparison-with-baseline)
16. [Future Work](#-future-work)
17. [References](#-references)

---

## 🎯 Project Overview

### Purpose

This project implements **Split Learning** for federated collaborative filtering, where model parameters are divided into:

- **LOCAL Parameters**: User embeddings & biases → Stay on clients (private)
- **GLOBAL Parameters**: Item embeddings & biases → Aggregated via FedAvg/FedProx

This is the **middle step** in the thesis progression:

```
Baseline → Personalized (Split) → Adaptive Personalized
(all global)  (THIS PROJECT)        (multi-factor α)
```

### Key Innovation: Split Learning

```
╔══════════════════════════════════════════════════════════════════════╗
║                    SPLIT LEARNING ARCHITECTURE                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   LOCAL Parameters (Private)     │    GLOBAL Parameters (Shared)     ║
║   ─────────────────────────────  │    ──────────────────────────     ║
║   • user_embeddings.weight       │    • item_embeddings.weight       ║
║   • user_bias.weight             │    • item_bias.weight             ║
║                                  │    • global_bias                  ║
║                                  │                                   ║
║   → Stay on client devices       │    → Aggregated on server         ║
║   → Accumulated over rounds      │    → Shared across all clients    ║
║   → Cached locally               │    → FedAvg/FedProx aggregation   ║
║                                  │                                   ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Why This Matters

| Aspect | Baseline (All Global) | Split Learning (This Project) |
|--------|----------------------|------------------------------|
| **User Embeddings** | Aggregated on server ❌ | Private on clients ✅ |
| **Privacy** | Limited | **Enhanced** |
| **Personalization** | None (averaged out) | **Preserved** |
| **Communication** | All parameters sent | Only item embeddings sent |
| **User History** | Reset each round | **Accumulated over time** |

---

## 🔬 Research Context

### Thesis Progression

| Project | Approach | User Embeddings | Key Feature |
|---------|----------|-----------------|-------------|
| **federated-baseline-cf** | Standard FedAvg | Global (aggregated) | Baseline comparison |
| **federated-personalized-cf** | Split Learning | **Local (private)** | Privacy + Personalization |
| **federated-adaptive-personalized-cf** | Split + Adaptive α | Local + Adaptive | Multi-factor personalization |

### RecSys 2024 Best Practices

This implementation follows **RecSys 2024 best practices**:

- **BPR-MF** (Bayesian Personalized Ranking) for ranking optimization
- **Xavier initialization** for embeddings (critical for performance)
- **Negative sampling** done correctly (exclude rated items)
- **Split learning** for privacy-preserving personalization

### Privacy Benefits

```
✅ User embeddings NEVER leave the client device
✅ Server only sees item embeddings (public knowledge)
✅ User preferences encoded in local embeddings
✅ Reduced communication cost (fewer parameters)
✅ Cumulative personalization over federated rounds
```

---

## 🏗️ Split Learning Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Flower Server                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               GLOBAL Parameters Only (Aggregated)              │  │
│  │                                                                 │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │  │
│  │  │ Item Embeddings │  │   Item Bias     │  │  Global Bias  │  │  │
│  │  │  (3706 × 128)   │  │   (3706 × 1)    │  │     (1,)      │  │  │
│  │  │     GLOBAL      │  │     GLOBAL      │  │    GLOBAL     │  │  │
│  │  └─────────────────┘  └─────────────────┘  └───────────────┘  │  │
│  │                                                                 │  │
│  │                 SplitFedAvg / SplitFedProx                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                    Only GLOBAL params exchanged
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
  │   Client 0    │        │   Client 1    │        │   Client N    │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ LOCAL (cached)│        │ LOCAL (cached)│        │ LOCAL (cached)│
  │ • user_emb    │        │ • user_emb    │        │ • user_emb    │
  │ • user_bias   │        │ • user_bias   │        │ • user_bias   │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ Local Data    │        │ Local Data    │        │ Local Data    │
  │ (partition)   │        │ (partition)   │        │ (partition)   │
  └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
          │                        │                        │
          └───────► .embedding_cache/ (persisted locally) ◄─┘
```

### Training Flow (Per Round)

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Round k Flow                                 │
└──────────────────────────────────────────────────────────────────────┘

SERVER → CLIENT:
  1. Server sends GLOBAL params (item_emb, item_bias, global_bias)
  2. Client loads GLOBAL params into model

CLIENT SETUP:
  3. Client loads LOCAL params from cache (.embedding_cache/)
     • Round 1: Use default Xavier initialization
     • Round 2+: Load previous user embeddings

CLIENT TRAINING:
  4. Train on local data (all parameters updated)
  5. If FedProx: Proximal term ONLY on GLOBAL params
     L = L_task + (μ/2) × ||w_global - w_server||²
     (User embeddings NOT regularized toward server!)

CLIENT → SERVER:
  6. Save LOCAL params to cache (for next round)
  7. Send ONLY GLOBAL params to server

SERVER AGGREGATION:
  8. Aggregate GLOBAL params using weighted average
  9. Update global model
```

### Parameter Classification

| Parameter | Dimensions | Type | Sent to Server | Cached Locally |
|-----------|------------|------|----------------|----------------|
| `user_embeddings.weight` | (6040, 128) | **LOCAL** | ❌ No | ✅ Yes |
| `user_bias.weight` | (6040, 1) | **LOCAL** | ❌ No | ✅ Yes |
| `item_embeddings.weight` | (3706, 128) | **GLOBAL** | ✅ Yes | ❌ No |
| `item_bias.weight` | (3706, 1) | **GLOBAL** | ✅ Yes | ❌ No |
| `global_bias` | (1,) | **GLOBAL** | ✅ Yes | ❌ No |

**Communication Savings**: Only ~485K parameters transmitted vs ~874K in baseline (44% reduction)

---

## 🔑 Key Components

### Directory Structure

```
federated-personalized-cf/
├── federated_personalized_cf/
│   ├── __init__.py
│   ├── dataset.py              # Data loading & Dirichlet partitioning
│   ├── task.py                 # Training, evaluation, ranking metrics
│   ├── client_app.py           # Flower client with split learning
│   ├── server_app.py           # Flower server with W&B logging
│   ├── strategy.py             # SplitFedAvg & SplitFedProx strategies
│   └── models/
│       ├── __init__.py
│       ├── basic_mf.py         # BasicMF with LOCAL/GLOBAL param methods
│       ├── bpr_mf.py           # BPRMF with LOCAL/GLOBAL param methods
│       └── losses.py           # MSELoss, BPRLoss implementations
├── .embedding_cache/           # Persisted user embeddings (created at runtime)
│   ├── partition_0/
│   │   └── user_embeddings.pt
│   ├── partition_1/
│   │   └── user_embeddings.pt
│   └── ...
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
| `task.py` | ~740 | Training (BasicMF/BPRMF), testing, ranking evaluation |
| `client_app.py` | ~395 | Split learning client, embedding caching, FedProx support |
| `server_app.py` | ~340 | Flower server, W&B logging, results export |
| `strategy.py` | ~125 | SplitFedAvg, SplitFedProx custom strategies |
| `basic_mf.py` | ~345 | BasicMF with get/set_local/global_parameters() |
| `bpr_mf.py` | ~467 | BPRMF with negative sampling + split learning |
| `losses.py` | ~132 | MSE and BPR loss implementations |

**Total**: ~2,986 lines of Python code

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
from federated_personalized_cf.task import load_data

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

Both models use **embedding-based matrix factorization** with **split learning support**:

```
Prediction = global_bias + user_bias[u] + item_bias[i] + dot(user_emb[u], item_emb[i])
```

### 1. BasicMF (MSE Loss)

**File**: `models/basic_mf.py`

**Purpose**: Traditional rating prediction (optimize RMSE/MAE)

```python
class BasicMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, dropout=0.1):
        # LOCAL parameters (stay on client)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)

        # GLOBAL parameters (aggregated on server)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

    # Split learning methods
    def get_global_parameters(self) -> OrderedDict:
        """Extract only GLOBAL params for server aggregation."""
        return {
            'item_embeddings.weight': self.item_embeddings.weight.data,
            'item_bias.weight': self.item_bias.weight.data,
            'global_bias': self.global_bias.data,
        }

    def set_global_parameters(self, state_dict: dict):
        """Load GLOBAL params from server."""
        self.item_embeddings.weight.data = state_dict['item_embeddings.weight']
        self.item_bias.weight.data = state_dict['item_bias.weight']
        self.global_bias.data = state_dict['global_bias']

    def get_local_parameters(self) -> OrderedDict:
        """Extract LOCAL params for caching."""
        return {
            'user_embeddings.weight': self.user_embeddings.weight.data,
            'user_bias.weight': self.user_bias.weight.data,
        }

    def set_local_parameters(self, state_dict: dict, strict=False):
        """Load LOCAL params from cache."""
        # Handles shape mismatches (partial loading)
```

**Loss Function**:
```python
loss = MSELoss(predictions, ratings)
```

### 2. BPRMF (BPR Loss) - Recommended

**File**: `models/bpr_mf.py`

**Purpose**: Ranking optimization (optimize top-K recommendations)

**Key Difference**: Optimizes **pairwise ranking** instead of absolute ratings.

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

### Client App (Split Learning)

**File**: `client_app.py`

```python
from flwr.client import ClientApp

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # 1. Create model with default initialization
    model = get_model(model_type, embedding_dim)

    # 2. Load GLOBAL params from server
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # 3. Load LOCAL params from cache (key innovation!)
    partition_id = context.node_config["partition-id"]
    loaded = load_local_user_embeddings(model, partition_id)
    # Round 1: loaded=False (use default init)
    # Round 2+: loaded=True (restore previous user embeddings!)

    # 4. FedProx: Only regularize GLOBAL params
    if proximal_mu > 0:
        global_param_names = model.get_global_parameter_names()
        global_params = [p.clone() for name, p in model.named_parameters()
                         if name in global_param_names]

    # 5. Load local data partition
    trainloader, _ = load_data(partition_id, num_partitions)

    # 6. Train locally
    train_loss = train_fn(model, trainloader, epochs, lr,
                          proximal_mu=proximal_mu,
                          global_params=global_params,
                          global_param_names=global_param_names)

    # 7. Save LOCAL params to cache (for next round!)
    save_local_user_embeddings(model, partition_id)

    # 8. Return ONLY GLOBAL params to server
    global_params = model.get_global_parameters()
    return Message(content={
        "arrays": ArrayRecord(global_params),  # Only GLOBAL!
        "metrics": {"train_loss": train_loss}
    })
```

### Server App

**File**: `server_app.py`

```python
from flwr.server import ServerApp
from federated_personalized_cf.strategy import SplitFedAvg, SplitFedProx

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context):
    # 1. Initialize global model
    global_model = get_model(model_type, embedding_dim)

    # 2. Extract only GLOBAL params for initial distribution
    initial_arrays = global_model.get_global_parameters()

    # 3. Select split learning strategy
    if strategy_name == "fedprox":
        strategy = SplitFedProx(fraction_fit, proximal_mu=proximal_mu)
    else:
        strategy = SplitFedAvg(fraction_fit)

    # 4. Run federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # 5. Log to W&B and save results
    wandb.log({"final/rmse": final_metrics['rmse'], ...})
    save_results_json(final_metrics)
```

---

## 💾 User Embedding Persistence

### Cache System

**Purpose**: Preserve user embeddings across federated rounds for personalization.

**Location**: `.embedding_cache/partition_{id}/user_embeddings.pt`

**File**: `client_app.py` - `save_local_user_embeddings()` and `load_local_user_embeddings()`

```python
def save_local_user_embeddings(model, partition_id):
    """Save user embeddings to local cache after training."""
    cache_dir = Path(".embedding_cache") / f"partition_{partition_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_params = model.get_local_parameters()
    local_params['_round'] = current_round
    local_params['_timestamp'] = datetime.now().isoformat()

    torch.save(local_params, cache_dir / "user_embeddings.pt")


def load_local_user_embeddings(model, partition_id):
    """Load user embeddings from cache before training."""
    cache_file = Path(".embedding_cache") / f"partition_{partition_id}" / "user_embeddings.pt"

    if not cache_file.exists():
        return False  # First round, use default init

    local_params = torch.load(cache_file)
    model.set_local_parameters(local_params, strict=False)
    return True
```

### Shape Mismatch Handling

**Scenario**: Client's user population changes between rounds.

**Solution**: Partial loading with graceful degradation.

```python
def set_local_parameters(self, state_dict, strict=False):
    """Load local params with shape mismatch handling."""
    saved_shape = state_dict['user_embeddings.weight'].shape
    current_shape = self.user_embeddings.weight.shape

    if saved_shape[0] < current_shape[0]:
        # Saved has fewer users: partial load, new users keep default init
        self.user_embeddings.weight.data[:saved_shape[0]] = saved_shape['user_embeddings.weight']
    elif saved_shape[0] > current_shape[0]:
        # Saved has more users: truncate to fit
        self.user_embeddings.weight.data = saved_shape['user_embeddings.weight'][:current_shape[0]]
    else:
        # Exact match
        self.user_embeddings.weight.data = saved_shape['user_embeddings.weight']
```

### Cache Contents

```python
# Saved to .embedding_cache/partition_0/user_embeddings.pt
{
    'user_embeddings.weight': tensor(shape=[num_users, embedding_dim]),
    'user_bias.weight': tensor(shape=[num_users, 1]),
    '_round': 5,
    '_timestamp': '2026-01-22T14:30:00'
}
```

---

## 🔧 Custom Strategies

### Strategy File

**File**: `strategy.py`

```python
from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx

# Parameter classification
GLOBAL_PARAM_KEYS = frozenset([
    'item_embeddings.weight',
    'item_bias.weight',
    'global_bias',
])

LOCAL_PARAM_KEYS = frozenset([
    'user_embeddings.weight',
    'user_bias.weight',
])
```

### SplitFedAvg

```python
class SplitFedAvg(BaseFedAvg):
    """FedAvg for Split Learning - only aggregates GLOBAL parameters.

    The "split" happens at the client level - clients only send GLOBAL params.
    Aggregation logic is unchanged from standard FedAvg.
    """

    def __init__(self, fraction_fit: float = 1.0, **kwargs):
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

    def __repr__(self):
        return f"SplitFedAvg(fraction_fit={self.fraction_fit})"
```

### SplitFedProx

```python
class SplitFedProx(BaseFedProx):
    """FedProx for Split Learning - proximal term only on GLOBAL parameters.

    In split learning with FedProx:
    - Proximal term ||w - w_global||² only applies to GLOBAL params
    - LOCAL params (user embeddings) are NOT regularized toward server
    - This enables true personalization while preventing drift on shared params
    """

    def __init__(self, fraction_fit=1.0, proximal_mu=0.01, **kwargs):
        super().__init__(fraction_fit=fraction_fit, proximal_mu=proximal_mu, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

    def __repr__(self):
        return f"SplitFedProx(fraction_fit={self.fraction_fit}, proximal_mu={self.proximal_mu})"
```

### Key Difference: FedProx with Split Learning

```
Standard FedProx (Baseline):
  L = L_task + (μ/2) × ||w_all - w_server||²
  → All parameters regularized toward server
  → User embeddings pulled toward global average

Split FedProx (This Project):
  L = L_task + (μ/2) × ||w_global - w_server||²
  → Only item embeddings regularized
  → User embeddings free to personalize!
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
cd federated-personalized-cf
pip install -e .
```

**Dependencies** (from `pyproject.toml`):
- `flwr[simulation]>=1.22.0`: Flower federated learning
- `torch>=2.7.1`: PyTorch deep learning
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing
- `scikit-learn>=1.3.0`: Machine learning utilities
- `wandb>=0.16.0`: Experiment tracking

### Basic Usage

**1. Run Federated Training** (default BPR-MF with SplitFedAvg):
```bash
flwr run .
```

**2. Run with SplitFedProx**:
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

### Clear User Embedding Cache

```bash
# Reset user embeddings to start fresh
rm -rf .embedding_cache/
```

### Visualize Data Partitions

```bash
python visualize_partitions.py
```

### Programmatic Usage

```python
from federated_personalized_cf.task import get_model, load_data, train, test

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

# Get GLOBAL params for server
global_params = model.get_global_parameters()

# Get LOCAL params for caching
local_params = model.get_local_parameters()
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
# Strategy Selection (Split Learning)
# =============================================================================
strategy = "fedavg"            # "fedavg" (SplitFedAvg) or "fedprox" (SplitFedProx)
proximal-mu = 0.01             # FedProx proximal term (only for GLOBAL params)

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
wandb-project = "federated-personalized-cf"
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

#### BPR-MF with Split Learning - Recommended

**Rating Metrics** (not optimized for):
| Round | Train Loss | RMSE | MAE |
|-------|------------|------|-----|
| 1 | 0.55 | 2.80 | 2.55 |
| 5 | 0.28 | 2.30 | 2.00 |
| 10 | 0.22 | 2.20 | 1.90 |

**Ranking Metrics** (optimized for):
- Hit Rate@10: **0.70-0.80** (improved from baseline 0.65-0.75)
- Precision@10: **0.10-0.14**
- NDCG@10: **0.18-0.28**

### Comparison: Baseline vs Split Learning

| Metric | Baseline (All Global) | Split Learning (This) | Improvement |
|--------|----------------------|----------------------|-------------|
| Hit Rate@10 | 0.65-0.75 | **0.70-0.80** | +5-7% |
| Precision@10 | 0.08-0.12 | **0.10-0.14** | +2-3% |
| NDCG@10 | 0.15-0.25 | **0.18-0.28** | +3-5% |
| Privacy | Limited | **Enhanced** | ✅ |
| Communication | 874K params | **485K params** | -44% |

### Key Insights

1. **User embeddings accumulate** - Performance improves as user embeddings are refined over rounds
2. **Privacy preserved** - User preferences never leave the device
3. **Communication reduced** - Only item embeddings transmitted
4. **Heterogeneity handled better** - Local user embeddings adapt to client-specific distributions

---

## 🔬 Technical Details

### Why Split Learning?

**Problem with Baseline** (all global):
```
User embeddings are AVERAGED across all clients:
→ Action fan + Romance fan = mediocre embeddings for both
→ Heavy users diluted by sparse users
→ No personalization possible
```

**Solution with Split Learning**:
```
User embeddings stay LOCAL:
→ Action fan keeps action-tuned embeddings
→ Romance fan keeps romance-tuned embeddings
→ Each client's users get personalized representations
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

### Matrix Factorization Math

**Model**:
```
R ≈ U @ V.T

Where:
- R: Rating matrix (users × items)
- U: User embeddings (LOCAL)
- V: Item embeddings (GLOBAL)
```

**Prediction**:
```
r̂_ui = μ + b_u + b_i + u_u^T × v_i

Where:
- μ: Global bias (GLOBAL)
- b_u: User bias (LOCAL)
- b_i: Item bias (GLOBAL)
- u_u: User embedding (LOCAL)
- v_i: Item embedding (GLOBAL)
```

---

## 📊 Comparison with Baseline

### Summary Table

| Aspect | Baseline | Split Learning (This) |
|--------|----------|----------------------|
| **User Embeddings** | Global (aggregated) | **Local (private)** |
| **Item Embeddings** | Global (aggregated) | Global (aggregated) |
| **User Biases** | Global (aggregated) | **Local (private)** |
| **Privacy** | Limited | **Enhanced** |
| **Personalization** | None | **User-level** |
| **Communication** | 874K params | **485K params** |
| **FedProx** | All params | **Global only** |
| **User History** | Reset each round | **Accumulated** |

### Architectural Difference

**Baseline**:
```
SERVER aggregates: ALL parameters
CLIENT sends: ALL parameters
Result: User preferences averaged away
```

**Split Learning**:
```
SERVER aggregates: Item embeddings only
CLIENT sends: Item embeddings only
CLIENT caches: User embeddings locally
Result: User preferences preserved!
```

---

## 🚀 Future Work

### 1. Adaptive Personalization (Thesis Contribution)

**See federated-adaptive-personalized-cf**:
- Multi-factor adaptive α
- Dual-level personalization (α-blending + PersonalMLP)
- Global prototype aggregation

### 2. Additional Enhancements

- **Differential Privacy**: Add noise for stronger privacy guarantees
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

4. **Split Learning: Distributed Learning without Sharing Raw Data**
   - Gupta & Raskar, 2018
   - Foundation for split learning approach

5. **Revisiting BPR: A Replicability Study**
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
**Project Status**: ✅ Split Learning Implemented, User Embedding Persistence Complete
**Role**: Middle step in thesis - Split Learning for Privacy + Personalization
**Thesis Author**: Dang Vinh
**Thesis Title**: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
