# Federated Adaptive Personalized Collaborative Filtering

> **Master Thesis: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems**
> MovieLens 1M | Split Learning | FedAvg/FedProx | BPR-MF | NDCG@K, HitRate@K

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Contributions](#research-contributions)
3. [Architecture](#architecture)
4. [Key Components](#key-components)
5. [Data Pipeline](#data-pipeline)
6. [Model Architecture](#model-architecture)
7. [Federated Learning Setup](#federated-learning-setup)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Usage Guide](#usage-guide)
10. [Configuration](#configuration)
11. [Proposed Experiments](#proposed-experiments)
12. [Technical Details](#technical-details)
13. [References](#references)

---

## Project Overview

### Purpose

This project implements **Personalized Federated Learning** for collaborative filtering recommendation systems as part of a master thesis. It uses **Split Learning Architecture** where:

- **User embeddings** remain **local** (private, never sent to server)
- **Item embeddings** are **global** (aggregated via FedAvg/FedProx)

### Research Focus

**Thesis Title**: *Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems*

**Key Research Questions**:
1. How does split learning architecture affect recommendation quality vs. privacy?
2. Can FedProx improve convergence on non-IID recommendation data?
3. What is the impact of adaptive data partitioning on personalization?

### Why This Matters

Traditional recommendation systems centralize user data, raising **privacy concerns**. Personalized Federated Learning enables:

- **Privacy preservation**: User preferences (embeddings) never leave devices
- **Personalization**: Local user embeddings adapt to individual preferences
- **Scalability**: Only item embeddings are communicated
- **Regulatory compliance**: GDPR, CCPA compatibility

---

## Research Contributions

### 1. Split Learning for Recommendations (Implemented)

Separating model parameters into local and global:

```python
# Global parameters (sent to server, aggregated)
GLOBAL_PARAMS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')

# Local parameters (stay on client, NOT aggregated)
LOCAL_PARAMS = ('user_embeddings.weight', 'user_bias.weight')
```

**Benefits**:
- User preferences never leave client devices
- Reduced communication (only item embeddings transmitted)
- Better personalization through local user embeddings

### 2. Split-Aware FedProx (Implemented)

Modified FedProx that applies proximal term **only to global parameters**:

```python
# FedProx loss with split learning
L_total = L_BPR + (mu/2) * ||w_global - w_server||^2

# Where w_global = {item_embeddings, item_bias, global_bias}
# User embeddings are NOT constrained
```

**Why**: User embeddings should be free to personalize without being pulled toward a global average.

### 3. Proposed: Adaptive Alpha (α) (Experiment)

Dynamic Dirichlet concentration based on user characteristics:

```python
# Proposed algorithm
def compute_adaptive_alpha(user_profile):
    genre_diversity = entropy(user_genre_distribution)
    activity_level = len(user_ratings) / avg_ratings

    # More diverse users → higher α (less genre clustering)
    # Niche users → lower α (stronger genre clustering)
    alpha = base_alpha * (1 + diversity_weight * genre_diversity)
    return alpha
```

### 4. Proposed: Popularity-Weighted Negative Sampling (Experiment)

Replace uniform negative sampling with popularity-aware strategy:

```python
# Proposed sampling strategies
def popularity_weighted_negative_sampling(user_id, pos_item, item_popularity):
    # Strategy 1: Hard negatives (popular items user hasn't rated)
    # - Popular items are "harder" to rank correctly
    # - Better gradient signal for learning

    # Strategy 2: Anti-popularity (unpopular items)
    # - Promotes diversity and long-tail recommendations
    # - Reduces popularity bias

    # Strategy 3: Mixed (α * popular + (1-α) * uniform)
    # - Balance between hard negatives and exploration
```

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Flower Server                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Global Parameters                                │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │  │
│  │  │ Item Embeddings  │  │   Item Biases    │  │   Global Bias    │  │  │
│  │  │  (3706 × 128)    │  │     (3706)       │  │       (1)        │  │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│                    FedAvg / FedProx Aggregation                          │
│                    (weighted by num_examples)                            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
  │   Client 0    │        │   Client 1    │        │   Client N    │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ GLOBAL (recv) │        │ GLOBAL (recv) │        │ GLOBAL (recv) │
  │ • Item Embeds │        │ • Item Embeds │        │ • Item Embeds │
  │ • Item Biases │        │ • Item Biases │        │ • Item Biases │
  │ • Global Bias │        │ • Global Bias │        │ • Global Bias │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ LOCAL (cache) │        │ LOCAL (cache) │        │ LOCAL (cache) │
  │ • User Embeds │        │ • User Embeds │        │ • User Embeds │
  │ • User Biases │        │ • User Biases │        │ • User Biases │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │   Local Data  │        │   Local Data  │        │   Local Data  │
  │  (990 ratings)│        │(8722 ratings) │        │(5899 ratings) │
  │   (8 users)   │        │  (108 users)  │        │  (60 users)   │
  └───────────────┘        └───────────────┘        └───────────────┘
```

### Split Learning Flow

```
Round t:
1. Server sends global params to selected clients
2. Client loads global params from server message
3. Client loads local params from cache (if exists)
4. Client trains on local data (BPR loss + optional FedProx proximal term)
5. Client saves local params to cache
6. Client returns ONLY global params to server
7. Server aggregates global params via FedAvg/FedProx
8. Repeat for round t+1
```

### Parameter Classification

| Parameter | Dimensions | Type | Privacy | Communication |
|-----------|------------|------|---------|---------------|
| `user_embeddings.weight` | (6040, 128) | Local | Private | Never sent |
| `user_bias.weight` | (6040, 1) | Local | Private | Never sent |
| `item_embeddings.weight` | (3706, 128) | Global | Shared | Each round |
| `item_bias.weight` | (3706, 1) | Global | Shared | Each round |
| `global_bias` | (1,) | Global | Shared | Each round |

**Total Parameters**: ~1.25M
- Local (not transmitted): ~773K (62%)
- Global (transmitted): ~478K (38%)

---

## Key Components

### Core Files

```
federated-adaptive-personalized-cf/
├── federated_adaptive_personalized_cf/
│   ├── __init__.py
│   ├── dataset.py              # Data loading & Dirichlet partitioning
│   ├── task.py                 # Training, evaluation & ranking metrics
│   ├── strategy.py             # SplitFedAvg & SplitFedProx strategies
│   ├── client_app.py           # Flower client with split learning
│   ├── server_app.py           # Flower server with wandb integration
│   └── models/
│       ├── __init__.py
│       ├── basic_mf.py         # Basic Matrix Factorization (MSE)
│       ├── bpr_mf.py           # BPR Matrix Factorization (ranking)
│       └── losses.py           # MSELoss, BPRLoss implementations
├── pyproject.toml              # Configuration & dependencies
├── test_dataset.py             # Dataset testing
├── test_models.py              # Model testing
└── visualize_partitions.py     # Partition visualization
```

---

## Data Pipeline

### 1. Dataset: MovieLens 1M

**Source**: `dataset.py:55-91`

- **1,000,209 ratings** from **6,040 users** on **3,883 movies**
- Rating scale: 1-5 stars
- 18 movie genres
- Timestamps, user demographics included

**Automatic Download**:
```python
from federated_adaptive_personalized_cf.dataset import download_movielens_1m

# Downloads from GroupLens, extracts to data/ml-1m
data_path = download_movielens_1m(data_dir="./data")
```

### 2. Dirichlet Partitioning

**Algorithm**: `dataset.py:183-283`

**Purpose**: Create **non-IID data distribution** across clients based on genre preferences.

**How It Works**:

```python
# Step 1: Compute user genre preferences
user_genre_dist = compute_user_genre_distribution(ratings_df, movies_df)
# → Each user: vector of 18 genre probabilities

# Step 2: Sample client genre distributions from Dirichlet(α)
client_genre_dist = np.random.dirichlet([alpha] * num_genres, num_clients)
# → α=0.5: high heterogeneity (realistic FL scenario)
# → α=1.0: moderate heterogeneity
# → α=10.0: nearly IID

# Step 3: Assign users to clients via KL divergence
for user in users:
    best_client = argmin(KL(user_pref || client_pref))
    assign(user → best_client)
```

**Key Parameters**:
- `alpha=0.5` (recommended): High non-IID, genre-based clustering
- `num_partitions=10`: Number of federated clients
- `test_ratio=0.2`: 80% train, 20% test split

**Partition Statistics** (α=0.5, 5 clients):
```
Client   Users   Ratings   Genre Focus
   0        8       990    Drama/Thriller
   1      108     8,722    Comedy/Romance
   2    3,427   619,815    Diverse (largest)
   3       44     3,897    Action/Adventure
   4       60     5,899    Sci-Fi/Fantasy
```

### 3. Data Loading

**Function**: `task.py`

```python
from federated_adaptive_personalized_cf.task import load_data

# Client-side data loading
trainloader, testloader = load_data(
    partition_id=0,        # Which client (0-N)
    num_partitions=5,      # Total clients
    alpha=0.5,             # Dirichlet concentration
    test_ratio=0.2,        # Train/test split
    batch_size=256,        # DataLoader batch size
)
```

**Output**:
- `trainloader`: PyTorch DataLoader with batches `{user, item, rating}`
- `testloader`: Separate test set for evaluation
- Caches `num_users`, `num_items`, `user2idx`, `item2idx`

---

## Model Architecture

### BPR-MF (Primary Model)

**File**: `models/bpr_mf.py`

**Architecture**:
```python
class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, dropout=0.1):
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
```

**Score Computation**:
```python
def _compute_score(self, user_ids, item_ids):
    user_emb = self.dropout(self.user_embeddings(user_ids))
    item_emb = self.dropout(self.item_embeddings(item_ids))

    interaction = torch.sum(user_emb * item_emb, dim=1)

    score = (self.global_bias +
             self.user_bias(user_ids).squeeze() +
             self.item_bias(item_ids).squeeze() +
             interaction)
    return score
```

**BPR Forward Pass**:
```python
def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
    pos_scores = self._compute_score(user_ids, pos_item_ids)

    if neg_item_ids is not None:
        neg_scores = self._compute_score(user_ids, neg_item_ids)
        return pos_scores, neg_scores

    return pos_scores  # For evaluation
```

**Split Learning Methods**:
```python
# Get only global parameters for server aggregation
def get_global_parameters(self):
    return {
        'item_embeddings.weight': self.item_embeddings.weight.data,
        'item_bias.weight': self.item_bias.weight.data,
        'global_bias': self.global_bias.data
    }

# Set global parameters from server, preserve local
def set_global_parameters(self, global_params):
    self.item_embeddings.weight.data.copy_(global_params['item_embeddings.weight'])
    self.item_bias.weight.data.copy_(global_params['item_bias.weight'])
    self.global_bias.data.copy_(global_params['global_bias'])

# Get local parameters for caching
def get_local_parameters(self):
    return {
        'user_embeddings.weight': self.user_embeddings.weight.data,
        'user_bias.weight': self.user_bias.weight.data
    }

# Load local parameters from cache
def set_local_parameters(self, local_params):
    self.user_embeddings.weight.data.copy_(local_params['user_embeddings.weight'])
    self.user_bias.weight.data.copy_(local_params['user_bias.weight'])
```

### Loss Functions

**File**: `models/losses.py`

**BPR Loss**:
```python
class BPRLoss(nn.Module):
    def __init__(self, margin=0.0):
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        diff = pos_scores - neg_scores - self.margin
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
        return loss
```

**Interpretation**:
- Maximizes score difference between positive and negative items
- User should prefer observed items over unobserved items
- Optimizes ranking, not absolute rating values

### Negative Sampling

**File**: `models/bpr_mf.py:246-314`

**Current Implementation** (Uniform):
```python
def sample_negatives(self, user_ids, pos_item_ids, user_rated_items, num_negatives=1):
    neg_items = []
    for user_id, pos_item in zip(user_ids, pos_item_ids):
        rated = user_rated_items.get(int(user_id), set())
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in rated:
                neg_items.append(neg_item)
                break
    return torch.LongTensor(neg_items)
```

**Proposed Enhancement** (Popularity-Weighted):
```python
# TODO: Implement popularity-weighted negative sampling
def sample_negatives_popularity(self, user_ids, pos_item_ids,
                                user_rated_items, item_popularity,
                                strategy='hard'):
    """
    strategy='hard': Sample popular items (harder negatives)
    strategy='soft': Sample unpopular items (diversity)
    strategy='mixed': α * popular + (1-α) * uniform
    """
    pass  # Proposed experiment
```

### Model Initialization

**Best Practices**: `models/bpr_mf.py`

```python
def _init_weights(self):
    # Xavier/Glorot initialization (critical!)
    init.xavier_uniform_(self.user_embeddings.weight)
    init.xavier_uniform_(self.item_embeddings.weight)

    # Small random biases
    init.normal_(self.user_bias.weight, mean=0.0, std=0.01)
    init.normal_(self.item_bias.weight, mean=0.0, std=0.01)

    # Zero global bias (learns mean rating)
    init.zeros_(self.global_bias)
```

---

## Federated Learning Setup

### 1. Client App

**File**: `client_app.py`

**Split Learning Train Flow**:
```python
@app.train()
def train(msg: Message, context: Context):
    # 1. Get configuration
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = context.run_config.get("embedding-dim", 128)

    # 2. Create model with default initialization
    model = get_model(model_type, embedding_dim, num_users, num_items)

    # 3. Load GLOBAL params from server
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state_dict)

    # 4. Load LOCAL params from cache (if exists)
    local_params = load_local_user_embeddings(partition_id)
    if local_params is not None:
        model.set_local_parameters(local_params)

    # 5. Train on local data
    train_loss = train_fn(
        model, trainloader,
        epochs=local_epochs, lr=lr,
        model_type=model_type,
        proximal_mu=proximal_mu,  # FedProx
        global_params=global_state_dict  # For proximal term
    )

    # 6. Save LOCAL params to cache
    save_local_user_embeddings(partition_id, model.get_local_parameters())

    # 7. Return ONLY global params to server
    return Message(
        content={"arrays": ArrayRecord(model.get_global_parameters()),
                 "metrics": {"train_loss": train_loss}}
    )
```

**User Embedding Persistence**:
```python
CACHE_DIR = ".embedding_cache/partition_{id}/"

def save_local_user_embeddings(partition_id, local_params, round_num=None):
    cache_dir = get_cache_dir(partition_id)
    cache_file = cache_dir / "user_embeddings.pt"

    # Atomic save with temp file
    temp_file = cache_file.with_suffix('.tmp')
    torch.save({
        'user_embeddings.weight': local_params['user_embeddings.weight'],
        'user_bias.weight': local_params['user_bias.weight'],
        '_round': round_num,
        '_timestamp': time.time()
    }, temp_file)
    temp_file.rename(cache_file)

def load_local_user_embeddings(partition_id):
    cache_file = get_cache_dir(partition_id) / "user_embeddings.pt"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)
    return None
```

### 2. Server App

**File**: `server_app.py`

**Main Function**:
```python
@app.main()
def main(grid: Grid, context: Context):
    # 1. Read configuration
    num_rounds = context.run_config.get("num-server-rounds", 10)
    strategy_name = context.run_config.get("strategy", "fedavg")
    proximal_mu = context.run_config.get("proximal-mu", 0.01)

    # 2. Initialize strategy
    if strategy_name == "fedprox":
        strategy = SplitFedProx(
            fraction_train=fraction_train,
            proximal_mu=proximal_mu
        )
    else:
        strategy = SplitFedAvg(fraction_train=fraction_train)

    # 3. Initialize global model (only global params)
    global_model = get_model(model_type, embedding_dim, num_users, num_items)
    initial_params = global_model.get_global_parameters()

    # 4. Run federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial_params),
        num_rounds=num_rounds
    )

    # 5. Log to wandb (if enabled)
    if wandb_enabled:
        wandb.log({"final_loss": result.metrics["eval_loss"]})

    # 6. Save results
    save_results(result, config)
```

### 3. Federated Strategies

**File**: `strategy.py`

**SplitFedAvg**:
```python
class SplitFedAvg(FedAvg):
    """FedAvg that only aggregates global parameters."""

    def __init__(self, fraction_train=1.0):
        super().__init__(fraction_train=fraction_train)
        self.global_param_keys = GLOBAL_PARAMS
        self.local_param_keys = LOCAL_PARAMS

    def aggregate(self, results):
        # Filter to only global parameters before aggregation
        global_results = [
            (extract_global_params(r.parameters), r.num_examples)
            for r in results
        ]
        return weighted_average(global_results)
```

**SplitFedProx**:
```python
class SplitFedProx:
    """FedProx with split-aware proximal term."""

    def __init__(self, fraction_train=1.0, proximal_mu=0.01):
        self.fraction_train = fraction_train
        self.proximal_mu = proximal_mu

    # Proximal term applied only to global params in client training
    # See task.py train_bpr_mf() for implementation
```

**FedProx Training** (in `task.py`):
```python
def train_bpr_mf(model, trainloader, epochs, lr, device,
                 proximal_mu=0.0, global_params=None, global_param_names=None):

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = BPRLoss()

    for epoch in range(epochs):
        for batch in trainloader:
            # ... forward pass, BPR loss ...

            # FedProx proximal term (only on global params)
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                global_param_set = set(global_param_names)

                idx = 0
                for name, local_w in model.named_parameters():
                    if name in global_param_set:
                        proximal_term += (local_w - global_params[idx]).norm(2) ** 2
                        idx += 1

                loss = bpr_loss + (proximal_mu / 2) * proximal_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Evaluation Metrics

### Rating Prediction Metrics

**File**: `task.py:293-371`

```python
def test(model, testloader, device, model_type="bpr"):
    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0

    with torch.no_grad():
        for batch in testloader:
            predictions = model(user_ids, item_ids, neg_item_ids=None)
            predictions = torch.clamp(predictions, min=1.0, max=5.0)

            total_squared_error += ((predictions - ratings) ** 2).sum()
            total_absolute_error += torch.abs(predictions - ratings).sum()

    rmse = torch.sqrt(total_squared_error / num_samples)
    mae = total_absolute_error / num_samples

    return {"rmse": rmse, "mae": mae}
```

### Ranking Metrics (Primary Focus)

**File**: `task.py:368-737`

| Metric | Formula | Implementation |
|--------|---------|----------------|
| **Hit Rate@K** | `hits / num_users` | `compute_hit_rate()` |
| **Precision@K** | `hits / K` | `compute_precision()` |
| **Recall@K** | `hits / num_relevant` | `compute_recall()` |
| **F1@K** | `2 * P * R / (P + R)` | `compute_f1()` |
| **NDCG@K** | `DCG / IDCG` | `compute_ndcg()` |
| **MAP@K** | `mean(AP@K)` | `compute_map()` |
| **MRR** | `mean(1 / rank_first_hit)` | `compute_mrr()` |
| **Coverage@K** | `unique_items / catalog_size` | `compute_coverage()` |
| **Novelty@K** | `mean(-log2(popularity))` | `compute_novelty()` |

**NDCG Implementation**:
```python
def compute_ndcg(recommended_items, relevant_items, k):
    """
    Normalized Discounted Cumulative Gain at K.

    DCG = sum(rel_i / log2(i + 1))  for i in 1..K
    IDCG = ideal DCG (all relevant items at top)
    NDCG = DCG / IDCG
    """
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

    # IDCG: assume perfect ranking
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))

    return dcg / idcg if idcg > 0 else 0.0
```

**Comprehensive Evaluation**:
```python
def evaluate_ranking(model, testloader, device, k_values=[5, 10, 20],
                     trainloader=None, item_popularity=None):
    """
    Compute all ranking metrics for given K values.

    Returns:
        {
            'hit_rate_5': 0.65, 'hit_rate_10': 0.72, 'hit_rate_20': 0.81,
            'precision_5': 0.12, 'precision_10': 0.09, 'precision_20': 0.06,
            'recall_5': 0.15, 'recall_10': 0.22, 'recall_20': 0.35,
            'f1_5': 0.13, 'f1_10': 0.13, 'f1_20': 0.13,
            'ndcg_5': 0.38, 'ndcg_10': 0.42, 'ndcg_20': 0.45,
            'map_5': 0.25, 'map_10': 0.28, 'map_20': 0.30,
            'mrr': 0.45,
            'coverage_10': 0.35,
            'novelty_10': 4.2
        }
    """
```

---

## Usage Guide

### Installation

```bash
cd federated-adaptive-personalized-cf
pip install -e .
```

**Dependencies** (from `pyproject.toml`):
- `flwr[simulation]>=1.22.0`: Flower federated learning
- `torch>=2.7.1`: PyTorch deep learning
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing
- `scikit-learn>=1.3.0`: ML utilities
- `wandb>=0.16.0`: Experiment tracking

### Basic Usage

**1. Run Federated Training (FedAvg)**:
```bash
flwr run .
```

**2. Run with FedProx**:
```bash
flwr run . --run-config "strategy='fedprox' proximal-mu=0.01"
```

**3. Custom Configuration**:
```bash
# 50 rounds, FedProx, higher proximal strength
flwr run . --run-config "num-server-rounds=50 strategy='fedprox' proximal-mu=0.1"

# Different embedding size
flwr run . --run-config "embedding-dim=256"

# More non-IID data
flwr run . --run-config "alpha=0.1"

# Disable wandb
flwr run . --run-config "wandb-enabled=false"
```

**4. Visualize Partitions**:
```bash
python visualize_partitions.py
```

### Programmatic Usage

```python
from federated_adaptive_personalized_cf.task import get_model, load_data, train, test
from federated_adaptive_personalized_cf.task import evaluate_ranking

# Load data for client 0
trainloader, testloader = load_data(
    partition_id=0,
    num_partitions=5,
    alpha=0.5
)

# Initialize model
model = get_model(model_type="bpr", embedding_dim=128,
                  num_users=6040, num_items=3706)

# Train
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loss = train(
    model, trainloader,
    epochs=5, lr=0.005, device=device,
    model_type="bpr"
)

# Evaluate rating prediction
eval_metrics = test(model, testloader, device, model_type="bpr")
print(f"RMSE: {eval_metrics['rmse']:.4f}, MAE: {eval_metrics['mae']:.4f}")

# Evaluate ranking
ranking_metrics = evaluate_ranking(
    model, testloader, device,
    k_values=[5, 10, 20],
    trainloader=trainloader
)
print(f"NDCG@10: {ranking_metrics['ndcg_10']:.4f}")
print(f"Hit Rate@10: {ranking_metrics['hit_rate_10']:.4f}")
```

---

## Configuration

### pyproject.toml

```toml
[tool.flwr.app.config]
# Federated Learning parameters
num-server-rounds = 10      # FL rounds (3-100)
fraction-train = 1.0        # Fraction of clients per round (0.1-1.0)
local-epochs = 5            # Local training epochs (1-10)
strategy = "fedavg"         # "fedavg" or "fedprox"
proximal-mu = 0.01          # FedProx strength (0.001-1.0)

# Model parameters
model-type = "bpr"          # "basic" (MSE) or "bpr" (ranking)
embedding-dim = 128         # Latent dimensions (32, 64, 128, 256)
dropout = 0.1               # Dropout rate (0.0-0.5)

# Training parameters
lr = 0.005                  # Learning rate (1e-4 to 1e-2)
weight-decay = 1e-5         # L2 regularization (1e-6 to 1e-3)
num-negatives = 1           # Negatives per positive for BPR (1-5)

# Data partitioning
alpha = 0.5                 # Dirichlet concentration (0.1-10.0)

# Evaluation
enable-ranking-eval = true
ranking-k-values = "5,10,20"

# Experiment tracking
wandb-enabled = true
wandb-project = "federated-adaptive-personalized-cf"
```

### Parameter Guidelines

| Parameter | Recommended | Purpose | Trade-offs |
|-----------|-------------|---------|------------|
| **num-server-rounds** | 10-50 | FL convergence | More = better performance, longer |
| **fraction-train** | 0.5-1.0 | Client sampling | Higher = faster convergence |
| **local-epochs** | 3-5 | Local training | More = better local fit, drift risk |
| **embedding-dim** | 64-128 | Model capacity | Higher = more expressive |
| **proximal-mu** | 0.01 | FedProx strength | Higher = more global constraint |
| **alpha** | 0.5 | Data heterogeneity | Lower = more non-IID |

---

## Proposed Experiments

### Experiment 1: Adaptive Alpha (α)

**Hypothesis**: Dynamic Dirichlet concentration based on user characteristics improves personalization.

**Proposed Algorithm**:
```python
def compute_adaptive_alpha(user_id, ratings_df, movies_df, base_alpha=0.5):
    """
    Compute per-user alpha based on:
    1. Genre diversity (entropy of genre distribution)
    2. Activity level (number of ratings)
    3. Rating variance (user's rating consistency)
    """
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    # Genre diversity
    genre_dist = compute_user_genre_distribution(user_ratings, movies_df)
    diversity = entropy(genre_dist)

    # Activity level
    activity = len(user_ratings) / ratings_df.groupby('user_id').size().mean()

    # More diverse users → higher α (less clustering)
    # More active users → higher α (more confident assignment)
    adaptive_alpha = base_alpha * (1 + 0.5 * diversity) * (1 + 0.2 * np.log1p(activity))

    return np.clip(adaptive_alpha, 0.1, 2.0)
```

**Evaluation**:
- Compare fixed α vs adaptive α on NDCG@10, Hit Rate@10
- Analyze per-client performance variance
- Measure personalization quality

### Experiment 2: Popularity-Weighted Negative Sampling

**Hypothesis**: Sampling popular items as negatives provides harder training signal.

**Proposed Strategies**:

```python
def popularity_weighted_negative_sampling(user_id, pos_item, item_popularity,
                                          num_items, user_rated_items,
                                          strategy='hard', temperature=1.0):
    """
    Strategy 1: 'hard' - Sample popular items (harder negatives)
    Strategy 2: 'soft' - Sample unpopular items (diversity)
    Strategy 3: 'mixed' - α * popular + (1-α) * uniform
    """
    rated = user_rated_items.get(user_id, set())

    if strategy == 'hard':
        # Popular items are harder negatives
        weights = np.array([item_popularity.get(i, 1e-6) for i in range(num_items)])
        weights = weights ** temperature
    elif strategy == 'soft':
        # Unpopular items for diversity
        weights = np.array([1.0 / (item_popularity.get(i, 1e-6) + 1e-6)
                           for i in range(num_items)])
        weights = weights ** temperature
    else:  # 'mixed'
        alpha = 0.5
        pop_weights = np.array([item_popularity.get(i, 1e-6) for i in range(num_items)])
        weights = alpha * pop_weights + (1 - alpha) * np.ones(num_items)

    # Zero out rated items
    for rated_item in rated:
        weights[rated_item] = 0

    # Normalize and sample
    weights = weights / weights.sum()
    neg_item = np.random.choice(num_items, p=weights)

    return neg_item
```

**Evaluation**:
- Compare uniform vs hard vs soft vs mixed on NDCG@10
- Measure novelty and coverage metrics
- Analyze ranking quality for long-tail items

### Experiment 3: FedAvg vs FedProx Comparison

**Objective**: Quantify the benefit of FedProx on non-IID recommendation data.

**Setup**:
- Fixed: α=0.5, embedding_dim=128, 10 rounds
- Variable: strategy (fedavg, fedprox), proximal_mu (0.001, 0.01, 0.1, 1.0)

**Metrics**:
- Convergence speed (rounds to target NDCG@10)
- Final performance (NDCG@10, Hit Rate@10)
- Per-client performance variance

### Experiment 4: Privacy-Utility Trade-off

**Objective**: Measure the cost of privacy in split learning.

**Setup**:
- Baseline: Centralized training (full MF)
- Split: Federated with split architecture
- Compare with varying num_rounds

**Metrics**:
- NDCG@10 gap (centralized vs federated)
- Communication cost (bytes transmitted)
- Personalization quality (per-user NDCG variance)

---

## Technical Details

### Why Split Learning for Recommendations?

**Traditional FL Problem**:
```
In standard FedAvg, user embeddings are aggregated:
- User 42 on Client A has embedding [0.1, 0.2, ...]
- User 42 on Client B has embedding [0.3, 0.4, ...]
- After aggregation: [0.2, 0.3, ...] (meaningless average!)
```

**Split Learning Solution**:
```
- User embeddings stay local (never aggregated)
- Each client has its own user embedding space
- Item embeddings are shared (useful for cold-start items)
```

### FedProx Deep Dive

**Standard FedAvg Problem**:
- Clients with different data drift in different directions
- Aggregation averages conflicting updates
- Slow convergence on non-IID data

**FedProx Solution**:
```
L_local = L_task + (μ/2) * ||w - w_global||²

Where:
- L_task: Task-specific loss (BPR for ranking)
- w: Local model weights
- w_global: Global model weights from server
- μ: Proximal strength (hyperparameter)
```

**Split-Aware FedProx**:
```python
# Only apply to global parameters
for name, param in model.named_parameters():
    if name in GLOBAL_PARAMS:
        proximal_term += (param - global_param).norm(2) ** 2

# User embeddings are FREE to personalize
# (no proximal constraint)
```

### Matrix Factorization Math

**Model**:
```
R ≈ U @ V.T

Where:
- R: Rating matrix (users × items)
- U: User embeddings (users × d)
- V: Item embeddings (items × d)
- d: Embedding dimension (128)
```

**Prediction with Biases**:
```
r̂_ui = μ + b_u + b_i + u_u^T v_i

Where:
- μ: Global mean rating
- b_u: User bias (tendency to rate high/low)
- b_i: Item bias (popularity)
- u_u: User embedding (preferences)
- v_i: Item embedding (attributes)
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

---

## References

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

5. **Federated Matrix Factorization for Recommendation**
   - Chai et al., 2019
   - Privacy-preserving recommendations

### Datasets

- **MovieLens 1M**: https://grouplens.org/datasets/movielens/1m/
  - 1 million ratings, 6,040 users, 3,883 movies

### Frameworks

- **Flower**: https://flower.ai - Federated learning framework
- **PyTorch**: https://pytorch.org - Deep learning framework
- **Weights & Biases**: https://wandb.ai - Experiment tracking

---

## License

Apache License 2.0

---

**Last Updated**: 2025-01-XX
**Project Status**: Baseline Complete, Proposed Experiments Pending
**Thesis Author**: Dang Vinh
**Thesis Title**: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
