# Federated Adaptive Personalized Collaborative Filtering

**Master Thesis: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems**

A Flower federated learning project implementing **Personalized Federated Learning** for collaborative filtering on the MovieLens 1M dataset, featuring **Split Learning Architecture** with **BPR-MF (Bayesian Personalized Ranking Matrix Factorization)**, **FedAvg/FedProx** strategies, and comprehensive ranking evaluation metrics.

## Research Contribution

This project implements and evaluates personalized federated learning approaches for recommendation systems, with focus on:

1. **Split Learning Architecture**: Separating user embeddings (local/private) from item embeddings (global/shared)
2. **Non-IID Data Handling**: Dirichlet-based partitioning to simulate realistic federated scenarios
3. **Privacy-Preserving Recommendations**: User preferences never leave client devices

### Proposed Experiments

- **Adaptive Alpha (α)**: Dynamic Dirichlet concentration based on user characteristics
- **Popularity-Weighted Negative Sampling**: Improved BPR training with popularity-aware sampling

## Features

- **Split Learning Architecture**: User embeddings stay local, item embeddings are globally aggregated
- **MovieLens 1M Dataset**: Automatic download and preprocessing
- **Dirichlet Partitioning**: Creates realistic non-IID data distribution based on genre preferences
- **Model**: BPR-MF (Bayesian Personalized Ranking Matrix Factorization) - PyTorch implementation
- **Federated Strategies**: FedAvg and FedProx with split-aware proximal term
- **Comprehensive Ranking Metrics**: NDCG@K, Hit Rate@K, Precision@K, Recall@K, MAP@K, MRR, Coverage, Novelty
- **Experiment Tracking**: Weights & Biases (wandb) integration
- **Visualization Tools**: Analyze data partitioning across clients

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Flower Server                  │
                    │  ┌─────────────────────────────────────┐│
                    │  │    Global Parameters (Aggregated)   ││
                    │  │  • Item Embeddings (3706 × 128)     ││
                    │  │  • Item Biases (3706)               ││
                    │  │  • Global Bias (1)                  ││
                    │  └─────────────────────────────────────┘│
                    │         FedAvg / FedProx                │
                    └────────────────┬────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
    ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
    │   Client 0    │        │   Client 1    │        │   Client N    │
    ├───────────────┤        ├───────────────┤        ├───────────────┤
    │ LOCAL (Private)│       │ LOCAL (Private)│       │ LOCAL (Private)│
    │ • User Embeds │        │ • User Embeds │        │ • User Embeds │
    │ • User Biases │        │ • User Biases │        │ • User Biases │
    │               │        │               │        │               │
    │ CACHED LOCALLY│        │ CACHED LOCALLY│        │ CACHED LOCALLY│
    └───────────────┘        └───────────────┘        └───────────────┘
```

### Split Learning Design

| Parameter Type | Location | Aggregation | Privacy |
|----------------|----------|-------------|---------|
| User Embeddings | Client | None (local only) | Private |
| User Biases | Client | None (local only) | Private |
| Item Embeddings | Server | FedAvg/FedProx | Shared |
| Item Biases | Server | FedAvg/FedProx | Shared |
| Global Bias | Server | FedAvg/FedProx | Shared |

## Dataset

**MovieLens 1M**:
- 1,000,209 ratings from 6,040 users on 3,883 movies
- Rating scale: 1-5 stars
- 18 movie genres
- Automatically downloaded on first run

## Quick Start

### Install dependencies

```bash
pip install -e .
```

### Run Federated Training (FedAvg)

```bash
flwr run .
```

### Run with FedProx

```bash
flwr run . --run-config "strategy='fedprox' proximal-mu=0.01"
```

### Run with Custom Configuration

```bash
# 50 rounds with FedProx
flwr run . --run-config "num-server-rounds=50 strategy='fedprox' proximal-mu=0.1"

# Different embedding size and more clients
flwr run . --run-config "embedding-dim=256"

# Adjust Dirichlet parameter (more non-IID)
flwr run . --run-config "alpha=0.1"
```

### Test Dataset Loading

```bash
python test_dataset.py
```

### Visualize Data Partitions

```bash
python visualize_partitions.py
```

## Evaluation Metrics

### Rating Prediction
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

### Ranking Metrics (Primary Focus)
| Metric | Description | Optimized By |
|--------|-------------|--------------|
| **NDCG@K** | Normalized Discounted Cumulative Gain | BPR-MF |
| **Hit Rate@K** | % of users with at least one relevant item in top-K | BPR-MF |
| **Precision@K** | Fraction of relevant items in top-K | BPR-MF |
| **Recall@K** | Fraction of relevant items retrieved | BPR-MF |
| **MAP@K** | Mean Average Precision | BPR-MF |
| **MRR** | Mean Reciprocal Rank | BPR-MF |
| **Coverage@K** | Catalog coverage in recommendations | - |
| **Novelty@K** | Average self-information of recommended items | - |
| **F1@K** | Harmonic mean of Precision and Recall | - |

## Configuration

Key parameters in `pyproject.toml`:

```toml
[tool.flwr.app.config]
# Federated Learning
num-server-rounds = 10
fraction-train = 1.0
local-epochs = 5
strategy = "fedavg"          # "fedavg" or "fedprox"
proximal-mu = 0.01           # FedProx strength (0.001, 0.01, 0.1, 1.0)

# Model
model-type = "bpr"           # "basic" (MSE) or "bpr" (ranking)
embedding-dim = 128          # Latent dimensions
dropout = 0.1

# Training
lr = 0.005
weight-decay = 1e-5
num-negatives = 1            # Negative samples per positive

# Data
alpha = 0.5                  # Dirichlet concentration (lower = more non-IID)

# Evaluation
enable-ranking-eval = true
ranking-k-values = "5,10,20"

# Experiment Tracking
wandb-enabled = true
wandb-project = "federated-adaptive-personalized-cf"
```

## Dirichlet Partitioning

The α parameter controls data heterogeneity:

| α Value | Heterogeneity | Description |
|---------|---------------|-------------|
| 0.1 | Very High | Extreme non-IID (clients specialize in few genres) |
| 0.5 | High | Recommended for realistic FL experiments |
| 1.0 | Moderate | Balanced but still non-IID |
| 10.0 | Low | Nearly IID distribution |

## Project Structure

```
federated-adaptive-personalized-cf/
├── federated_adaptive_personalized_cf/
│   ├── __init__.py
│   ├── dataset.py           # MovieLens 1M loader & Dirichlet partitioner
│   ├── task.py              # Training, evaluation & ranking metrics
│   ├── strategy.py          # SplitFedAvg & SplitFedProx strategies
│   ├── client_app.py        # Flower client with split learning
│   ├── server_app.py        # Flower server with wandb integration
│   └── models/
│       ├── __init__.py
│       ├── basic_mf.py      # Basic Matrix Factorization (MSE)
│       ├── bpr_mf.py        # BPR Matrix Factorization (ranking)
│       └── losses.py        # MSELoss, BPRLoss implementations
├── data/
│   └── ml-1m/               # MovieLens 1M dataset (auto-downloaded)
├── figures/                 # Generated visualizations
├── test_dataset.py          # Test dataset loading
├── test_models.py           # Test model implementations
├── visualize_partitions.py  # Generate visualizations
├── pyproject.toml           # Project config & dependencies
├── README.md                # This file
└── claude.md                # Detailed technical documentation
```

## Federated Strategies

### FedAvg (Federated Averaging)
Standard federated averaging where global parameters are aggregated using weighted average based on number of local samples.

### FedProx (Federated Proximal)
Adds a proximal term to handle heterogeneous data:
- Proximal term applied **only to global parameters** (item embeddings, biases)
- User embeddings remain fully personalized (no proximal constraint)
- Controls drift from global model with μ parameter

```
L_local = L_BPR + (μ/2) * ||w_global - w_server||²
```

## Expected Results

### BPR-MF Performance (Split Learning)

| Metric | Round 1 | Round 10 | Notes |
|--------|---------|----------|-------|
| Train Loss | ~0.50 | ~0.25 | BPR loss (lower is better) |
| Hit Rate@10 | ~0.55 | ~0.70 | Target metric |
| NDCG@10 | ~0.30 | ~0.45 | Ranking quality |
| Precision@10 | ~0.05 | ~0.10 | Sparse ground truth |

**Note**: RMSE/MAE are high for BPR-MF (~2.0+) because it optimizes ranking, not rating prediction.

## Future Work / Proposed Experiments

### 1. Adaptive Alpha (α)
Dynamic adjustment of Dirichlet concentration based on user characteristics:
- Users with diverse preferences → higher α (more balanced)
- Users with niche preferences → lower α (more specialized)

### 2. Popularity-Weighted Negative Sampling
Replace uniform negative sampling with popularity-aware sampling:
- Popular items as harder negatives
- Unpopular items for diversity
- Better model calibration

### 3. Additional Experiments
- [ ] Compare FedAvg vs FedProx convergence
- [ ] Analyze user embedding divergence across rounds
- [ ] Cold-start user performance
- [ ] Communication efficiency analysis

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{vinh2025personalizedfl,
  title={Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems},
  author={Dang Vinh},
  year={2025}
}
```

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/1m/)
- [BPR Paper](https://arxiv.org/abs/1205.2618)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)

## License

Apache License 2.0
