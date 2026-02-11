# Federated Adaptive Personalized Collaborative Filtering

> **Master Thesis: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems**
> MovieLens 1M | Split Learning | FedAvg/FedProx | BPR-MF | Hierarchical Conditional Alpha | Dual-Level Personalization | Global Prototype Aggregation

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Research Contributions](#-research-contributions)
3. [Architecture](#-architecture)
4. [Key Components](#-key-components)
5. [Adaptive Alpha Implementation](#-adaptive-alpha-implementation)
6. [Dual-Level Personalization](#-dual-level-personalization)
7. [Global Prototype Aggregation](#-global-prototype-aggregation)
8. [Data Pipeline](#-data-pipeline)
9. [Model Architecture](#-model-architecture)
10. [Federated Learning Setup](#-federated-learning-setup)
11. [Evaluation Metrics](#-evaluation-metrics)
12. [Usage Guide](#-usage-guide)
13. [Configuration](#-configuration)
14. [Experiments](#-experiments)
15. [Technical Details](#-technical-details)
16. [References](#-references)

---

## 🎯 Project Overview

### Purpose

This project implements **Personalized Federated Learning** for collaborative filtering recommendation systems as part of a master thesis. It introduces a novel **Multi-Factor Adaptive Alpha** approach and **Dual-Level Personalization** architecture:

- **User embeddings** remain **local** (private, never sent to server)
- **Item embeddings** are **global** (aggregated via FedAvg/FedProx)
- **Adaptive Alpha (α)** controls personalization level per client based on user characteristics
- **Global Prototype** helps sparse users via server-aggregated user representation

### Research Focus

**Thesis Title**: *Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems*

**Key Innovations**:
1. **Hierarchical Conditional Alpha**: Domain-aware alpha computation that addresses factor conflicts
2. **Multi-Factor Adaptive Alpha**: 4-factor weighted formula for personalization level
3. **Dual-Level Personalization**: Statistical (α-blending) + Neural (client-specific MLP)
4. **Global Prototype Aggregation**: EMA-based prototype for sparse user support

### Why This Matters

Traditional recommendation systems centralize user data, raising **privacy concerns**. This approach enables:

- ✅ **Privacy preservation**: User preferences (embeddings) never leave devices
- ✅ **Adaptive personalization**: Per-client α based on user characteristics
- ✅ **Sparse user support**: Global prototype helps users with few interactions
- ✅ **Scalability**: Only item embeddings communicated
- ✅ **Regulatory compliance**: GDPR, CCPA compatibility

---

## 🔬 Research Contributions

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

### 2. Hierarchical Conditional Alpha (Implemented - Key Thesis Contribution)

**File**: `models/adaptive_alpha.py`

Advanced alpha computation that addresses conflicts in the multi-factor approach through:
1. **Hierarchical factor aggregation** (resolves redundancy)
2. **Conditional rules** (handles domain-specific user archetypes)

**Identified Conflicts in Multi-Factor Approach:**
- **Quantity-Coverage Redundancy**: Correlation 0.8-1.0, combined 60% weight
- **Diversity-Consistency Contradiction**: Negative correlation -0.3 to -0.5

**Solution - Two-Stage Computation:**

```python
# Stage 1: Hierarchical Aggregation
data_volume = sqrt(f_quantity × f_coverage)           # Geometric mean (addresses redundancy)
preference_quality = 2×f_d×f_s / (f_d + f_s)         # Harmonic mean (balances conflict)

base_alpha = 0.55 × data_volume + 0.45 × preference_quality

# Stage 2: Conditional Rules (domain-aware adjustments)
# Rule 1: Sparse users (< 20 interactions) get penalty
# Rule 2: Niche specialists (low diversity, high quantity) get bonus
# Rule 3: Inconsistent raters (high variance) get penalty
# Rule 4: Completionists (high coverage, low diversity) get bonus
```

**Conditional Rules:**
| Rule | Condition | Adjustment | Rationale |
|------|-----------|------------|-----------|
| Sparse Users | n < 20 | penalty up to 50% | Insufficient data |
| Niche Specialists | f_d < 0.25 AND f_q > 0.6 | +0.15 bonus | Trust niche expertise |
| Inconsistent Raters | f_s < 0.3 | 30% penalty | Unreliable preferences |
| Completionists | f_c > 0.7 AND f_d < 0.3 | +0.1 bonus | Trust item exploration |

### 3. Multi-Factor Adaptive Alpha (Alternative Method)

**File**: `models/adaptive_alpha.py`

Simpler 4-factor weighted formula (use when hierarchical method is not needed):

```
α = w_q × f_quantity + w_d × f_diversity + w_c × f_coverage + w_s × f_consistency
α = clip(α, min_alpha, max_alpha)
```

**Default Weights** (sum to 1.0):
| Factor | Weight | Formula | Rationale |
|--------|--------|---------|-----------|
| **Quantity** | 0.40 | sigmoid((n - threshold) × temp) | More data = better local model |
| **Diversity** | 0.25 | genre_entropy / max_entropy | Diverse preferences need personalization |
| **Coverage** | 0.20 | min(n_unique_items / threshold, 1.0) | Wide coverage = reliable patterns |
| **Consistency** | 0.15 | 1 - (rating_std / max_std) | Stable preferences worth preserving |

**Known Issues** (addressed by hierarchical conditional):
- Quantity-Coverage redundancy (high correlation)
- Diversity-Consistency contradiction (negative correlation)
- Quantity dominance (40% weight)

### 4. Dual-Level Personalization (Implemented)

**File**: `models/dual_personalized_bpr_mf.py`

Novel architecture combining TWO levels of personalization:

```
Level 1 (Statistical - Embedding Space):
    p̃_u = α × p_local + (1 - α) × p_global

Level 2 (Neural - Function Space):
    score_cf = dot(p̃_u, q_i) + biases
    score_neural = PersonalMLP(p̃_u ⊙ q_i)
    final_score = Fusion(score_cf, score_neural)
```

**Why Dual-Level?**
- α is **interpretable** (computed from observable user behavior)
- MLP captures **non-linear patterns** (learned transformation)
- **Complementary**: α controls magnitude, MLP learns interaction

### 5. Global Prototype Aggregation (Implemented)

**File**: `strategy.py`

Server maintains EMA-based global user prototype:

```python
# EMA update: p_global = m × p_old + (1 - m) × p_new
new_prototype = weighted_average(client_prototypes)
global_prototype = momentum × global_prototype + (1 - momentum) × new_prototype
```

**Benefits**:
- Helps sparse users (low interaction count) by providing population average
- EMA ensures stability (momentum = 0.9 default)
- Privacy-preserving (aggregated across all clients)

### 6. Split-Aware FedProx (Implemented)

Modified FedProx that applies proximal term **only to global parameters**:

```python
L_total = L_BPR + (μ/2) × ||w_global - w_server||²

# User embeddings are NOT constrained - free to personalize
```

---

## 🏗️ Architecture

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
│  │                                                                     │  │
│  │                    Global User Prototype (EMA)                      │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │  p_global = 0.9 × p_old + 0.1 × weighted_avg(client_protos)  │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                    FedAvg / FedProx Aggregation                          │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
  │   Client 0    │        │   Client 1    │        │   Client N    │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ GLOBAL (recv) │        │ GLOBAL (recv) │        │ GLOBAL (recv) │
  │ • Item Embeds │        │ • Item Embeds │        │ • Item Embeds │
  │ • Global Proto│        │ • Global Proto│        │ • Global Proto│
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │ LOCAL (cache) │        │ LOCAL (cache) │        │ LOCAL (cache) │
  │ • User Embeds │        │ • User Embeds │        │ • User Embeds │
  │ • PersonalMLP │        │ • PersonalMLP │        │ • PersonalMLP │
  │ • α = 0.35    │        │ • α = 0.72    │        │ • α = 0.58    │
  ├───────────────┤        ├───────────────┤        ├───────────────┤
  │   Local Data  │        │   Local Data  │        │   Local Data  │
  │  (990 ratings)│        │(8722 ratings) │        │(5899 ratings) │
  └───────────────┘        └───────────────┘        └───────────────┘
```

### Dual-Level Personalization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dual-Level Personalization                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 1 (Statistical - Adaptive Alpha):                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  α = 0.40×f_quantity + 0.25×f_diversity + 0.20×f_coverage          │ │
│  │      + 0.15×f_consistency                                          │ │
│  │                                                                     │ │
│  │  p̃_u = α × p_local + (1 - α) × p_global                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                           │
│  Level 2 (Neural - Client-Specific MLP):                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  score_cf = dot(p̃_u, q_i) + b_u + b_i + μ                          │ │
│  │  score_mlp = PersonalMLP(p̃_u ⊙ q_i)                                │ │
│  │                                                                     │ │
│  │  Fusion Types:                                                      │ │
│  │  • "add":    final = score_cf + score_mlp                          │ │
│  │  • "gate":   final = σ(g) × score_cf + (1-σ(g)) × score_mlp        │ │
│  │  • "concat": final = Linear([score_cf; score_mlp])                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Parameter Classification

| Parameter | Dimensions | Type | Privacy | Communication |
|-----------|------------|------|---------|---------------|
| `user_embeddings.weight` | (6040, 128) | Local | Private | Never sent |
| `user_bias.weight` | (6040, 1) | Local | Private | Never sent |
| `personal_mlp.*` | ~16K params | Local | Private | Never sent |
| `fusion_gate/layer` | 1-3 params | Local | Private | Never sent |
| `item_embeddings.weight` | (3706, 128) | Global | Shared | Each round |
| `item_bias.weight` | (3706, 1) | Global | Shared | Each round |
| `global_bias` | (1,) | Global | Shared | Each round |

**Communication Savings**: Only ~38% of parameters transmitted per round.

---

## 🔑 Key Components

### Directory Structure

```
federated-adaptive-personalized-cf/
├── federated_adaptive_personalized_cf/
│   ├── __init__.py
│   ├── dataset.py                  # Data loading & Dirichlet partitioning
│   ├── task.py                     # Training, evaluation, alpha computation
│   ├── strategy.py                 # SplitFedAvg & SplitFedProx with prototype
│   ├── client_app.py               # Flower client with split learning + adaptive α
│   ├── server_app.py               # Flower server with wandb + alpha analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── basic_mf.py             # Basic Matrix Factorization (MSE)
│   │   ├── bpr_mf.py               # BPR Matrix Factorization (ranking)
│   │   ├── dual_personalized_bpr_mf.py  # Dual-Level Personalized BPR (NOVEL)
│   │   ├── adaptive_alpha.py       # DataQuantityAlpha & MultiFactorAlpha (NOVEL)
│   │   └── losses.py               # MSELoss, BPRLoss implementations
│   └── evaluation/                 # Analysis modules
│       ├── __init__.py
│       ├── alpha_analysis.py       # Alpha distribution & correlation analysis
│       └── user_groups.py          # Per-user-group metrics (sparse/medium/dense)
├── scripts/
│   └── analyze_sweep_results.py    # Hyperparameter sweep analysis
├── pyproject.toml                  # Configuration & dependencies
├── test_dataset.py                 # Dataset testing
├── test_models.py                  # Model testing
├── visualize_partitions.py         # Partition visualization
└── run_fusion_experiments.sh       # Experiment runner script
```

---

## 🎛️ Adaptive Alpha Implementation

**File**: `models/adaptive_alpha.py`

Alpha (α) controls the personalization level:
- α → 1: Fully personalized (use local user embedding)
- α → 0: Fully global (use global prototype)

### AlphaConfig Dataclass

```python
@dataclass
class AlphaConfig:
    # Method selection
    method: str = "hierarchical_conditional"  # "data_quantity", "multi_factor", or "hierarchical_conditional"

    # Common parameters
    min_alpha: float = 0.1         # Minimum personalization
    max_alpha: float = 0.95        # Maximum personalization
    quantity_threshold: int = 100  # Sigmoid midpoint
    quantity_temperature: float = 0.05  # Sigmoid steepness

    # Multi-factor weights (must sum to 1.0)
    factor_weights: Dict[str, float] = {
        'quantity': 0.40,
        'diversity': 0.25,
        'coverage': 0.20,
        'consistency': 0.15,
    }

    # Normalization thresholds
    max_entropy: float = 3.0       # ~log2(18) for MovieLens genres
    coverage_threshold: int = 100  # Items for full coverage credit
    max_rating_std: float = 1.5    # Typical max std for 1-5 ratings

@dataclass
class HierarchicalConditionalAlphaConfig:
    # Base parameters
    min_alpha: float = 0.1
    max_alpha: float = 0.95

    # Hierarchical weights (must sum to 1.0)
    data_volume_weight: float = 0.55      # Weight for sqrt(quantity × coverage)
    preference_quality_weight: float = 0.45  # Weight for harmonic(diversity, consistency)

    # Factor thresholds (same as multi-factor)
    quantity_threshold: int = 100
    quantity_temperature: float = 0.05
    max_entropy: float = 3.0
    coverage_threshold: int = 100
    max_rating_std: float = 1.5

    # Conditional rule thresholds
    sparse_threshold: int = 20           # Users below this get penalty
    sparse_penalty_max: float = 0.5      # Max penalty factor (50%)
    niche_diversity_threshold: float = 0.25
    niche_quantity_threshold: float = 0.6
    niche_bonus: float = 0.15
    inconsistent_threshold: float = 0.3
    inconsistent_penalty: float = 0.3    # 30% penalty
    completionist_coverage: float = 0.7
    completionist_diversity: float = 0.3
    completionist_bonus: float = 0.1
```

### Method 1: DataQuantityAlpha (Single Factor)

```python
class DataQuantityAlpha:
    """Compute alpha based on interaction count only."""

    def compute(self, n_interactions: int) -> float:
        x = (n_interactions - threshold) * temperature
        alpha_raw = sigmoid(x)
        return clip(alpha_raw, min_alpha, max_alpha)
```

**Example** (threshold=100, temperature=0.05):
- n=50 → α ≈ 0.076 → clipped to 0.1 (min)
- n=100 → α = 0.5 (midpoint)
- n=150 → α ≈ 0.92

### Method 2: MultiFactorAlpha (4 Factors - Key Contribution)

```python
class MultiFactorAlpha:
    """Compute alpha from 4 user characteristics."""

    def compute_from_stats(self, user_stats: Dict) -> float:
        # Factor 1: Quantity (sigmoid normalized)
        f_quantity = sigmoid((n - threshold) * temperature)

        # Factor 2: Diversity (genre entropy)
        f_diversity = min(genre_entropy / max_entropy, 1.0)

        # Factor 3: Coverage (unique items)
        f_coverage = min(n_unique_items / coverage_threshold, 1.0)

        # Factor 4: Consistency (inverse of rating std)
        f_consistency = 1.0 - min(rating_std / max_rating_std, 1.0)

        # Weighted combination
        alpha = (w_q * f_quantity + w_d * f_diversity +
                 w_c * f_coverage + w_s * f_consistency)

        return clip(alpha, min_alpha, max_alpha)
```

### Method 3: HierarchicalConditionalAlpha (Recommended)

```python
class HierarchicalConditionalAlpha:
    """
    Two-stage alpha computation:
    1. Hierarchical factor aggregation (addresses redundancy)
    2. Conditional adjustments (handles conflicts and edge cases)
    """

    def compute_from_stats(self, user_stats: Dict) -> float:
        # Get base factors (reuse existing computation)
        f_q = self._compute_quantity_factor(user_stats.get('n_interactions', 0))
        f_d = self._compute_diversity_factor(user_stats.get('genre_entropy', 1.5))
        f_c = self._compute_coverage_factor(user_stats.get('n_unique_items', 0))
        f_s = self._compute_consistency_factor(user_stats.get('rating_std', 0.75))

        # Stage 1: Hierarchical aggregation
        data_volume = np.sqrt(f_q * f_c)  # Geometric mean
        preference_quality = 2 * f_d * f_s / (f_d + f_s)  # Harmonic mean

        base_alpha = (0.55 * data_volume + 0.45 * preference_quality)

        # Stage 2: Conditional rules
        n = user_stats.get('n_interactions', 0)

        # Rule 1: Sparse users penalty
        if n < 20:
            penalty = 0.5 * (1 - n / 20)
            base_alpha *= (1 - penalty)

        # Rule 2: Niche specialists bonus
        if f_d < 0.25 and f_q > 0.6:
            base_alpha = min(base_alpha + 0.15, 1.0)

        # Rule 3: Inconsistent raters penalty
        if f_s < 0.3:
            base_alpha *= 0.7

        # Rule 4: Completionists bonus
        if f_c > 0.7 and f_d < 0.3:
            base_alpha = min(base_alpha + 0.1, 1.0)

        return float(np.clip(base_alpha, 0.1, 0.95))
```

**Example Outputs:**
- Sparse user (n=10, moderate diversity) → α ≈ 0.15 (low - insufficient data)
- Niche specialist (n=200, low diversity) → α ≈ 0.85 (high - trust niche expertise)
- Inconsistent explorer (n=150, high diversity, high variance) → α ≈ 0.40 (moderate)
- Completionist (n=300, high coverage, low diversity) → α ≈ 0.90 (high - trust exploration)

### Factory Function

```python
from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    create_alpha_computer, AlphaConfig, HierarchicalConditionalAlphaConfig
)

# Create hierarchical conditional alpha computer (recommended)
config = AlphaConfig(method="hierarchical_conditional")
hc_config = HierarchicalConditionalAlphaConfig()
alpha_computer = create_alpha_computer(config, hc_config=hc_config)

# Or create multi-factor alpha computer
config = AlphaConfig(method="multi_factor")
alpha_computer = create_alpha_computer(config)

# Compute alpha for a user
user_stats = {
    'n_interactions': 75,
    'genre_entropy': 2.5,
    'n_unique_items': 60,
    'rating_std': 0.8
}
alpha = alpha_computer.compute_from_stats(user_stats)  # e.g., 0.52
```

---

## 🧠 Dual-Level Personalization

**File**: `models/dual_personalized_bpr_mf.py`

### Architecture

```python
class DualPersonalizedBPRMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_hidden_dims: List[int] = None,  # Default: [dim, dim//2]
        dropout: float = 0.0,
        use_bias: bool = True,
        fusion_type: str = "add",  # "add", "gate", or "concat"
    ):
        # Embeddings (same as BPRMF)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)  # LOCAL
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)  # GLOBAL
        self.user_bias = nn.Embedding(num_users, 1)  # LOCAL
        self.item_bias = nn.Embedding(num_items, 1)  # GLOBAL
        self.global_bias = nn.Parameter(torch.zeros(1))  # GLOBAL

        # PersonalMLP (LOCAL - client-specific)
        self.personal_mlp = Sequential(
            Linear(embedding_dim, hidden_dims[0]),
            ReLU(),
            Linear(hidden_dims[0], hidden_dims[1]),
            ReLU(),
            Linear(hidden_dims[1], 1)
        )

        # Fusion mechanism
        if fusion_type == "gate":
            self.fusion_gate = nn.Parameter(torch.zeros(1))
        elif fusion_type == "concat":
            self.fusion_layer = nn.Linear(2, 1)

        # Adaptive personalization state
        self._alpha: float = 1.0
        self._global_prototype: Optional[torch.Tensor] = None
```

### Score Computation

```python
def _compute_score(self, user_ids, item_ids):
    # Level 1: Get α-blended user embeddings
    user_emb = self.get_effective_embedding(user_ids)  # α * local + (1-α) * global
    item_emb = self.item_embeddings(item_ids)

    # Path 1: Collaborative Filtering score
    cf_score = dot(user_emb, item_emb) + biases

    # Path 2: Neural personalized score
    interaction = user_emb * item_emb  # Element-wise product
    mlp_score = self.personal_mlp(interaction)

    # Fusion
    return self._fuse_scores(cf_score, mlp_score)

def get_effective_embedding(self, user_ids):
    """Level 1: α-blended embeddings."""
    local_emb = self.user_embeddings(user_ids)

    if self._global_prototype is None or self._alpha == 1.0:
        return local_emb

    return self._alpha * local_emb + (1 - self._alpha) * self._global_prototype
```

### Fusion Types

| Type | Formula | Parameters | Use Case |
|------|---------|------------|----------|
| **add** | `cf + mlp` | 0 | Simple, fast |
| **gate** | `σ(g) × cf + (1-σ(g)) × mlp` | 1 | Learnable balance |
| **concat** | `Linear([cf; mlp])` | 3 | Most flexible |

### Parameter Counts (ML-1M, dim=64)

```python
params = model.count_parameters()
# Returns:
{
    'global': 478,081,        # Item embeddings + biases + global bias
    'local_embeddings': 392,640,  # User embeddings + biases
    'local_mlp': 4,225,       # PersonalMLP weights
    'local_fusion': 3,        # Fusion layer (concat)
    'total_local': 396,868,
    'total': 874,949
}
```

---

## 🌐 Global Prototype Aggregation

**File**: `strategy.py`

### Purpose

Global prototype helps sparse users (low interaction count) by providing a "population average" user embedding to blend with their local embedding.

### Server-Side Aggregation (SplitFedAvg/SplitFedProx)

```python
class SplitFedAvg(BaseFedAvg):
    def __init__(self, fraction_fit=1.0, prototype_momentum=0.9):
        self.prototype_momentum = prototype_momentum
        self._global_prototype = None

    def aggregate_fit(self, server_round, results, failures):
        # Standard parameter aggregation
        aggregated_params, metrics = super().aggregate_fit(...)

        # Aggregate user prototypes
        self._aggregate_prototypes(results)

        return aggregated_params, metrics

    def _aggregate_prototypes(self, results):
        # Collect prototypes from clients
        prototypes_and_weights = []
        for _, fit_res in results:
            if USER_PROTOTYPE_KEY in fit_res.metrics:
                prototype = np.array(fit_res.metrics[USER_PROTOTYPE_KEY])
                weight = fit_res.num_examples
                prototypes_and_weights.append((prototype, weight))

        # Weighted average
        new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight

        # EMA update
        if self._global_prototype is None:
            self._global_prototype = new_prototype
        else:
            self._global_prototype = (
                self.prototype_momentum * self._global_prototype +
                (1 - self.prototype_momentum) * new_prototype
            )
```

### Client-Side Usage

```python
# In client_app.py train()

# Compute user prototype (mean of user embeddings)
user_prototype = model.compute_user_prototype()

# Send to server in metrics
metrics[USER_PROTOTYPE_KEY] = user_prototype.tolist()

# Receive global prototype from server (next round)
if global_prototype is not None:
    model.set_global_prototype(torch.tensor(global_prototype))
```

---

## 📊 Data Pipeline

### Dataset: MovieLens 1M

**File**: `dataset.py`

- **1,000,209 ratings** from **6,040 users** on **3,883 movies**
- Rating scale: 1-5 stars
- 18 movie genres
- Timestamps, user demographics included

### Dirichlet Partitioning

```python
# Non-IID data distribution based on genre preferences

# Step 1: Compute user genre preferences
user_genre_dist = compute_user_genre_distribution(ratings_df, movies_df)

# Step 2: Sample client genre distributions
client_genre_dist = np.random.dirichlet([alpha] * num_genres, num_clients)

# Step 3: Assign users to clients via KL divergence
for user in users:
    best_client = argmin(KL(user_pref || client_pref))
    assign(user → best_client)
```

**Key Parameters**:
- `alpha=0.5` (default): High non-IID, realistic FL scenario
- `num_partitions=5`: Number of federated clients
- `test_ratio=0.2`: 80% train, 20% test split

---

## 📈 Evaluation Metrics

### Rating Prediction Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

### Ranking Metrics (Primary Focus)

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

### Evaluation Modules

**File**: `evaluation/alpha_analysis.py`

```python
from federated_adaptive_personalized_cf.evaluation import AlphaAnalyzer

analyzer = AlphaAnalyzer()
analyzer.add_client_data(client_id=0, alpha=0.3, metrics={'ndcg@10': 0.15})
analyzer.add_client_data(client_id=1, alpha=0.8, metrics={'ndcg@10': 0.25})

stats = analyzer.compute_statistics()  # AlphaStatistics dataclass
correlations = analyzer.compute_correlations()  # alpha vs each metric
group_analysis = analyzer.group_by_alpha_range()  # low/mid/high alpha groups
```

**File**: `evaluation/user_groups.py`

```python
from federated_adaptive_personalized_cf.evaluation import (
    classify_user_group,
    aggregate_metrics_by_group,
    UserGroupConfig
)

config = UserGroupConfig(
    sparse=(0, 30),    # 0-30 interactions
    medium=(30, 100),  # 30-100 interactions
    dense=(100, 10000) # 100+ interactions
)

group = classify_user_group(n_interactions=25, config=config)  # "sparse"

# Aggregate metrics per group
group_metrics = aggregate_metrics_by_group(user_metrics, user_stats, config)
# Returns: {'sparse': {'ndcg@10': 0.12}, 'medium': {...}, 'dense': {...}}
```

---

## 📖 Usage Guide

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
- `wandb>=0.19.0`: Experiment tracking

### Basic Usage

**1. Run Federated Training (default BPR-MF with hierarchical conditional alpha)**:
```bash
flwr run .
```

**2. Run with FedProx**:
```bash
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
```

**3. Run with Dual-Level Personalization**:
```bash
flwr run . --run-config "model-type=dual fusion-type=concat"
```

**4. Custom Configuration**:
```bash
# 50 rounds, FedProx, hierarchical conditional alpha (default)
flwr run . --run-config "num-server-rounds=50 strategy=fedprox alpha-method=hierarchical_conditional"

# Use multi-factor alpha instead
flwr run . --run-config "alpha-method=multi_factor"

# Different embedding size
flwr run . --run-config "embedding-dim=256"

# Disable wandb
flwr run . --run-config "wandb-enabled=false"
```

### Programmatic Usage

```python
from federated_adaptive_personalized_cf.task import get_model, load_data, train, test
from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    create_alpha_computer, AlphaConfig, HierarchicalConditionalAlphaConfig
)

# Load data for client 0
trainloader, testloader = load_data(
    partition_id=0,
    num_partitions=5,
    alpha=0.5
)

# Initialize model
model = get_model(model_type="dual", embedding_dim=128,
                  num_users=6040, num_items=3706)

# Create hierarchical conditional alpha computer (recommended)
alpha_config = AlphaConfig(method="hierarchical_conditional")
hc_config = HierarchicalConditionalAlphaConfig()
alpha_computer = create_alpha_computer(alpha_config, hc_config=hc_config)

# Compute client alpha
client_alpha = alpha_computer.compute_from_stats(user_stats)
model.set_alpha(client_alpha)

# Train
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loss = train(model, trainloader, epochs=5, lr=0.005, device=device)

# Evaluate
eval_metrics = test(model, testloader, device)
```

---

## ⚙️ Configuration

### pyproject.toml

```toml
[tool.flwr.app.config]

# =============================================================================
# Federated Learning Parameters
# =============================================================================
num-server-rounds = 50
fraction-train = 1.0
local-epochs = 12
strategy = "fedavg"         # "fedavg" or "fedprox"
proximal-mu = 0.01          # FedProx strength

# =============================================================================
# Model Parameters
# =============================================================================
model-type = "bpr"          # "basic", "bpr", or "dual"
embedding-dim = 128
dropout = 0.1

# Dual Model Specific
mlp-hidden-dims = "512,256,128"  # PersonalMLP hidden layers
fusion-type = "concat"           # "add", "gate", or "concat"

# =============================================================================
# Adaptive Alpha Configuration (Key Thesis Feature)
# =============================================================================
alpha-method = "hierarchical_conditional"  # "data_quantity", "multi_factor", or "hierarchical_conditional"
alpha-min = 0.1
alpha-max = 0.95
alpha-quantity-threshold = 100
alpha-quantity-temperature = 0.05

# Multi-factor weights (only used when alpha-method = "multi_factor")
alpha-weight-quantity = 0.40
alpha-weight-diversity = 0.25
alpha-weight-coverage = 0.20
alpha-weight-consistency = 0.15

# Normalization thresholds (used by multi_factor and hierarchical_conditional)
alpha-max-entropy = 3.0
alpha-coverage-threshold = 100
alpha-max-rating-std = 1.5

# =============================================================================
# Hierarchical Conditional Alpha (only used when alpha-method = "hierarchical_conditional")
# =============================================================================
# Addresses factor conflicts in multi-factor approach:
# 1. Quantity-Coverage redundancy → geometric mean
# 2. Diversity-Consistency contradiction → harmonic mean

# Hierarchical weights (must sum to 1.0)
alpha-hc-data-volume-weight = 0.55    # Weight for data_volume = sqrt(quantity × coverage)
alpha-hc-preference-weight = 0.45     # Weight for preference_quality = harmonic(diversity, consistency)

# Conditional rule thresholds (domain-aware adjustments)
alpha-hc-sparse-threshold = 20        # Users below this get penalty
alpha-hc-sparse-penalty-max = 0.5     # Max penalty factor (50%)
alpha-hc-niche-diversity-threshold = 0.25
alpha-hc-niche-quantity-threshold = 0.6
alpha-hc-niche-bonus = 0.15
alpha-hc-inconsistent-threshold = 0.3
alpha-hc-inconsistent-penalty = 0.3
alpha-hc-completionist-coverage = 0.7
alpha-hc-completionist-diversity = 0.3
alpha-hc-completionist-bonus = 0.1

# =============================================================================
# Global Prototype Aggregation
# =============================================================================
prototype-momentum = 0.9

# =============================================================================
# User Group Boundaries (for per-group metrics)
# =============================================================================
user-group-sparse = "0,30"
user-group-medium = "30,100"
user-group-dense = "100,10000"

# =============================================================================
# Evaluation Configuration
# =============================================================================
enable-ranking-eval = true
ranking-k-values = "5,10,20"
eval-num-negatives = 99      # Sampled evaluation (NCF protocol)

# =============================================================================
# Experiment Tracking
# =============================================================================
wandb-enabled = true
wandb-project = "federated-adaptive-personalized-cf"

# =============================================================================
# Early Stopping
# =============================================================================
early-stopping-enabled = false     # Enable for hyperparameter sweeps
early-stopping-patience = 10       # Rounds without improvement
early-stopping-metric = "sampled_ndcg@10"
early-stopping-mode = "max"
early-stopping-min-delta = 0.001
```

### Early Stopping Usage

Enable early stopping to automatically stop training when metrics plateau:

```bash
# Enable early stopping with default patience (10 rounds)
flwr run . --run-config "early-stopping-enabled=true"

# Custom patience and metric
flwr run . --run-config "early-stopping-enabled=true early-stopping-patience=15 early-stopping-metric=ndcg@10"
```

### Hyperparameter Sweeps with wandb

The project includes full wandb sweep support for hyperparameter tuning:

**1. Create a sweep**:
```bash
cd federated-adaptive-personalized-cf
wandb sweep sweep.yaml
# Note the SWEEP_ID from output
```

**2. Run sweep agents** (can run multiple in parallel):
```bash
wandb agent <YOUR_ENTITY>/federated-adaptive-personalized-cf/<SWEEP_ID>

# Or run specific number of experiments
wandb agent --count 20 <SWEEP_ID>
```

**3. Use convenience scripts**:
```bash
# Load helper functions
source scripts/sweep_commands.sh

# Test sweep config locally (dry run)
test_sweep_config

# Create sweep
create_sweep

# Run agent
run_sweep_agent <SWEEP_ID>
```

**Sweep configuration** (`sweep.yaml`) includes:
- Bayesian optimization for efficient search
- Hyperband early termination for poor runs
- Key hyperparameters: lr, embedding_dim, model_type, fusion_type, alpha weights
- Automatically enables early stopping

### Parameter Guidelines

| Parameter | Recommended | Purpose | Trade-offs |
|-----------|-------------|---------|------------|
| **alpha-method** | hierarchical_conditional | Addresses factor conflicts in multi-factor | Requires user stats |
| **alpha-hc-data-volume-weight** | 0.55 | Data sufficiency weight | Balance with preference quality |
| **alpha-hc-preference-weight** | 0.45 | Preference quality weight | Balance with data volume |
| **prototype-momentum** | 0.9 | Stable prototype | Higher = slower adaptation |
| **fusion-type** | concat | Most flexible | More parameters |
| **model-type** | dual | Full personalization | More computation |

---

## 🧪 Experiments

### Implemented Experiments

#### 1. Hierarchical Conditional vs Multi-Factor vs Data-Quantity Alpha

**Status**: ✅ Implemented

**Configuration**:
```bash
# Hierarchical Conditional (recommended)
flwr run . --run-config "alpha-method=hierarchical_conditional"

# Multi-factor
flwr run . --run-config "alpha-method=multi_factor"

# Data-quantity only
flwr run . --run-config "alpha-method=data_quantity"
```

**Hypothesis**: Hierarchical conditional alpha outperforms multi-factor by addressing:
- Quantity-Coverage redundancy (geometric mean aggregation)
- Diversity-Consistency contradiction (harmonic mean balance)
- Domain-specific user archetypes (conditional rules)

#### 2. Dual-Level Personalization Ablation

**Status**: ✅ Implemented

**Configuration**:
```bash
# BPR only (no dual)
flwr run . --run-config "model-type=bpr"

# Dual with different fusions
flwr run . --run-config "model-type=dual fusion-type=add"
flwr run . --run-config "model-type=dual fusion-type=gate"
flwr run . --run-config "model-type=dual fusion-type=concat"
```

**Questions**:
- Does PersonalMLP improve over α-only?
- Which fusion type works best?

#### 3. FedAvg vs FedProx Comparison

**Status**: ✅ Implemented

**Configuration**:
```bash
flwr run . --run-config "strategy=fedavg"
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "strategy=fedprox proximal-mu=0.1"
```

### Proposed Experiments

#### 4. Popularity-Weighted Negative Sampling

**Status**: ❌ Not Implemented

**Proposed Strategies**:
- `'hard'`: Sample popular items (harder negatives)
- `'soft'`: Sample unpopular items (diversity)
- `'mixed'`: α × popular + (1-α) × uniform

---

## 🔬 Technical Details

### Why Multi-Factor Alpha?

**Problem with Single-Factor (Quantity-Only)**:

Users with many interactions tend to also have:
- Higher genre diversity (more movies = more genres)
- Higher item coverage (naturally)
- More stable rating patterns (regression to mean)

This creates **correlation problem**:
- Quantity dominates all other factors
- Alpha becomes essentially just f(n_interactions)
- Loses ability to capture diverse user characteristics

**Solution: Weighted Multi-Factor**:

By using diversity (25%), coverage (20%), and consistency (15%) alongside quantity (40%), we capture orthogonal user characteristics:

```
User A: 50 interactions, HIGH diversity → moderate α (diverse preferences)
User B: 50 interactions, LOW diversity  → lower α (predictable, use global)
User C: 150 interactions, LOW diversity → moderate α (not automatic high)
```

### Mathematical Interpretation

The multi-factor formula can be interpreted as:

```
α = E[quality of local model given user characteristics]
```

Where each factor estimates a different aspect of local model quality:
- **quantity**: "Do I have enough data?"
- **diversity**: "Is my data diverse enough to generalize?"
- **coverage**: "Have I explored the item space?"
- **consistency**: "Are my preferences stable?"

### Why Dual-Level Personalization?

Single-level approaches have limitations:

1. **α-only (APFL-style)**: Can only control HOW MUCH to personalize, not HOW
2. **MLP-only (PFedRec-style)**: Black-box, not interpretable

**Dual-level combines both**:
- α is **interpretable** (computed from observable user behavior)
- MLP captures **non-linear patterns** (learned transformation)
- **Complementary**: α controls magnitude, MLP learns transformation

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

5. **PFedRec: Privacy-Preserving Federated Recommendation via User-Specific MLP**
   - IJCAI 2023
   - Inspiration for client-specific neural scoring

6. **Adaptive Personalized Federated Learning (APFL)**
   - NeurIPS 2020
   - Adaptive mixing of local and global models

7. **Neural Collaborative Filtering (NCF)**
   - WWW 2017
   - Evaluation protocol: leave-one-out with 99 negatives

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

**Last Updated**: 2026-01-23
**Project Status**: Hierarchical Conditional Alpha Implemented, Multi-Factor Adaptive Alpha Implemented, Dual-Level Personalization Implemented, Experiments In Progress
**Thesis Author**: Dang Vinh
**Thesis Title**: Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
