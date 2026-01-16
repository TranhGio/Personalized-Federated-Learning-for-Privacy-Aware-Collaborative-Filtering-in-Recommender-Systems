# Adaptive-α BPR-FedMF: Complete Implementation Plan

## Project Overview

**Title:** Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems

**Core Contribution:** Adaptive personalization level (α) that automatically adjusts per-user based on their data characteristics, balancing local personalization with global knowledge transfer in federated BPR Matrix Factorization.

**Key Innovation:** Unlike fixed-split personalization (FedPer), our method dynamically determines HOW MUCH each user should rely on their local model vs. global model based on data quantity, diversity, and quality metrics.

---

## 1. Project Structure

```
adaptive_bpr_fedmf/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── default_config.yaml
│   └── experiment_configs/
│       ├── baseline_fedavg.yaml
│       ├── baseline_local_only.yaml
│       ├── fixed_alpha_05.yaml
│       ├── fixed_alpha_08.yaml
│       ├── adaptive_alpha_quantity.yaml
│       ├── adaptive_alpha_multifactor.yaml
│       └── adaptive_alpha_learned.yaml
├── data/
│   ├── __init__.py
│   ├── download.py              # Download MovieLens 1M
│   ├── preprocessing.py         # Data preprocessing
│   ├── dataset.py               # Dataset classes
│   ├── triplet_sampler.py       # BPR triplet sampling
│   └── federated_partition.py   # Partition data by user
├── models/
│   ├── __init__.py
│   ├── bpr_mf.py                # Base BPR-MF model
│   ├── adaptive_alpha.py        # Alpha computation strategies
│   └── adaptive_bpr_mf.py       # Full adaptive model
├── federated/
│   ├── __init__.py
│   ├── client.py                # Flower client implementation
│   ├── server.py                # Flower server + strategy
│   ├── strategy.py              # Custom FedAvg strategy
│   └── simulation.py            # Simulation runner
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # NDCG, HitRate, AUC, MRR
│   ├── evaluator.py             # Evaluation pipeline
│   └── analysis.py              # Results analysis + visualization
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py        # Main experiment runner
│   ├── hyperparameter_search.py # HP tuning
│   └── ablation_study.py        # Ablation experiments
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Logging utilities
│   ├── seed.py                  # Reproducibility
│   ├── io.py                    # Save/load utilities
│   └── visualization.py         # Plotting functions
├── scripts/
│   ├── download_data.sh
│   ├── run_all_experiments.sh
│   └── generate_tables.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_alpha_analysis.ipynb
│   └── 03_results_visualization.ipynb
└── tests/
    ├── test_bpr_mf.py
    ├── test_adaptive_alpha.py
    ├── test_federated.py
    └── test_metrics.py
```

---

## 2. Dependencies (requirements.txt)

```
# Core
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Deep Learning (optional, for learned alpha)
torch>=1.9.0

# Federated Learning
flwr>=1.0.0

# Evaluation & Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
pyyaml>=5.4.0
tqdm>=4.62.0
loguru>=0.5.0
hydra-core>=1.1.0

# Development
pytest>=6.2.0
black>=21.7b0
isort>=5.9.0

# Jupyter (for notebooks)
jupyter>=1.0.0
ipywidgets>=7.6.0
```

---

## 3. Configuration System

### 3.1 Default Configuration (config/default_config.yaml)

```yaml
# ═══════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION FOR ADAPTIVE-α BPR-FEDMF
# ═══════════════════════════════════════════════════════════════

# Experiment metadata
experiment:
  name: "adaptive_alpha_bpr_fedmf"
  seed: 42
  output_dir: "./outputs"
  log_level: "INFO"

# Data configuration
data:
  dataset: "movielens-1m"
  data_path: "./data/ml-1m"
  min_interactions: 5          # Filter users with < 5 interactions
  test_ratio: 0.2              # 20% for testing
  val_ratio: 0.1               # 10% for validation (from train)
  negative_sample_ratio: 4     # 4 negatives per positive for BPR

# Model configuration
model:
  n_factors: 64                # Latent factor dimension
  learning_rate: 0.01
  reg_lambda: 0.01             # L2 regularization
  init_std: 0.1                # Initialization std

# Adaptive Alpha configuration
alpha:
  method: "data_quantity"      # Options: "data_quantity", "multi_factor", "learned", "fixed"
  fixed_value: 0.5             # Used when method="fixed"
  min_alpha: 0.1               # Minimum personalization level
  max_alpha: 0.95              # Maximum personalization level
  
  # For data_quantity method
  quantity_threshold: 50       # Midpoint for sigmoid
  quantity_temperature: 0.1    # Steepness of sigmoid
  
  # For multi_factor method
  weights:
    quantity: 0.4
    diversity: 0.2
    local_quality: 0.25
    coverage: 0.15

# Federated Learning configuration
federated:
  num_rounds: 100              # Total FL rounds
  clients_per_round: 50        # Clients sampled per round
  local_epochs: 5              # Local training epochs
  local_batch_size: 256        # Batch size for local training
  server_learning_rate: 1.0    # Server-side learning rate for aggregation
  
  # Client selection
  client_selection: "random"   # Options: "random", "weighted", "all"
  min_available_clients: 10
  
  # Aggregation
  aggregation: "fedavg"        # Options: "fedavg", "weighted_fedavg"
  
  # Global prototype update
  prototype_momentum: 0.9      # EMA for global user prototype

# Evaluation configuration
evaluation:
  metrics: ["ndcg", "hitrate", "mrr", "auc"]
  k_values: [5, 10, 20]        # Top-K values
  eval_every: 5                # Evaluate every N rounds
  
  # User group analysis
  user_groups:
    sparse: [0, 30]            # 0-29 interactions
    medium: [30, 100]          # 30-99 interactions
    dense: [100, 10000]        # 100+ interactions

# Logging and checkpoints
logging:
  wandb: false                 # Use Weights & Biases
  tensorboard: true
  save_checkpoints: true
  checkpoint_every: 10
```

---

## 4. Data Module Implementation

### 4.1 Data Download (data/download.py)

```python
"""
MovieLens 1M Dataset Download and Extraction
"""
import os
import urllib.request
import zipfile
from pathlib import Path
from loguru import logger


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(data_dir: str = "./data") -> Path:
    """
    Download and extract MovieLens 1M dataset.
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the extracted dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_path / "ml-1m.zip"
    extract_path = data_path / "ml-1m"
    
    # Check if already exists
    if extract_path.exists() and (extract_path / "ratings.dat").exists():
        logger.info(f"Dataset already exists at {extract_path}")
        return extract_path
    
    # Download
    logger.info(f"Downloading MovieLens 1M from {MOVIELENS_1M_URL}")
    urllib.request.urlretrieve(MOVIELENS_1M_URL, zip_path)
    logger.info(f"Downloaded to {zip_path}")
    
    # Extract
    logger.info("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
    # Cleanup
    zip_path.unlink()
    logger.info(f"Dataset ready at {extract_path}")
    
    return extract_path


if __name__ == "__main__":
    download_movielens_1m()
```

### 4.2 Data Preprocessing (data/preprocessing.py)

```python
"""
Data Preprocessing for MovieLens 1M
- Load ratings
- Filter users/items
- Create train/val/test splits
- Encode user/item IDs
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class PreprocessedData:
    """Container for preprocessed data"""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    n_users: int
    n_items: int
    user_id_map: Dict[int, int]      # original -> encoded
    item_id_map: Dict[int, int]      # original -> encoded
    user_id_reverse: Dict[int, int]  # encoded -> original
    item_id_reverse: Dict[int, int]  # encoded -> original
    user_stats: Dict[int, Dict]      # Statistics per user


def load_ratings(data_path: Path) -> pd.DataFrame:
    """Load ratings.dat file"""
    ratings_file = data_path / "ratings.dat"
    
    df = pd.read_csv(
        ratings_file,
        sep='::',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1'
    )
    
    logger.info(f"Loaded {len(df)} ratings from {len(df['user_id'].unique())} users "
                f"and {len(df['item_id'].unique())} items")
    
    return df


def filter_data(df: pd.DataFrame, min_user_interactions: int = 5, 
                min_item_interactions: int = 5) -> pd.DataFrame:
    """Filter users and items with minimum interactions"""
    
    # Iteratively filter until convergence
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        
        # Filter users
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]
    
    logger.info(f"After filtering: {len(df)} ratings, "
                f"{len(df['user_id'].unique())} users, "
                f"{len(df['item_id'].unique())} items")
    
    return df


def encode_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict]:
    """Encode user and item IDs to consecutive integers starting from 0"""
    
    # Create mappings
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_id_map = {old: new for new, old in enumerate(unique_users)}
    item_id_map = {old: new for new, old in enumerate(unique_items)}
    
    user_id_reverse = {new: old for old, new in user_id_map.items()}
    item_id_reverse = {new: old for old, new in item_id_map.items()}
    
    # Apply encoding
    df = df.copy()
    df['user_id'] = df['user_id'].map(user_id_map)
    df['item_id'] = df['item_id'].map(item_id_map)
    
    return df, user_id_map, item_id_map, user_id_reverse, item_id_reverse


def compute_user_stats(df: pd.DataFrame, item_genres: Dict[int, List[str]] = None) -> Dict[int, Dict]:
    """
    Compute statistics for each user (used for alpha computation)
    
    Returns:
        Dict mapping user_id to stats dict with:
        - n_interactions: number of interactions
        - n_unique_items: number of unique items
        - rating_mean: mean rating (if available)
        - rating_std: rating std (if available)
        - genre_entropy: entropy of genre distribution (if genres available)
    """
    user_stats = {}
    
    for user_id, group in df.groupby('user_id'):
        stats = {
            'n_interactions': len(group),
            'n_unique_items': group['item_id'].nunique(),
            'rating_mean': group['rating'].mean() if 'rating' in group.columns else None,
            'rating_std': group['rating'].std() if 'rating' in group.columns else None,
        }
        
        # Compute genre entropy if genres are available
        if item_genres is not None:
            genre_counts = {}
            for item_id in group['item_id']:
                if item_id in item_genres:
                    for genre in item_genres[item_id]:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            if genre_counts:
                total = sum(genre_counts.values())
                probs = np.array(list(genre_counts.values())) / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                stats['genre_entropy'] = entropy
                stats['n_genres'] = len(genre_counts)
            else:
                stats['genre_entropy'] = 0.0
                stats['n_genres'] = 0
        
        user_stats[user_id] = stats
    
    return user_stats


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2, 
                   val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally - for each user, last interactions go to test/val
    
    This is more realistic for recommendation systems than random split.
    """
    train_list = []
    val_list = []
    test_list = []
    
    for user_id, group in df.groupby('user_id'):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        n = len(group)
        
        # Calculate split points
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        
        if n_train < 1:
            # Not enough data - put all in train
            train_list.append(group)
            continue
        
        train_list.append(group.iloc[:n_train])
        val_list.append(group.iloc[n_train:n_train + n_val])
        test_list.append(group.iloc[n_train + n_val:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True)
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def preprocess_movielens(data_path: str, 
                         min_interactions: int = 5,
                         test_ratio: float = 0.2,
                         val_ratio: float = 0.1) -> PreprocessedData:
    """
    Full preprocessing pipeline for MovieLens 1M
    """
    data_path = Path(data_path)
    
    # Load and filter
    df = load_ratings(data_path)
    df = filter_data(df, min_user_interactions=min_interactions, 
                     min_item_interactions=min_interactions)
    
    # Encode IDs
    df, user_id_map, item_id_map, user_id_reverse, item_id_reverse = encode_ids(df)
    
    # Compute user stats BEFORE splitting (using all data for stats)
    user_stats = compute_user_stats(df)
    
    # Split data
    train_df, val_df, test_df = temporal_split(df, test_ratio, val_ratio)
    
    return PreprocessedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_users=len(user_id_map),
        n_items=len(item_id_map),
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        user_id_reverse=user_id_reverse,
        item_id_reverse=item_id_reverse,
        user_stats=user_stats
    )
```

### 4.3 BPR Triplet Sampling (data/triplet_sampler.py)

```python
"""
BPR Triplet Sampling for Implicit Feedback
"""
import numpy as np
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
import pandas as pd


@dataclass
class UserTriplets:
    """Container for a user's BPR training triplets"""
    user_id: int
    triplets: List[Tuple[int, int, int]]  # (user, positive_item, negative_item)
    positive_items: Set[int]
    n_items: int


class BPRTripletSampler:
    """
    Sample BPR triplets for training.
    
    For each positive interaction (u, i), sample negative items j
    that user u has NOT interacted with.
    """
    
    def __init__(self, 
                 n_items: int, 
                 negative_ratio: int = 4,
                 seed: int = 42):
        """
        Args:
            n_items: Total number of items
            negative_ratio: Number of negative samples per positive
            seed: Random seed
        """
        self.n_items = n_items
        self.negative_ratio = negative_ratio
        self.rng = np.random.RandomState(seed)
        self.all_items = set(range(n_items))
    
    def sample_negatives(self, positive_items: Set[int], n_samples: int) -> List[int]:
        """Sample negative items that are not in positive_items"""
        negative_pool = list(self.all_items - positive_items)
        
        if len(negative_pool) == 0:
            return []
        
        # Sample with replacement if needed
        if n_samples > len(negative_pool):
            return list(self.rng.choice(negative_pool, n_samples, replace=True))
        else:
            return list(self.rng.choice(negative_pool, n_samples, replace=False))
    
    def create_triplets_for_user(self, 
                                  user_id: int, 
                                  user_interactions: pd.DataFrame) -> UserTriplets:
        """
        Create BPR triplets for a single user.
        
        Args:
            user_id: The user ID
            user_interactions: DataFrame with user's interactions (must have 'item_id' column)
            
        Returns:
            UserTriplets object
        """
        positive_items = set(user_interactions['item_id'].values)
        n_positives = len(positive_items)
        
        triplets = []
        for pos_item in positive_items:
            # Sample negative items for this positive
            n_neg = self.negative_ratio
            neg_items = self.sample_negatives(positive_items, n_neg)
            
            for neg_item in neg_items:
                triplets.append((user_id, pos_item, neg_item))
        
        return UserTriplets(
            user_id=user_id,
            triplets=triplets,
            positive_items=positive_items,
            n_items=self.n_items
        )
    
    def create_all_triplets(self, train_df: pd.DataFrame) -> Dict[int, UserTriplets]:
        """
        Create triplets for all users.
        
        Args:
            train_df: Training DataFrame with 'user_id' and 'item_id' columns
            
        Returns:
            Dict mapping user_id to UserTriplets
        """
        user_triplets = {}
        
        for user_id, group in train_df.groupby('user_id'):
            user_triplets[user_id] = self.create_triplets_for_user(user_id, group)
        
        return user_triplets
```

### 4.4 Federated Data Partition (data/federated_partition.py)

```python
"""
Partition data for Federated Learning simulation.
Each user = one client.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .triplet_sampler import BPRTripletSampler, UserTriplets


@dataclass 
class ClientData:
    """Data for a single federated client (user)"""
    client_id: int                      # Same as user_id
    user_id: int
    train_triplets: List[Tuple[int, int, int]]
    val_items: List[int]                # Items for validation
    test_items: List[int]               # Items for testing
    positive_items_train: set           # Items in training set
    user_stats: Dict                    # User statistics for alpha computation


class FederatedDataPartition:
    """
    Partition recommendation data for FL simulation.
    
    In recommendation FL:
    - Each user is a client
    - Client's local data = their interactions
    - No data sharing between clients (privacy)
    """
    
    def __init__(self, 
                 preprocessed_data,  # PreprocessedData object
                 negative_ratio: int = 4,
                 seed: int = 42):
        self.data = preprocessed_data
        self.negative_ratio = negative_ratio
        self.seed = seed
        
        self.triplet_sampler = BPRTripletSampler(
            n_items=preprocessed_data.n_items,
            negative_ratio=negative_ratio,
            seed=seed
        )
        
        self.client_data: Dict[int, ClientData] = {}
        self._partition()
    
    def _partition(self):
        """Create client data for each user"""
        
        # Get all user triplets
        user_triplets = self.triplet_sampler.create_all_triplets(self.data.train_df)
        
        # Group val/test by user
        val_by_user = self.data.val_df.groupby('user_id')['item_id'].apply(list).to_dict()
        test_by_user = self.data.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
        
        for user_id in user_triplets.keys():
            ut = user_triplets[user_id]
            
            self.client_data[user_id] = ClientData(
                client_id=user_id,
                user_id=user_id,
                train_triplets=ut.triplets,
                val_items=val_by_user.get(user_id, []),
                test_items=test_by_user.get(user_id, []),
                positive_items_train=ut.positive_items,
                user_stats=self.data.user_stats.get(user_id, {})
            )
    
    def get_client_ids(self) -> List[int]:
        """Get all client IDs"""
        return list(self.client_data.keys())
    
    def get_client_data(self, client_id: int) -> ClientData:
        """Get data for a specific client"""
        return self.client_data[client_id]
    
    def sample_clients(self, n_clients: int, seed: int = None) -> List[int]:
        """Randomly sample clients for a round"""
        rng = np.random.RandomState(seed)
        all_clients = self.get_client_ids()
        n_clients = min(n_clients, len(all_clients))
        return list(rng.choice(all_clients, n_clients, replace=False))
    
    def get_statistics(self) -> Dict:
        """Get statistics about the partition"""
        n_triplets = [len(cd.train_triplets) for cd in self.client_data.values()]
        n_interactions = [cd.user_stats.get('n_interactions', 0) 
                          for cd in self.client_data.values()]
        
        return {
            'n_clients': len(self.client_data),
            'n_items': self.data.n_items,
            'triplets_per_client': {
                'mean': np.mean(n_triplets),
                'std': np.std(n_triplets),
                'min': np.min(n_triplets),
                'max': np.max(n_triplets),
                'median': np.median(n_triplets)
            },
            'interactions_per_client': {
                'mean': np.mean(n_interactions),
                'std': np.std(n_interactions),
                'min': np.min(n_interactions),
                'max': np.max(n_interactions),
                'median': np.median(n_interactions)
            }
        }
```

---

## 5. Model Implementation

### 5.1 Adaptive Alpha Computation (models/adaptive_alpha.py)

```python
"""
Adaptive Alpha Computation Strategies

Alpha (α) controls the personalization level:
- α → 1: Fully personalized (use local user embedding)
- α → 0: Fully global (use global prototype)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class AlphaConfig:
    """Configuration for alpha computation"""
    method: str = "data_quantity"
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    fixed_value: float = 0.5
    
    # For data_quantity method
    quantity_threshold: int = 50
    quantity_temperature: float = 0.1
    
    # For multi_factor method
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'quantity': 0.4,
                'diversity': 0.2,
                'local_quality': 0.25,
                'coverage': 0.15
            }


class AlphaComputer(ABC):
    """Abstract base class for alpha computation"""
    
    def __init__(self, config: AlphaConfig):
        self.config = config
    
    @abstractmethod
    def compute(self, user_stats: Dict) -> float:
        """Compute alpha for a user given their statistics"""
        pass
    
    def _clip(self, alpha: float) -> float:
        """Clip alpha to valid range"""
        return np.clip(alpha, self.config.min_alpha, self.config.max_alpha)


class FixedAlpha(AlphaComputer):
    """Fixed alpha for all users (baseline)"""
    
    def compute(self, user_stats: Dict) -> float:
        return self.config.fixed_value


class DataQuantityAlpha(AlphaComputer):
    """
    Alpha based on number of interactions.
    
    More interactions → Higher alpha (more personalization)
    Fewer interactions → Lower alpha (rely on global)
    
    Uses sigmoid: α = σ((n - threshold) * temperature)
    """
    
    def compute(self, user_stats: Dict) -> float:
        n = user_stats.get('n_interactions', 0)
        threshold = self.config.quantity_threshold
        temp = self.config.quantity_temperature
        
        # Sigmoid function
        alpha_raw = 1 / (1 + np.exp(-(n - threshold) * temp))
        
        return self._clip(alpha_raw)


class MultiFactorAlpha(AlphaComputer):
    """
    Alpha based on multiple user characteristics.
    
    Factors considered:
    1. Data quantity (n_interactions)
    2. Diversity (genre entropy or item coverage)
    3. Local model quality (if available)
    4. Coverage (unique items / total possible)
    """
    
    def __init__(self, config: AlphaConfig, n_items: int = 1000, max_entropy: float = 3.0):
        super().__init__(config)
        self.n_items = n_items
        self.max_entropy = max_entropy
    
    def compute(self, user_stats: Dict) -> float:
        weights = self.config.weights
        
        # Feature 1: Data quantity (sigmoid normalized)
        n = user_stats.get('n_interactions', 0)
        f_quantity = 1 / (1 + np.exp(-(n - self.config.quantity_threshold) * 
                                      self.config.quantity_temperature))
        
        # Feature 2: Diversity (genre entropy)
        entropy = user_stats.get('genre_entropy', self.max_entropy / 2)
        f_diversity = min(entropy / self.max_entropy, 1.0)
        
        # Feature 3: Local quality (AUC if available, else default 0.5)
        f_quality = user_stats.get('local_auc', 0.5)
        
        # Feature 4: Coverage (unique items)
        n_unique = user_stats.get('n_unique_items', n)
        f_coverage = min(n_unique / 100, 1.0)  # Normalize by 100 items
        
        # Weighted sum
        alpha_raw = (
            weights['quantity'] * f_quantity +
            weights['diversity'] * f_diversity +
            weights['local_quality'] * f_quality +
            weights['coverage'] * f_coverage
        )
        
        return self._clip(alpha_raw)


class LearnedAlpha(AlphaComputer):
    """
    Learned alpha using a small neural network.
    
    This requires pre-training or joint training with the main model.
    """
    
    def __init__(self, config: AlphaConfig, model_path: str = None):
        super().__init__(config)
        self.model = None
        self.feature_names = ['n_interactions', 'n_unique_items', 
                              'genre_entropy', 'rating_std']
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load pre-trained alpha model"""
        # TODO: Implement model loading
        pass
    
    def _extract_features(self, user_stats: Dict) -> np.ndarray:
        """Extract feature vector from user stats"""
        features = []
        for name in self.feature_names:
            val = user_stats.get(name, 0.0)
            features.append(val if val is not None else 0.0)
        return np.array(features)
    
    def compute(self, user_stats: Dict) -> float:
        if self.model is None:
            # Fallback to data quantity if model not loaded
            return DataQuantityAlpha(self.config).compute(user_stats)
        
        features = self._extract_features(user_stats)
        alpha_raw = self.model.predict(features.reshape(1, -1))[0]
        return self._clip(alpha_raw)


def create_alpha_computer(config: AlphaConfig, **kwargs) -> AlphaComputer:
    """Factory function to create alpha computer"""
    
    method = config.method.lower()
    
    if method == "fixed":
        return FixedAlpha(config)
    elif method == "data_quantity":
        return DataQuantityAlpha(config)
    elif method == "multi_factor":
        return MultiFactorAlpha(config, **kwargs)
    elif method == "learned":
        return LearnedAlpha(config, **kwargs)
    else:
        raise ValueError(f"Unknown alpha method: {method}")
```

### 5.2 BPR-MF Base Model (models/bpr_mf.py)

```python
"""
Bayesian Personalized Ranking Matrix Factorization (BPR-MF)

Reference: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BPRMFConfig:
    """Configuration for BPR-MF model"""
    n_factors: int = 64
    learning_rate: float = 0.01
    reg_user: float = 0.01
    reg_item: float = 0.01
    reg_bias: float = 0.01
    init_std: float = 0.1
    use_bias: bool = True


class BPRMF:
    """
    BPR Matrix Factorization model.
    
    Prediction: x̂_ui = b_u + b_i + p_u^T * q_i
    
    BPR Optimization:
    - For triplet (u, i, j) where user u prefers i over j:
    - Maximize: ln σ(x̂_uij) where x̂_uij = x̂_ui - x̂_uj
    """
    
    def __init__(self, 
                 n_users: int, 
                 n_items: int, 
                 config: BPRMFConfig = None):
        self.n_users = n_users
        self.n_items = n_items
        self.config = config or BPRMFConfig()
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        cfg = self.config
        
        # User embeddings: P ∈ R^(n_users × n_factors)
        self.user_embeddings = np.random.normal(
            0, cfg.init_std, (self.n_users, cfg.n_factors)
        )
        
        # Item embeddings: Q ∈ R^(n_items × n_factors)
        self.item_embeddings = np.random.normal(
            0, cfg.init_std, (self.n_items, cfg.n_factors)
        )
        
        # Biases
        if cfg.use_bias:
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
        else:
            self.user_bias = None
            self.item_bias = None
    
    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict score for a user-item pair"""
        score = np.dot(self.user_embeddings[user_id], self.item_embeddings[item_id])
        
        if self.config.use_bias:
            score += self.user_bias[user_id] + self.item_bias[item_id]
        
        return score
    
    def predict_preference(self, user_id: int, item_i: int, item_j: int) -> float:
        """Predict x̂_uij = x̂_ui - x̂_uj"""
        return self.predict_score(user_id, item_i) - self.predict_score(user_id, item_j)
    
    def compute_bpr_loss(self, triplets: List[Tuple[int, int, int]]) -> float:
        """Compute BPR loss over triplets"""
        total_loss = 0.0
        
        for (u, i, j) in triplets:
            x_uij = self.predict_preference(u, i, j)
            # BPR loss = -ln(σ(x_uij))
            loss = -np.log(self._sigmoid(x_uij) + 1e-10)
            total_loss += loss
        
        # Add regularization
        reg_loss = (
            self.config.reg_user * np.sum(self.user_embeddings ** 2) +
            self.config.reg_item * np.sum(self.item_embeddings ** 2)
        )
        if self.config.use_bias:
            reg_loss += self.config.reg_bias * (
                np.sum(self.user_bias ** 2) + np.sum(self.item_bias ** 2)
            )
        
        return (total_loss / len(triplets)) + reg_loss
    
    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid"""
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)
    
    def train_step(self, triplet: Tuple[int, int, int]) -> float:
        """
        Single SGD step for one triplet.
        
        Returns:
            loss for this triplet
        """
        u, i, j = triplet
        cfg = self.config
        
        # Forward pass
        x_uij = self.predict_preference(u, i, j)
        
        # Gradient coefficient
        sigmoid_neg = self._sigmoid(-x_uij)  # σ(-x̂_uij)
        
        # Get current parameters
        p_u = self.user_embeddings[u]
        q_i = self.item_embeddings[i]
        q_j = self.item_embeddings[j]
        
        # Compute gradients (negative gradient for maximization)
        # ∂BPR/∂p_u = σ(-x̂_uij) * (q_j - q_i) + λ * p_u
        grad_p_u = sigmoid_neg * (q_j - q_i) + cfg.reg_user * p_u
        
        # ∂BPR/∂q_i = σ(-x̂_uij) * (-p_u) + λ * q_i
        grad_q_i = sigmoid_neg * (-p_u) + cfg.reg_item * q_i
        
        # ∂BPR/∂q_j = σ(-x̂_uij) * (p_u) + λ * q_j
        grad_q_j = sigmoid_neg * p_u + cfg.reg_item * q_j
        
        # Update parameters (gradient descent on negative log-likelihood)
        self.user_embeddings[u] -= cfg.learning_rate * grad_p_u
        self.item_embeddings[i] -= cfg.learning_rate * grad_q_i
        self.item_embeddings[j] -= cfg.learning_rate * grad_q_j
        
        # Update biases if used
        if cfg.use_bias:
            grad_b_i = sigmoid_neg * (-1) + cfg.reg_bias * self.item_bias[i]
            grad_b_j = sigmoid_neg * (1) + cfg.reg_bias * self.item_bias[j]
            
            self.item_bias[i] -= cfg.learning_rate * grad_b_i
            self.item_bias[j] -= cfg.learning_rate * grad_b_j
        
        # Return loss
        return -np.log(self._sigmoid(x_uij) + 1e-10)
    
    def train_epoch(self, triplets: List[Tuple[int, int, int]], 
                    shuffle: bool = True) -> float:
        """Train one epoch over all triplets"""
        if shuffle:
            triplets = np.random.permutation(triplets).tolist()
        
        total_loss = 0.0
        for triplet in triplets:
            loss = self.train_step(tuple(triplet))
            total_loss += loss
        
        return total_loss / len(triplets)
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding for a user"""
        return self.user_embeddings[user_id].copy()
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings"""
        return self.item_embeddings.copy()
    
    def set_item_embeddings(self, embeddings: np.ndarray):
        """Set item embeddings (from server)"""
        self.item_embeddings = embeddings.copy()
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_items: set = None) -> List[int]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n_items: Number of items to recommend
            exclude_items: Items to exclude (e.g., already interacted)
            
        Returns:
            List of item IDs
        """
        # Compute scores for all items
        scores = np.dot(self.user_embeddings[user_id], self.item_embeddings.T)
        
        if self.config.use_bias:
            scores += self.item_bias
        
        # Exclude items
        if exclude_items:
            for item_id in exclude_items:
                scores[item_id] = -np.inf
        
        # Get top-N
        top_items = np.argsort(scores)[::-1][:n_items]
        
        return top_items.tolist()
```

### 5.3 Adaptive BPR-MF Model (models/adaptive_bpr_mf.py)

```python
"""
Adaptive-α BPR Matrix Factorization

Key innovation: Adaptive personalization level per user.
p̃_u = α_u * p_u_local + (1 - α_u) * p̄_global
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .bpr_mf import BPRMF, BPRMFConfig
from .adaptive_alpha import AlphaComputer, AlphaConfig, create_alpha_computer


@dataclass
class AdaptiveBPRMFConfig(BPRMFConfig):
    """Configuration for Adaptive BPR-MF"""
    alpha_config: AlphaConfig = None
    
    def __post_init__(self):
        if self.alpha_config is None:
            self.alpha_config = AlphaConfig()


class AdaptiveBPRMF:
    """
    Adaptive-α BPR Matrix Factorization for Federated Learning.
    
    Each user (client) has:
    - Local user embedding (p_u): Trained only on local data, never shared
    - Access to global item embeddings (Q): Received from server
    - Access to global user prototype (p̄): Aggregated user representation
    - Adaptive alpha (α_u): Personalization level based on user characteristics
    
    Effective embedding: p̃_u = α_u * p_u + (1 - α_u) * p̄
    """
    
    def __init__(self,
                 user_id: int,
                 n_items: int,
                 config: AdaptiveBPRMFConfig = None):
        self.user_id = user_id
        self.n_items = n_items
        self.config = config or AdaptiveBPRMFConfig()
        
        # Initialize alpha computer
        self.alpha_computer = create_alpha_computer(
            self.config.alpha_config,
            n_items=n_items
        )
        
        # Local parameters (NEVER SHARED)
        self.user_embedding = np.random.normal(
            0, self.config.init_std, self.config.n_factors
        )
        self.user_bias = 0.0 if self.config.use_bias else None
        
        # Global parameters (RECEIVED FROM SERVER)
        self.item_embeddings = None  # Will be set by set_global_parameters
        self.item_bias = None
        self.global_prototype = None
        
        # Current alpha value
        self.alpha = None
        self.user_stats = None
    
    def set_global_parameters(self,
                               item_embeddings: np.ndarray,
                               global_prototype: np.ndarray,
                               item_bias: np.ndarray = None):
        """Receive global parameters from server"""
        self.item_embeddings = item_embeddings.copy()
        self.global_prototype = global_prototype.copy()
        if item_bias is not None:
            self.item_bias = item_bias.copy()
    
    def set_user_stats(self, user_stats: Dict):
        """Set user statistics for alpha computation"""
        self.user_stats = user_stats
        self.alpha = self.alpha_computer.compute(user_stats)
    
    def get_effective_embedding(self) -> np.ndarray:
        """
        Compute effective user embedding.
        
        p̃_u = α_u * p_u_local + (1 - α_u) * p̄_global
        """
        if self.global_prototype is None:
            return self.user_embedding
        
        return (self.alpha * self.user_embedding + 
                (1 - self.alpha) * self.global_prototype)
    
    def predict_score(self, item_id: int) -> float:
        """Predict score for an item"""
        p_eff = self.get_effective_embedding()
        score = np.dot(p_eff, self.item_embeddings[item_id])
        
        if self.config.use_bias and self.item_bias is not None:
            score += self.user_bias + self.item_bias[item_id]
        
        return score
    
    def predict_preference(self, item_i: int, item_j: int) -> float:
        """Predict x̂_uij for BPR"""
        return self.predict_score(item_i) - self.predict_score(item_j)
    
    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid"""
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)
    
    def train_step(self, triplet: Tuple[int, int, int]) -> Tuple[Dict, float]:
        """
        Single training step for one triplet.
        
        Returns:
            item_gradients: Dict mapping item_id to gradient
            loss: BPR loss for this triplet
        """
        u, i, j = triplet
        cfg = self.config
        
        # Get effective embedding
        p_eff = self.get_effective_embedding()
        
        # Forward pass
        x_uij = np.dot(p_eff, self.item_embeddings[i] - self.item_embeddings[j])
        if cfg.use_bias and self.item_bias is not None:
            x_uij += self.item_bias[i] - self.item_bias[j]
        
        # Gradient coefficient
        sigmoid_neg = self._sigmoid(-x_uij)
        
        # Get current item embeddings
        q_i = self.item_embeddings[i]
        q_j = self.item_embeddings[j]
        
        # ═══════════════════════════════════════════════════════════
        # GRADIENT FOR LOCAL USER EMBEDDING
        # Note: We only update the LOCAL part, not the global prototype
        # ∂L/∂p_u_local = α_u * σ(-x̂_uij) * (q_j - q_i) + λ * p_u_local
        # ═══════════════════════════════════════════════════════════
        grad_p_local = (self.alpha * sigmoid_neg * (q_j - q_i) + 
                        cfg.reg_user * self.user_embedding)
        
        # Update local user embedding
        self.user_embedding -= cfg.learning_rate * grad_p_local
        
        # ═══════════════════════════════════════════════════════════
        # GRADIENTS FOR ITEM EMBEDDINGS (TO BE SENT TO SERVER)
        # ∂L/∂q_i = σ(-x̂_uij) * (-p̃_u) + λ * q_i
        # ∂L/∂q_j = σ(-x̂_uij) * (p̃_u) + λ * q_j
        # ═══════════════════════════════════════════════════════════
        grad_q_i = sigmoid_neg * (-p_eff) + cfg.reg_item * q_i
        grad_q_j = sigmoid_neg * p_eff + cfg.reg_item * q_j
        
        item_gradients = {
            i: grad_q_i,
            j: grad_q_j
        }
        
        # Update local item embeddings (for better local fit)
        self.item_embeddings[i] -= cfg.learning_rate * grad_q_i
        self.item_embeddings[j] -= cfg.learning_rate * grad_q_j
        
        # Handle biases
        if cfg.use_bias and self.item_bias is not None:
            grad_b_i = sigmoid_neg * (-1) + cfg.reg_bias * self.item_bias[i]
            grad_b_j = sigmoid_neg * (1) + cfg.reg_bias * self.item_bias[j]
            
            self.item_bias[i] -= cfg.learning_rate * grad_b_i
            self.item_bias[j] -= cfg.learning_rate * grad_b_j
            
            item_gradients['bias'] = {i: grad_b_i, j: grad_b_j}
        
        # Compute loss
        loss = -np.log(self._sigmoid(x_uij) + 1e-10)
        
        return item_gradients, loss
    
    def train_epoch(self, triplets: List[Tuple[int, int, int]], 
                    shuffle: bool = True) -> Tuple[np.ndarray, float]:
        """
        Train one epoch on local triplets.
        
        Returns:
            accumulated_item_gradients: Gradients to send to server
            avg_loss: Average loss over epoch
        """
        if shuffle:
            np.random.shuffle(triplets)
        
        # Accumulator for item gradients
        grad_accumulator = np.zeros_like(self.item_embeddings)
        grad_counts = np.zeros(self.n_items)
        
        total_loss = 0.0
        
        for triplet in triplets:
            item_grads, loss = self.train_step(triplet)
            total_loss += loss
            
            # Accumulate gradients
            for item_id, grad in item_grads.items():
                if item_id != 'bias':
                    grad_accumulator[item_id] += grad
                    grad_counts[item_id] += 1
        
        # Average gradients per item
        nonzero_mask = grad_counts > 0
        grad_accumulator[nonzero_mask] /= grad_counts[nonzero_mask, np.newaxis]
        
        avg_loss = total_loss / len(triplets)
        
        return grad_accumulator, avg_loss
    
    def get_user_embedding_for_aggregation(self) -> np.ndarray:
        """Get local user embedding for global prototype update"""
        return self.user_embedding.copy()
    
    def recommend(self, n_items: int = 10, exclude_items: set = None) -> List[int]:
        """Generate recommendations"""
        p_eff = self.get_effective_embedding()
        scores = np.dot(p_eff, self.item_embeddings.T)
        
        if self.config.use_bias and self.item_bias is not None:
            scores += self.item_bias
        
        if exclude_items:
            for item_id in exclude_items:
                scores[item_id] = -np.inf
        
        return np.argsort(scores)[::-1][:n_items].tolist()
    
    def get_current_alpha(self) -> float:
        """Get current alpha value"""
        return self.alpha
    
    def get_local_parameters(self) -> Dict:
        """Get local parameters (for checkpointing)"""
        return {
            'user_embedding': self.user_embedding.copy(),
            'user_bias': self.user_bias,
            'alpha': self.alpha
        }
    
    def set_local_parameters(self, params: Dict):
        """Set local parameters (for loading checkpoint)"""
        self.user_embedding = params['user_embedding'].copy()
        self.user_bias = params.get('user_bias', 0.0)
        self.alpha = params.get('alpha', 0.5)
```

---

## 6. Federated Learning Module

### 6.1 Flower Client (federated/client.py)

```python
"""
Flower Client for Adaptive-α BPR-FedMF
"""
import numpy as np
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import (
    Parameters, 
    FitRes, 
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

from models.adaptive_bpr_mf import AdaptiveBPRMF, AdaptiveBPRMFConfig
from data.federated_partition import ClientData


class AdaptiveBPRClient(fl.client.NumPyClient):
    """
    Flower client for Adaptive-α BPR-FedMF.
    
    Each client represents one user in the recommendation system.
    """
    
    def __init__(self,
                 client_data: ClientData,
                 n_items: int,
                 config: AdaptiveBPRMFConfig,
                 local_epochs: int = 5):
        """
        Args:
            client_data: ClientData object with training triplets and stats
            n_items: Total number of items
            config: Model configuration
            local_epochs: Number of local training epochs per round
        """
        self.client_data = client_data
        self.n_items = n_items
        self.config = config
        self.local_epochs = local_epochs
        
        # Initialize model
        self.model = AdaptiveBPRMF(
            user_id=client_data.user_id,
            n_items=n_items,
            config=config
        )
        
        # Set user stats for alpha computation
        self.model.set_user_stats(client_data.user_stats)
        
        # Training history
        self.round_losses = []
    
    def get_parameters(self, config: Dict = None) -> List[np.ndarray]:
        """
        Return parameters to send to server.
        
        Note: We DON'T send user embeddings (privacy).
        This is called by Flower but we handle parameter exchange in fit().
        """
        return []
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Receive global parameters from server.
        
        Expected parameters:
        - parameters[0]: Item embeddings (n_items × n_factors)
        - parameters[1]: Global user prototype (n_factors,)
        - parameters[2]: Item biases (n_items,) [optional]
        """
        item_embeddings = parameters[0]
        global_prototype = parameters[1]
        item_bias = parameters[2] if len(parameters) > 2 else None
        
        self.model.set_global_parameters(
            item_embeddings=item_embeddings,
            global_prototype=global_prototype,
            item_bias=item_bias
        )
    
    def fit(self, parameters: List[np.ndarray], 
            config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training round.
        
        Args:
            parameters: Global parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (local_parameters, num_examples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Local training
        total_loss = 0.0
        accumulated_grads = np.zeros((self.n_items, self.config.n_factors))
        
        for epoch in range(self.local_epochs):
            grads, loss = self.model.train_epoch(self.client_data.train_triplets)
            accumulated_grads += grads
            total_loss += loss
        
        # Average gradients over epochs
        accumulated_grads /= self.local_epochs
        avg_loss = total_loss / self.local_epochs
        
        # Get user embedding for prototype aggregation
        user_embedding = self.model.get_user_embedding_for_aggregation()
        
        # Prepare return values
        # parameters[0]: Item gradients (n_items × n_factors)
        # parameters[1]: User embedding for prototype update (n_factors,)
        local_params = [accumulated_grads, user_embedding]
        
        num_examples = len(self.client_data.train_triplets)
        
        metrics = {
            "loss": float(avg_loss),
            "alpha": float(self.model.get_current_alpha()),
            "n_interactions": self.client_data.user_stats.get('n_interactions', 0)
        }
        
        self.round_losses.append(avg_loss)
        
        return local_params, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], 
                 config: Dict) -> Tuple[float, int, Dict]:
        """
        Local evaluation.
        
        Computes metrics on validation/test set.
        """
        self.set_parameters(parameters)
        
        # Get recommendations
        exclude = self.client_data.positive_items_train
        recommendations = self.model.recommend(n_items=20, exclude_items=exclude)
        
        # Compute metrics
        test_items = set(self.client_data.test_items)
        
        # Hit Rate @ K
        def hit_rate_at_k(recs, ground_truth, k):
            return 1.0 if any(r in ground_truth for r in recs[:k]) else 0.0
        
        # NDCG @ K
        def ndcg_at_k(recs, ground_truth, k):
            dcg = 0.0
            for i, item in enumerate(recs[:k]):
                if item in ground_truth:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
            return dcg / idcg if idcg > 0 else 0.0
        
        metrics = {
            "hitrate@10": hit_rate_at_k(recommendations, test_items, 10),
            "ndcg@10": ndcg_at_k(recommendations, test_items, 10),
            "hitrate@5": hit_rate_at_k(recommendations, test_items, 5),
            "ndcg@5": ndcg_at_k(recommendations, test_items, 5),
            "alpha": float(self.model.get_current_alpha())
        }
        
        # Loss as evaluation metric
        loss = 0.0  # Could compute BPR loss on val set
        
        return loss, len(self.client_data.test_items), metrics


def create_client_fn(client_data_dict: Dict[int, ClientData],
                     n_items: int,
                     config: AdaptiveBPRMFConfig,
                     local_epochs: int = 5):
    """
    Factory function to create clients for Flower simulation.
    """
    def client_fn(cid: str) -> AdaptiveBPRClient:
        client_id = int(cid)
        return AdaptiveBPRClient(
            client_data=client_data_dict[client_id],
            n_items=n_items,
            config=config,
            local_epochs=local_epochs
        )
    
    return client_fn
```

### 6.2 Custom Federated Strategy (federated/strategy.py)

```python
"""
Custom FedAvg Strategy for Adaptive-α BPR-FedMF
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class AdaptiveFedAvgStrategy(FedAvg):
    """
    Custom FedAvg strategy for Adaptive-α BPR-FedMF.
    
    Key differences from standard FedAvg:
    1. Maintains global item embeddings AND global user prototype
    2. Aggregates item gradients (not full parameters)
    3. Updates global user prototype via weighted averaging
    4. Tracks per-round metrics including alpha distribution
    """
    
    def __init__(self,
                 n_items: int,
                 n_factors: int,
                 server_learning_rate: float = 1.0,
                 prototype_momentum: float = 0.9,
                 init_std: float = 0.1,
                 use_bias: bool = True,
                 **kwargs):
        """
        Args:
            n_items: Number of items
            n_factors: Latent factor dimension
            server_learning_rate: Learning rate for server-side gradient update
            prototype_momentum: EMA momentum for global prototype update
            init_std: Standard deviation for initialization
            use_bias: Whether to use item biases
        """
        super().__init__(**kwargs)
        
        self.n_items = n_items
        self.n_factors = n_factors
        self.server_lr = server_learning_rate
        self.prototype_momentum = prototype_momentum
        self.use_bias = use_bias
        
        # Initialize global parameters
        self.item_embeddings = np.random.normal(0, init_std, (n_items, n_factors))
        self.global_prototype = np.zeros(n_factors)  # Start with zeros
        self.item_bias = np.zeros(n_items) if use_bias else None
        
        # Metrics tracking
        self.round_metrics = []
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Return initial global parameters"""
        params = [self.item_embeddings, self.global_prototype]
        if self.use_bias:
            params.append(self.item_bias)
        return ndarrays_to_parameters(params)
    
    def configure_fit(self, 
                      server_round: int,
                      parameters: Parameters,
                      client_manager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training"""
        
        # Get current parameters
        params = [self.item_embeddings, self.global_prototype]
        if self.use_bias:
            params.append(self.item_bias)
        
        # Create FitIns with current parameters
        fit_ins = fl.common.FitIns(
            ndarrays_to_parameters(params),
            {"round": server_round}
        )
        
        # Sample clients
        sample_size = self.min_fit_clients
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates.
        
        Expected from each client:
        - parameters[0]: Item gradients (n_items × n_factors)
        - parameters[1]: User embedding for prototype update (n_factors,)
        """
        if not results:
            return None, {}
        
        # Compute total examples
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        
        # Aggregate item gradients (weighted by num_examples)
        aggregated_item_grads = np.zeros_like(self.item_embeddings)
        aggregated_user_embeddings = np.zeros_like(self.global_prototype)
        
        # Collect metrics
        round_losses = []
        round_alphas = []
        round_interactions = []
        
        for _, fit_res in results:
            weight = fit_res.num_examples / total_examples
            params = parameters_to_ndarrays(fit_res.parameters)
            
            item_grads = params[0]
            user_embedding = params[1]
            
            aggregated_item_grads += weight * item_grads
            aggregated_user_embeddings += weight * user_embedding
            
            # Collect metrics
            metrics = fit_res.metrics
            round_losses.append(metrics.get("loss", 0.0))
            round_alphas.append(metrics.get("alpha", 0.5))
            round_interactions.append(metrics.get("n_interactions", 0))
        
        # Update global item embeddings using aggregated gradients
        self.item_embeddings -= self.server_lr * aggregated_item_grads
        
        # Update global user prototype using momentum (EMA)
        self.global_prototype = (
            self.prototype_momentum * self.global_prototype +
            (1 - self.prototype_momentum) * aggregated_user_embeddings
        )
        
        # Prepare return parameters
        params = [self.item_embeddings, self.global_prototype]
        if self.use_bias:
            params.append(self.item_bias)
        
        # Compute aggregated metrics
        metrics = {
            "avg_loss": float(np.mean(round_losses)),
            "avg_alpha": float(np.mean(round_alphas)),
            "std_alpha": float(np.std(round_alphas)),
            "min_alpha": float(np.min(round_alphas)),
            "max_alpha": float(np.max(round_alphas)),
            "num_clients": len(results),
            "avg_interactions": float(np.mean(round_interactions))
        }
        
        # Store for analysis
        self.round_metrics.append({
            "round": server_round,
            **metrics
        })
        
        logger.info(
            f"Round {server_round}: "
            f"loss={metrics['avg_loss']:.4f}, "
            f"α={metrics['avg_alpha']:.3f}±{metrics['std_alpha']:.3f}, "
            f"clients={metrics['num_clients']}"
        )
        
        return ndarrays_to_parameters(params), metrics
    
    def configure_evaluate(self, 
                           server_round: int,
                           parameters: Parameters,
                           client_manager) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure evaluation round"""
        if server_round % 5 != 0:  # Evaluate every 5 rounds
            return []
        
        params = [self.item_embeddings, self.global_prototype]
        if self.use_bias:
            params.append(self.item_bias)
        
        eval_ins = fl.common.EvaluateIns(
            ndarrays_to_parameters(params),
            {"round": server_round}
        )
        
        # Sample clients for evaluation
        sample_size = min(50, self.min_available_clients)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min(10, self.min_available_clients)
        )
        
        return [(client, eval_ins) for client in clients]
    
    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}
        
        # Aggregate metrics
        total_examples = sum(res.num_examples for _, res in results)
        
        aggregated_metrics = {}
        metric_keys = ["hitrate@10", "ndcg@10", "hitrate@5", "ndcg@5"]
        
        for key in metric_keys:
            weighted_sum = sum(
                res.num_examples * res.metrics.get(key, 0.0) 
                for _, res in results
            )
            aggregated_metrics[key] = weighted_sum / total_examples
        
        # Alpha statistics
        alphas = [res.metrics.get("alpha", 0.5) for _, res in results]
        aggregated_metrics["avg_alpha"] = float(np.mean(alphas))
        
        logger.info(
            f"Eval Round {server_round}: "
            f"HR@10={aggregated_metrics['hitrate@10']:.4f}, "
            f"NDCG@10={aggregated_metrics['ndcg@10']:.4f}"
        )
        
        return 0.0, aggregated_metrics
    
    def get_round_metrics(self) -> List[Dict]:
        """Get all round metrics for analysis"""
        return self.round_metrics
```

### 6.3 Simulation Runner (federated/simulation.py)

```python
"""
Federated Learning Simulation Runner
"""
import os
from typing import Dict, Optional
from pathlib import Path
from loguru import logger

import flwr as fl
from flwr.simulation import start_simulation

from data.preprocessing import preprocess_movielens
from data.federated_partition import FederatedDataPartition
from models.adaptive_bpr_mf import AdaptiveBPRMFConfig
from models.adaptive_alpha import AlphaConfig
from federated.client import create_client_fn
from federated.strategy import AdaptiveFedAvgStrategy


def run_simulation(
    data_path: str,
    output_dir: str,
    # Data config
    min_interactions: int = 5,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    negative_ratio: int = 4,
    # Model config
    n_factors: int = 64,
    learning_rate: float = 0.01,
    reg_lambda: float = 0.01,
    # Alpha config
    alpha_method: str = "data_quantity",
    alpha_threshold: int = 50,
    alpha_temperature: float = 0.1,
    fixed_alpha: float = 0.5,
    # Federated config
    num_rounds: int = 100,
    clients_per_round: int = 50,
    local_epochs: int = 5,
    server_learning_rate: float = 1.0,
    prototype_momentum: float = 0.9,
    # Other
    seed: int = 42,
    ray_init_args: Dict = None
) -> Dict:
    """
    Run federated learning simulation.
    
    Returns:
        Dict with results and metrics
    """
    # Set seed
    import numpy as np
    np.random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════
    # 1. DATA PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    logger.info("Step 1: Preprocessing data...")
    
    preprocessed = preprocess_movielens(
        data_path=data_path,
        min_interactions=min_interactions,
        test_ratio=test_ratio,
        val_ratio=val_ratio
    )
    
    logger.info(f"Users: {preprocessed.n_users}, Items: {preprocessed.n_items}")
    
    # ═══════════════════════════════════════════════════════════════
    # 2. FEDERATED DATA PARTITION
    # ═══════════════════════════════════════════════════════════════
    logger.info("Step 2: Partitioning data for FL...")
    
    partition = FederatedDataPartition(
        preprocessed_data=preprocessed,
        negative_ratio=negative_ratio,
        seed=seed
    )
    
    stats = partition.get_statistics()
    logger.info(f"Partition stats: {stats}")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. CONFIGURE MODEL AND ALPHA
    # ═══════════════════════════════════════════════════════════════
    logger.info("Step 3: Configuring model...")
    
    alpha_config = AlphaConfig(
        method=alpha_method,
        fixed_value=fixed_alpha,
        quantity_threshold=alpha_threshold,
        quantity_temperature=alpha_temperature
    )
    
    model_config = AdaptiveBPRMFConfig(
        n_factors=n_factors,
        learning_rate=learning_rate,
        reg_user=reg_lambda,
        reg_item=reg_lambda,
        alpha_config=alpha_config
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 4. CREATE STRATEGY
    # ═══════════════════════════════════════════════════════════════
    logger.info("Step 4: Creating federated strategy...")
    
    strategy = AdaptiveFedAvgStrategy(
        n_items=preprocessed.n_items,
        n_factors=n_factors,
        server_learning_rate=server_learning_rate,
        prototype_momentum=prototype_momentum,
        min_fit_clients=clients_per_round,
        min_available_clients=clients_per_round,
        min_evaluate_clients=min(50, clients_per_round)
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 5. CREATE CLIENT FUNCTION
    # ═══════════════════════════════════════════════════════════════
    logger.info("Step 5: Creating client function...")
    
    client_fn = create_client_fn(
        client_data_dict=partition.client_data,
        n_items=preprocessed.n_items,
        config=model_config,
        local_epochs=local_epochs
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 6. RUN SIMULATION
    # ═══════════════════════════════════════════════════════════════
    logger.info(f"Step 6: Starting simulation for {num_rounds} rounds...")
    
    # Ray configuration
    if ray_init_args is None:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False
        }
    
    # Run Flower simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(partition.client_data),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 7. COLLECT RESULTS
    # ═══════════════════════════════════════════════════════════════
    logger.info("Step 7: Collecting results...")
    
    results = {
        "history": history,
        "round_metrics": strategy.get_round_metrics(),
        "config": {
            "n_factors": n_factors,
            "learning_rate": learning_rate,
            "alpha_method": alpha_method,
            "num_rounds": num_rounds,
            "clients_per_round": clients_per_round,
            "local_epochs": local_epochs
        },
        "data_stats": stats,
        "final_item_embeddings": strategy.item_embeddings,
        "final_global_prototype": strategy.global_prototype
    }
    
    # Save results
    import pickle
    with open(output_path / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {output_path}")
    
    return results
```

---

## 7. Evaluation Module

### 7.1 Metrics Implementation (evaluation/metrics.py)

```python
"""
Evaluation Metrics for Recommendation Systems

Metrics implemented:
- Hit Rate @ K
- NDCG @ K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- AUC (Area Under ROC Curve)
- Precision @ K
- Recall @ K
"""
import numpy as np
from typing import List, Set, Dict
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric results"""
    value: float
    k: int = None
    name: str = ""


def hit_rate_at_k(recommendations: List[int], 
                  ground_truth: Set[int], 
                  k: int) -> float:
    """
    Hit Rate @ K: 1 if any relevant item in top-K, 0 otherwise.
    
    Args:
        recommendations: Ranked list of recommended item IDs
        ground_truth: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        1.0 or 0.0
    """
    top_k = recommendations[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def precision_at_k(recommendations: List[int],
                   ground_truth: Set[int],
                   k: int) -> float:
    """
    Precision @ K: Fraction of relevant items in top-K.
    """
    top_k = recommendations[:k]
    relevant = sum(1 for item in top_k if item in ground_truth)
    return relevant / k


def recall_at_k(recommendations: List[int],
                ground_truth: Set[int],
                k: int) -> float:
    """
    Recall @ K: Fraction of relevant items that are in top-K.
    """
    if len(ground_truth) == 0:
        return 0.0
    
    top_k = recommendations[:k]
    relevant = sum(1 for item in top_k if item in ground_truth)
    return relevant / len(ground_truth)


def ndcg_at_k(recommendations: List[int],
              ground_truth: Set[int],
              k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    
    DCG = Σ (2^rel_i - 1) / log2(i + 2) for i in 0..k-1
    For binary relevance: DCG = Σ 1 / log2(i + 2) for relevant items
    
    IDCG = DCG with perfect ranking (all relevant first)
    NDCG = DCG / IDCG
    """
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(recommendations[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i + 2 because log2(1) = 0
    
    # Compute IDCG (perfect ranking)
    n_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr(recommendations: List[int],
        ground_truth: Set[int]) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of first relevant item.
    """
    for i, item in enumerate(recommendations):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def auc(positive_scores: List[float],
        negative_scores: List[float]) -> float:
    """
    AUC: Probability that a random positive is ranked higher than a random negative.
    
    Computed using the Wilcoxon-Mann-Whitney statistic.
    """
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Count pairs where positive > negative
    n_correct = 0
    n_ties = 0
    
    for pos_score in positive_scores:
        for neg_score in negative_scores:
            if pos_score > neg_score:
                n_correct += 1
            elif pos_score == neg_score:
                n_ties += 1
    
    # AUC = (correct + 0.5 * ties) / total_pairs
    return (n_correct + 0.5 * n_ties) / (n_pos * n_neg)


def compute_all_metrics(recommendations: List[int],
                        ground_truth: Set[int],
                        k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute all metrics for a single user.
    
    Returns:
        Dict with all metric values
    """
    results = {}
    
    for k in k_values:
        results[f"hitrate@{k}"] = hit_rate_at_k(recommendations, ground_truth, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommendations, ground_truth, k)
        results[f"precision@{k}"] = precision_at_k(recommendations, ground_truth, k)
        results[f"recall@{k}"] = recall_at_k(recommendations, ground_truth, k)
    
    results["mrr"] = mrr(recommendations, ground_truth)
    
    return results


def aggregate_metrics(user_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across users.
    
    Returns mean of each metric.
    """
    if not user_metrics:
        return {}
    
    aggregated = {}
    metric_keys = user_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in user_metrics]
        aggregated[f"mean_{key}"] = np.mean(values)
        aggregated[f"std_{key}"] = np.std(values)
    
    return aggregated
```

### 7.2 Evaluator Class (evaluation/evaluator.py)

```python
"""
Comprehensive Evaluator for Federated Recommendation
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from .metrics import compute_all_metrics, aggregate_metrics


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    k_values: List[int] = None
    user_groups: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20]
        
        if self.user_groups is None:
            self.user_groups = {
                "sparse": (0, 30),
                "medium": (30, 100),
                "dense": (100, 10000)
            }


class FederatedEvaluator:
    """
    Evaluator for Federated Recommendation System.
    
    Supports:
    - Overall metrics
    - Per-user-group metrics (sparse, medium, dense)
    - Alpha analysis
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results_history = []
    
    def evaluate_user(self,
                      user_id: int,
                      recommendations: List[int],
                      ground_truth: set,
                      user_stats: Dict,
                      alpha: float) -> Dict:
        """Evaluate a single user"""
        metrics = compute_all_metrics(
            recommendations=recommendations,
            ground_truth=ground_truth,
            k_values=self.config.k_values
        )
        
        metrics["user_id"] = user_id
        metrics["alpha"] = alpha
        metrics["n_interactions"] = user_stats.get("n_interactions", 0)
        metrics["n_test_items"] = len(ground_truth)
        
        return metrics
    
    def evaluate_all_users(self,
                           user_recommendations: Dict[int, List[int]],
                           user_ground_truth: Dict[int, set],
                           user_stats: Dict[int, Dict],
                           user_alphas: Dict[int, float]) -> Dict:
        """
        Evaluate all users and compute aggregate metrics.
        
        Returns:
            Dict with:
            - overall: Overall aggregated metrics
            - by_group: Metrics per user group
            - alpha_analysis: Analysis of alpha values
            - per_user: Raw per-user metrics
        """
        per_user_results = []
        
        for user_id in user_recommendations.keys():
            if user_id not in user_ground_truth:
                continue
            
            metrics = self.evaluate_user(
                user_id=user_id,
                recommendations=user_recommendations[user_id],
                ground_truth=user_ground_truth[user_id],
                user_stats=user_stats.get(user_id, {}),
                alpha=user_alphas.get(user_id, 0.5)
            )
            per_user_results.append(metrics)
        
        if not per_user_results:
            logger.warning("No users to evaluate!")
            return {}
        
        # Overall metrics
        overall = aggregate_metrics(per_user_results)
        
        # Group users
        grouped_results = self._group_users(per_user_results)
        
        # Metrics by group
        by_group = {}
        for group_name, group_results in grouped_results.items():
            if group_results:
                by_group[group_name] = aggregate_metrics(group_results)
                by_group[group_name]["n_users"] = len(group_results)
                by_group[group_name]["avg_alpha"] = np.mean(
                    [r["alpha"] for r in group_results]
                )
        
        # Alpha analysis
        alpha_analysis = self._analyze_alpha(per_user_results)
        
        result = {
            "overall": overall,
            "by_group": by_group,
            "alpha_analysis": alpha_analysis,
            "per_user": per_user_results,
            "n_users_evaluated": len(per_user_results)
        }
        
        self.results_history.append(result)
        
        return result
    
    def _group_users(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by user category"""
        grouped = defaultdict(list)
        
        for r in results:
            n = r.get("n_interactions", 0)
            
            for group_name, (low, high) in self.config.user_groups.items():
                if low <= n < high:
                    grouped[group_name].append(r)
                    break
        
        return grouped
    
    def _analyze_alpha(self, results: List[Dict]) -> Dict:
        """Analyze alpha distribution and correlation with metrics"""
        alphas = [r["alpha"] for r in results]
        ndcgs = [r["ndcg@10"] for r in results]
        n_interactions = [r["n_interactions"] for r in results]
        
        analysis = {
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "alpha_median": float(np.median(alphas)),
        }
        
        # Correlation between alpha and metrics
        if len(set(alphas)) > 1:  # Need variance for correlation
            analysis["alpha_ndcg_correlation"] = float(
                np.corrcoef(alphas, ndcgs)[0, 1]
            )
            analysis["alpha_interactions_correlation"] = float(
                np.corrcoef(alphas, n_interactions)[0, 1]
            )
        
        # Alpha distribution by quantile
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            analysis[f"alpha_q{int(q*100)}"] = float(np.quantile(alphas, q))
        
        return analysis
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        # Overall
        overall = results.get("overall", {})
        logger.info("\nOverall Metrics:")
        for k in self.config.k_values:
            hr = overall.get(f"mean_hitrate@{k}", 0)
            ndcg = overall.get(f"mean_ndcg@{k}", 0)
            logger.info(f"  K={k}: HR={hr:.4f}, NDCG={ndcg:.4f}")
        
        # By group
        by_group = results.get("by_group", {})
        logger.info("\nMetrics by User Group:")
        for group_name, metrics in by_group.items():
            n = metrics.get("n_users", 0)
            hr = metrics.get("mean_hitrate@10", 0)
            ndcg = metrics.get("mean_ndcg@10", 0)
            avg_alpha = metrics.get("avg_alpha", 0)
            logger.info(
                f"  {group_name}: n={n}, HR@10={hr:.4f}, "
                f"NDCG@10={ndcg:.4f}, α={avg_alpha:.3f}"
            )
        
        # Alpha analysis
        alpha = results.get("alpha_analysis", {})
        logger.info("\nAlpha Analysis:")
        logger.info(
            f"  Mean: {alpha.get('alpha_mean', 0):.3f} ± "
            f"{alpha.get('alpha_std', 0):.3f}"
        )
        logger.info(
            f"  Range: [{alpha.get('alpha_min', 0):.3f}, "
            f"{alpha.get('alpha_max', 0):.3f}]"
        )
        if "alpha_ndcg_correlation" in alpha:
            logger.info(
                f"  Correlation with NDCG@10: "
                f"{alpha.get('alpha_ndcg_correlation', 0):.3f}"
            )
        
        logger.info("=" * 60)
```

---

## 8. Experiment Runner

### 8.1 Main Experiment Script (experiments/run_experiment.py)

```python
"""
Main Experiment Runner

Usage:
    python experiments/run_experiment.py --config config/experiment_configs/adaptive_alpha_quantity.yaml
"""
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.download import download_movielens_1m
from federated.simulation import run_simulation


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Deep merge two configs, override takes precedence"""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_logging(output_dir: Path, experiment_name: str):
    """Configure logging"""
    log_file = output_dir / f"{experiment_name}.log"
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG")
    
    logger.info(f"Logging to {log_file}")


def run_experiment(config_path: str, override_args: dict = None):
    """
    Run a single experiment with given configuration.
    """
    # Load base config
    base_config_path = Path("config/default_config.yaml")
    if base_config_path.exists():
        base_config = load_config(str(base_config_path))
    else:
        base_config = {}
    
    # Load experiment config
    exp_config = load_config(config_path)
    
    # Merge configs
    config = merge_configs(base_config, exp_config)
    
    # Apply command-line overrides
    if override_args:
        config = merge_configs(config, override_args)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment", {}).get("name", "experiment")
    output_dir = Path(config.get("experiment", {}).get("output_dir", "./outputs"))
    run_dir = output_dir / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(run_dir, exp_name)
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Output directory: {run_dir}")
    
    # Download data if needed
    data_config = config.get("data", {})
    data_path = Path(data_config.get("data_path", "./data/ml-1m"))
    
    if not data_path.exists():
        logger.info("Downloading MovieLens 1M dataset...")
        download_movielens_1m(str(data_path.parent))
    
    # Extract configs for simulation
    model_config = config.get("model", {})
    alpha_config = config.get("alpha", {})
    fed_config = config.get("federated", {})
    
    # Run simulation
    results = run_simulation(
        data_path=str(data_path),
        output_dir=str(run_dir),
        # Data
        min_interactions=data_config.get("min_interactions", 5),
        test_ratio=data_config.get("test_ratio", 0.2),
        val_ratio=data_config.get("val_ratio", 0.1),
        negative_ratio=data_config.get("negative_sample_ratio", 4),
        # Model
        n_factors=model_config.get("n_factors", 64),
        learning_rate=model_config.get("learning_rate", 0.01),
        reg_lambda=model_config.get("reg_lambda", 0.01),
        # Alpha
        alpha_method=alpha_config.get("method", "data_quantity"),
        alpha_threshold=alpha_config.get("quantity_threshold", 50),
        alpha_temperature=alpha_config.get("quantity_temperature", 0.1),
        fixed_alpha=alpha_config.get("fixed_value", 0.5),
        # Federated
        num_rounds=fed_config.get("num_rounds", 100),
        clients_per_round=fed_config.get("clients_per_round", 50),
        local_epochs=fed_config.get("local_epochs", 5),
        server_learning_rate=fed_config.get("server_learning_rate", 1.0),
        prototype_momentum=fed_config.get("prototype_momentum", 0.9),
        # Other
        seed=config.get("experiment", {}).get("seed", 42)
    )
    
    logger.info("Experiment completed!")
    logger.info(f"Results saved to {run_dir}")
    
    return results, run_dir


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive-α BPR-FedMF experiment")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--n_factors",
        type=int,
        default=None,
        help="Override: number of latent factors"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Override: number of federated rounds"
    )
    parser.add_argument(
        "--alpha_method",
        type=str,
        default=None,
        choices=["fixed", "data_quantity", "multi_factor", "learned"],
        help="Override: alpha computation method"
    )
    
    args = parser.parse_args()
    
    # Build override dict from command-line args
    overrides = {}
    if args.n_factors:
        overrides.setdefault("model", {})["n_factors"] = args.n_factors
    if args.num_rounds:
        overrides.setdefault("federated", {})["num_rounds"] = args.num_rounds
    if args.alpha_method:
        overrides.setdefault("alpha", {})["method"] = args.alpha_method
    
    run_experiment(args.config, overrides)


if __name__ == "__main__":
    main()
```

### 8.2 Run All Experiments Script (scripts/run_all_experiments.sh)

```bash
#!/bin/bash
# Run all experiments for the thesis

set -e

echo "=========================================="
echo "Adaptive-α BPR-FedMF Experiments"
echo "=========================================="

# Create output directory
OUTPUT_DIR="./outputs/thesis_experiments_$(date +%Y%m%d)"
mkdir -p $OUTPUT_DIR

# Baseline 1: FedAvg (no personalization)
echo "Running Baseline: FedAvg..."
python experiments/run_experiment.py \
    --config config/experiment_configs/baseline_fedavg.yaml

# Baseline 2: Local Only (full personalization)
echo "Running Baseline: Local Only..."
python experiments/run_experiment.py \
    --config config/experiment_configs/baseline_local_only.yaml

# Baseline 3: Fixed Alpha = 0.5
echo "Running Baseline: Fixed Alpha 0.5..."
python experiments/run_experiment.py \
    --config config/experiment_configs/fixed_alpha_05.yaml

# Baseline 4: Fixed Alpha = 0.8
echo "Running Baseline: Fixed Alpha 0.8..."
python experiments/run_experiment.py \
    --config config/experiment_configs/fixed_alpha_08.yaml

# Proposed Method 1: Adaptive Alpha (Data Quantity)
echo "Running Proposed: Adaptive Alpha (Quantity)..."
python experiments/run_experiment.py \
    --config config/experiment_configs/adaptive_alpha_quantity.yaml

# Proposed Method 2: Adaptive Alpha (Multi-Factor)
echo "Running Proposed: Adaptive Alpha (Multi-Factor)..."
python experiments/run_experiment.py \
    --config config/experiment_configs/adaptive_alpha_multifactor.yaml

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
```

---

## 9. Testing

### 9.1 Test BPR-MF (tests/test_bpr_mf.py)

```python
"""
Unit tests for BPR-MF model
"""
import pytest
import numpy as np

from models.bpr_mf import BPRMF, BPRMFConfig
from models.adaptive_bpr_mf import AdaptiveBPRMF, AdaptiveBPRMFConfig
from models.adaptive_alpha import AlphaConfig, DataQuantityAlpha


class TestBPRMF:
    """Tests for base BPR-MF model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = BPRMF(n_users=100, n_items=1000)
        
        assert model.user_embeddings.shape == (100, 64)
        assert model.item_embeddings.shape == (1000, 64)
        assert model.user_bias.shape == (100,)
        assert model.item_bias.shape == (1000,)
    
    def test_prediction(self):
        """Test score prediction"""
        model = BPRMF(n_users=100, n_items=1000)
        
        score = model.predict_score(user_id=0, item_id=0)
        assert isinstance(score, float)
    
    def test_preference_prediction(self):
        """Test pairwise preference prediction"""
        model = BPRMF(n_users=100, n_items=1000)
        
        # Set embeddings to known values
        model.user_embeddings[0] = np.ones(64)
        model.item_embeddings[0] = np.ones(64) * 2  # Higher
        model.item_embeddings[1] = np.ones(64) * 1  # Lower
        
        pref = model.predict_preference(user_id=0, item_i=0, item_j=1)
        assert pref > 0  # Should prefer item 0
    
    def test_training_step(self):
        """Test single training step"""
        model = BPRMF(n_users=100, n_items=1000)
        
        # Store initial embeddings
        p_u_before = model.user_embeddings[0].copy()
        q_i_before = model.item_embeddings[0].copy()
        
        # Training step
        triplet = (0, 0, 1)  # user 0 prefers item 0 over item 1
        loss = model.train_step(triplet)
        
        # Check that parameters changed
        assert not np.allclose(model.user_embeddings[0], p_u_before)
        assert not np.allclose(model.item_embeddings[0], q_i_before)
        assert loss > 0
    
    def test_recommendation(self):
        """Test recommendation generation"""
        model = BPRMF(n_users=100, n_items=1000)
        
        recs = model.recommend(user_id=0, n_items=10, exclude_items={0, 1})
        
        assert len(recs) == 10
        assert 0 not in recs
        assert 1 not in recs


class TestAdaptiveAlpha:
    """Tests for adaptive alpha computation"""
    
    def test_data_quantity_alpha(self):
        """Test data quantity based alpha"""
        config = AlphaConfig(
            method="data_quantity",
            quantity_threshold=50,
            quantity_temperature=0.1
        )
        computer = DataQuantityAlpha(config)
        
        # Sparse user should have low alpha
        sparse_stats = {"n_interactions": 10}
        alpha_sparse = computer.compute(sparse_stats)
        assert alpha_sparse < 0.3
        
        # Dense user should have high alpha
        dense_stats = {"n_interactions": 200}
        alpha_dense = computer.compute(dense_stats)
        assert alpha_dense > 0.8
        
        # Alpha should be monotonic in n_interactions
        assert alpha_sparse < alpha_dense


class TestAdaptiveBPRMF:
    """Tests for adaptive BPR-MF model"""
    
    def test_effective_embedding(self):
        """Test effective embedding computation"""
        config = AdaptiveBPRMFConfig(
            alpha_config=AlphaConfig(method="fixed", fixed_value=0.5)
        )
        model = AdaptiveBPRMF(user_id=0, n_items=1000, config=config)
        
        # Set up global parameters
        item_embeddings = np.random.randn(1000, 64)
        global_prototype = np.ones(64)
        model.set_global_parameters(item_embeddings, global_prototype)
        model.set_user_stats({"n_interactions": 50})
        
        # Set local embedding
        model.user_embedding = np.zeros(64)
        
        # Effective should be 0.5 * zeros + 0.5 * ones = 0.5
        effective = model.get_effective_embedding()
        assert np.allclose(effective, 0.5 * np.ones(64))
    
    def test_training_preserves_privacy(self):
        """Test that training doesn't expose user embedding"""
        config = AdaptiveBPRMFConfig()
        model = AdaptiveBPRMF(user_id=0, n_items=100, config=config)
        
        # Set global parameters
        model.set_global_parameters(
            np.random.randn(100, 64),
            np.zeros(64)
        )
        model.set_user_stats({"n_interactions": 50})
        
        # Train
        triplets = [(0, i, i+1) for i in range(0, 50, 2)]
        grads, loss = model.train_epoch(triplets)
        
        # Gradients should be for items, not user
        assert grads.shape == (100, 64)
        
        # User embedding should be updated locally
        user_emb = model.get_user_embedding_for_aggregation()
        assert user_emb.shape == (64,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 10. Execution Checklist

### Phase 1: Setup (Day 1)
- [ ] Create project structure
- [ ] Install dependencies
- [ ] Download MovieLens 1M dataset
- [ ] Verify data loading works

### Phase 2: Core Implementation (Days 2-4)
- [ ] Implement `data/preprocessing.py`
- [ ] Implement `data/triplet_sampler.py`
- [ ] Implement `data/federated_partition.py`
- [ ] Implement `models/bpr_mf.py`
- [ ] Implement `models/adaptive_alpha.py`
- [ ] Implement `models/adaptive_bpr_mf.py`
- [ ] Write unit tests for models

### Phase 3: Federated Learning (Days 5-6)
- [ ] Implement `federated/client.py`
- [ ] Implement `federated/strategy.py`
- [ ] Implement `federated/simulation.py`
- [ ] Test FL simulation with small config

### Phase 4: Evaluation (Day 7)
- [ ] Implement `evaluation/metrics.py`
- [ ] Implement `evaluation/evaluator.py`
- [ ] Verify metrics are correct

### Phase 5: Experiments (Days 8-10)
- [ ] Create all experiment configs
- [ ] Run baseline experiments
- [ ] Run proposed method experiments
- [ ] Collect and analyze results

### Phase 6: Analysis (Days 11-12)
- [ ] Create visualization notebooks
- [ ] Generate tables and figures
- [ ] Statistical significance tests
- [ ] Write analysis report

---

## 11. Expected Outputs

After running all experiments, you should have:

1. **Quantitative Results:**
   - Table comparing all methods on NDCG@K, HR@K, MRR, AUC
   - Results broken down by user group (sparse, medium, dense)
   - Statistical significance tests (paired t-test or Wilcoxon)

2. **Alpha Analysis:**
   - Distribution of alpha values across users
   - Correlation between alpha and user characteristics
   - Correlation between alpha and recommendation quality

3. **Visualizations:**
   - Learning curves (loss vs rounds)
   - Performance vs user data size
   - Alpha distribution histograms
   - Improvement over baselines by user group

4. **Ablation Studies:**
   - Impact of alpha computation method
   - Impact of threshold parameter
   - Impact of prototype momentum

---

## 12. Key Research Questions to Answer

1. **RQ1:** Does adaptive α improve overall recommendation quality compared to fixed personalization?

2. **RQ2:** Which user groups benefit most from adaptive personalization?

3. **RQ3:** What is the relationship between learned α and user characteristics?

4. **RQ4:** How does adaptive α affect convergence speed?

5. **RQ5:** Is the computational overhead of adaptive α justified by performance gains?

---

## Notes for Claude Code

1. **Start with data module** - ensure data pipeline works before models
2. **Test incrementally** - run unit tests after each component
3. **Use small configs first** - debug with 10 rounds, 10 clients before full experiments
4. **Log extensively** - use loguru for debugging
5. **Save intermediate results** - checkpoint models and metrics
6. **Memory management** - MovieLens 1M fits in memory, but watch for triplet explosion

Good luck with the implementation!
