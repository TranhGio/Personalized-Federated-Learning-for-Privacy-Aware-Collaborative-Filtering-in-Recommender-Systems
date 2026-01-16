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