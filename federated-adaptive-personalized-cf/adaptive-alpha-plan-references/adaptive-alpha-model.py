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