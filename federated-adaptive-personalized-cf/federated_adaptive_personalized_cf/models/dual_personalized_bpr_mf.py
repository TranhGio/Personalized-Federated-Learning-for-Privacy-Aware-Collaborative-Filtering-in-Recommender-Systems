"""Dual-Level Personalized BPR Matrix Factorization.

Novel architecture combining TWO levels of personalization:

Level 1 - Statistical Adaptive Alpha (Embedding Space):
    p̃_u = α * p_local + (1 - α) * p_global

    Where α is computed from user statistics:
    α = w_q * f_quantity + w_d * f_diversity + w_c * f_coverage + w_s * f_consistency

Level 2 - Client-Specific Neural Scoring (Function Space):
    score_neural = PersonalMLP(p̃_u ⊙ q_i)

    Where PersonalMLP is local to each client (not aggregated in FL)

Final Score Fusion:
    score = FusionGate(score_cf, score_neural)

    Options: "add", "gate" (learnable), "mlp" (both paths)

This dual approach is novel because:
1. α is interpretable and computed from observable user behavior
2. MLP captures non-linear user-item interaction patterns
3. Both personalization mechanisms are complementary

Split Learning Architecture:
    GLOBAL (aggregated via FedAvg/FedProx):
        - item_embeddings.weight
        - item_bias.weight
        - global_bias
        - global_prototype

    LOCAL (private, not aggregated):
        - user_embeddings.weight
        - user_bias.weight
        - personal_mlp.*  (client-specific neural layers)
        - fusion_gate (if using gate fusion)

Reference:
    - PFedRec (IJCAI 2023): Client-specific MLPs
    - APFL: Adaptive personalization (but learned, not statistical)
    - This work: Combines statistical α with neural personalization
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class DualPersonalizedBPRMF(nn.Module):
    """
    Dual-Level Personalized BPR Matrix Factorization.

    Combines:
    - Level 1: Multi-factor adaptive α for embedding interpolation
    - Level 2: Client-specific MLP for non-linear scoring

    This enables both:
    - HOW MUCH to personalize (via α)
    - HOW to score interactions (via PersonalMLP)

    Args:
        num_users: Total number of users
        num_items: Total number of items
        embedding_dim: Latent factor dimensionality
        mlp_hidden_dims: Hidden layer dimensions for PersonalMLP
                        Default [64, 32] creates: input → 64 → 32 → 1
        dropout: Dropout rate for regularization
        use_bias: Whether to use bias terms (recommended)
        fusion_type: How to combine CF and MLP scores
                    "add" - Simple addition (default)
                    "gate" - Learnable gating mechanism
                    "concat" - Concatenate and project
    """

    # Parameter classification for split learning
    # MLP parameters are LOCAL (client-specific, not aggregated)
    _GLOBAL_PARAMS_BASE = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')
    _LOCAL_PARAMS_BASE = ('user_embeddings.weight', 'user_bias.weight')

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_hidden_dims: List[int] = None,
        dropout: float = 0.0,
        use_bias: bool = True,
        fusion_type: str = "add",
    ):
        super().__init__()

        # Default MLP architecture: lightweight but expressive
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [embedding_dim, embedding_dim // 2]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.use_bias = use_bias
        self.fusion_type = fusion_type

        # =====================================================================
        # Embedding Layers (same as BPRMF)
        # =====================================================================

        # User embeddings (LOCAL - not shared in FL)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # Item embeddings (GLOBAL - aggregated via FedAvg/FedProx)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # =====================================================================
        # Personal MLP (LOCAL - client-specific, not aggregated)
        # =====================================================================
        # This is the key novelty: personalized non-linear scoring function
        # Input: element-wise product of user and item embeddings
        # Output: scalar score

        self.personal_mlp = self._build_mlp(
            input_dim=embedding_dim,  # element-wise product keeps same dim
            hidden_dims=mlp_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

        # =====================================================================
        # Fusion Layer (combines CF score and MLP score)
        # =====================================================================

        if fusion_type == "gate":
            # Learnable gate: score = σ(g) * cf_score + (1-σ(g)) * mlp_score
            # Gate is LOCAL (client-specific fusion preference)
            self.fusion_gate = nn.Parameter(torch.zeros(1))
        elif fusion_type == "concat":
            # Concatenate both scores and project
            self.fusion_layer = nn.Linear(2, 1)
        # "add" fusion doesn't need extra parameters

        # =====================================================================
        # Adaptive Personalization Parameters (Level 1)
        # =====================================================================

        self._alpha: float = 1.0  # Default: fully personalized
        self._global_prototype: Optional[torch.Tensor] = None

        # Initialize weights
        self._init_weights()

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build the personal MLP network."""
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection to scalar
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights following best practices."""
        # Xavier initialization for embeddings
        init.xavier_uniform_(self.user_embeddings.weight)
        init.xavier_uniform_(self.item_embeddings.weight)

        # Bias initialization
        if self.use_bias:
            init.normal_(self.user_bias.weight, mean=0.0, std=0.01)
            init.normal_(self.item_bias.weight, mean=0.0, std=0.01)
            init.zeros_(self.global_bias)

        # MLP initialization (Xavier for linear layers)
        for module in self.personal_mlp.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        # Fusion gate initialization (start balanced)
        if self.fusion_type == "gate":
            init.zeros_(self.fusion_gate)  # σ(0) = 0.5, balanced fusion
        elif self.fusion_type == "concat":
            init.xavier_uniform_(self.fusion_layer.weight)
            init.zeros_(self.fusion_layer.bias)

    # =========================================================================
    # Core Forward Pass
    # =========================================================================

    def _compute_score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute dual-personalized score for user-item pairs.

        Score = Fusion(CF_score, MLP_score)

        Where:
        - CF_score = dot(p̃_u, q_i) + biases  (collaborative filtering)
        - MLP_score = PersonalMLP(p̃_u ⊙ q_i)  (neural personalization)

        Args:
            user_ids: User indices, shape (batch_size,)
            item_ids: Item indices, shape (batch_size,) or (batch_size, num_samples)

        Returns:
            scores: Predicted scores
        """
        # Get effective user embeddings (Level 1: α-blended)
        user_emb = self.get_effective_embedding(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Apply dropout
        if self.dropout is not None:
            user_emb = self.dropout(user_emb)
            item_emb = self.dropout(item_emb)

        # Handle shape for multiple samples
        is_multi_sample = item_emb.dim() == 3
        if is_multi_sample:
            # Expand user embedding: (batch, dim) → (batch, num_samples, dim)
            user_emb = user_emb.unsqueeze(1).expand_as(item_emb)

        # =====================================================================
        # Path 1: Collaborative Filtering Score (dot product)
        # =====================================================================
        if is_multi_sample:
            cf_score = torch.sum(user_emb * item_emb, dim=2)  # (batch, num_samples)
        else:
            cf_score = torch.sum(user_emb * item_emb, dim=1)  # (batch,)

        # Add biases to CF score
        if self.use_bias:
            user_b = self.user_bias(user_ids).squeeze(-1)
            item_b = self.item_bias(item_ids).squeeze(-1)

            if is_multi_sample:
                user_b = user_b.unsqueeze(1).expand_as(cf_score)

            cf_score = self.global_bias + user_b + item_b + cf_score

        # =====================================================================
        # Path 2: Neural Personalized Score (MLP on interaction)
        # =====================================================================
        # Element-wise product captures interaction patterns
        interaction = user_emb * item_emb  # (batch, dim) or (batch, num_samples, dim)

        if is_multi_sample:
            # Flatten for MLP, then reshape
            batch_size, num_samples, dim = interaction.shape
            interaction_flat = interaction.view(batch_size * num_samples, dim)
            mlp_score_flat = self.personal_mlp(interaction_flat).squeeze(-1)
            mlp_score = mlp_score_flat.view(batch_size, num_samples)
        else:
            mlp_score = self.personal_mlp(interaction).squeeze(-1)

        # =====================================================================
        # Fusion: Combine CF and MLP scores
        # =====================================================================
        scores = self._fuse_scores(cf_score, mlp_score)

        return scores

    def _fuse_scores(
        self,
        cf_score: torch.Tensor,
        mlp_score: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse collaborative filtering and MLP scores."""

        if self.fusion_type == "add":
            # Simple addition (equal weight)
            return cf_score + mlp_score

        elif self.fusion_type == "gate":
            # Learnable gate: g ∈ [0, 1] controls contribution
            gate = torch.sigmoid(self.fusion_gate)
            return gate * cf_score + (1 - gate) * mlp_score

        elif self.fusion_type == "concat":
            # Concatenate and project
            if cf_score.dim() == 1:
                combined = torch.stack([cf_score, mlp_score], dim=1)
                return self.fusion_layer(combined).squeeze(-1)
            else:
                # Multi-sample case
                combined = torch.stack([cf_score, mlp_score], dim=2)
                batch_size, num_samples, _ = combined.shape
                combined_flat = combined.view(batch_size * num_samples, 2)
                fused_flat = self.fusion_layer(combined_flat).squeeze(-1)
                return fused_flat.view(batch_size, num_samples)

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for BPR training or prediction.

        Args:
            user_ids: User indices, shape (batch_size,)
            pos_item_ids: Positive item indices, shape (batch_size,)
            neg_item_ids: Negative item indices (optional)

        Returns:
            If neg_item_ids is None: pos_scores only
            Else: (pos_scores, neg_scores) for BPR loss
        """
        pos_scores = self._compute_score(user_ids, pos_item_ids)

        if neg_item_ids is None:
            return pos_scores

        if neg_item_ids.dim() == 1:
            neg_scores = self._compute_score(user_ids, neg_item_ids)
        else:
            # Multiple negatives per positive
            batch_size, num_neg = neg_item_ids.shape
            user_ids_expanded = user_ids.unsqueeze(1).expand(batch_size, num_neg)
            neg_scores = self._compute_score(user_ids_expanded.reshape(-1), neg_item_ids.reshape(-1))
            neg_scores = neg_scores.view(batch_size, num_neg)

        return pos_scores, neg_scores

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict scores for user-item pairs (inference mode)."""
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids, item_ids, neg_item_ids=None)
        return scores

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_items: set = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate top-K recommendations for a user."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device

            user_ids = torch.LongTensor([user_id] * self.num_items).to(device)
            all_item_ids = torch.arange(self.num_items, device=device)

            scores = self._compute_score(user_ids, all_item_ids)

            if exclude_items is not None:
                scores[list(exclude_items)] = float('-inf')

            top_scores, top_items = torch.topk(scores, k=min(top_k, len(scores)))

        return top_items.cpu().numpy(), top_scores.cpu().numpy()

    # =========================================================================
    # Adaptive Personalization Methods (Level 1)
    # =========================================================================

    def set_alpha(self, alpha: float) -> None:
        """Set the personalization level (α)."""
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    def get_alpha(self) -> float:
        """Get the current personalization level."""
        return self._alpha

    def set_global_prototype(self, prototype: torch.Tensor) -> None:
        """Set the global user prototype from server."""
        if prototype.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Prototype dim {prototype.shape[-1]} != model dim {self.embedding_dim}"
            )
        device = next(self.parameters()).device
        self._global_prototype = prototype.to(device)

    def get_global_prototype(self) -> Optional[torch.Tensor]:
        """Get the current global prototype."""
        return self._global_prototype

    def clear_global_prototype(self) -> None:
        """Clear the global prototype."""
        self._global_prototype = None

    def get_effective_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Get α-blended user embeddings (Level 1 personalization).

        p̃_u = α * p_local + (1 - α) * p_global
        """
        local_emb = self.user_embeddings(user_ids)

        if self._global_prototype is None or self._alpha == 1.0:
            return local_emb

        if self._alpha == 0.0:
            return self._global_prototype.expand_as(local_emb)

        global_expanded = self._global_prototype.expand_as(local_emb)
        return self._alpha * local_emb + (1 - self._alpha) * global_expanded

    def compute_user_prototype(self) -> torch.Tensor:
        """Compute mean user embedding for global aggregation."""
        return self.user_embeddings.weight.mean(dim=0)

    # =========================================================================
    # Split Learning Methods
    # =========================================================================

    @property
    def _GLOBAL_PARAMS(self) -> tuple:
        """Global parameter names (shared across clients)."""
        if self.use_bias:
            return self._GLOBAL_PARAMS_BASE
        return ('item_embeddings.weight',)

    @property
    def _LOCAL_PARAMS(self) -> tuple:
        """Local parameter names (client-specific, including MLP)."""
        # Base local params
        if self.use_bias:
            base = list(self._LOCAL_PARAMS_BASE)
        else:
            base = ['user_embeddings.weight']

        # Add MLP parameters (all are local)
        mlp_params = [name for name, _ in self.personal_mlp.named_parameters()]
        base.extend([f'personal_mlp.{name}' for name in mlp_params])

        # Add fusion parameters if applicable
        if self.fusion_type == "gate":
            base.append('fusion_gate')
        elif self.fusion_type == "concat":
            base.extend(['fusion_layer.weight', 'fusion_layer.bias'])

        return tuple(base)

    def get_global_parameters(self) -> OrderedDict:
        """Get global parameters for federated aggregation."""
        global_params = OrderedDict()
        full_state = self.state_dict()

        for name in self._GLOBAL_PARAMS:
            if name in full_state:
                global_params[name] = full_state[name].cpu().clone()

        return global_params

    def set_global_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Update global parameters from server aggregation."""
        current_state = self.state_dict()

        for name in self._GLOBAL_PARAMS:
            if name in global_state_dict:
                current_state[name] = global_state_dict[name]

        self.load_state_dict(current_state, strict=True)

    def get_local_parameters(self) -> OrderedDict:
        """Get local parameters (user embeddings + MLP) for persistence."""
        local_params = OrderedDict()
        full_state = self.state_dict()

        for name in self._LOCAL_PARAMS:
            if name in full_state:
                local_params[name] = full_state[name].cpu().clone()

        return local_params

    def set_local_parameters(
        self,
        local_state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Load local parameters from persistence."""
        loaded_keys = []
        missing_keys = []
        current_state = self.state_dict()

        for name in self._LOCAL_PARAMS:
            if name not in local_state_dict:
                missing_keys.append(name)
                continue

            saved_tensor = local_state_dict[name]
            current_tensor = current_state[name]

            if saved_tensor.shape == current_tensor.shape:
                current_state[name] = saved_tensor
                loaded_keys.append(name)
            elif name.startswith('user_') and saved_tensor.shape[0] < current_tensor.shape[0]:
                # New users - partial load
                num_saved = saved_tensor.shape[0]
                current_state[name][:num_saved] = saved_tensor
                loaded_keys.append(f"{name}[:{num_saved}]")
            elif strict:
                raise ValueError(f"Shape mismatch for {name}")
            else:
                # Truncate to fit
                current_state[name] = saved_tensor[:current_tensor.shape[0]]
                loaded_keys.append(f"{name}[:truncated]")

        self.load_state_dict(current_state, strict=True)
        return loaded_keys, missing_keys

    def get_global_parameter_names(self) -> List[str]:
        """Return list of global parameter names."""
        return list(self._GLOBAL_PARAMS)

    def get_local_parameter_names(self) -> List[str]:
        """Return list of local parameter names."""
        return list(self._LOCAL_PARAMS)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_embedding_weights(self) -> Dict[str, torch.Tensor]:
        """Get embedding weights for analysis."""
        return {
            'user_embeddings': self.user_embeddings.weight,
            'item_embeddings': self.item_embeddings.weight,
        }

    def get_fusion_weight(self) -> float:
        """Get the current fusion gate weight (for gate fusion only)."""
        if self.fusion_type == "gate":
            return torch.sigmoid(self.fusion_gate).item()
        return 0.5  # Default for non-gate fusion

    def sample_negatives(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        num_negatives: int = 1,
        user_rated_items: Dict = None,
        sampling_strategy: str = 'uniform',
    ) -> torch.Tensor:
        """Sample negative items for BPR training."""
        batch_size = user_ids.shape[0]
        device = user_ids.device

        if num_negatives == 1:
            neg_items = []
            for user_id, pos_item in zip(user_ids.cpu().numpy(), pos_item_ids.cpu().numpy()):
                rated = user_rated_items.get(int(user_id), set()) if user_rated_items else {int(pos_item)}
                while True:
                    neg_item = np.random.randint(0, self.num_items)
                    if neg_item not in rated:
                        neg_items.append(neg_item)
                        break
            neg_item_ids = torch.LongTensor(neg_items).to(device)
        else:
            neg_items = []
            for user_id, pos_item in zip(user_ids.cpu().numpy(), pos_item_ids.cpu().numpy()):
                rated = user_rated_items.get(int(user_id), set()) if user_rated_items else {int(pos_item)}
                user_negs = []
                while len(user_negs) < num_negatives:
                    neg_item = np.random.randint(0, self.num_items)
                    if neg_item not in rated:
                        user_negs.append(neg_item)
                neg_items.append(user_negs)
            neg_item_ids = torch.LongTensor(neg_items).to(device)

        return neg_item_ids

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category."""
        global_params = sum(
            p.numel() for name, p in self.named_parameters()
            if any(g in name for g in ['item_embeddings', 'item_bias', 'global_bias'])
        )

        local_base = sum(
            p.numel() for name, p in self.named_parameters()
            if any(l in name for l in ['user_embeddings', 'user_bias'])
        )

        mlp_params = sum(p.numel() for p in self.personal_mlp.parameters())

        fusion_params = 0
        if self.fusion_type == "gate":
            fusion_params = 1
        elif self.fusion_type == "concat":
            fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())

        return {
            'global': global_params,
            'local_embeddings': local_base,
            'local_mlp': mlp_params,
            'local_fusion': fusion_params,
            'total_local': local_base + mlp_params + fusion_params,
            'total': global_params + local_base + mlp_params + fusion_params,
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"DualPersonalizedBPRMF(\n"
            f"  num_users={self.num_users}, num_items={self.num_items},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  mlp_hidden_dims={self.mlp_hidden_dims},\n"
            f"  fusion_type='{self.fusion_type}',\n"
            f"  use_bias={self.use_bias},\n"
            f"  params: global={params['global']:,}, local={params['total_local']:,}\n"
            f")"
        )
