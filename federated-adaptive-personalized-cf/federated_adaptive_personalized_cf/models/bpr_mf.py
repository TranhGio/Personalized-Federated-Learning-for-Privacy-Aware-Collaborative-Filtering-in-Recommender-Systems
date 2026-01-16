"""BPR Matrix Factorization - State-of-the-art baseline (RecSys 2024) with Split Architecture.

Supports Adaptive Personalization (α) for federated learning:
    - α → 1: Fully personalized (use local user embedding)
    - α → 0: Fully global (use global prototype)

Effective embedding: p̃_u = α_u * p_u_local + (1 - α_u) * p̄_global
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class BPRMF(nn.Module):
    """
    Bayesian Personalized Ranking Matrix Factorization.

    State-of-the-art baseline based on RecSys 2024 research:
    - Proper implementation critical (50% performance variance)
    - Optimizes for ranking (not rating prediction)

    Reference:
        "BPR: Bayesian Personalized Ranking from Implicit Feedback"
        Rendle et al., UAI 2009

        "Revisiting BPR: A Replicability Study" RecSys 2024

    Architecture:
        score = global_bias + user_bias + item_bias + dot(user_emb, item_emb)
        loss = BPR(score_positive, score_negative)

    Adaptive Personalization:
        With adaptive alpha, the effective user embedding becomes:
        p̃_u = α * p_u_local + (1 - α) * p̄_global

        Where:
        - α: Personalization level (0 to 1), computed from user statistics
        - p_u_local: Local user embedding (learned from user's data)
        - p̄_global: Global user prototype (aggregated from all clients)

        Sparse users (low α) benefit from the global prototype.
        Dense users (high α) use their personalized embeddings.

    Split Learning Parameter Classification:
        GLOBAL (aggregated via FedAvg/FedProx):
            - item_embeddings.weight: Item latent factors
            - item_bias.weight: Item popularity bias (if use_bias=True)
            - global_bias: Overall bias (if use_bias=True)
            - global_prototype: Aggregated user representation (for adaptive α)

        LOCAL (private, not aggregated):
            - user_embeddings.weight: User latent factors (personalized)
            - user_bias.weight: User rating tendencies (if use_bias=True)
    """

    # Parameter classification for split learning (with bias)
    _GLOBAL_PARAMS_WITH_BIAS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')
    _LOCAL_PARAMS_WITH_BIAS = ('user_embeddings.weight', 'user_bias.weight')

    # Parameter classification for split learning (without bias)
    _GLOBAL_PARAMS_NO_BIAS = ('item_embeddings.weight',)
    _LOCAL_PARAMS_NO_BIAS = ('user_embeddings.weight',)

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        """
        Initialize BPR Matrix Factorization model.

        Args:
            num_users: Total number of users
            num_items: Total number of items
            embedding_dim: Latent factor dimensionality
                          Typical values: 32, 64, 128, 256
                          Higher = more capacity, but more overfitting risk
            dropout: Dropout rate (0.0 = no dropout)
            use_bias: Whether to use bias terms
                     Biases help but add parameters
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        # User embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms (optional but recommended)
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

        # Adaptive personalization parameters
        # These are set externally via set_alpha() and set_global_prototype()
        self._alpha: float = 1.0  # Default: fully personalized (no global influence)
        self._global_prototype: Optional[torch.Tensor] = None  # Received from server

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following RecSys 2024 best practices.

        Critical for BPR performance:
            - Proper initialization prevents early saturation
            - Xavier/Glorot works well for embeddings
            - Small random values for biases
        """
        # Xavier initialization for embeddings
        init.xavier_uniform_(self.user_embeddings.weight)
        init.xavier_uniform_(self.item_embeddings.weight)

        # Bias initialization
        if self.use_bias:
            init.normal_(self.user_bias.weight, mean=0.0, std=0.01)
            init.normal_(self.item_bias.weight, mean=0.0, std=0.01)
            init.zeros_(self.global_bias)

    def _compute_score(self, user_ids, item_ids):
        """
        Compute score for user-item pairs.

        This is the core prediction function used for both
        positive and negative samples in BPR.

        Uses adaptive personalization when global prototype is set:
            p̃_u = α * p_u_local + (1 - α) * p̄_global

        Args:
            user_ids: User indices, shape (batch_size,) or (batch_size, num_samples)
            item_ids: Item indices, shape (batch_size,) or (batch_size, num_samples)

        Returns:
            scores: Predicted scores
        """
        # Get effective user embeddings (blended with global prototype if available)
        user_emb = self.get_effective_embedding(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Apply dropout
        if self.dropout is not None:
            user_emb = self.dropout(user_emb)
            item_emb = self.dropout(item_emb)

        # Dot product
        if user_emb.dim() == 2:
            # Standard case: (batch_size, embedding_dim)
            interaction = torch.sum(user_emb * item_emb, dim=1)
        else:
            # Multiple samples: (batch_size, num_samples, embedding_dim)
            interaction = torch.sum(user_emb * item_emb, dim=2)

        # Add biases
        if self.use_bias:
            user_b = self.user_bias(user_ids).squeeze(-1)
            item_b = self.item_bias(item_ids).squeeze(-1)
            scores = self.global_bias + user_b + item_b + interaction
        else:
            scores = interaction

        return scores

    def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
        """
        Forward pass for BPR training or prediction.

        Args:
            user_ids: User indices, shape (batch_size,)
            pos_item_ids: Positive (observed) item indices, shape (batch_size,)
            neg_item_ids: Negative (unobserved) item indices, shape (batch_size,) or (batch_size, num_neg)
                         If None, only returns positive scores (for prediction)

        Returns:
            If neg_item_ids is None:
                pos_scores: Scores for positive items only
            Else:
                (pos_scores, neg_scores): Tuple of positive and negative scores for BPR loss
        """
        # Compute positive scores
        pos_scores = self._compute_score(user_ids, pos_item_ids)

        # If no negative samples, return only positive scores (prediction mode)
        if neg_item_ids is None:
            return pos_scores

        # Compute negative scores for BPR training
        if neg_item_ids.dim() == 1:
            # Single negative per positive
            neg_scores = self._compute_score(user_ids, neg_item_ids)
        else:
            # Multiple negatives per positive: (batch_size, num_neg)
            # Expand user_ids to match
            batch_size, num_neg = neg_item_ids.shape
            user_ids_expanded = user_ids.unsqueeze(1).expand(batch_size, num_neg)
            neg_scores = self._compute_score(user_ids_expanded, neg_item_ids)

        return pos_scores, neg_scores

    def predict(self, user_ids, item_ids):
        """
        Predict scores for user-item pairs (inference mode).

        Args:
            user_ids: User indices
            item_ids: Item indices

        Returns:
            Predicted scores
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids, item_ids, neg_item_ids=None)
        return scores

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """
        Generate top-K recommendations for a user.

        This is where BPR shines - ranking items by predicted score.

        Args:
            user_id: User index (single user)
            top_k: Number of recommendations
            exclude_items: Items to exclude (already rated)

        Returns:
            top_items: Top-K item indices
            top_scores: Corresponding scores
        """
        self.eval()
        with torch.no_grad():
            # Get device from model parameters
            device = next(self.parameters()).device

            # Expand user_id for all items
            user_ids = torch.LongTensor([user_id] * self.num_items).to(device)
            all_item_ids = torch.arange(self.num_items, device=device)

            # Get scores for all items
            scores = self._compute_score(user_ids, all_item_ids)

            # Exclude items
            if exclude_items is not None:
                scores[list(exclude_items)] = float('-inf')

            # Get top-K
            top_scores, top_items = torch.topk(scores, k=min(top_k, len(scores)))

        return top_items.cpu().numpy(), top_scores.cpu().numpy()

    def sample_negatives(
        self,
        user_ids,
        pos_item_ids,
        num_negatives=1,
        user_rated_items=None,
        sampling_strategy='uniform',
    ):
        """
        Sample negative items for BPR training.

        CRITICAL for BPR performance (RecSys 2024 finding):
            - Sampling strategy significantly affects results
            - Uniform sampling is simple and effective
            - Popularity-based sampling can help but adds complexity

        Args:
            user_ids: User indices, shape (batch_size,)
            pos_item_ids: Positive item indices, shape (batch_size,)
            num_negatives: Number of negative samples per positive
            user_rated_items: Dict[user_id -> Set[item_ids]] of rated items to exclude
            sampling_strategy: 'uniform' or 'popularity'

        Returns:
            neg_item_ids: Negative item indices, shape (batch_size, num_negatives) or (batch_size,)
        """
        device = user_ids.device

        if num_negatives == 1:
            # Single negative per positive (most common)
            neg_items = []

            for user_id, pos_item in zip(user_ids.cpu().numpy(), pos_item_ids.cpu().numpy()):
                # Get items rated by this user
                rated = user_rated_items.get(int(user_id), set()) if user_rated_items else {int(pos_item)}

                # Sample until we get an unrated item
                while True:
                    if sampling_strategy == 'uniform':
                        neg_item = np.random.randint(0, self.num_items)
                    else:
                        # Placeholder for popularity-based sampling
                        neg_item = np.random.randint(0, self.num_items)

                    if neg_item not in rated:
                        neg_items.append(neg_item)
                        break

            neg_item_ids = torch.LongTensor(neg_items).to(device)

        else:
            # Multiple negatives per positive
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

    def get_embedding_weights(self):
        """
        Get embedding weights for regularization or analysis.

        Returns:
            Dictionary of embedding weights
        """
        return {
            'user_embeddings': self.user_embeddings.weight,
            'item_embeddings': self.item_embeddings.weight,
        }

    # =========================================================================
    # Adaptive Personalization Methods
    # =========================================================================

    def set_alpha(self, alpha: float) -> None:
        """
        Set the personalization level for this model.

        Args:
            alpha: Personalization level in [0, 1].
                   - α → 1: Fully personalized (use local embedding)
                   - α → 0: Fully global (use global prototype)
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    def get_alpha(self) -> float:
        """Get the current personalization level."""
        return self._alpha

    def set_global_prototype(self, prototype: torch.Tensor) -> None:
        """
        Set the global user prototype received from the server.

        Args:
            prototype: Global user prototype tensor of shape (embedding_dim,)
                      This is the aggregated user representation from all clients.
        """
        if prototype.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Prototype embedding dim {prototype.shape[-1]} doesn't match "
                f"model embedding dim {self.embedding_dim}"
            )
        # Store on the same device as model parameters
        device = next(self.parameters()).device
        self._global_prototype = prototype.to(device)

    def get_global_prototype(self) -> Optional[torch.Tensor]:
        """Get the current global user prototype."""
        return self._global_prototype

    def clear_global_prototype(self) -> None:
        """Clear the global prototype (revert to fully local mode)."""
        self._global_prototype = None

    def get_effective_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the effective user embedding using adaptive personalization.

        Blends local user embeddings with the global prototype:
            p̃_u = α * p_u_local + (1 - α) * p̄_global

        Args:
            user_ids: User indices, shape (batch_size,) or (batch_size, num_samples)

        Returns:
            Effective user embeddings with the same shape as local embeddings
        """
        # Get local user embeddings
        local_emb = self.user_embeddings(user_ids)

        # If no global prototype or fully personalized, return local only
        if self._global_prototype is None or self._alpha == 1.0:
            return local_emb

        # If fully global, return only global prototype (broadcasted)
        if self._alpha == 0.0:
            # Expand global prototype to match local embedding shape
            return self._global_prototype.expand_as(local_emb)

        # Blend: α * local + (1 - α) * global
        global_expanded = self._global_prototype.expand_as(local_emb)
        return self._alpha * local_emb + (1 - self._alpha) * global_expanded

    def compute_user_prototype(self) -> torch.Tensor:
        """
        Compute the average user embedding for this client.

        This is sent to the server for global prototype aggregation.

        Returns:
            Mean of all user embeddings, shape (embedding_dim,)
        """
        return self.user_embeddings.weight.mean(dim=0)

    # =========================================================================
    # Split Learning Methods
    # =========================================================================

    @property
    def _GLOBAL_PARAMS(self) -> tuple:
        """Get global parameter names based on use_bias setting."""
        return self._GLOBAL_PARAMS_WITH_BIAS if self.use_bias else self._GLOBAL_PARAMS_NO_BIAS

    @property
    def _LOCAL_PARAMS(self) -> tuple:
        """Get local parameter names based on use_bias setting."""
        return self._LOCAL_PARAMS_WITH_BIAS if self.use_bias else self._LOCAL_PARAMS_NO_BIAS

    def get_global_parameters(self) -> OrderedDict:
        """
        Get only global parameters for federated aggregation.

        Returns:
            OrderedDict with keys: item_embeddings.weight, [item_bias.weight, global_bias]
            Values are detached tensor copies on CPU.

        Note:
            - Returns OrderedDict to maintain consistent key ordering for ArrayRecord
            - Tensors are moved to CPU for serialization
            - If use_bias=False, only returns item_embeddings.weight
        """
        global_params = OrderedDict()
        full_state = self.state_dict()

        for name in self._GLOBAL_PARAMS:
            if name in full_state:
                global_params[name] = full_state[name].cpu().clone()

        return global_params

    def set_global_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update only global parameters from aggregated server weights.

        Args:
            global_state_dict: Dictionary containing item_embeddings.weight,
                              [item_bias.weight, global_bias]

        Note:
            Local parameters (user embeddings) are preserved.
        """
        current_state = self.state_dict()

        for name in self._GLOBAL_PARAMS:
            if name in global_state_dict:
                current_state[name] = global_state_dict[name]

        self.load_state_dict(current_state, strict=True)

    def get_local_parameters(self) -> OrderedDict:
        """
        Get local (user) parameters for client-side persistence.

        Returns:
            OrderedDict with keys: user_embeddings.weight, [user_bias.weight]
            Values are detached tensor copies on CPU.

        Note:
            Used to save user embeddings between federated rounds.
        """
        local_params = OrderedDict()
        full_state = self.state_dict()

        for name in self._LOCAL_PARAMS:
            if name in full_state:
                local_params[name] = full_state[name].cpu().clone()

        return local_params

    def set_local_parameters(
        self,
        local_state_dict: Dict[str, torch.Tensor],
        strict: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Load local (user) parameters from persistence.

        Args:
            local_state_dict: Saved local parameters
            strict: If True, raise error on shape mismatch.
                   If False, partially load what fits.

        Returns:
            Tuple of (loaded_keys, missing_keys)

        Edge case handling:
            - If saved embeddings have fewer users than model, new users get
              initialized with Xavier uniform (kept from model init)
            - If saved embeddings have more users, extras are ignored
        """
        loaded_keys = []
        missing_keys = []
        current_state = self.state_dict()

        for name in self._LOCAL_PARAMS:
            if name not in local_state_dict:
                missing_keys.append(name)
                continue

            saved_tensor = local_state_dict[name]
            current_tensor = current_state[name]

            # Check if shapes match
            if saved_tensor.shape == current_tensor.shape:
                # Perfect match - load directly
                current_state[name] = saved_tensor
                loaded_keys.append(name)
            elif saved_tensor.shape[0] < current_tensor.shape[0]:
                # New users in this round - partial load
                num_saved = saved_tensor.shape[0]
                current_state[name][:num_saved] = saved_tensor
                loaded_keys.append(f"{name}[:{num_saved}]")
            elif strict:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"saved {saved_tensor.shape} vs current {current_tensor.shape}"
                )
            else:
                # Saved has more users than model expects (unusual)
                # Truncate to fit
                current_state[name] = saved_tensor[:current_tensor.shape[0]]
                loaded_keys.append(f"{name}[:truncated]")

        self.load_state_dict(current_state, strict=True)
        return loaded_keys, missing_keys

    def get_global_parameter_names(self) -> List[str]:
        """Return list of global parameter names in consistent order."""
        return list(self._GLOBAL_PARAMS)

    def get_local_parameter_names(self) -> List[str]:
        """Return list of local parameter names in consistent order."""
        return list(self._LOCAL_PARAMS)
