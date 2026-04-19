"""Basic Matrix Factorization with MSE loss — Single-row Split Architecture (D-01, D-03, PSN-06).

Phase 3 Plan 01 refactor: BasicMF mirrors BPRMF's collapse from a full per-user embedding
table to a single ``local_user_row`` nn.Parameter + ``local_user_bias`` nn.Parameter. The
client holds one user's row; forward no longer accepts ``user_ids``.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple


class BasicMF(nn.Module):
    """Basic Matrix Factorization (single-row per-client contract).

    Architecture:
        prediction = global_bias + local_user_bias + item_bias[i] + dot(local_user_row, item_emb[i])

    Split Learning Parameter Classification (D-03):
        GLOBAL (aggregated via FedAvg/FedProx):
            - item_embeddings.weight
            - item_bias.weight
            - global_bias

        LOCAL (private, cached client-side):
            - local_user_row   (shape: (embedding_dim,))
            - local_user_bias  (shape: (1,))
    """

    # Parameter classification for split learning
    _GLOBAL_PARAMS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')
    _LOCAL_PARAMS = ('local_user_row', 'local_user_bias')

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ):
        """Initialize Basic Matrix Factorization model.

        Parameters
        ----------
        num_users : int
            Retained for API compatibility but NOT stored as ``self.num_users`` —
            a client holds one user under D-01 (single-row contract).
        num_items : int
            Catalog size.
        embedding_dim : int
            Latent factor dimensionality.
        dropout : float
            Dropout rate (0.0 disables).
        """
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # D-01 single-user local row (replaces the pre-refactor ghost user table)
        self.local_user_row = nn.Parameter(torch.empty(embedding_dim))
        self.local_user_bias = nn.Parameter(torch.zeros(1))

        # Item embeddings (GLOBAL)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)

        # Global bias (overall rating mean, GLOBAL)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Weight init
        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform init on embeddings; zero init on biases."""
        init.xavier_uniform_(self.local_user_row.view(1, -1))
        init.xavier_uniform_(self.item_embeddings.weight)
        init.zeros_(self.local_user_bias)
        init.normal_(self.item_bias.weight, mean=0.0, std=0.01)
        init.zeros_(self.global_bias)

    def forward(self, item_ids):
        """Forward pass: predict ratings for the client user against given items.

        Parameters
        ----------
        item_ids : torch.LongTensor
            Item indices, shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Predicted ratings, shape ``(batch_size,)``.
        """
        item_emb = self.item_embeddings(item_ids)  # (batch_size, embedding_dim)
        user_row = self.local_user_row            # (embedding_dim,)

        if self.dropout is not None:
            user_row = self.dropout(user_row)
            item_emb = self.dropout(item_emb)

        item_b = self.item_bias(item_ids).squeeze(-1)  # (batch_size,)
        interaction = torch.sum(item_emb * user_row, dim=-1)  # (batch_size,)
        predictions = self.global_bias + self.local_user_bias + item_b + interaction
        return predictions

    def predict(self, item_ids):
        """Predict ratings (inference mode). Clamped to MovieLens rating range [1, 5]."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(item_ids)
            predictions = torch.clamp(predictions, min=1.0, max=5.0)
        return predictions

    def recommend(self, top_k: int = 10, exclude_items: Optional[Set[int]] = None):
        """Generate top-K recommendations for the client user.

        Parameters
        ----------
        top_k : int
            Number of recommendations.
        exclude_items : Set[int], optional
            Items to exclude (already rated by the client user).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(top_items, top_scores)``, both CPU numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            all_item_ids = torch.arange(self.num_items, device=device)
            item_embs = self.item_embeddings(all_item_ids)  # (num_items, embedding_dim)
            item_bs = self.item_bias(all_item_ids).squeeze(-1)  # (num_items,)

            scores = (
                self.global_bias
                + self.local_user_bias
                + item_bs
                + torch.matmul(item_embs, self.local_user_row)
            )

            if exclude_items is not None:
                scores[list(exclude_items)] = float('-inf')

            top_scores, top_items = torch.topk(scores, k=min(top_k, len(scores)))

        return top_items.cpu().numpy(), top_scores.cpu().numpy()

    def get_embedding_weights(self):
        """Return key model tensors for regularization or analysis."""
        return {
            'local_user_row': self.local_user_row,
            'item_embeddings': self.item_embeddings.weight,
        }

    # =========================================================================
    # Split Learning Methods
    # =========================================================================

    def get_global_parameters(self) -> OrderedDict:
        """Return only GLOBAL parameters for federated aggregation."""
        global_params = OrderedDict()
        full_state = self.state_dict()
        for name in self._GLOBAL_PARAMS:
            if name in full_state:
                global_params[name] = full_state[name].cpu().clone()
        return global_params

    def set_global_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Update only GLOBAL parameters from the aggregated server tensors."""
        current_state = self.state_dict()
        for name in self._GLOBAL_PARAMS:
            if name in global_state_dict:
                current_state[name] = global_state_dict[name]
        self.load_state_dict(current_state, strict=True)

    def get_local_parameters(self) -> OrderedDict:
        """Return the 2-key LOCAL parameter payload (local_user_row, local_user_bias)."""
        state = OrderedDict()
        state["local_user_row"] = self.local_user_row.detach().cpu().clone()
        state["local_user_bias"] = self.local_user_bias.detach().cpu().clone()
        return state

    def set_local_parameters(
        self,
        local_state_dict: Dict[str, torch.Tensor],
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Load the 2-key LOCAL parameter payload back into the model.

        Parameters
        ----------
        local_state_dict : Dict[str, torch.Tensor]
            Must include ``local_user_row`` and ``local_user_bias``.
        strict : bool
            Retained for API parity with the ghost-table version; unused.

        Returns
        -------
        Tuple[List[str], List[str]]
            ``(loaded_keys, missing_keys)``.
        """
        loaded: List[str] = []
        missing: List[str] = []

        if "local_user_row" in local_state_dict:
            self.local_user_row.data.copy_(local_state_dict["local_user_row"])
            loaded.append("local_user_row")
        else:
            missing.append("local_user_row")

        if "local_user_bias" in local_state_dict:
            self.local_user_bias.data.copy_(local_state_dict["local_user_bias"])
            loaded.append("local_user_bias")
        else:
            missing.append("local_user_bias")

        return loaded, missing

    def get_global_parameter_names(self) -> List[str]:
        """Return list of global parameter names in consistent order."""
        return list(self._GLOBAL_PARAMS)

    def get_local_parameter_names(self) -> List[str]:
        """Return list of local parameter names in consistent order."""
        return list(self._LOCAL_PARAMS)
