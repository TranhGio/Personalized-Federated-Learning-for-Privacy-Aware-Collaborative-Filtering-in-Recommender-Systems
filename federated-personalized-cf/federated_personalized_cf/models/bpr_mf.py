"""BPR Matrix Factorization - Single-row (per-client) Split Architecture (D-01, D-03, PSN-06).

Phase 3 Plan 01 refactor: the old ghost user table is collapsed to a single nn.Parameter
tensor per client. The client IS one user under the cross-device protocol (1 user = 1
client); carrying a full 6040xD user table was dead weight and a privacy smell. Forward
methods no longer take user_ids — there is only one user row.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple


class BPRMF(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization (single-row local contract).

    Under the cross-device protocol (1 user = 1 client, D-01), the user representation
    is a SINGLE row (``local_user_row``), not a full per-user embedding table. The client
    never holds any other user's parameters.

    Architecture:
        score = global_bias + local_user_bias + item_bias[i] + dot(local_user_row, item_emb[i])
        loss  = BPR(score_positive, score_negative)

    Split Learning Parameter Classification (D-03):
        GLOBAL (aggregated via FedAvg/FedProx, sent each round):
            - item_embeddings.weight:  Item latent factors
            - item_bias.weight:        Item popularity bias (if use_bias=True)
            - global_bias:             Overall bias scalar (if use_bias=True)

        LOCAL (private; never aggregated, cached on client):
            - local_user_row:   Single-user latent factor, shape=(embedding_dim,)
            - local_user_bias:  Single-user rating tendency, shape=(1,) [if use_bias=True]
                                Registered as a persistent buffer when use_bias=False
                                so the get/set_local_parameters contract is stable.

    References:
        - "BPR: Bayesian Personalized Ranking from Implicit Feedback" — Rendle et al., UAI 2009
        - "Revisiting BPR: A Replicability Study" — RecSys 2024
    """

    # Parameter classification for split learning — D-03 single-row contract.
    _GLOBAL_PARAMS_WITH_BIAS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')
    _LOCAL_PARAMS_WITH_BIAS = ('local_user_row', 'local_user_bias')

    _GLOBAL_PARAMS_NO_BIAS = ('item_embeddings.weight',)
    _LOCAL_PARAMS_NO_BIAS = ('local_user_row',)

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        """Initialize BPR Matrix Factorization model.

        Parameters
        ----------
        num_users : int
            Retained as constructor arg for API compatibility but NOT stored as
            ``self.num_users`` attribute — the client holds one user, not a table.
        num_items : int
            Catalog size; item_embeddings remains an nn.Embedding (GLOBAL).
        embedding_dim : int
            Latent factor dimensionality. Typical: 32 / 64 / 128 / 256.
        dropout : float
            Dropout rate applied to user/item vectors (0.0 disables).
        use_bias : bool
            Enables user/item/global bias terms.
        """
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        # D-01 single-user local row (replaces the pre-refactor ghost user table)
        self.local_user_row = nn.Parameter(torch.empty(embedding_dim))

        # Item embeddings (GLOBAL)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms
        if use_bias:
            self.local_user_bias = nn.Parameter(torch.zeros(1))
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            # Register as buffer so get_local_parameters returns a consistent key set
            # when the caller opts into the no-bias configuration.
            self.register_buffer("local_user_bias", torch.zeros(1), persistent=False)
            self.item_bias = None
            self.global_bias = None

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following RecSys 2024 best practices.

        Xavier-uniform initialization of ``local_user_row`` is done through a
        (1, embedding_dim) view so the init.xavier_uniform_ fan-in math matches
        the original nn.Embedding path. Item embeddings keep their original
        Xavier init.
        """
        # Xavier init for the single user row (via 2D view — Xavier needs >=2 dims)
        init.xavier_uniform_(self.local_user_row.view(1, -1))
        init.xavier_uniform_(self.item_embeddings.weight)

        if self.use_bias:
            # Small zero init for the single user bias; item_bias gets small random init.
            init.zeros_(self.local_user_bias)
            init.normal_(self.item_bias.weight, mean=0.0, std=0.01)
            init.zeros_(self.global_bias)

    def _compute_score(self, item_ids):
        """Compute score for the single-client user against the given items.

        Parameters
        ----------
        item_ids : torch.LongTensor
            Item indices. Shape ``(batch_size,)`` or ``(batch_size, num_samples)``.

        Returns
        -------
        torch.Tensor
            Predicted scores with the same leading shape as ``item_ids``.
        """
        item_emb = self.item_embeddings(item_ids)  # (..., embedding_dim)

        # Broadcast the single user row across whatever leading dims item_ids has.
        user_row = self.local_user_row
        if self.dropout is not None:
            # Dropout is applied to both the user row (as a 1D vector broadcast)
            # and item embeddings so the training signal matches the previous impl.
            user_row = self.dropout(user_row)
            item_emb = self.dropout(item_emb)

        # Dot product across the embedding_dim axis (last).
        interaction = torch.sum(item_emb * user_row, dim=-1)

        if self.use_bias:
            item_b = self.item_bias(item_ids).squeeze(-1)
            # local_user_bias is shape (1,) — broadcasts across batch automatically.
            scores = self.global_bias + self.local_user_bias + item_b + interaction
        else:
            scores = interaction

        return scores

    def forward(self, pos_item_ids, neg_item_ids=None):
        """Forward pass for BPR training or prediction.

        The single-row user representation means the signature no longer takes
        ``user_ids`` — the client IS one user.

        Parameters
        ----------
        pos_item_ids : torch.LongTensor
            Positive (observed) item indices, shape ``(batch_size,)``.
        neg_item_ids : torch.LongTensor, optional
            Negative item indices, shape ``(batch_size,)`` or ``(batch_size, num_neg)``.
            If ``None``, only positive scores are returned (prediction mode).

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Positive scores if ``neg_item_ids`` is None, else ``(pos, neg)``.
        """
        pos_scores = self._compute_score(pos_item_ids)
        if neg_item_ids is None:
            return pos_scores
        neg_scores = self._compute_score(neg_item_ids)
        return pos_scores, neg_scores

    def predict(self, item_ids):
        """Predict scores for the client user against the given items (inference mode)."""
        self.eval()
        with torch.no_grad():
            scores = self.forward(item_ids, neg_item_ids=None)
        return scores

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
            ``(top_items, top_scores)`` both on CPU as numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            all_item_ids = torch.arange(self.num_items, device=device)
            scores = self._compute_score(all_item_ids)

            if exclude_items is not None:
                scores[list(exclude_items)] = float('-inf')

            top_scores, top_items = torch.topk(scores, k=min(top_k, len(scores)))

        return top_items.cpu().numpy(), top_scores.cpu().numpy()

    def sample_negatives(
        self,
        pos_item_ids,
        num_negatives: int = 1,
        user_rated_items: Optional[Set[int]] = None,
        sampling_strategy: str = 'uniform',
    ):
        """Sample negative items for the single-client user.

        Because the client IS one user, ``user_rated_items`` is a flat set (not a dict
        keyed by user_id). The drawing loop rejects any item in ``user_rated_items``.

        Parameters
        ----------
        pos_item_ids : torch.LongTensor
            Positive item indices, shape ``(batch_size,)``. Used only for batch size
            + device reference; the per-row positive is automatically excluded if
            ``user_rated_items`` is None.
        num_negatives : int
            Number of negatives per positive.
        user_rated_items : Set[int], optional
            Items the single-client user has rated; excluded from sampling.
        sampling_strategy : str
            'uniform' (default) or 'popularity' (placeholder).

        Returns
        -------
        torch.LongTensor
            Negative item indices of shape ``(batch_size,)`` or ``(batch_size, num_negatives)``.
        """
        batch_size = pos_item_ids.shape[0]
        device = pos_item_ids.device

        if num_negatives == 1:
            neg_items: List[int] = []
            for pos_item in pos_item_ids.cpu().numpy():
                rated = user_rated_items if user_rated_items is not None else {int(pos_item)}
                while True:
                    neg_item = int(np.random.randint(0, self.num_items))
                    if neg_item not in rated:
                        neg_items.append(neg_item)
                        break
            return torch.LongTensor(neg_items).to(device)

        neg_items_matrix: List[List[int]] = []
        for pos_item in pos_item_ids.cpu().numpy():
            rated = user_rated_items if user_rated_items is not None else {int(pos_item)}
            user_negs: List[int] = []
            while len(user_negs) < num_negatives:
                neg_item = int(np.random.randint(0, self.num_items))
                if neg_item not in rated:
                    user_negs.append(neg_item)
            neg_items_matrix.append(user_negs)
        return torch.LongTensor(neg_items_matrix).to(device)

    def get_embedding_weights(self):
        """Return key model tensors for regularization or analysis.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``{'local_user_row': ..., 'item_embeddings': item_embeddings.weight}``.
        """
        return {
            'local_user_row': self.local_user_row,
            'item_embeddings': self.item_embeddings.weight,
        }

    # =========================================================================
    # Split Learning Methods
    # =========================================================================

    @property
    def _GLOBAL_PARAMS(self) -> tuple:
        """Global parameter name tuple, conditional on use_bias."""
        return self._GLOBAL_PARAMS_WITH_BIAS if self.use_bias else self._GLOBAL_PARAMS_NO_BIAS

    @property
    def _LOCAL_PARAMS(self) -> tuple:
        """Local parameter name tuple, conditional on use_bias."""
        return self._LOCAL_PARAMS_WITH_BIAS if self.use_bias else self._LOCAL_PARAMS_NO_BIAS

    def get_global_parameters(self) -> OrderedDict:
        """Return only GLOBAL parameters (for Flower aggregation).

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Keys are ``item_embeddings.weight`` [+ ``item_bias.weight`` + ``global_bias``
            when use_bias]. Tensors are detached clones on CPU.
        """
        global_params = OrderedDict()
        full_state = self.state_dict()
        for name in self._GLOBAL_PARAMS:
            if name in full_state:
                global_params[name] = full_state[name].cpu().clone()
        return global_params

    def set_global_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Update only GLOBAL parameters from the aggregated server tensors.

        Local tensors (``local_user_row``, ``local_user_bias``) are preserved.
        """
        current_state = self.state_dict()
        for name in self._GLOBAL_PARAMS:
            if name in global_state_dict:
                current_state[name] = global_state_dict[name]
        self.load_state_dict(current_state, strict=True)

    def get_local_parameters(self) -> OrderedDict:
        """Return the 2-key LOCAL parameter payload for client-side persistence.

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Exactly the keys listed in ``_LOCAL_PARAMS`` — ``local_user_row``
            (+ ``local_user_bias`` when use_bias). Tensors are detached clones on CPU.
        """
        state = OrderedDict()
        state["local_user_row"] = self.local_user_row.detach().cpu().clone()
        # local_user_bias is either an nn.Parameter or a buffer — both support detach().
        lub = self.local_user_bias
        state["local_user_bias"] = lub.detach().cpu().clone()
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
            Saved single-row local tensors. Keys must include ``local_user_row``
            (+ ``local_user_bias`` when use_bias).
        strict : bool
            Unused on the single-row contract (retained for API parity with the
            ghost-table version); included so callers don't break.

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
            if isinstance(self.local_user_bias, nn.Parameter):
                self.local_user_bias.data.copy_(local_state_dict["local_user_bias"])
            else:
                # Buffer branch (use_bias=False).
                self.local_user_bias.copy_(local_state_dict["local_user_bias"])
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
