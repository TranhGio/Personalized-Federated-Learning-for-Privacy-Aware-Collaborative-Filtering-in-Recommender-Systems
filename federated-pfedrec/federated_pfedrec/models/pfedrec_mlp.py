"""PFedRec MLP Model - Client-local score function with global item embeddings.

Architecture from IJCAI-23 PFedRec paper:
    score(item) = sigmoid(Linear(Embedding(item)))

Key design: NO explicit user embedding. Personalization comes from the
client-local affine_output (Linear) layer. Each user has their own
affine_output weights, while item embeddings are shared globally.

Split Learning Parameter Classification:
    GLOBAL (aggregated via FedAvg):
        - embedding_item.weight: Item latent factors
        - affine_output.bias: User score function bias (D-01:
          IJCAI-23-PFedRec/engine.py:143 deletes only ``affine_output.weight``
          before aggregation, so bias is aggregated server-side; updated from
          prior LOCAL classification to align with reference).

    LOCAL (private, per-user, never sent to server):
        - affine_output.weight: User's personalized score function weight only.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple


class PFedRecMLP(nn.Module):
    """PFedRec MLP: Personalized Federated Recommendation via local score function.

    Parameters
    ----------
    num_items : int
        Total number of items in the catalog.
    latent_dim : int
        Embedding dimensionality (default: 32, paper default).
    """

    # D-01: bias is GLOBAL (engine.py:143 only deletes affine_output.weight).
    _GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')
    # D-01: only affine_output.weight is LOCAL (per-user score function).
    _LOCAL_PARAMS = ('affine_output.weight',)

    def __init__(self, num_items: int, latent_dim: int = 32):
        super().__init__()

        self.num_items = num_items
        self.latent_dim = latent_dim

        # GLOBAL: Item embeddings (aggregated on server)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=latent_dim
        )

        # LOCAL: Score function (stays on client, per-user)
        self.affine_output = nn.Linear(in_features=latent_dim, out_features=1)

        # Sigmoid for implicit feedback prediction [0, 1]
        self.logistic = nn.Sigmoid()

    def forward(self, item_indices: torch.LongTensor) -> torch.Tensor:
        """Forward pass: item indices -> predicted scores.

        Parameters
        ----------
        item_indices : torch.LongTensor
            Item indices, shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Predicted scores in [0, 1], shape (batch_size, 1).
        """
        item_embedding = self.embedding_item(item_indices)
        logits = self.affine_output(item_embedding)
        rating = self.logistic(logits)
        return rating

    def predict(self, item_indices: torch.LongTensor) -> torch.Tensor:
        """Predict scores for items (inference mode).

        Parameters
        ----------
        item_indices : torch.LongTensor
            Item indices.

        Returns
        -------
        torch.Tensor
            Predicted scores, shape (batch_size,).
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(item_indices)
        return scores.view(-1)

    # =========================================================================
    # Split Learning Methods
    # =========================================================================

    def get_global_parameters(self) -> OrderedDict:
        """Get global parameters for server aggregation (D-01).

        Returns
        -------
        OrderedDict
            Contains 'embedding_item.weight' AND 'affine_output.bias' tensors
            on CPU (D-01: bias is GLOBAL and aggregated server-side).
        """
        global_params = OrderedDict()
        full_state = self.state_dict()
        for name in self._GLOBAL_PARAMS:
            if name in full_state:
                global_params[name] = full_state[name].cpu().clone()
        return global_params

    def set_global_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Load global parameters from server (D-01).

        Parameters
        ----------
        global_state_dict : dict
            Dictionary containing 'embedding_item.weight' AND
            'affine_output.bias' (D-01: bias is GLOBAL).
        """
        current_state = self.state_dict()
        for name in self._GLOBAL_PARAMS:
            if name in global_state_dict:
                current_state[name] = global_state_dict[name]
        self.load_state_dict(current_state, strict=True)

    def get_local_parameters(self) -> OrderedDict:
        """Get local parameters for client-side persistence (D-01 / D-20).

        Returns
        -------
        OrderedDict
            Contains 'affine_output.weight' on CPU at native PyTorch shape
            ``(1, latent_dim)`` (D-20). Bias is no longer included — it is
            GLOBAL per D-01.
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
        strict: bool = True,
        run_id: str = "<run_id>",
    ) -> Tuple[List[str], List[str]]:
        """Load local parameters with D-21 strict=True hard-fail semantics.

        Parameters
        ----------
        local_state_dict : Dict[str, torch.Tensor]
            Saved local parameters. MUST contain exactly the keys in
            ``self._LOCAL_PARAMS`` (after D-01: only ``affine_output.weight``).
        strict : bool, optional
            If True (D-21 default), raise ``RuntimeError`` on missing key or
            shape mismatch with per-field delta and a literal
            ``rm -rf .embedding_cache/{run_id}/`` hint. If False (legacy
            back-compat — NOT used by Phase 5 client_app), partial-load and
            report the missing keys via ``missing_keys``.
        run_id : str, optional
            Threaded into the rm -rf hint when strict=True. Defaults to the
            placeholder ``"<run_id>"``; client_app passes the real run_id.

        Returns
        -------
        Tuple[List[str], List[str]]
            ``(loaded_keys, missing_keys)``. Under strict=True ``missing_keys``
            is always empty when this method returns (errors raise instead).

        Raises
        ------
        RuntimeError
            (strict=True) When any LOCAL key is missing from
            ``local_state_dict`` or when its shape does not match the live
            model parameter. The message surfaces the offending key, the
            saved shape, the current shape, and the ``rm -rf`` hint.
        """
        loaded_keys: List[str] = []
        missing_keys: List[str] = []
        current_state = self.state_dict()

        for name in self._LOCAL_PARAMS:
            if name not in local_state_dict:
                if strict:
                    raise RuntimeError(
                        f"D-21 missing local key {name!r} on cache load. "
                        f"Run: rm -rf .embedding_cache/{run_id}/"
                    )
                missing_keys.append(name)
                continue

            saved_tensor = local_state_dict[name]
            current_tensor = current_state[name]
            if saved_tensor.shape != current_tensor.shape:
                if strict:
                    raise RuntimeError(
                        f"D-21 shape mismatch for {name!r}: "
                        f"saved shape {tuple(saved_tensor.shape)} vs current "
                        f"shape {tuple(current_tensor.shape)}. "
                        f"Run: rm -rf .embedding_cache/{run_id}/"
                    )
                missing_keys.append(name)
                continue

            current_state[name] = saved_tensor
            loaded_keys.append(name)

        self.load_state_dict(current_state, strict=True)
        return loaded_keys, missing_keys

    def get_global_parameter_names(self) -> List[str]:
        """Return list of global parameter names."""
        return list(self._GLOBAL_PARAMS)

    def get_local_parameter_names(self) -> List[str]:
        """Return list of local parameter names."""
        return list(self._LOCAL_PARAMS)
