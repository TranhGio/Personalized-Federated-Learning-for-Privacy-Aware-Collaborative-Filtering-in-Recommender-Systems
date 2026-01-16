"""Split Learning Strategies for Federated Personalized Collaborative Filtering.

These custom strategies handle the split architecture where:
- GLOBAL params (item embeddings, item bias, global bias) are aggregated
- LOCAL params (user embeddings, user bias) stay on clients

Supports Adaptive Personalization:
- Clients send user prototypes along with training results
- Server aggregates prototypes into global prototype using EMA
- Global prototype is sent back to clients for embedding blending

The actual aggregation logic is inherited from Flower's FedAvg/FedProx.
The split happens at the client level (clients only send global params).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from logging import WARNING

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx
from flwr.server.strategy.aggregate import aggregate


# Define which parameters are global (aggregated) vs local (private)
GLOBAL_PARAM_KEYS = frozenset([
    'item_embeddings.weight',
    'item_bias.weight',
    'global_bias',
])

LOCAL_PARAM_KEYS = frozenset([
    'user_embeddings.weight',
    'user_bias.weight',
])

# Key used for user prototype in metrics
USER_PROTOTYPE_KEY = 'user_prototype'


class SplitFedAvg(BaseFedAvg):
    """
    FedAvg for Split Learning - only aggregates global parameters.

    The split architecture works as follows:
    1. Server initializes and sends only GLOBAL params to clients
    2. Clients merge global params with their LOCAL params
    3. Clients train on local data
    4. Clients send back only GLOBAL params to server
    5. Server aggregates GLOBAL params using weighted average

    Adaptive Personalization:
    - Aggregates user prototypes from clients into global prototype
    - Uses EMA (Exponential Moving Average) for prototype stability
    - Sends global prototype to clients for embedding blending

    The aggregation logic is unchanged from standard FedAvg.
    The "split" happens in client_app.py which only sends global params.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        prototype_momentum: float = 0.9,
        **kwargs
    ):
        """
        Initialize SplitFedAvg strategy.

        Args:
            fraction_fit: Fraction of clients to use per round
            prototype_momentum: EMA momentum for global prototype update.
                              Higher values = more stable prototype.
                              Formula: p_global = m * p_old + (1-m) * p_new
            **kwargs: Additional arguments passed to base FedAvg
        """
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

        # Adaptive personalization state
        self.prototype_momentum = prototype_momentum
        self._global_prototype: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"SplitFedAvg(fraction_fit={self.fraction_fit}, "
            f"prototype_momentum={self.prototype_momentum})"
        )

    def get_global_prototype(self) -> Optional[np.ndarray]:
        """Get the current global user prototype."""
        return self._global_prototype

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results and user prototypes.

        Extends base FedAvg aggregation to also aggregate user prototypes
        from clients into a global prototype using EMA.

        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failed client results

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        # First, do the standard FedAvg aggregation for model parameters
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Then, aggregate user prototypes from client metrics
        self._aggregate_prototypes(results)

        # Add prototype stats to metrics
        if self._global_prototype is not None:
            metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))

        return aggregated_params, metrics

    def _aggregate_prototypes(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> None:
        """
        Aggregate user prototypes from client fit results.

        Uses weighted average of client prototypes, then applies EMA
        for stable updates.

        Args:
            results: List of (client_proxy, fit_result) tuples
        """
        prototypes_and_weights = []

        for _, fit_res in results:
            # Check if client sent a user prototype
            if fit_res.metrics and USER_PROTOTYPE_KEY in fit_res.metrics:
                # Prototype is serialized as list in metrics
                prototype_list = fit_res.metrics[USER_PROTOTYPE_KEY]
                if isinstance(prototype_list, (list, tuple)):
                    prototype = np.array(prototype_list, dtype=np.float32)
                    weight = fit_res.num_examples
                    prototypes_and_weights.append((prototype, weight))

        if not prototypes_and_weights:
            # No prototypes received, keep existing
            return

        # Weighted average of client prototypes
        total_weight = sum(w for _, w in prototypes_and_weights)
        new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight

        # Apply EMA update
        if self._global_prototype is None:
            # First round: just use the new prototype
            self._global_prototype = new_prototype
        else:
            # EMA: p_global = m * p_old + (1-m) * p_new
            self._global_prototype = (
                self.prototype_momentum * self._global_prototype +
                (1 - self.prototype_momentum) * new_prototype
            )

        log(
            WARNING,
            f"Updated global prototype: norm={np.linalg.norm(self._global_prototype):.4f}, "
            f"from {len(prototypes_and_weights)} clients"
        )


class SplitFedProx(BaseFedProx):
    """
    FedProx for Split Learning - proximal term only on global parameters.

    In split learning with FedProx:
    1. The proximal term ||w - w_global||^2 should only apply to global params
    2. Local params (user embeddings) are personalized and shouldn't be
       regularized toward the server's version

    Adaptive Personalization:
    - Aggregates user prototypes from clients into global prototype
    - Uses EMA (Exponential Moving Average) for prototype stability
    - Sends global prototype to clients for embedding blending

    The proximal term computation happens in client's train_fn (task.py).
    This strategy signals to use split-aware proximal regularization.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        proximal_mu: float = 0.01,
        prototype_momentum: float = 0.9,
        **kwargs
    ):
        """
        Initialize SplitFedProx strategy.

        Args:
            fraction_fit: Fraction of clients to use per round
            proximal_mu: Proximal term coefficient (only applied to global params)
            prototype_momentum: EMA momentum for global prototype update.
                              Higher values = more stable prototype.
            **kwargs: Additional arguments passed to base FedProx
        """
        super().__init__(
            fraction_fit=fraction_fit,
            proximal_mu=proximal_mu,
            **kwargs
        )
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

        # Adaptive personalization state
        self.prototype_momentum = prototype_momentum
        self._global_prototype: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"SplitFedProx(fraction_fit={self.fraction_fit}, "
            f"proximal_mu={self.proximal_mu}, "
            f"prototype_momentum={self.prototype_momentum})"
        )

    def get_global_prototype(self) -> Optional[np.ndarray]:
        """Get the current global user prototype."""
        return self._global_prototype

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results and user prototypes.

        Extends base FedProx aggregation to also aggregate user prototypes
        from clients into a global prototype using EMA.

        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failed client results

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        # First, do the standard FedProx aggregation for model parameters
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Then, aggregate user prototypes from client metrics
        self._aggregate_prototypes(results)

        # Add prototype stats to metrics
        if self._global_prototype is not None:
            metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))

        return aggregated_params, metrics

    def _aggregate_prototypes(
        self,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> None:
        """
        Aggregate user prototypes from client fit results.

        Uses weighted average of client prototypes, then applies EMA
        for stable updates.

        Args:
            results: List of (client_proxy, fit_result) tuples
        """
        prototypes_and_weights = []

        for _, fit_res in results:
            # Check if client sent a user prototype
            if fit_res.metrics and USER_PROTOTYPE_KEY in fit_res.metrics:
                # Prototype is serialized as list in metrics
                prototype_list = fit_res.metrics[USER_PROTOTYPE_KEY]
                if isinstance(prototype_list, (list, tuple)):
                    prototype = np.array(prototype_list, dtype=np.float32)
                    weight = fit_res.num_examples
                    prototypes_and_weights.append((prototype, weight))

        if not prototypes_and_weights:
            # No prototypes received, keep existing
            return

        # Weighted average of client prototypes
        total_weight = sum(w for _, w in prototypes_and_weights)
        new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight

        # Apply EMA update
        if self._global_prototype is None:
            # First round: just use the new prototype
            self._global_prototype = new_prototype
        else:
            # EMA: p_global = m * p_old + (1-m) * p_new
            self._global_prototype = (
                self.prototype_momentum * self._global_prototype +
                (1 - self.prototype_momentum) * new_prototype
            )

        log(
            WARNING,
            f"Updated global prototype: norm={np.linalg.norm(self._global_prototype):.4f}, "
            f"from {len(prototypes_and_weights)} clients"
        )


def extract_global_params(state_dict: dict) -> dict:
    """
    Extract only global parameters from a full model state_dict.

    Args:
        state_dict: Full model state dictionary

    Returns:
        Dictionary containing only global parameters
    """
    return {k: v for k, v in state_dict.items() if k in GLOBAL_PARAM_KEYS}


def extract_local_params(state_dict: dict) -> dict:
    """
    Extract only local parameters from a full model state_dict.

    Args:
        state_dict: Full model state dictionary

    Returns:
        Dictionary containing only local parameters
    """
    return {k: v for k, v in state_dict.items() if k in LOCAL_PARAM_KEYS}
