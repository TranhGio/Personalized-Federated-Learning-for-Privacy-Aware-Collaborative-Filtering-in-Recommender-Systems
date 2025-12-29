"""Split Learning Strategies for Federated Personalized Collaborative Filtering.

These custom strategies handle the split architecture where:
- GLOBAL params (item embeddings, item bias, global bias) are aggregated
- LOCAL params (user embeddings, user bias) stay on clients

The actual aggregation logic is inherited from Flower's FedAvg/FedProx.
The split happens at the client level (clients only send global params).
"""

from flwr.serverapp.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx


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


class SplitFedAvg(BaseFedAvg):
    """
    FedAvg for Split Learning - only aggregates global parameters.

    The split architecture works as follows:
    1. Server initializes and sends only GLOBAL params to clients
    2. Clients merge global params with their LOCAL params
    3. Clients train on local data
    4. Clients send back only GLOBAL params to server
    5. Server aggregates GLOBAL params using weighted average

    The aggregation logic is unchanged from standard FedAvg.
    The "split" happens in client_app.py which only sends global params.
    """

    def __init__(self, fraction_train: float = 1.0, **kwargs):
        """
        Initialize SplitFedAvg strategy.

        Args:
            fraction_train: Fraction of clients to use per round
            **kwargs: Additional arguments passed to base FedAvg
        """
        super().__init__(fraction_train=fraction_train, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

    def __repr__(self) -> str:
        return f"SplitFedAvg(fraction_train={self.fraction_train})"


class SplitFedProx(BaseFedProx):
    """
    FedProx for Split Learning - proximal term only on global parameters.

    In split learning with FedProx:
    1. The proximal term ||w - w_global||^2 should only apply to global params
    2. Local params (user embeddings) are personalized and shouldn't be
       regularized toward the server's version

    The proximal term computation happens in client's train_fn (task.py).
    This strategy signals to use split-aware proximal regularization.
    """

    def __init__(
        self,
        fraction_train: float = 1.0,
        proximal_mu: float = 0.01,
        **kwargs
    ):
        """
        Initialize SplitFedProx strategy.

        Args:
            fraction_train: Fraction of clients to use per round
            proximal_mu: Proximal term coefficient (only applied to global params)
            **kwargs: Additional arguments passed to base FedProx
        """
        super().__init__(
            fraction_train=fraction_train,
            proximal_mu=proximal_mu,
            **kwargs
        )
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

    def __repr__(self) -> str:
        return f"SplitFedProx(fraction_train={self.fraction_train}, proximal_mu={self.proximal_mu})"


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
