"""Aggregation weight policy abstraction (FND-05).

Each federated round, the server must decide how to weight per-client model
updates when averaging. Different protocols (FedAvg variants, paper-compat
modes) use different conventions:

- ``uniform``: every client counts the same (FedAvg-classic with equal weights).
- ``num_positives``: weight by count of positive train samples (the
  baseline / personalized / adaptive default per ``CONTEXT.md`` D-Discretion).
- ``num_training_examples``: weight by total train sample count (positives + negatives).

This module provides a single resolver so Phase 2-5 strategies don't re-invent
the wheel. The expected metric keys come from ``fit_metrics.FitMetricsContract``
(CR-4) which every client's ``@app.train()`` handler must populate.

Per-module wiring happens in Phase 2-5 (each module's strategy.py / server_app.py
reads ``weight-policy`` from ``context.run_config`` and calls
``compute_aggregation_weight(fit_res.metrics, policy)``).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Union


class WeightPolicy(str, Enum):
    """Aggregation weight policy identifier (FND-05).

    Attributes
    ----------
    UNIFORM : str
        Every client contributes weight 1.0 regardless of dataset size.
    NUM_POSITIVES : str
        Weight by the client's count of positive training interactions.
        Default for baseline / personalized / adaptive modules per D-Discretion.
    NUM_TRAINING_EXAMPLES : str
        Weight by the client's total training sample count (positives + negatives).
    """

    UNIFORM = "uniform"
    NUM_POSITIVES = "num_positives"
    NUM_TRAINING_EXAMPLES = "num_training_examples"


def compute_aggregation_weight(
    client_metrics: Dict[str, Union[int, float]],
    policy: str,
) -> float:
    """Compute this client's aggregation weight from its returned metrics.

    Expected metric keys (produced by each client's ``@app.train()`` handler
    via ``FitMetricsContract.to_dict()``):

    - ``num_positives`` (int): count of positive train samples (for
      ``NUM_POSITIVES``).
    - ``num_training_examples`` (int): total train sample count (for
      ``NUM_TRAINING_EXAMPLES``).

    ``UNIFORM`` returns ``1.0`` unconditionally.

    Parameters
    ----------
    client_metrics : Dict[str, int | float]
        The client's ``FitRes.metrics`` dict. For all non-UNIFORM policies,
        the expected key(s) must be present.
    policy : str
        One of the ``WeightPolicy`` values:
        ``"uniform"`` / ``"num_positives"`` / ``"num_training_examples"``.

    Returns
    -------
    float
        Aggregation weight for this client.

    Raises
    ------
    ValueError
        If the policy is unknown, or if a required metric key is missing.
    """
    try:
        p = WeightPolicy(policy)
    except ValueError as e:
        raise ValueError(f"Unknown weight policy: {policy!r}") from e

    if p is WeightPolicy.UNIFORM:
        return 1.0
    if p is WeightPolicy.NUM_POSITIVES:
        if "num_positives" not in client_metrics:
            raise ValueError(
                "weight-policy=num_positives requires 'num_positives' metric; "
                f"got keys={sorted(client_metrics.keys())}."
            )
        return float(client_metrics["num_positives"])
    if p is WeightPolicy.NUM_TRAINING_EXAMPLES:
        if "num_training_examples" not in client_metrics:
            raise ValueError(
                "weight-policy=num_training_examples requires "
                "'num_training_examples' metric; "
                f"got keys={sorted(client_metrics.keys())}."
            )
        return float(client_metrics["num_training_examples"])
    # Defensive: unreachable because WeightPolicy(policy) above catches typos.
    raise ValueError(f"Unknown weight policy: {policy!r}")
