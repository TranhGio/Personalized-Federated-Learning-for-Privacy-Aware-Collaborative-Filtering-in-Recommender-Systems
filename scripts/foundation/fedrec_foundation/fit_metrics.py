"""Client-side fit-metrics contract (FND-05 + Codex CR-4).

Every federated module's ``@app.train()`` handler MUST return a dict whose
keys include ``FIT_METRICS_REQUIRED_KEYS`` so the weight-policy abstraction
in ``weight_policy.py`` has the inputs it needs. Modules construct a
``FitMetricsContract``, call ``.to_dict()``, and merge with per-module metrics.

Codex CR-4 flagged that clients currently return only ``num-examples``; this
module is the fixed import surface Phases 2-5 populate so
``compute_aggregation_weight`` has real data to consume.

``from_dict`` uses an explicit ``try/except`` so a missing required field
surfaces as a clear ``ValueError`` (not a cryptic dataclass ``TypeError``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Union

FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")


@dataclass
class FitMetricsContract:
    """Minimum metrics a federated client MUST return from ``@app.train()``.

    Phases 2-5 populate this in each module's client_app.py train handler.
    Weight-policy resolution consumes ``num_positives`` /
    ``num_training_examples`` (see ``weight_policy.compute_aggregation_weight``).

    Attributes
    ----------
    train_loss : float
        Final training loss (after last local epoch).
    num_positives : int
        Count of positive train samples for this client (for
        ``WeightPolicy.NUM_POSITIVES``).
    num_training_examples : int
        Total train sample count including negatives (for
        ``WeightPolicy.NUM_TRAINING_EXAMPLES``).
    round_num : Optional[int]
        Current round number (optional; some modules log it here, some in
        ``FitRes``).
    """

    train_loss: float
    num_positives: int
    num_training_examples: int
    # Per-module extension fields below (optional; modules add their own).
    round_num: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Return a dict suitable for a Flower ``FitRes.metrics`` merge.

        ``None`` values are dropped so downstream aggregators don't see null.

        Returns
        -------
        Dict[str, int | float]
            A dict with all non-None dataclass fields.
        """
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Union[int, float]]) -> "FitMetricsContract":
        """Construct from a dict, ignoring unknown keys (forward-compat).

        Parameters
        ----------
        d : Dict[str, int | float]
            Flat dict containing at least every required dataclass field.
            Unknown keys are ignored so modules can safely add new metrics
            without breaking contract deserialization.

        Returns
        -------
        FitMetricsContract
            A populated instance.

        Raises
        ------
        ValueError
            If the input dict is missing a required dataclass field. The
            underlying dataclass ``TypeError`` is caught and re-raised as a
            clear ``ValueError`` so callers don't see cryptic
            ``__init__ missing N required positional arguments`` messages.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        try:
            return cls(**filtered)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError(
                f"FitMetricsContract.from_dict missing required field: {e}"
            ) from e


def validate_fit_metrics(metrics: Dict[str, Union[int, float]]) -> None:
    """Raise ValueError if metrics dict is missing a required contract key.

    Checks every entry in ``FIT_METRICS_REQUIRED_KEYS`` for presence and
    int/float type. Intended as a cheap server-side guard before calling
    ``compute_aggregation_weight``.

    Parameters
    ----------
    metrics : Dict[str, int | float]
        The client's returned ``FitRes.metrics``.

    Raises
    ------
    ValueError
        If any required key is missing or is not an int/float.
    """
    for key in FIT_METRICS_REQUIRED_KEYS:
        if key not in metrics:
            raise ValueError(
                f"Client metrics missing required key {key!r}. "
                f"Expected at least {list(FIT_METRICS_REQUIRED_KEYS)}; "
                f"got {sorted(metrics.keys())}."
            )
        if not isinstance(metrics[key], (int, float)):
            raise ValueError(
                f"Client metrics[{key!r}] must be int|float, "
                f"got {type(metrics[key]).__name__}."
            )
