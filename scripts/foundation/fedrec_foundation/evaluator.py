"""Primary evaluator selector (FND-04).

Phase 1 provides a **config-level constant** + **resolver function** for the
primary evaluator protocol. Phase 1 does NOT replace evaluator implementations
(`evaluate_ranking_sampled` already exists in every module's task.py). The per-
module surgical wiring happens in Phases 2-5.

D-12: Primary evaluator is locked to ``sampled_loo_99`` for every mode. The
``allrank`` variant exists only as a namespace prefix for secondary metrics
(e.g., ``allrank_ndcg@10`` is never mixed into thesis tables).

Downstream code imports ``EvalProtocol.SAMPLED_LOO_99.value`` as the single
authoritative primary-evaluator string — we deliberately avoid scattering
``"sampled_loo_99"`` literals across modules.
"""
from __future__ import annotations

from enum import Enum


class EvalProtocol(str, Enum):
    """Evaluation protocol identifier (D-12).

    Attributes
    ----------
    SAMPLED_LOO_99 : str
        Leave-one-out with 99 negative samples (NCF protocol). Primary metric
        namespace across all four federated modules.
    ALLRANK : str
        All-items ranking, namespaced as a secondary metric. Its value is only
        useful as a metric prefix; ``get_primary_evaluator`` never returns it.
    """

    SAMPLED_LOO_99 = "sampled_loo_99"
    ALLRANK = "allrank"


# Whitelist of recognized mode strings. Keeping this explicit (rather than
# silently accepting any mode) prevents typos in run config from silently
# defaulting to the primary evaluator — matching the CONVENTIONS.md rule that
# factory functions raise ValueError on unknown enum-like strings.
_KNOWN_MODES = frozenset(
    {
        "benchmark_cross_device",
        "paper_compat_pfedrec",
        "cross_silo_legacy",
    }
)


def get_primary_evaluator(mode: str) -> str:
    """Return the primary evaluator string for a given mode (FND-04).

    For all three recognized modes the primary evaluator is
    ``sampled_loo_99``; this function exists so that future modes (e.g., a
    paper-compat mode with a different protocol) have a clean extension point.

    Parameters
    ----------
    mode : str
        One of ``benchmark_cross_device``, ``paper_compat_pfedrec``,
        ``cross_silo_legacy``.

    Returns
    -------
    str
        ``EvalProtocol.SAMPLED_LOO_99.value`` (``"sampled_loo_99"``).

    Raises
    ------
    ValueError
        If ``mode`` is not in the recognized whitelist.
    """
    if mode not in _KNOWN_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of {sorted(_KNOWN_MODES)}."
        )
    return EvalProtocol.SAMPLED_LOO_99.value
