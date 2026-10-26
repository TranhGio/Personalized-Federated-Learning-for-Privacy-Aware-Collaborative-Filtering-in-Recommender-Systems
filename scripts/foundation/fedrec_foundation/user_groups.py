"""User-group bucket classifier with FROZEN half-open semantics.

Half-open: sparse = [0, 30), medium = [30, 100), dense = [100, inf).
Boundary value 30 lands in "medium", not "sparse" -- the half-open
decision is recorded in ``split_manifest.json`` as ``bucket_semantics``
so future readers never have to infer (Codex IMP-4).
"""
from __future__ import annotations

USER_GROUP_BOUNDARIES = (30, 100)
BUCKET_SEMANTICS = "half_open"


def classify_user_group(n_interactions: int) -> str:
    """Return "sparse" | "medium" | "dense" using half-open semantics.

    Parameters
    ----------
    n_interactions : int
        Number of training interactions (TRAIN-ONLY per Codex CR-5).

    Returns
    -------
    str
        "sparse" if n < 30, "medium" if 30 <= n < 100, "dense" if n >= 100.
    """
    sparse_hi, medium_hi = USER_GROUP_BOUNDARIES
    if n_interactions < sparse_hi:
        return "sparse"
    if n_interactions < medium_hi:
        return "medium"
    return "dense"
