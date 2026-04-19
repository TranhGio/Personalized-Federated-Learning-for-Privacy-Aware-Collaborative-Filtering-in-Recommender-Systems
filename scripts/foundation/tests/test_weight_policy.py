"""Tests for fedrec_foundation.weight_policy (implemented in Plan 03)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 03 implements fedrec_foundation.weight_policy")


def test_num_positives() -> None:
    """FND-05-a: num_positives policy weights clients by count of positive interactions."""
    raise NotImplementedError("Plan 03 fills this in")


def test_unknown_policy_raises() -> None:
    """FND-05-b: an unknown policy string raises ValueError with the bad value echoed."""
    raise NotImplementedError("Plan 03 fills this in")


def test_fit_metrics_contract() -> None:
    """FND-05-c + CR-4: client FitRes.metrics carries the documented policy key."""
    raise NotImplementedError("Plan 03 fills this in")


def test_from_dict_missing_required_raises() -> None:
    """CR-4: from_dict(...) raises on missing required key rather than silently defaulting."""
    raise NotImplementedError("Plan 03 fills this in")
