"""Tests for fedrec_foundation.evaluator (FND-04)."""
from __future__ import annotations

import pytest

from fedrec_foundation.evaluator import EvalProtocol, get_primary_evaluator


def test_primary_evaluator_all_modes() -> None:
    """FND-04-a: every mode profile declares primary_evaluator == 'sampled_loo_99'."""
    for mode in ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"):
        assert get_primary_evaluator(mode) == "sampled_loo_99"
        assert get_primary_evaluator(mode) == EvalProtocol.SAMPLED_LOO_99.value


def test_unknown_mode_raises() -> None:
    """FND-04-a: unknown mode string raises ValueError with "Unknown mode" message."""
    with pytest.raises(ValueError, match="Unknown mode"):
        get_primary_evaluator("not_a_mode")


def test_allrank_is_namespaced() -> None:
    """FND-04: ALLRANK exists but is not returned by get_primary_evaluator; it's only
    useful as a metric prefix for namespaced secondary metrics (D-12)."""
    assert EvalProtocol.ALLRANK.value == "allrank"
