"""Tests for fedrec_foundation.evaluator (implemented in Plan 03)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 03 implements fedrec_foundation.evaluator")


def test_primary_evaluator_all_modes() -> None:
    """FND-04-a: every mode profile declares primary_evaluator == 'sampled_loo_99'."""
    raise NotImplementedError("Plan 03 fills this in")
