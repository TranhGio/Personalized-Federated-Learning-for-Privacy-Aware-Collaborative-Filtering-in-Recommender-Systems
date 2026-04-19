"""Tests for fedrec_foundation.exclusion (implemented in Plan 02)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 02 implements fedrec_foundation.exclusion")


def test_includes_test_item() -> None:
    """FND-03-a: exclusion_for(u) includes the LOO test item (fix train-neg leak)."""
    raise NotImplementedError("Plan 02 fills this in")


def test_safe_load() -> None:
    """FND-03-b + D-05: loader uses np.load(allow_pickle=False); no pickle."""
    raise NotImplementedError("Plan 02 fills this in")


def test_indptr_layout() -> None:
    """FND-03-c + IMP-3: indices/indptr CSR-style layout matches documented schema."""
    raise NotImplementedError("Plan 02 fills this in")


def test_module_level_exclusion_for() -> None:
    """CR-3: module-level exclusion_for(user_idx) helper returns the set correctly."""
    raise NotImplementedError("Plan 02 fills this in")
