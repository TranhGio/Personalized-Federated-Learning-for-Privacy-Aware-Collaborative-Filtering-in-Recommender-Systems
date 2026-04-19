"""Tests for fedrec_foundation.mapping (implemented in Plan 02)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 02 implements fedrec_foundation.mapping")


def test_sort_order() -> None:
    """FND-01-b: user_idx / item_idx assigned in sorted raw-ID order (deterministic)."""
    raise NotImplementedError("Plan 02 fills this in")


def test_item_mapping_from_ratings_only() -> None:
    """FND-01-c + CR-1: item_idx built from ratings.dat, NOT the union with movies.dat."""
    raise NotImplementedError("Plan 02 fills this in")


def test_roundtrip() -> None:
    """FND-01-a: build -> save -> load round-trip preserves raw<->idx bijection."""
    raise NotImplementedError("Plan 02 fills this in")
