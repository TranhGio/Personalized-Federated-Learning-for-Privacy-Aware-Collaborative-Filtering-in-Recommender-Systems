"""Tests for fedrec_foundation.split (implemented in Plan 02)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 02 implements fedrec_foundation.split")


def test_hash_deterministic() -> None:
    """FND-02-a: split_hash is stable under repeated builds of the same ratings."""
    raise NotImplementedError("Plan 02 fills this in")


def test_timestamp_tiebreak() -> None:
    """FND-02-b: ties broken by (timestamp DESC, item_idx ASC) deterministically."""
    raise NotImplementedError("Plan 02 fills this in")


def test_split_lock_refuses_overwrite() -> None:
    """FND-02-c + D-04: builder refuses to overwrite an existing manifest with a new hash."""
    raise NotImplementedError("Plan 02 fills this in")


def test_train_only_user_stats() -> None:
    """FND-02-d + CR-5: user_stats computed on train-only (held-out item excluded)."""
    raise NotImplementedError("Plan 02 fills this in")
