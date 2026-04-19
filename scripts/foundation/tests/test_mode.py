"""Tests for fedrec_foundation.mode (implemented in Plan 05)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 05 implements fedrec_foundation.mode")


def test_override_logging() -> None:
    """mode-a + D-10: CLI overrides are recorded in manifest.overrides and printed loudly."""
    raise NotImplementedError("Plan 05 fills this in")


def test_assertion_flags() -> None:
    """mode-b + D-11 + CR-2: benchmark_cross_device asserts num_users_in_client == 1."""
    raise NotImplementedError("Plan 05 fills this in")
