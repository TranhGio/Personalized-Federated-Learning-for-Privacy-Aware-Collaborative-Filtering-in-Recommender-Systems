"""Tests for fedrec_foundation launcher (implemented in Plan 05)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plan 05 implements the mode-aware launcher")


def test_launcher_sets_num_supernodes() -> None:
    """mode-c + D-06 + CR-2: benchmark launcher passes num-supernodes=6040 to flwr run."""
    raise NotImplementedError("Plan 05 fills this in")
