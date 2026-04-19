"""Tests for fedrec_foundation.rng (implemented in Plan 04)."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skip(reason="Plan 04 implements fedrec_foundation.rng")


def test_derive_rng_stable_across_processes() -> None:
    """FND-06-a + CR-3: seeds match across fresh Python processes with different PYTHONHASHSEED."""
    script = textwrap.dedent(
        """
        from fedrec_foundation.rng import _derive_seed
        print(_derive_seed(42, 123, 7, "train_neg"))
        """
    )
    outputs = []
    for hashseed in ("0", "1", "random"):
        env = {"PYTHONHASHSEED": hashseed} if hashseed != "random" else {}
        r = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        outputs.append(r.stdout.strip())
    assert len(set(outputs)) == 1, (
        f"RNG seed must be stable across PYTHONHASHSEED; got {outputs}"
    )


def test_tuple_uniqueness() -> None:
    """FND-06-b: distinct (user_id, round, purpose) tuples produce distinct seeds."""
    raise NotImplementedError("Plan 04 fills this in")


def test_all_three_rng_factories() -> None:
    """FND-06-c + CR-3: server / per-user / torch-generator factories all derive from run_seed."""
    raise NotImplementedError("Plan 04 fills this in")


def test_torch_generator_reproducible() -> None:
    """FND-06-d + CR-3: torch.Generator seeded from run_seed yields reproducible outputs."""
    raise NotImplementedError("Plan 04 fills this in")


def test_sample_reproducible() -> None:
    """FND-06-e: per-user RNG produces the same sample sequence for the same (user, round)."""
    raise NotImplementedError("Plan 04 fills this in")
