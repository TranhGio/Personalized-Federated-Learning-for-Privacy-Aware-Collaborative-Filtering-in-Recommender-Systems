"""Tests for fedrec_foundation.rng (FND-06 + CR-3)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedrec_foundation.rng import (
    _ALLOWED_PURPOSES,
    _derive_seed,
    derive_rng,
    np_rng,
    py_rng,
    server_rng,
    torch_gen,
)


def test_derive_rng_stable_across_processes() -> None:
    """FND-06-a + CR-3: seeds match across fresh Python processes with different PYTHONHASHSEED."""
    script = textwrap.dedent(
        """
        from fedrec_foundation.rng import _derive_seed
        print(_derive_seed("py", 42, 123, 7, "train_neg"))
        """
    )
    outputs = []
    for hashseed in ("0", "1", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(r.stdout.strip())
    assert len(set(outputs)) == 1, (
        f"RNG seed must be stable across PYTHONHASHSEED; got {outputs}"
    )


def test_tuple_uniqueness() -> None:
    """FND-06-b: distinct (user_id, round, purpose) tuples produce distinct seeds."""
    seeds = {
        _derive_seed("py", 42, u, r, p)
        for u in (1, 2, 3)
        for r in (0, 1, 2)
        for p in ("train_neg", "eval_neg", "model_init")
    }
    # 27 tuples -> 27 distinct seeds (collision-free by sha256).
    assert len(seeds) == 27


def test_all_three_rng_factories() -> None:
    """FND-06-c + CR-3: py_rng / np_rng / torch_gen produce INDEPENDENT streams."""
    py = py_rng(42, 1, 0, "train_neg")
    np_r = np_rng(42, 1, 0, "train_neg")
    torch_g = torch_gen(42, 1, 0, "train_neg")
    # Each is a distinct type.
    import random as _r

    assert isinstance(py, _r.Random)
    assert isinstance(np_r, np.random.Generator)
    assert isinstance(torch_g, torch.Generator)
    # Same inputs, different outputs (namespaces disambiguate).
    py_val = py.random()
    np_val = float(np_r.random())
    torch_val = float(torch.rand(1, generator=torch_g).item())
    assert py_val != np_val
    assert np_val != torch_val


def test_torch_generator_reproducible() -> None:
    """FND-06-d + CR-3: torch.Generator seeded from run_seed yields reproducible outputs."""
    g1 = torch_gen(42, 1, 0, "model_init")
    g2 = torch_gen(42, 1, 0, "model_init")
    t1 = torch.randn(5, 5, generator=g1)
    t2 = torch.randn(5, 5, generator=g2)
    assert torch.equal(t1, t2)


def test_sample_reproducible() -> None:
    """FND-06-e: per-user RNG produces the same sample sequence for the same (user, round)."""
    s1 = py_rng(42, -1, 0, "server_sample")
    s2 = py_rng(42, -1, 0, "server_sample")
    assert s1.sample(range(6040), 200) == s2.sample(range(6040), 200)


def test_dataloader_iteration_order() -> None:
    """CR-3 fourth assertion: DataLoader(generator=torch_gen(...)) order is deterministic."""
    ds = TensorDataset(torch.arange(100))
    g1 = torch_gen(42, 1, 0, "dataloader")
    g2 = torch_gen(42, 1, 0, "dataloader")
    loader1 = DataLoader(ds, batch_size=10, shuffle=True, generator=g1)
    loader2 = DataLoader(ds, batch_size=10, shuffle=True, generator=g2)
    batches1 = [b[0].tolist() for b in loader1]
    batches2 = [b[0].tolist() for b in loader2]
    assert batches1 == batches2


def test_unknown_purpose_raises() -> None:
    """Unknown purpose raises ValueError with the bad value in the message."""
    with pytest.raises(ValueError, match="Unknown RNG purpose"):
        _derive_seed("py", 42, 1, 0, "not_a_purpose")


def test_allowed_purposes_includes_dataloader() -> None:
    """`dataloader` purpose is REQUIRED for CR-3 DataLoader seeding."""
    assert "dataloader" in _ALLOWED_PURPOSES
    assert "server_sample" in _ALLOWED_PURPOSES


def test_server_rng_reproducible() -> None:
    """server_rng(run_seed) is reproducible."""
    s1 = server_rng(42)
    s2 = server_rng(42)
    assert s1.random() == s2.random()


def test_derive_rng_is_py_rng_alias() -> None:
    """derive_rng must produce identical stream to py_rng for same inputs (back-compat)."""
    d = derive_rng(42, 1, 0, "train_neg")
    p = py_rng(42, 1, 0, "train_neg")
    assert d.random() == p.random()
