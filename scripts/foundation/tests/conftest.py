"""Shared fixtures for foundation tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator

import pandas as pd
import pytest


@pytest.fixture
def synthetic_ratings_df() -> pd.DataFrame:
    """Tiny deterministic ratings DataFrame for unit tests.

    5 users, 4 items, 12 interactions — small enough to hand-verify.
    Columns match MovieLens-1M: user_id, movie_id, rating, timestamp.
    """
    rows = [
        # (user_id, movie_id, rating, timestamp)
        (1, 10, 5, 1000),
        (1, 20, 4, 1001),
        (1, 30, 3, 1002),
        (2, 10, 5, 2000),
        (2, 40, 4, 2001),
        (3, 20, 3, 3000),
        (3, 30, 4, 3001),
        (3, 40, 5, 3002),
        (4, 10, 2, 4000),
        (4, 20, 5, 4001),
        (5, 30, 4, 5000),
        (5, 40, 3, 5001),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


@pytest.fixture
def tmp_derived_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp data/derived/ directory and point the env override at it."""
    derived = tmp_path / "derived"
    derived.mkdir()
    monkeypatch.setenv("FEDREC_FOUNDATION_DATA_DIR", str(derived))
    return derived


@pytest.fixture
def pythonhashseed_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force PYTHONHASHSEED=random for the current process.

    Note: PYTHONHASHSEED is read at interpreter startup; setting it here
    only affects CHILD subprocesses spawned after the fixture runs.
    Use this in combination with subprocess.run tests.
    """
    monkeypatch.setenv("PYTHONHASHSEED", "random")
