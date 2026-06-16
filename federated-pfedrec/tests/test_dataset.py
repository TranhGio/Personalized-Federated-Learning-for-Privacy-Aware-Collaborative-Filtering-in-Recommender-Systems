"""Phase 5 D-09 frozen-cross-silo guard: dataset.py raises at both entry points.

Mirrors Phase-3 Plan-02 ``test_dataset_adapter.py`` and Phase-4 Plan-02
``test_dataset_adapter.py`` patterns:

  - D-09 guards both ``load_partition_data`` and ``load_full_data`` against
    cross-silo (``partition_mode='dirichlet'``) invocations.
  - Error message includes the literal tokens 'D-09', 'cross-device', and
    'pre-Phase-5' so the regression guard catches any future loosening of
    the message text.
  - D-17 rip targets are absent: legacy module-local mapping/split helpers
    no longer live in dataset.py (foundation owns them now).
"""
from __future__ import annotations

import pytest


def test_load_partition_data_raises_on_non_natural() -> None:
    """D-09: partition_mode='dirichlet' raises NotImplementedError BEFORE any data load."""
    from federated_pfedrec.dataset import load_partition_data

    with pytest.raises(NotImplementedError) as exc_info:
        load_partition_data(partition_id=0, num_partitions=6040, partition_mode="dirichlet")

    msg = str(exc_info.value)
    assert "D-09" in msg, msg
    assert "cross-device" in msg, msg
    assert "pre-Phase-5" in msg, msg


def test_load_full_data_raises_on_non_natural() -> None:
    """D-09: load_full_data also guards (Phase-3/4 tightening pattern: BOTH entry points)."""
    from federated_pfedrec.dataset import load_full_data

    with pytest.raises(NotImplementedError) as exc_info:
        load_full_data(partition_mode="dirichlet")

    msg = str(exc_info.value)
    assert "D-09" in msg, msg
    assert "cross-device" in msg, msg
    assert "pre-Phase-5" in msg, msg


def test_dataset_uses_foundation_adapter_imports() -> None:
    """D-17 rip-and-replace: dataset.py imports from fedrec_foundation
    and the legacy mapping/split/exclusion helpers are removed."""
    import inspect

    import federated_pfedrec.dataset as ds

    src = inspect.getsource(ds)
    assert "from fedrec_foundation" in src

    # Legacy helpers must be removed (D-17 rip targets).
    assert "def create_global_mappings" not in src
    assert "def create_leave_one_out_split" not in src
    assert "def dirichlet_partition_users" not in src
    assert "def create_train_test_split" not in src
    assert "def compute_user_genre_distribution" not in src
