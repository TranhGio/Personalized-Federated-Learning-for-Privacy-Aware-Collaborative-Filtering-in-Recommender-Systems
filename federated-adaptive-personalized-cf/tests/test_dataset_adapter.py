"""Adapter tests for federated_adaptive_personalized_cf.dataset (Phase 4 Plan 02).

D-17 rip-and-replace: mapping / split / exclusion delegated to fedrec_foundation.
D-18 preservation: MovieLensDataset, download_movielens_1m, load_movielens_1m,
  and natural_partition_users retained verbatim.
D-02 tightening (Phase-3 Plan 02 pattern): partition_mode='dirichlet' raises
  NotImplementedError at BOTH load_partition_data AND load_full_data.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "data" / "derived" / "foundation_index.json").exists():
            return p
    raise RuntimeError("repo_root not found")


_FOUNDATION_INDEX = (
    Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json"
)

pytestmark = pytest.mark.skipif(
    not _FOUNDATION_INDEX.exists(),
    reason="foundation bundle not committed (data/derived/foundation_index.json missing)",
)


def test_load_partition_data_uses_foundation_mapping() -> None:
    """Per D-17: mapping comes from fedrec_foundation, not a module-local helper."""
    from federated_adaptive_personalized_cf.dataset import load_partition_data
    from fedrec_foundation.mapping import load_mapping

    bundle_path = _repo_root() / "data" / "derived" / "mapping.json"
    foundation_mapping = load_mapping(str(bundle_path))

    trainloader, testloader, num_users, num_items, user2idx, item2idx, user_stats = (
        load_partition_data(
            partition_id=0,
            num_partitions=foundation_mapping.num_users,
            alpha=0.5,
            compute_stats=True,
            split_mode="leave-one-out",
            partition_mode="natural",
            batch_size=32,
        )
    )

    assert num_users == foundation_mapping.num_users == 6040
    assert num_items == foundation_mapping.num_items == 3706
    # Spot-check a raw user ID whose canonical idx is known.
    some_raw_id = next(iter(foundation_mapping.user2idx))
    assert user2idx[some_raw_id] == foundation_mapping.user2idx[some_raw_id]
    # user_stats dict carries the 4 fields adaptive compute_client_alpha expects.
    assert user_stats is not None
    pid_stats = user_stats[0]
    for key in ("n_interactions", "genre_entropy", "n_unique_items", "rating_std"):
        assert key in pid_stats, (
            f"user_stats missing '{key}' — adaptive compute_client_alpha "
            f"relies on this field name"
        )


def test_load_partition_data_test_item_from_foundation_split() -> None:
    """Per D-17: test_item for partition_id == user_idx matches split_manifest.test_item_per_user."""
    from federated_adaptive_personalized_cf.dataset import load_partition_data
    from fedrec_foundation.split import load_split_manifest

    split_path = _repo_root() / "data" / "derived" / "split_manifest.json"
    split = load_split_manifest(split_path)

    # Pick a user_idx with a known test item.
    user_idx = int(next(iter(split.test_item_per_user.keys())))
    expected_test_item = int(split.test_item_per_user[user_idx])

    _train, testloader, _nu, _ni, _u2i, _i2i, _us = load_partition_data(
        partition_id=user_idx,
        num_partitions=6040,
        alpha=0.5,
        compute_stats=False,
        split_mode="leave-one-out",
        partition_mode="natural",
        batch_size=32,
    )
    test_items = [int(b["item"].item()) for b in testloader]
    assert expected_test_item in test_items, (
        f"user_idx={user_idx} expected test item {expected_test_item} "
        f"not found in testloader items {test_items[:5]}"
    )


def test_removed_helpers_gone_and_d18_preserved() -> None:
    """Per D-17: rip targets absent. Per D-18: pre-existing WIP preserved."""
    from federated_adaptive_personalized_cf import dataset

    # D-17: these helpers must NOT appear as module attributes.
    for name in (
        "create_global_mappings",
        "create_leave_one_out_split",
        "compute_user_genre_distribution",
        "compute_user_stats",
        "compute_partition_user_stats",
        "dirichlet_partition_users",
        "create_train_test_split",
        "_partition_cache",
    ):
        assert not hasattr(dataset, name), (
            f"{name} should have been removed per D-17 but is still present; "
            f"rip-and-replace incomplete."
        )

    # D-18: these 4 symbols MUST remain in the module for backwards compatibility.
    for name in (
        "MovieLensDataset",
        "download_movielens_1m",
        "load_movielens_1m",
        "natural_partition_users",
    ):
        assert hasattr(dataset, name), (
            f"{name} was incorrectly removed; D-18 surgical discipline requires "
            f"preserving pre-existing WIP."
        )


def test_dirichlet_raises_at_both_entry_points() -> None:
    """Per D-02 (Phase-3 tightening): BOTH load_partition_data AND load_full_data raise."""
    from federated_adaptive_personalized_cf.dataset import load_partition_data, load_full_data

    with pytest.raises(NotImplementedError) as excinfo1:
        load_partition_data(
            partition_id=0,
            num_partitions=5,
            alpha=0.5,
            partition_mode="dirichlet",
            split_mode="leave-one-out",
        )
    msg1 = str(excinfo1.value)
    assert "D-02" in msg1 or "cross-device" in msg1 or "pre-Phase-4" in msg1, (
        f"load_partition_data NotImplementedError should reference "
        f"D-02 / cross-device / pre-Phase-4; got: {msg1!r}"
    )

    with pytest.raises(NotImplementedError) as excinfo2:
        load_full_data(partition_mode="dirichlet")
    msg2 = str(excinfo2.value)
    assert "D-02" in msg2 or "cross-device" in msg2 or "pre-Phase-4" in msg2, (
        f"load_full_data NotImplementedError should reference "
        f"D-02 / cross-device / pre-Phase-4; got: {msg2!r}"
    )
