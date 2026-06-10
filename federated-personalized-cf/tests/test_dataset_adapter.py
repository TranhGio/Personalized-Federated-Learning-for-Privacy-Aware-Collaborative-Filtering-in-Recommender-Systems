"""Tests for federated_personalized_cf.dataset (Phase 3 Plan 02 — D-17 foundation adapter + D-02)."""
from __future__ import annotations

from pathlib import Path

import pytest


# Skip entire file when the committed foundation bundle is not present.
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "data" / "derived" / "foundation_index.json").exists():
            return p
    raise RuntimeError("repo_root not found")


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed at data/derived/foundation_index.json",
)


def test_load_partition_data_uses_foundation_mapping() -> None:
    """Per D-17: mapping comes from fedrec_foundation, not a module-local helper."""
    from federated_personalized_cf.dataset import load_partition_data
    from fedrec_foundation.mapping import load_mapping

    bundle_path = _repo_root() / "data" / "derived" / "mapping.json"
    foundation_mapping = load_mapping(str(bundle_path))

    # partition_id=0 with natural partitioning == first user in the canonical mapping.
    _trainloader, _testloader, num_users, num_items, user2idx, item2idx = load_partition_data(
        partition_id=0, num_partitions=foundation_mapping.num_users,
        partition_mode="natural", split_mode="leave-one-out", batch_size=32,
    )

    assert num_users == foundation_mapping.num_users == 6040
    assert num_items == foundation_mapping.num_items == 3706
    # Spot-check a raw user ID whose canonical idx is known.
    some_raw_id = next(iter(foundation_mapping.user2idx))
    assert user2idx[some_raw_id] == foundation_mapping.user2idx[some_raw_id]


def test_load_partition_data_test_item_from_foundation_split() -> None:
    """Per D-17: test_item for partition_id == user_idx matches split_manifest.test_item_per_user."""
    from federated_personalized_cf.dataset import load_partition_data
    from fedrec_foundation.split import load_split_manifest

    split_path = _repo_root() / "data" / "derived" / "split_manifest.json"
    split = load_split_manifest(split_path)

    # Pick a user_idx with a known test item.
    user_idx = next(iter(split.test_item_per_user.keys()))
    expected_test_item = int(split.test_item_per_user[user_idx])

    _train, testloader, _nu, _ni, _u2i, _i2i = load_partition_data(
        partition_id=int(user_idx), num_partitions=6040,
        partition_mode="natural", split_mode="leave-one-out", batch_size=32,
    )
    test_items = [int(b["item"].item()) for b in testloader]
    assert expected_test_item in test_items, (
        f"user_idx={user_idx} expected test item {expected_test_item} not found in "
        f"testloader items {test_items[:5]}"
    )


def test_removed_helpers_gone_and_d02_raises() -> None:
    """Per D-17: rip targets absent. Per D-02: partition_mode='dirichlet' raises NotImplementedError."""
    from federated_personalized_cf import dataset
    from federated_personalized_cf.dataset import load_partition_data

    # D-17: these helpers must NOT appear as module attributes.
    for name in (
        "create_global_mappings",
        "create_leave_one_out_split",
        "compute_user_genre_distribution",
        "dirichlet_partition_users",
        "create_train_test_split",
        "_partition_cache",
    ):
        assert not hasattr(dataset, name), (
            f"{name} should have been removed per D-17 but is still present; "
            f"rip-and-replace incomplete."
        )

    # D-02: dirichlet must raise NotImplementedError with a pointer to pre-Phase-3 commits.
    with pytest.raises(NotImplementedError) as excinfo:
        load_partition_data(
            partition_id=0, num_partitions=5, partition_mode="dirichlet",
            split_mode="leave-one-out",
        )
    msg = str(excinfo.value)
    assert "D-02" in msg or "cross-device" in msg or "pre-Phase-3" in msg, (
        f"NotImplementedError message should reference D-02 / cross-device / pre-Phase-3; got: {msg!r}"
    )
