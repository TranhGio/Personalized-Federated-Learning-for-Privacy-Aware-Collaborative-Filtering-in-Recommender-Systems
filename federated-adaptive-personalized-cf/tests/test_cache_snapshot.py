"""Unit tests for cache_snapshot.py (Bug 3 / Path B).

Verifies the snapshot/restore/cleanup contract without spinning up Flower.
The helper module has no torch / numpy / flwr imports, so these tests run
in milliseconds against a small synthetic ``.embedding_cache/`` layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from federated_adaptive_personalized_cf.cache_snapshot import (
    cleanup_snapshots,
    restore_cache,
    snapshot_cache,
)


def _make_fake_cache(root: Path, num_partitions: int, payload: bytes) -> None:
    """Create a synthetic ``.embedding_cache/{run_id}/`` directory.

    Writes ``num_partitions`` ``partition_{i}.pt`` files with the given
    payload, plus a ``manifest.json``.
    """
    root.mkdir(parents=True, exist_ok=True)
    for i in range(num_partitions):
        (root / f"partition_{i}.pt").write_bytes(payload)
    (root / "manifest.json").write_text(json.dumps({"schema_version": 2, "tag": payload.decode()}))


def test_snapshot_returns_none_when_cache_missing(tmp_path: Path):
    cache_root = tmp_path / "missing"
    assert snapshot_cache(cache_root, round_num=5) is None


def test_snapshot_creates_round_suffixed_dir(tmp_path: Path):
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=3, payload=b"v1")

    snap = snapshot_cache(cache_root, round_num=7)

    assert snap is not None
    assert snap.is_dir()
    assert snap.name == "_best_snapshot_round_7"
    # Snapshot contains all partition files + manifest.
    assert (snap / "partition_0.pt").read_bytes() == b"v1"
    assert (snap / "partition_2.pt").read_bytes() == b"v1"
    assert json.loads((snap / "manifest.json").read_text())["tag"] == "v1"
    # Live cache untouched.
    assert (cache_root / "partition_0.pt").read_bytes() == b"v1"


def test_snapshot_overwrites_previous_snapshot(tmp_path: Path):
    """Single rolling snapshot — old _best_snapshot_round_* must be removed."""
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=2, payload=b"v1")

    snap_a = snapshot_cache(cache_root, round_num=3)
    assert snap_a is not None and snap_a.exists()

    # Mutate the live cache and snapshot at a new round.
    _make_fake_cache(cache_root, num_partitions=2, payload=b"v2")
    snap_b = snapshot_cache(cache_root, round_num=11)

    assert snap_b is not None
    assert snap_b.name == "_best_snapshot_round_11"
    assert snap_b.exists()
    # The old snapshot must be gone.
    assert not snap_a.exists()
    # The new snapshot has v2 payload.
    assert (snap_b / "partition_0.pt").read_bytes() == b"v2"


def test_snapshot_excludes_existing_snapshot_dirs(tmp_path: Path):
    """Snapshot must not nest snapshots inside snapshots."""
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=2, payload=b"v1")

    snap_a = snapshot_cache(cache_root, round_num=3)
    assert snap_a is not None

    # Write some new files, then snapshot at round 5. The round_3 snapshot
    # should be removed (rolling), and the new snapshot should NOT contain
    # any nested snapshot dir even if the cleanup ordering differed.
    _make_fake_cache(cache_root, num_partitions=2, payload=b"v3")
    snap_b = snapshot_cache(cache_root, round_num=5)
    assert snap_b is not None

    nested_snapshots = [p for p in snap_b.iterdir() if p.is_dir() and p.name.startswith("_best_snapshot_")]
    assert nested_snapshots == [], f"Nested snapshot dirs leaked into snapshot: {nested_snapshots}"


def test_restore_overwrites_live_cache_with_snapshot(tmp_path: Path):
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=4, payload=b"best_round_v")

    snap = snapshot_cache(cache_root, round_num=9)
    assert snap is not None

    # Mutate the live cache to simulate later rounds drifting away from best.
    _make_fake_cache(cache_root, num_partitions=4, payload=b"drifted_v")
    assert (cache_root / "partition_0.pt").read_bytes() == b"drifted_v"

    ok = restore_cache(cache_root, round_num=9)
    assert ok is True
    # All partition files are back to the snapshot payload.
    for i in range(4):
        assert (cache_root / f"partition_{i}.pt").read_bytes() == b"best_round_v"
    # Manifest restored too.
    assert json.loads((cache_root / "manifest.json").read_text())["tag"] == "best_round_v"


def test_restore_returns_false_when_snapshot_missing(tmp_path: Path):
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=1, payload=b"v")

    # No snapshot ever taken.
    ok = restore_cache(cache_root, round_num=42)
    assert ok is False
    # Live cache untouched.
    assert (cache_root / "partition_0.pt").read_bytes() == b"v"


def test_cleanup_removes_all_snapshots_but_keeps_live_cache(tmp_path: Path):
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=3, payload=b"v")

    snap = snapshot_cache(cache_root, round_num=2)
    assert snap is not None and snap.exists()

    cleanup_snapshots(cache_root)

    assert not snap.exists()
    # Live cache fully intact.
    assert (cache_root / "partition_0.pt").read_bytes() == b"v"
    assert (cache_root / "partition_2.pt").read_bytes() == b"v"
    assert (cache_root / "manifest.json").exists()


def test_cleanup_no_op_when_cache_root_missing(tmp_path: Path):
    cleanup_snapshots(tmp_path / "absent")  # Must not raise.


def test_full_lifecycle_snapshot_drift_restore_cleanup(tmp_path: Path):
    """End-to-end: snapshot at best round, drift, restore, then cleanup."""
    cache_root = tmp_path / "run_id_x"
    _make_fake_cache(cache_root, num_partitions=5, payload=b"best")

    snap = snapshot_cache(cache_root, round_num=10)
    assert snap is not None

    # Subsequent (worse) rounds mutate cache.
    _make_fake_cache(cache_root, num_partitions=5, payload=b"later")
    snap2 = snapshot_cache(cache_root, round_num=15)  # but this round is also a new best
    assert snap2 is not None
    # First snapshot rolled out.
    assert not snap.exists()

    # Imagine round 15 was actually NOT the best — restoring requires we
    # snapshotted the right round. Simulate restoring the round 15 snapshot.
    _make_fake_cache(cache_root, num_partitions=5, payload=b"final_drift")
    ok = restore_cache(cache_root, round_num=15)
    assert ok is True
    for i in range(5):
        assert (cache_root / f"partition_{i}.pt").read_bytes() == b"later"

    cleanup_snapshots(cache_root)
    assert not snap2.exists()
    # Live cache contents preserved (whatever was there post-restore).
    for i in range(5):
        assert (cache_root / f"partition_{i}.pt").read_bytes() == b"later"


# =============================================================================
# D-06.5 source-level integration guards (Bug 3 / Path B).
# Mirror the D-05 / D-07 source-proximity asserts in test_server_integration.py:
# proves the snapshot/restore hooks are wired into the right branches without
# spinning up a live Flower Grid.
# =============================================================================

def _read_server_app_source() -> str:
    from pathlib import Path
    here = Path(__file__).resolve().parent
    src = here.parent / "federated_adaptive_personalized_cf" / "server_app.py"
    return src.read_text()


def test_snapshot_cache_called_inside_best_round_restore_branch() -> None:
    """D-06.5: ``snapshot_cache(cache_root, ...)`` fires AFTER the
    ``best_metric = current_ndcg`` assignment AND only under the
    ``best_round_restore`` branch (not under ``best_round`` or ``last_round``)."""
    src = _read_server_app_source()

    # Helper must be imported.
    assert "from federated_adaptive_personalized_cf.cache_snapshot import" in src
    assert "snapshot_cache" in src and "restore_cache" in src and "cleanup_snapshots" in src

    best_metric_assign_idx = src.find("best_metric = current_ndcg")
    assert best_metric_assign_idx != -1
    snapshot_call_idx = src.find("snapshot_cache(cache_root", best_metric_assign_idx)
    assert snapshot_call_idx != -1, (
        "D-06.5 violated: snapshot_cache() call missing inside best-metric branch"
    )
    # The snapshot call must be guarded by an explicit best_round_restore check.
    window = src[best_metric_assign_idx:snapshot_call_idx]
    assert 'checkpoint_rule == "best_round_restore"' in window, (
        "D-06.5 violated: snapshot_cache() must be guarded by "
        "checkpoint_rule == 'best_round_restore' to avoid burning ~24 GB on "
        "best_round / last_round runs that never use the snapshot."
    )


def test_restore_cache_called_after_arrays_restore() -> None:
    """D-06.5: ``restore_cache(cache_root, ...)`` fires AFTER ``arrays = best_arrays``
    and AFTER the prototype restore — i.e. inside the same best_round_restore
    branch, BEFORE the D-06 extra-eval-round broadcast."""
    src = _read_server_app_source()

    arrays_restore_idx = src.find("arrays = best_arrays")
    assert arrays_restore_idx != -1
    restore_call_idx = src.find("restore_cache(cache_root", arrays_restore_idx)
    assert restore_call_idx != -1, (
        "D-06.5 violated: restore_cache() call missing AFTER `arrays = best_arrays`"
    )
    # Restore must happen BEFORE the D-06 extra-eval-round broadcast block so
    # clients see the snapshotted local state during the canonical full-pop eval.
    d06_broadcast_idx = src.find("[D-06] Broadcasting extra eval round")
    assert d06_broadcast_idx != -1
    assert restore_call_idx < d06_broadcast_idx, (
        "D-06.5 violated: restore_cache() must run BEFORE the D-06 extra-eval "
        "broadcast — otherwise clients evaluate against de-synchronized cache."
    )


def test_cleanup_snapshots_called_at_end_of_main() -> None:
    """D-06.5: snapshot dir is cleaned up at end of main() to free ~24 GB."""
    src = _read_server_app_source()
    assert "cleanup_snapshots(cache_root)" in src, (
        "D-06.5 violated: cleanup_snapshots() not called at end of main(); "
        "the snapshot dir would persist after each run and leak disk."
    )
