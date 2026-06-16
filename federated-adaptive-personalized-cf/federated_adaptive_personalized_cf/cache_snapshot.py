"""Bug 3 fix (Path B): snapshot/restore the per-client embedding cache at best-round.

Background
----------
Under split learning the server holds GLOBAL item embeddings and the per-client
on-disk cache (``.embedding_cache/{run_id}/partition_{pid}.pt``) holds LOCAL user
embeddings + per-user MLP/fusion/alpha state. The existing ``best_round_restore``
checkpoint rule rolls back ONLY the in-memory GLOBAL params at end of run; each
client's cache file is left at "whenever that user was last trained" (median
~R/k rounds before the best round under cross-device with N=6040 and
fraction_train=0.1). The mismatch desynchronizes the user/item embedding spaces
for ~90% of users → near-random predictions on the final D-06 full-pop eval.

Fix
---
At the SAME moment the in-memory GLOBAL params are snapshotted (the
``[CHECKPOINT] New best ...`` block), copy the entire client cache directory
into a sibling ``_best_snapshot_round_{N}/`` (single rolling snapshot, overwrite
on each new best). At the existing best_round_restore block, swap the snapshot
back into place. Clean up at end of run.

Operational properties
----------------------
- Hook point: AFTER ``grid.send_and_receive(eval_messages)`` returns and
  ``aggregate_evaluate`` has run for the round. All clients have finished
  writing round-N ``partition_{pid}.pt`` files; the server is mid-bookkeeping
  with no in-flight client writes.
- Atomicity: write into ``_best_snapshot_NEW/`` then ``os.rename`` to
  ``_best_snapshot_round_{N}/`` (removing any previous snapshot first).
- Disk cost: ~24 GB for ML-1M (4 MB × 6040). Reflink (``cp -r --reflink=auto``)
  is preferred — near-instant on CoW filesystems (ext4 with reflink, xfs, btrfs).
  Fallback: ``shutil.copytree``.
- Backwards-compatible: the helper is invoked ONLY under
  ``checkpoint_rule == "best_round_restore"``; for ``best_round`` and
  ``last_round`` the snapshot machinery is a no-op.
- Telemetry: emits ``[D-06.5]`` log lines on snapshot + restore.

This module is intentionally self-contained (no Flower / torch / numpy imports)
so it can be unit-tested without spinning up a federation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


_SNAPSHOT_PREFIX = "_best_snapshot_round_"
_SNAPSHOT_TMP = "_best_snapshot_NEW"


def _is_snapshot_dir(path: Path) -> bool:
    """Return True if ``path.name`` matches the snapshot dir naming convention."""
    name = path.name
    return name.startswith(_SNAPSHOT_PREFIX) or name == _SNAPSHOT_TMP


def _list_existing_snapshots(cache_root: Path) -> list[Path]:
    """Return all snapshot directories under ``cache_root`` (rolling + tmp)."""
    if not cache_root.is_dir():
        return []
    return [p for p in cache_root.iterdir() if p.is_dir() and _is_snapshot_dir(p)]


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy ``src`` directory to ``dst`` using reflink when available.

    Tries ``cp -r --reflink=auto`` first (CoW filesystems take O(metadata)
    instead of O(bytes)); falls back to ``shutil.copytree`` on any failure.
    Only the top-level direct children of ``src`` that are NOT themselves
    snapshot dirs are copied — we never nest snapshots inside snapshots.
    """
    dst.mkdir(parents=True, exist_ok=False)

    # Filter children: exclude any pre-existing snapshot dirs so we never
    # snapshot-of-snapshot or balloon disk usage.
    children = [c for c in src.iterdir() if not (c.is_dir() and _is_snapshot_dir(c))]

    # Try reflink first via cp.
    try:
        cmd = ["cp", "-a", "--reflink=auto"] + [str(c) for c in children] + [str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fall through to pure-Python copy.
        pass

    for child in children:
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def snapshot_cache(cache_root: Path, round_num: int) -> Optional[Path]:
    """Snapshot the live cache directory into ``_best_snapshot_round_{N}/``.

    Single rolling snapshot — any previous ``_best_snapshot_round_*`` is
    removed first. Atomically renames a temp dir into place so an interrupted
    snapshot never leaves a half-written ``round_{N}`` dir.

    Parameters
    ----------
    cache_root : Path
        ``<module>/.embedding_cache/{run_id}/`` — the directory containing the
        live ``partition_*.pt`` files and ``manifest.json``.
    round_num : int
        Round number this snapshot represents (encoded in the dir name).

    Returns
    -------
    Optional[Path]
        Path to the new snapshot dir on success, or ``None`` if ``cache_root``
        does not exist (no live cache to snapshot — e.g. round 1 with no
        clients yet written).
    """
    if not cache_root.is_dir():
        return None

    target = cache_root / f"{_SNAPSHOT_PREFIX}{int(round_num)}"
    tmp = cache_root / _SNAPSHOT_TMP

    # Clean up any leftover tmp from a prior interrupted snapshot.
    if tmp.exists():
        shutil.rmtree(tmp)

    t0 = time.monotonic()
    _copy_tree(cache_root, tmp)

    # Remove all prior snapshot dirs (single rolling snapshot semantics).
    for old in _list_existing_snapshots(cache_root):
        if old == tmp:
            continue
        shutil.rmtree(old, ignore_errors=True)

    os.rename(tmp, target)
    elapsed = time.monotonic() - t0

    # Best-effort size reporting.
    size_gb: Optional[float] = None
    try:
        total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        size_gb = total / (1024 ** 3)
    except OSError:
        pass

    size_str = f"{size_gb:.2f} GB" if size_gb is not None else "unknown size"
    print(
        f"  [D-06.5] Snapshotted client cache at round {round_num} "
        f"-> {target.name}/ (size: {size_str}, took {elapsed:.1f}s)"
    )
    return target


def restore_cache(cache_root: Path, round_num: int) -> bool:
    """Restore the cache from ``_best_snapshot_round_{N}/`` back into ``cache_root``.

    Replaces the live ``partition_*.pt`` files with the snapshot copies. The
    ``manifest.json`` from the snapshot is also restored so signature checks
    in subsequent reads see the same fingerprint that produced the best round.
    Other files at ``cache_root`` (sibling snapshot dirs, unrelated paths) are
    left alone.

    Parameters
    ----------
    cache_root : Path
        Live cache root.
    round_num : int
        Best round number — must match the suffix of the existing snapshot dir.

    Returns
    -------
    bool
        True if the restore happened, False if no matching snapshot exists.
    """
    snapshot = cache_root / f"{_SNAPSHOT_PREFIX}{int(round_num)}"
    if not snapshot.is_dir():
        return False

    # Replace partition_*.pt and manifest.json from snapshot.
    for src in snapshot.iterdir():
        if not src.is_file():
            continue
        dst = cache_root / src.name
        # Atomic file replace.
        shutil.copy2(src, dst.with_suffix(dst.suffix + ".restoretmp"))
        os.replace(dst.with_suffix(dst.suffix + ".restoretmp"), dst)

    print(
        f"  [D-06.5] Restored client cache from snapshot at round {round_num} "
        f"(best_round_restore active)"
    )
    return True


def cleanup_snapshots(cache_root: Path) -> None:
    """Remove all snapshot dirs under ``cache_root``.

    Called at end of run (success or failure) to free disk. The live cache
    (top-level ``partition_*.pt`` + ``manifest.json``) is left intact.
    """
    if not cache_root.is_dir():
        return
    for snap in _list_existing_snapshots(cache_root):
        shutil.rmtree(snap, ignore_errors=True)
