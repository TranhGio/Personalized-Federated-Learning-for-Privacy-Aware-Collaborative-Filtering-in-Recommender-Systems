#!/usr/bin/env python3
"""Prune old run-scoped embedding caches under .embedding_cache/.

D-10 helper (manual only; never invoked by automation). Keeps the newest N
run_id-scoped subdirectories (sorted by mtime) and deletes all older ones.

Content-hash ``sig_*`` directories (D-09 reuse-cache opt-in layout, keyed by
sha256 of the non-run_id signature fields) are NEVER touched by this helper —
those are user-managed via ``--run-config reuse-cache=true`` collisions and
must be preserved even when ``--keep 0``.

Usage
-----
    python scripts/clean_cache.py [--keep N] [--cache-root PATH] [--dry-run]

Examples
--------
    # Preview: show which run dirs would be deleted under the default keep=5.
    python scripts/clean_cache.py --dry-run

    # Actually prune: keep only the 3 newest run dirs.
    python scripts/clean_cache.py --keep 3

    # Point at a non-default cache root (e.g. per-module cache).
    python scripts/clean_cache.py --cache-root federated-personalized-cf/.embedding_cache

Exit codes: 0 on success, non-zero on invalid args.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _list_run_dirs(cache_root: Path) -> List[Path]:
    """Return all run-id-scoped subdirs (i.e. NOT ``sig_*`` content-hash dirs).

    A run-id subdir is any direct child of ``cache_root`` that is a directory
    AND whose name does NOT start with ``sig_``. Content-hash dirs (D-09
    reuse-cache) are the opt-in cross-run-sharing layout and must be preserved
    by this helper regardless of ``--keep``.

    Parameters
    ----------
    cache_root : Path
        Path to the embedding cache root. If the directory does not exist or
        is not a directory, an empty list is returned.

    Returns
    -------
    List[Path]
        All run-id-scoped subdirectories of ``cache_root``. Order is not
        guaranteed — callers must sort.
    """
    if not cache_root.exists() or not cache_root.is_dir():
        return []
    return [
        p for p in cache_root.iterdir()
        if p.is_dir() and not p.name.startswith("sig_")
    ]


def prune(cache_root: Path, keep: int, dry_run: bool) -> List[Path]:
    """Prune all but the ``keep`` newest run-id dirs under ``cache_root``.

    Parameters
    ----------
    cache_root : Path
        Embedding cache root (e.g. ``./.embedding_cache``).
    keep : int
        Number of newest run-id dirs to preserve. Must be >= 0.
    dry_run : bool
        If True, print the directories that WOULD be deleted without actually
        removing them. No filesystem mutation.

    Returns
    -------
    List[Path]
        The directories that were deleted (or, under ``dry_run``, the
        directories that WOULD have been deleted).
    """
    dirs = _list_run_dirs(cache_root)
    # mtime descending (newest first); ties broken by name for determinism.
    dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    keep_set = set(dirs[: max(0, int(keep))])
    to_delete = [d for d in dirs if d not in keep_set]
    for d in to_delete:
        if dry_run:
            print(f"[DRY-RUN] would delete {d}")
        else:
            shutil.rmtree(d)
            print(f"deleted {d}")
    return to_delete


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        CLI arguments excluding ``sys.argv[0]``. If None, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prune old run-id-scoped subdirectories under .embedding_cache/. "
            "Keeps the newest N by mtime; never touches sig_* content-hash dirs (D-09)."
        ),
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of newest run dirs to preserve (default: 5).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".embedding_cache"),
        help="Path to the embedding cache root (default: ./.embedding_cache).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without deleting anything.",
    )
    args = parser.parse_args(argv)

    if args.keep < 0:
        parser.error("--keep must be >= 0")

    deleted = prune(args.cache_root, args.keep, args.dry_run)
    # Under dry_run, no deletion happened, so _list_run_dirs still holds the
    # full pre-prune set; subtract len(deleted) to report the would-be-kept
    # count. Under real pruning, _list_run_dirs already reflects survivors.
    current = _list_run_dirs(args.cache_root)
    kept = len(current) - (len(deleted) if args.dry_run else 0)
    action_word = "would delete" if args.dry_run else "deleted"
    print(
        f"{action_word} {len(deleted)} run-dir(s); "
        f"kept {kept} newest under {args.cache_root}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
