"""SHA-256 hashing helpers for foundation artifacts (FND-02, FND-07)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a single file read in 65536-byte chunks.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase 64-character hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_raw_data_hash(ml1m_dir: Path) -> str:
    """SHA-256 of ratings.dat || movies.dat || users.dat (in that order).

    The concatenation order is LOCKED — changing it changes the raw-data
    fingerprint for every committed artifact.

    Parameters
    ----------
    ml1m_dir : pathlib.Path
        Directory containing the three ML-1M .dat files.

    Returns
    -------
    str
        Lowercase 64-character hex digest.
    """
    h = hashlib.sha256()
    for fname in ("ratings.dat", "movies.dat", "users.dat"):
        with open(ml1m_dir / fname, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()
