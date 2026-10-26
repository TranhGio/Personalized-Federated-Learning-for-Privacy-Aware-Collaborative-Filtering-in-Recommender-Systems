"""Tests for fedrec_foundation.hashing (implemented in Plan 01)."""
from __future__ import annotations

from pathlib import Path

from fedrec_foundation.hashing import sha256_file, compute_raw_data_hash


def test_sha256_file_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_raw_data_hash_order_matters(tmp_path: Path) -> None:
    (tmp_path / "ratings.dat").write_bytes(b"R")
    (tmp_path / "movies.dat").write_bytes(b"M")
    (tmp_path / "users.dat").write_bytes(b"U")
    h = compute_raw_data_hash(tmp_path)
    assert len(h) == 64
    # Flip one file; hash must change.
    (tmp_path / "ratings.dat").write_bytes(b"X")
    h2 = compute_raw_data_hash(tmp_path)
    assert h != h2
