"""Tests for fedrec_foundation.atomic (Phase 7 D-17 / Pattern 6)."""
from __future__ import annotations

from pathlib import Path

from fedrec_foundation.atomic import atomic_write_text


def test_atomic_write_text(tmp_path: Path) -> None:
    """Phase 7: atomic_write_text writes UTF-8 content via tempfile+os.replace; no .tmp-* leftovers."""
    target = tmp_path / "out.md"
    payload = "# Header\n\n| col1 | col2 |\n|---|---|\n| 0.4123 ± 0.0089 | 0.7290 ± 0.0123 |\n"
    atomic_write_text(str(target), payload)
    # File exists + content is byte-identical.
    assert target.exists()
    assert target.read_text(encoding="utf-8") == payload
    # No .tmp-* leftovers in the parent dir (atomicity contract).
    leftovers = list(tmp_path.glob(".tmp-*"))
    assert leftovers == [], f"Expected no .tmp-* leftovers; found {leftovers}"


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    """Phase 7: parent directories auto-created if absent (matches atomic_write_json semantics)."""
    target = tmp_path / "deeply" / "nested" / "dir" / "out.csv"
    atomic_write_text(str(target), "module,ndcg10_mean\nbaseline,0.4123\n")
    assert target.exists()
    assert "0.4123" in target.read_text(encoding="utf-8")


def test_atomic_write_text_overwrites_existing(tmp_path: Path) -> None:
    """Phase 7: atomic write replaces existing file content (idempotent re-aggregation)."""
    target = tmp_path / "out.md"
    atomic_write_text(str(target), "first")
    atomic_write_text(str(target), "second")
    assert target.read_text(encoding="utf-8") == "second"
