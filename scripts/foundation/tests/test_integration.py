"""End-to-end tests for the foundation build (implemented in Plans 02 + 06)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Plans 02 and 06 implement the end-to-end build")


def test_build_idempotent() -> None:
    """bundle-a + D-04 + N-3: running the builder twice produces byte-identical artifacts."""
    raise NotImplementedError("Plan 02 fills this in")


def test_bundle_atomic_publication() -> None:
    """bundle-b + N-3: artifacts appear atomically (no partial write visible mid-build)."""
    raise NotImplementedError("Plan 02 fills this in")


def test_build_creates_all_artifacts() -> None:
    """build-e2e: mapping.json, split_manifest.json, exclusion_items.npz all present after build."""
    raise NotImplementedError("Plan 02 fills this in")


def test_ml1m_counts_6040_3706() -> None:
    """empirical-a (Codex anchor): mapping built on data/ml-1m/ yields 6040 users and 3706 items."""
    raise NotImplementedError("Plan 06 fills this in")
