"""Phase 5 test conftest: shared fixtures for Wave-2 tests.

- Foundation-bundle skip marker: integration tests requiring the committed
  ``data/derived/foundation_index.json`` bundle are skipped on minimal clones
  where the bundle is absent.
- ``run_seed`` fixture pinning the canonical FND-06 root seed (42).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_PATH = _REPO_ROOT / "data" / "derived" / "foundation_index.json"


def pytest_collection_modifyitems(config, items):
    """Skip tests requiring the foundation bundle if it's not committed."""
    if _BUNDLE_PATH.exists():
        return
    skip_marker = pytest.mark.skip(
        reason="foundation bundle not committed (skip Phase-5 integration tests)"
    )
    for item in items:
        if "foundation_bundle_required" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def run_seed() -> int:
    """Canonical FND-06 root seed for Phase-5 deterministic tests."""
    return 42
