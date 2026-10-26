"""End-to-end integration tests for the foundation bundle + ML-1M anchors.

Plan 02 ships:
- test_build_idempotent
- test_bundle_atomic_publication
- test_build_creates_all_artifacts
- test_ml1m_counts_6040_3706 (empirical anchor; skipped if real ML-1M absent)

Plan 06 appends (IMP-1 cross-module integration):
- test_cross_module_imports (parametrized across 4 federated-*-cf/ modules)
- test_pyproject_declares_foundation_dep (pyproject.toml dep declaration)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedrec_foundation.bundle import publish_bundle, verify_bundle
from fedrec_foundation.exclusion import build_exclusion
from fedrec_foundation.mapping import build_mapping
from fedrec_foundation.paths import ml1m_dir
from fedrec_foundation.split import build_split


@pytest.fixture
def synthetic_movies_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (10, "A", "Action"),
            (20, "B", "Drama"),
            (30, "C", "Comedy"),
            (40, "D", "Action"),
        ],
        columns=["movie_id", "title", "genres"],
    )


def _vectorized_train_split(
    df_canonical: pd.DataFrame, test_item_per_user: dict
) -> pd.DataFrame:
    """Vectorized train filter via a merge on (user_idx, item_idx) pairs."""
    test_pairs = pd.DataFrame(
        [(u, i) for u, i in test_item_per_user.items()],
        columns=["user_idx", "item_idx"],
    )
    test_pairs["is_test"] = True
    merged = df_canonical.merge(test_pairs, on=["user_idx", "item_idx"], how="left")
    return merged[merged["is_test"].isna()].drop(columns=["is_test"]).copy()


def _build_small_bundle(df: pd.DataFrame, movies_df: pd.DataFrame, derived: Path):
    m = build_mapping(df)
    s = build_split(df, m, movies_df, mapping_sha256="a" * 64, raw_data_hash="b" * 64)
    df_c = df.copy()
    df_c["user_idx"] = df_c["user_id"].map(m.user2idx)
    df_c["item_idx"] = df_c["movie_id"].map(m.item2idx)
    train_c = _vectorized_train_split(df_c, s.test_item_per_user)
    excl = build_exclusion(train_c, s)
    return publish_bundle(derived, m, s, excl)  # 4-arg signature (IMP-2)


def test_build_idempotent(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """bundle-a + D-04 + N-3: re-running the builder produces the same hashes."""
    derived = tmp_path / "derived"
    idx1 = _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    idx2 = _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    assert idx1.split_hash == idx2.split_hash
    assert idx1.foundation_contract_sha256 == idx2.foundation_contract_sha256
    assert idx1.mapping_sha256 == idx2.mapping_sha256
    assert idx1.exclusion_sha256 == idx2.exclusion_sha256


def test_bundle_atomic_publication(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """bundle-b + N-3: verify_bundle fails if any payload is missing."""
    derived = tmp_path / "derived"
    _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    verify_bundle(derived)  # happy path OK
    # Delete one payload; verify_bundle must fail.
    (derived / "exclusion_items.npz").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_bundle(derived)


def test_build_creates_all_artifacts(
    synthetic_ratings_df: pd.DataFrame,
    synthetic_movies_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """build-e2e: mapping.json + split_manifest.json + exclusion_items.npz + foundation_index.json."""
    derived = tmp_path / "derived"
    _build_small_bundle(synthetic_ratings_df, synthetic_movies_df, derived)
    for name in (
        "mapping.json",
        "split_manifest.json",
        "exclusion_items.npz",
        "foundation_index.json",
    ):
        assert (derived / name).exists(), f"{name} missing"


def test_ml1m_counts_6040_3706() -> None:
    """empirical-a (Codex anchor): real ML-1M produces 6040 users + 3706 items."""
    ml1m = ml1m_dir()
    if not (ml1m / "ratings.dat").exists():
        pytest.skip("real ML-1M not present in data/ml-1m/")
    ratings = pd.read_csv(
        ml1m / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    m = build_mapping(ratings)
    assert m.num_users == 6040
    assert m.num_items == 3706


# ---------------------------------------------------------------------------
# Cross-module import smoke test (Plan 06 / IMP-1)
# ---------------------------------------------------------------------------

_MODULES = (
    "federated-baseline-cf",
    "federated-pfedrec",
    "federated-personalized-cf",
    "federated-adaptive-personalized-cf",
)

_FOUNDATION_SUBMODULES = (
    "fedrec_foundation",
    "fedrec_foundation.mapping",
    "fedrec_foundation.split",
    "fedrec_foundation.exclusion",
    "fedrec_foundation.evaluator",
    "fedrec_foundation.weight_policy",
    "fedrec_foundation.fit_metrics",
    "fedrec_foundation.rng",
    "fedrec_foundation.manifest",
    "fedrec_foundation.mode",
)


def _repo_root() -> Path:
    """Walk up from this test to the repo root (containing data/ml-1m)."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "data" / "ml-1m").exists():
            return p
    pytest.skip("Repo root with data/ml-1m not located")


@pytest.mark.parametrize("module_dir", _MODULES)
def test_cross_module_imports(module_dir: str) -> None:
    """Each of the four federated modules can import every foundation submodule.

    Runs a subprocess with cwd set to the module's directory to mirror
    `flwr run .` behavior. Requires that the user has already run
    `pip install -e scripts/foundation/` (documented in docs/setup.md).
    """
    root = _repo_root()
    mod_path = root / module_dir
    if not mod_path.exists():
        pytest.skip(f"{mod_path} not present")
    import_stmts = "; ".join(f"import {m}" for m in _FOUNDATION_SUBMODULES)
    script = f"{import_stmts}; print('ok')"
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(mod_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, (
        f"Cross-module import failed in {module_dir}:\n"
        f"STDOUT={r.stdout!r}\nSTDERR={r.stderr!r}\n"
        f"Hint: run `pip install -e scripts/foundation/` and "
        f"`pip install -e {module_dir}/` (see docs/setup.md)."
    )
    assert "ok" in r.stdout


def test_pyproject_declares_foundation_dep() -> None:
    """IMP-1: each module's pyproject.toml declares fedrec-foundation as a dep."""
    root = _repo_root()
    for mod in _MODULES:
        pyproject = root / mod / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip(f"{pyproject} not present")
        content = pyproject.read_text()
        assert "fedrec-foundation" in content, (
            f"{mod}/pyproject.toml missing fedrec-foundation dependency"
        )
