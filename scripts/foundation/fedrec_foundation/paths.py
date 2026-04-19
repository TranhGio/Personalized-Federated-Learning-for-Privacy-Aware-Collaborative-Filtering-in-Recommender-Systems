"""Path helpers for foundation artifacts and raw data.

Locates the repo root by walking up from this file until a directory
containing ``data/ml-1m/`` is found. This keeps the foundation module
usable from any cwd (Flower subprocesses may chdir).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_OVERRIDE = "FEDREC_FOUNDATION_DATA_DIR"


def repo_root() -> Path:
    """Return the repo root by walking up from this file.

    Returns
    -------
    pathlib.Path
        The first ancestor directory that contains ``data/ml-1m``.

    Raises
    ------
    RuntimeError
        If no such ancestor exists.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "data" / "ml-1m").exists():
            return parent
    raise RuntimeError(
        f"Could not locate repo root from {here}. Expected an ancestor "
        f"containing data/ml-1m/."
    )


def data_derived() -> Path:
    """Return the data/derived/ directory (override via env var).

    Env var ``FEDREC_FOUNDATION_DATA_DIR`` overrides the default for CI
    or remote environments.
    """
    override: Optional[str] = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)
    return repo_root() / "data" / "derived"


def ml1m_dir() -> Path:
    """Return the data/ml-1m/ directory (not overridable)."""
    return repo_root() / "data" / "ml-1m"
