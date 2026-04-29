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


_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})


def module_run_results_dir(module: str, run_id: str) -> Path:
    """Return ``<repo>/results/federated/<module>/<run_id>/`` (creating it).

    Used by every per-module ``server_app.py`` to resolve the canonical write
    path for a Phase-6 cross-device run. The directory is created with
    ``parents=True, exist_ok=True`` so callers never see ``FileNotFoundError``
    on the parent path. The directory IS the run identifier (D-01 -- one
    directory per run; results.json + manifest.json live inside).

    The path is repo-root anchored via :func:`repo_root` (D-02). This makes
    the helper safe to call from any cwd -- Flower simulation may chdir
    subprocesses; the returned path is independent of cwd.

    Parameters
    ----------
    module : str
        One of ``"baseline"`` / ``"personalized"`` / ``"adaptive"`` /
        ``"pfedrec"``. Matches the literal value passed to
        :func:`fedrec_foundation.manifest.build_run_manifest` ``module=`` kwarg
        (manifest.py:80 comment is the source of truth).
    run_id : str
        Same string as ``RunManifest.run_id`` (from
        :func:`fedrec_foundation.manifest.generate_run_id`).

    Returns
    -------
    pathlib.Path
        Absolute, resolved path to the per-run directory.

    Raises
    ------
    ValueError
        If ``module`` is not in the allowed-modules whitelist (Pitfall 6 --
        typos in literals like ``"basline"`` must fail loud at runtime so
        results never land in ``/results/federated/basline/<run_id>/``).
    """
    if module not in _ALLOWED_MODULES:
        raise ValueError(
            f"Unknown module {module!r}. Expected one of "
            f"{sorted(_ALLOWED_MODULES)}."
        )
    out = repo_root() / "results" / "federated" / module / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out
