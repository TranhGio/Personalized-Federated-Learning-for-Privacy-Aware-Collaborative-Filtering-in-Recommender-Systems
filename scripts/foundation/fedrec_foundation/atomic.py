"""Atomic JSON write helpers for foundation artifacts.

Matches the tempfile + ``os.replace`` pattern already used elsewhere in
the repo (see CLAUDE.md > "Atomic cache saves"). No pickle; JSON only
(D-05 in 01-CONTEXT.md).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str, data: object) -> None:
    """Write JSON atomically via tempfile + ``os.replace``.

    Works on POSIX and Windows. On crash mid-write, the destination
    file is either untouched or fully-new — never partial.

    Parameters
    ----------
    path : str
        Destination path for the JSON file. Parent directories are
        created if absent.
    data : object
        JSON-serializable payload. Numpy scalars and ``pathlib.Path``
        objects are handled by ``_json_default``.

    Returns
    -------
    None
    """
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem required for atomic replace.
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _json_default(obj: Any) -> Any:
    """Handle numpy scalars and Path objects that ``json.dumps`` rejects."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
