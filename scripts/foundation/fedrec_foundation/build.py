"""Enable ``python -m fedrec_foundation.build``.

Thin shim re-exporting ``scripts/build_derived.py::main`` so the CLI
is discoverable as a module entry point. We add the sibling
``scripts/`` directory to ``sys.path`` at import time so the shim
works regardless of cwd.
"""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/foundation/scripts/ lives next to scripts/foundation/fedrec_foundation/.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_derived import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
