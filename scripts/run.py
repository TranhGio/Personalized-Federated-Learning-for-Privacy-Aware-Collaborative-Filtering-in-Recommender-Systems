#!/usr/bin/env python
"""Launcher for Flower app runs with mode-locked federation-level config (Codex CR-2).

Usage
-----
    python scripts/run.py <module> <mode> [--run-config KEY=VAL ...]
    python scripts/run.py --dry-run <module> <mode>

Modules: baseline, personalized, adaptive, pfedrec
Modes:   benchmark_cross_device, paper_compat_pfedrec, cross_silo_legacy

The mode selector in each module's pyproject.toml is an app-level
assertion. ``num-supernodes`` cannot be set from inside a Flower app
via ``Context.run_config`` because it is resolved at federation
construction time, strictly BEFORE ``ServerApp``/``ClientApp`` see any
runtime config. This launcher is therefore the SINGLE correct entry
point for cross-device and paper-compat runs.

Examples
--------
    # Cross-device BPR-MF baseline (N=6040):
    python scripts/run.py baseline benchmark_cross_device

    # PFedRec paper reproduction (SGD lr=0.1, 100 rounds, N=6040):
    python scripts/run.py pfedrec paper_compat_pfedrec --run-config "run-seed=42"

    # Legacy cross-silo reproduction (N=5):
    python scripts/run.py adaptive cross_silo_legacy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Sequence


# ============================================================================
# Module -> app-directory map
# ----------------------------------------------------------------------------
# The four Flower apps live at the repo root. This launcher resolves CLI
# aliases (baseline/personalized/adaptive/pfedrec) to the canonical
# federated-*-cf directory that ``flwr run`` expects.
# ============================================================================

MODULE_DIR = {
    "baseline": "federated-baseline-cf",
    "personalized": "federated-personalized-cf",
    "adaptive": "federated-adaptive-personalized-cf",
    "pfedrec": "federated-pfedrec",
}


# ============================================================================
# Mode -> num-supernodes map (CR-2)
# ----------------------------------------------------------------------------
# Locked at the federation level. The in-app ``mode`` assertion (see
# ``fedrec_foundation.mode.assert_benchmark_one_user_per_client``) verifies
# the launcher got this right: mismatched launch (app-mode != launcher
# num-supernodes) raises AssertionError on the first round.
# ============================================================================

MODE_NUM_SUPERNODES = {
    "benchmark_cross_device": 6040,
    "paper_compat_pfedrec": 6040,
    "cross_silo_legacy": 5,
}


def _build_run_config(mode: str, extra_kv: Sequence[str]) -> str:
    """Build a single space-separated key=value string for ``--run-config``.

    Always includes ``num-supernodes`` (from the mode table) and ``mode``
    (so the in-app assertion can verify the launcher agreed).

    Parameters
    ----------
    mode : str
        Mode identifier (key of :data:`MODE_NUM_SUPERNODES`).
    extra_kv : Sequence[str]
        User-supplied ``KEY=VAL`` strings (from ``--run-config`` flags).

    Returns
    -------
    str
        Space-separated ``key=value key2=value2 ...`` string ready for
        ``flwr run --run-config``.

    Raises
    ------
    SystemExit
        If any element of ``extra_kv`` is missing the ``=`` separator.
    """
    base = {
        "num-supernodes": str(MODE_NUM_SUPERNODES[mode]),
        "mode": mode,
    }
    for item in extra_kv:
        if "=" not in item:
            raise SystemExit(f"--run-config expects KEY=VAL pairs; got {item!r}")
        k, _, v = item.partition("=")
        base[k] = v
    return " ".join(f"{k}={v}" for k, v in base.items())


def main(argv: List[str]) -> int:
    """Entry point.

    Parameters
    ----------
    argv : List[str]
        CLI args (excluding ``sys.argv[0]``).

    Returns
    -------
    int
        Subprocess return code, or 0 on successful ``--dry-run``.
    """
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("module", choices=sorted(MODULE_DIR.keys()))
    parser.add_argument("mode", choices=sorted(MODE_NUM_SUPERNODES.keys()))
    parser.add_argument(
        "--run-config",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="extra Flower run_config override",
    )
    parser.add_argument(
        "--federation",
        default=None,
        help=(
            "Flower federation name (e.g. 'local-sim-gpu' or 'local-simulation'). "
            "When omitted, flwr uses the module's pyproject [tool.flwr.federations] default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the flwr command without executing (for tests/CI)",
    )
    args = parser.parse_args(argv)

    module_dir = MODULE_DIR[args.module]
    run_config = _build_run_config(args.mode, args.run_config)

    cmd = ["flwr", "run", f"./{module_dir}"]
    if args.federation is not None:
        cmd.extend(["--federation", args.federation])
    cmd.extend(["--run-config", run_config])
    print(
        f"[launcher] invoking: "
        f"{' '.join(repr(x) if ' ' in x else x for x in cmd)}"
    )
    if args.dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
