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
import re
import subprocess
import sys
from typing import List, Sequence

# Mirrors flwr.common.config.parse_config_args regex: matches space-separated
# KEY=VAL pairs where VAL may be single-quoted, double-quoted, or bare.
_PAIR_RE = re.compile(r"(\S+?)=(\'[^\']*\'|\"[^\"]*\"|\S+)")


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
    "thesis_crossdevice_main": 6040,  # Phase 7 D-04
    "paper_compat_pfedrec": 6040,
    "cross_silo_legacy": 5,
}


def _quote_value_for_flwr(v: str) -> str:
    """Return ``v`` in a form ``flwr run --run-config`` accepts.

    Flower's ``parse_config_args`` (flwr/common/config.py) rebuilds the
    ``--run-config`` string into TOML before parsing with ``tomli``. Bare-word
    values like ``benchmark_cross_device`` or ``bpr`` are not valid TOML values,
    so we must quote non-numeric / non-boolean strings. Values that are already
    quoted, numeric, or boolean pass through unchanged.
    """
    if not v:
        return '""'
    if (v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'"):
        return v
    try:
        float(v)
        return v
    except ValueError:
        pass
    if v.lower() in ("true", "false"):
        return v.lower()
    return f'"{v}"'


def _build_run_config(mode: str, extra_kv: Sequence[str]) -> str:
    """Build a single space-separated key=value string for ``--run-config``.

    Always includes ``num-supernodes`` (from the mode table) and ``mode``
    (so the in-app assertion can verify the launcher agreed).

    String values are auto-quoted so they survive Flower's TOML parser.

    Parameters
    ----------
    mode : str
        Mode identifier (key of :data:`MODE_NUM_SUPERNODES`).
    extra_kv : Sequence[str]
        User-supplied ``KEY=VAL`` strings (from ``--run-config`` flags).
        Each element is one KEY=VAL pair; pass multiple pairs by supplying
        ``--run-config`` multiple times on the command line.

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
    # num-supernodes is a FEDERATION-level option (set in pyproject
    # [tool.flwr.federations.<name>] options.num-supernodes), NOT an app
    # run_config key. Emitting it here breaks flwr's fuse_dicts validation
    # ("Key 'num-supernodes' is not present in the main dictionary"). The
    # in-app mode assertion reads num-supernodes from the Flower Context
    # (populated by the federation) and cross-checks it against the mode
    # profile's expected value — no need to duplicate it in run_config.
    base = {
        "mode": mode,
    }
    for item in extra_kv:
        if "=" not in item:
            raise SystemExit(f"--run-config expects KEY=VAL pairs; got {item!r}")
        # Support both single-pair ("a=1") and space-separated multi-pair
        # ("a=1 b=2 c=3") forms in one --run-config flag, matching flwr's own
        # regex semantics.
        matches = _PAIR_RE.findall(item)
        if not matches:
            raise SystemExit(f"--run-config could not parse pairs: {item!r}")
        for k, v in matches:
            base[k] = v
    return " ".join(
        f"{k}={_quote_value_for_flwr(v)}" for k, v in base.items()
    )


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
