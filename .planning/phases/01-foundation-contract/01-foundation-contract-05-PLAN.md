---
phase: 01-foundation-contract
plan: 05
type: execute
wave: 2
depends_on: [01-foundation-contract-01, 01-foundation-contract-03]
files_modified:
  - scripts/foundation/fedrec_foundation/mode.py
  - scripts/run.py
  - scripts/foundation/tests/test_mode.py
  - scripts/foundation/tests/test_launcher.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "`resolve_mode_defaults(mode)` returns a complete `ModeProfile` dataclass for each of `benchmark_cross_device`, `paper_compat_pfedrec`, `cross_silo_legacy` (D-06, D-07)."
    - "`log_mode_and_overrides(mode, profile, run_config)` prints a `[MODE OVERRIDE]` warning for every run_config key that deviates from the mode default AND handles kebab→snake conversion (D-10, Pitfall 6)."
    - "`cross_silo_legacy` sets `assert_one_user_per_client=False`; `benchmark_cross_device` and `paper_compat_pfedrec` set it to `True` (D-11 + Pitfall 8)."
    - "`scripts/run.py` launcher (CR-2) takes `(module, mode)` and invokes `flwr run` with the correct `--federation` value so that `num-supernodes` is set OUTSIDE the app (benchmark=6040, cross_silo_legacy=5)."
    - "App-level assertion catches any mismatch between the launcher-set `num-supernodes` and the `mode` declared in the client's run_config."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/mode.py"
      provides: "ModeProfile dataclass, resolve_mode_defaults, log_mode_and_overrides, assert_benchmark_one_user_per_client helper"
      exports: ["ModeProfile", "resolve_mode_defaults", "log_mode_and_overrides", "assert_benchmark_one_user_per_client", "MODE_NAMES"]
    - path: "scripts/run.py"
      provides: "CR-2 launcher: `python scripts/run.py <module> <mode> [--run-config KEY=VAL ...]`"
  key_links:
    - from: "scripts/foundation/fedrec_foundation/mode.py::log_mode_and_overrides"
      to: "kebab→snake conversion (Pitfall 6)"
      via: "`key.replace('-', '_')` before hasattr check"
      pattern: "replace\\([\"']-[\"'], [\"']_[\"']\\)"
    - from: "scripts/run.py"
      to: "flwr run --federation {benchmark|cross_silo_legacy}"
      via: "launcher maps mode → federation → num-supernodes"
      pattern: "num-supernodes"
---

<objective>
Implement the `mode` resolver (D-06..D-11) AND a standalone launcher `scripts/run.py` (Codex CR-2) that sets `num-supernodes` at the Flower federation level — which `Context.run_config` cannot do from inside the app. The launcher is the only path that can lock `num-supernodes=6040` for benchmark mode and `num-supernodes=5` for `cross_silo_legacy`; the in-app `mode` selector becomes an assertion layer that verifies the launcher got it right.

Purpose: CR-2 flagged that `Context.run_config` is app-level and `num-supernodes` is federation-level, so no amount of `resolve_mode_defaults(mode)` inside `server_app.py` can actually change the supernode count. Phase 1 must provide a launcher that sits OUTSIDE the Flower app and invokes `flwr run . --federation {...}` with the correct federation-level options, plus an in-app assertion that fails loudly on mismatch.

Output: Two fully-implemented modules (`mode.py`, `scripts/run.py`) + two test files flipped GREEN (`test_mode.py`, `test_launcher.py`). Plans 2–5 will use `resolve_mode_defaults` inside `server_app.py`/`client_app.py` to read the mode's non-supernode fields; they also inherit the `scripts/run.py` launcher for starting runs.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-foundation-contract/01-CONTEXT.md
@.planning/phases/01-foundation-contract/01-RESEARCH.md
@.planning/phases/01-foundation-contract/01-VALIDATION.md
@CLAUDE.md

<interfaces>
From scripts/foundation/fedrec_foundation/mode.py:
```python
from dataclasses import dataclass
from typing import Dict, Optional

MODE_NAMES = ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy")

@dataclass(frozen=True)
class ModeProfile:
    mode: str
    num_supernodes: int
    partition_mode: str       # "natural" | "dirichlet"
    weight_policy: str        # fedrec_foundation.weight_policy.WeightPolicy values
    primary_evaluator: str    # fedrec_foundation.evaluator.EvalProtocol values
    fraction_train: float
    fraction_eval: float
    num_train_negatives: int
    num_eval_negatives: int
    embedding_dim: int
    optimizer: str            # "adam" | "sgd"
    lr: float
    local_epochs: int
    num_server_rounds: int
    checkpoint_rule: str      # "best_round" | "last_round"
    assert_one_user_per_client: bool

def resolve_mode_defaults(mode: str, module_overrides: Optional[Dict[str, object]] = None) -> ModeProfile: ...
def log_mode_and_overrides(mode: str, profile: ModeProfile, run_config: Dict[str, object]) -> Dict[str, object]: ...
def assert_benchmark_one_user_per_client(profile: ModeProfile, num_users_in_client: int, overrides: Dict[str, object]) -> None: ...
```

scripts/run.py (CR-2 launcher, new file at REPO ROOT, not in scripts/foundation/):
```python
# Usage:
#   python scripts/run.py baseline benchmark_cross_device
#   python scripts/run.py pfedrec paper_compat_pfedrec --run-config "run-seed=42"
#   python scripts/run.py adaptive cross_silo_legacy
#
# Resolves (module, mode) -> (flwr federation name, num-supernodes), then
# subprocess.run(["flwr", "run", "./federated-<module>-cf", "--federation", <fed>, ...]).
```
</interfaces>

@federated-baseline-cf/pyproject.toml
@federated-pfedrec/pyproject.toml
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement mode resolver + log_mode_and_overrides (D-06..D-11) + flip test_mode green</name>
  <files>
    scripts/foundation/fedrec_foundation/mode.py
    scripts/foundation/tests/test_mode.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md D-06..D-11 (mode selector LOCKED; override semantics)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 7: Mode Resolver (D-06 to D-11)" (lines 824-986)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §CR-2 (`mode` cannot lock num-supernodes from inside app — app becomes assertion layer)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pitfall 6: mode override-logging misses snake_case vs kebab-case mismatch" (key.replace('-','_'))
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pitfall 8: Benchmark-mode assertion triggers in cross_silo_legacy by accident"
    - CLAUDE.md (dataclass-first, typing.Dict/Optional)
  </read_first>
  <behavior>
    - `ModeProfile` frozen dataclass with 17 fields (see interface above).
    - Three registered profiles:
      - `_BENCHMARK_CROSS_DEVICE`: `num_supernodes=6040`, `partition_mode="natural"`, `weight_policy="num_positives"`, `primary_evaluator="sampled_loo_99"`, `fraction_train=0.1`, `fraction_eval=1.0`, `num_train_negatives=4`, `num_eval_negatives=99`, `embedding_dim=64`, `optimizer="adam"`, `lr=0.001`, `local_epochs=1`, `num_server_rounds=100`, `checkpoint_rule="best_round"`, `assert_one_user_per_client=True`.
      - `_PAPER_COMPAT_PFEDREC`: `num_supernodes=6040`, `partition_mode="natural"`, `weight_policy="num_positives"` (deferred confirmation to PFR-02), `primary_evaluator="sampled_loo_99"`, `fraction_train=1.0`, `fraction_eval=1.0`, `num_train_negatives=4`, `num_eval_negatives=99`, `embedding_dim=32`, `optimizer="sgd"`, `lr=0.1`, `local_epochs=1`, `num_server_rounds=100`, `checkpoint_rule="best_round"`, `assert_one_user_per_client=True`.
      - `_CROSS_SILO_LEGACY`: `num_supernodes=5`, `partition_mode="dirichlet"`, `weight_policy="num_training_examples"`, `primary_evaluator="sampled_loo_99"`, `fraction_train=1.0`, `fraction_eval=1.0`, `num_train_negatives=1`, `num_eval_negatives=99`, `embedding_dim=128`, `optimizer="adam"`, `lr=0.001`, `local_epochs=5`, `num_server_rounds=10`, `checkpoint_rule="last_round"`, `assert_one_user_per_client=False`.
    - `resolve_mode_defaults(mode, module_overrides=None)` looks up profile; applies `dataclasses.replace` if overrides provided; raises `ValueError` on unknown mode.
    - `log_mode_and_overrides(mode, profile, run_config)` converts each run_config kebab key to snake (`weight-policy` → `weight_policy`) and compares to profile; prints `[MODE OVERRIDE]` for each override; returns overrides dict.
    - `assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)` raises `AssertionError` if `profile.assert_one_user_per_client and num_users_in_client != 1 and "num_supernodes" not in overrides`. Overrides disable the assertion (D-10 spirit: visible overrides bypass locks).
    - Tests flip green: `test_override_logging`, `test_assertion_flags`.
  </behavior>
  <action>
Create `scripts/foundation/fedrec_foundation/mode.py` by lifting research Pattern 7 verbatim (lines 830-986) with small adaptations:
- Import `from fedrec_foundation.weight_policy import WeightPolicy` and `from fedrec_foundation.evaluator import EvalProtocol` for enum values.
- Expose `MODE_NAMES = tuple(_REGISTRY.keys())` at module level.
- Add a new helper `assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)` that Phases 2-5 will call inside their `@app.train()`:

```python
def assert_benchmark_one_user_per_client(
    profile: "ModeProfile",
    num_users_in_client: int,
    overrides: Dict[str, object],
) -> None:
    """Phase 2-5 client_app.py assertion entry point.

    Raises AssertionError if profile requires single-user clients AND
    no num-supernodes override is in play AND num_users_in_client != 1.

    Parameters
    ----------
    profile : ModeProfile
    num_users_in_client : int
        The count of distinct users in this client's partition, typically
        ``len(client_partition_df["user_idx"].unique())``.
    overrides : Dict[str, object]
        Return value of ``log_mode_and_overrides(...)`` — if num_supernodes
        was overridden, the assertion is skipped (visible override
        bypasses the lock per D-10).
    """
    if not profile.assert_one_user_per_client:
        return
    if "num_supernodes" in overrides or "num-supernodes" in overrides:
        print(
            f"[MODE] single-user-per-client assertion SKIPPED because "
            f"num-supernodes was overridden (value={num_users_in_client})"
        )
        return
    if num_users_in_client != 1:
        raise AssertionError(
            f"Benchmark mode {profile.mode!r} requires exactly one user per "
            f"client; got {num_users_in_client}. Either run via `scripts/run.py "
            f"<module> {profile.mode}` (which sets num-supernodes=6040) or "
            f"override num-supernodes explicitly."
        )
```

Flip `tests/test_mode.py` to GREEN:
```python
"""Tests for fedrec_foundation.mode (D-06..D-11 + CR-2 + Pitfalls 6, 8)."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import pytest

from fedrec_foundation.mode import (
    MODE_NAMES, ModeProfile,
    resolve_mode_defaults, log_mode_and_overrides,
    assert_benchmark_one_user_per_client,
)


def test_all_three_modes_registered() -> None:
    assert set(MODE_NAMES) == {
        "benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy",
    }


def test_benchmark_profile() -> None:
    p = resolve_mode_defaults("benchmark_cross_device")
    assert p.num_supernodes == 6040
    assert p.partition_mode == "natural"
    assert p.weight_policy == "num_positives"
    assert p.primary_evaluator == "sampled_loo_99"
    assert p.assert_one_user_per_client is True


def test_cross_silo_legacy_profile() -> None:
    p = resolve_mode_defaults("cross_silo_legacy")
    assert p.num_supernodes == 5
    assert p.partition_mode == "dirichlet"
    assert p.assert_one_user_per_client is False


def test_paper_compat_profile() -> None:
    p = resolve_mode_defaults("paper_compat_pfedrec")
    assert p.num_supernodes == 6040
    assert p.embedding_dim == 32
    assert p.optimizer == "sgd"
    assert p.lr == 0.1


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        resolve_mode_defaults("made_up_mode")


def test_module_override() -> None:
    p = resolve_mode_defaults(
        "paper_compat_pfedrec", module_overrides={"weight_policy": "uniform"}
    )
    assert p.weight_policy == "uniform"
    # Other fields unchanged.
    assert p.embedding_dim == 32


def test_override_logging() -> None:
    """Pitfall 6: kebab keys (from run_config) convert to snake before comparison."""
    p = resolve_mode_defaults("benchmark_cross_device")
    run_config = {
        "weight-policy": "uniform",          # kebab override of snake field
        "num-server-rounds": 50,             # different value -> override
        "embedding-dim": 64,                 # same value as default -> NOT an override
    }
    buf = StringIO()
    with redirect_stdout(buf):
        overrides = log_mode_and_overrides("benchmark_cross_device", p, run_config)
    assert "weight_policy" in overrides
    assert overrides["weight_policy"] == "uniform"
    assert "num_server_rounds" in overrides
    assert overrides["num_server_rounds"] == 50
    assert "embedding_dim" not in overrides  # same as default
    # Stdout carries the loud warning prefix.
    out = buf.getvalue()
    assert "[MODE OVERRIDE]" in out


def test_assertion_flags_benchmark() -> None:
    p = resolve_mode_defaults("benchmark_cross_device")
    # One user: passes.
    assert_benchmark_one_user_per_client(p, num_users_in_client=1, overrides={})
    # More than one: raises.
    with pytest.raises(AssertionError, match="exactly one user"):
        assert_benchmark_one_user_per_client(p, num_users_in_client=5, overrides={})


def test_assertion_flags_cross_silo_legacy_skipped() -> None:
    """Pitfall 8: cross_silo_legacy must NOT trigger the one-user assertion."""
    p = resolve_mode_defaults("cross_silo_legacy")
    # 1200 users in client -> no error because assert_one_user_per_client=False.
    assert_benchmark_one_user_per_client(p, num_users_in_client=1200, overrides={})


def test_assertion_skipped_on_override() -> None:
    """D-10: explicit override bypasses the lock (and emits a visible skip log)."""
    p = resolve_mode_defaults("benchmark_cross_device")
    # Simulate the override dict returned by log_mode_and_overrides.
    overrides = {"num_supernodes": 10}
    assert_benchmark_one_user_per_client(p, num_users_in_client=604, overrides=overrides)
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_mode.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/mode.py` defines `ModeProfile`, `resolve_mode_defaults`, `log_mode_and_overrides`, `assert_benchmark_one_user_per_client`, `MODE_NAMES`.
    - `grep -E "replace\\([\"']-[\"'], ?[\"']_[\"']\\)" scripts/foundation/fedrec_foundation/mode.py` matches (kebab→snake).
    - `grep "num_supernodes=6040" scripts/foundation/fedrec_foundation/mode.py` matches (benchmark profile).
    - `grep "num_supernodes=5" scripts/foundation/fedrec_foundation/mode.py` matches (cross-silo legacy profile).
    - `grep "assert_one_user_per_client=False" scripts/foundation/fedrec_foundation/mode.py` matches (legacy profile).
    - `cd scripts/foundation && pytest tests/test_mode.py -v` prints 10+ passed.
  </acceptance_criteria>
  <done>Mode resolver with three registered profiles, override logging (kebab→snake), and single-user assertion helper all implemented.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement CR-2 launcher scripts/run.py + flip test_launcher green</name>
  <files>
    scripts/run.py
    scripts/foundation/tests/test_launcher.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §CR-2 (launcher pattern: shell wrapper or Python — we're doing Python)
    - .planning/phases/01-foundation-contract/01-VALIDATION.md (test row `test_launcher_sets_num_supernodes`)
    - federated-baseline-cf/pyproject.toml (shows `[tool.flwr.federations.local-simulation]` with `options.num-supernodes = 5` — the launcher must PASS IN an alternate federation or override this value)
    - CLAUDE.md (Python 3.9+ typing, type hints)
  </read_first>
  <behavior>
    - `scripts/run.py` is a Python launcher at the REPO ROOT (not inside scripts/foundation/ — so it can live with the other orchestration scripts).
    - CLI: `python scripts/run.py <module> <mode> [--run-config KEY=VAL]...`.
    - Module → directory map: `baseline -> ./federated-baseline-cf`, `personalized -> ./federated-personalized-cf`, `adaptive -> ./federated-adaptive-personalized-cf`, `pfedrec -> ./federated-pfedrec`.
    - Mode → flwr-run-arguments map:
      - `benchmark_cross_device` and `paper_compat_pfedrec`: `--federation local-simulation` AND `--run-config "num-supernodes=6040 mode=<mode>"` — overriding the default `5` from pyproject.toml.
      - `cross_silo_legacy`: `--federation local-simulation --run-config "num-supernodes=5 mode=cross_silo_legacy"`.
    - Launcher appends user's `--run-config KEY=VAL` pairs to the single Flower `--run-config` string.
    - Launcher prints the full `flwr run` command before invoking, for auditability.
    - Exits with the subprocess's return code.
    - Tests flip green: `test_launcher_sets_num_supernodes` — use `subprocess.run([sys.executable, "scripts/run.py", ...], check=False)` with a flag that causes the launcher to DRY-RUN (print the `flwr run` command to stdout without executing); assert stdout contains `num-supernodes=6040` for benchmark mode, `num-supernodes=5` for cross-silo.
  </behavior>
  <action>
Create `scripts/run.py` at the repo root:
```python
#!/usr/bin/env python
"""Launcher for Flower app runs with mode-locked federation-level config (Codex CR-2).

Usage
-----
    python scripts/run.py <module> <mode> [--run-config KEY=VAL ...]
    python scripts/run.py --dry-run <module> <mode>

Modules: baseline, personalized, adaptive, pfedrec
Modes:   benchmark_cross_device, paper_compat_pfedrec, cross_silo_legacy

The mode selector in each module's pyproject.toml is an app-level
assertion. `num-supernodes` cannot be set from inside a Flower app
via Context.run_config — it must be passed to the `flwr run`
invocation. This launcher is therefore the SINGLE correct entry
point for cross-device and paper-compat runs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

MODULE_DIR = {
    "baseline": "federated-baseline-cf",
    "personalized": "federated-personalized-cf",
    "adaptive": "federated-adaptive-personalized-cf",
    "pfedrec": "federated-pfedrec",
}

MODE_NUM_SUPERNODES = {
    "benchmark_cross_device": 6040,
    "paper_compat_pfedrec": 6040,
    "cross_silo_legacy": 5,
}


def _build_run_config(mode: str, extra_kv: Sequence[str]) -> str:
    """Build a single space-separated key=value string for --run-config.

    Always includes `num-supernodes` (from the mode table) and `mode`
    (so the in-app assertion can verify the launcher agreed).
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
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("module", choices=sorted(MODULE_DIR.keys()))
    parser.add_argument("mode", choices=sorted(MODE_NUM_SUPERNODES.keys()))
    parser.add_argument(
        "--run-config", action="append", default=[],
        metavar="KEY=VAL", help="extra Flower run_config override",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the flwr command without executing (for tests/CI)",
    )
    args = parser.parse_args(argv)

    module_dir = MODULE_DIR[args.module]
    run_config = _build_run_config(args.mode, args.run_config)

    cmd = [
        "flwr", "run", f"./{module_dir}",
        "--federation", "local-simulation",
        "--run-config", run_config,
    ]
    print(f"[launcher] invoking: {' '.join(repr(x) if ' ' in x else x for x in cmd)}")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

Flip `tests/test_launcher.py` to GREEN:
```python
"""Tests for scripts/run.py launcher (Codex CR-2)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _launcher_path() -> Path:
    """Locate scripts/run.py at the repo root."""
    # This test file lives at scripts/foundation/tests/test_launcher.py;
    # repo root is three parents up.
    return Path(__file__).resolve().parents[3] / "scripts" / "run.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_launcher_path()), *args],
        capture_output=True, text=True, check=False,
    )


def test_launcher_exists() -> None:
    assert _launcher_path().exists(), f"{_launcher_path()} missing"


def test_launcher_sets_num_supernodes_benchmark() -> None:
    r = _run("--dry-run", "baseline", "benchmark_cross_device")
    assert r.returncode == 0, r.stderr
    assert "num-supernodes=6040" in r.stdout
    assert "mode=benchmark_cross_device" in r.stdout
    assert "federated-baseline-cf" in r.stdout


def test_launcher_sets_num_supernodes_cross_silo_legacy() -> None:
    r = _run("--dry-run", "pfedrec", "cross_silo_legacy")
    assert r.returncode == 0, r.stderr
    assert "num-supernodes=5" in r.stdout
    assert "mode=cross_silo_legacy" in r.stdout
    assert "federated-pfedrec" in r.stdout


def test_launcher_paper_compat_pfedrec() -> None:
    r = _run("--dry-run", "pfedrec", "paper_compat_pfedrec")
    assert r.returncode == 0, r.stderr
    assert "num-supernodes=6040" in r.stdout
    assert "mode=paper_compat_pfedrec" in r.stdout


def test_launcher_passes_extra_run_config() -> None:
    r = _run(
        "--dry-run", "adaptive", "benchmark_cross_device",
        "--run-config", "run-seed=999", "--run-config", "lr=0.005",
    )
    assert r.returncode == 0, r.stderr
    assert "run-seed=999" in r.stdout
    assert "lr=0.005" in r.stdout
    assert "num-supernodes=6040" in r.stdout


def test_launcher_unknown_mode_rejected() -> None:
    r = _run("--dry-run", "baseline", "not_a_mode")
    assert r.returncode != 0
    assert "invalid choice" in r.stderr.lower() or "invalid" in r.stderr.lower()


def test_launcher_malformed_run_config_rejected() -> None:
    r = _run(
        "--dry-run", "baseline", "benchmark_cross_device",
        "--run-config", "no_equals_sign",
    )
    assert r.returncode != 0
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_launcher.py -v &amp;&amp; python ../../scripts/run.py --dry-run baseline benchmark_cross_device</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/run.py` exists at the repo root.
    - `grep "num-supernodes" scripts/run.py` matches.
    - `grep "MODULE_DIR" scripts/run.py` matches.
    - `grep "dry-run" scripts/run.py` matches.
    - Running `python scripts/run.py --dry-run baseline benchmark_cross_device` exits 0 AND stdout contains `num-supernodes=6040` AND `mode=benchmark_cross_device` AND `federated-baseline-cf`.
    - Running `python scripts/run.py --dry-run pfedrec cross_silo_legacy` stdout contains `num-supernodes=5`.
    - `cd scripts/foundation && pytest tests/test_launcher.py -v` prints 6+ passed.
  </acceptance_criteria>
  <done>CR-2 launcher in place; federation-level num-supernodes is set OUTSIDE the Flower app; in-app assertion (from Task 1) can verify the launcher got it right.</done>
</task>

</tasks>

<verification>
- `cd scripts/foundation && pytest tests/test_mode.py tests/test_launcher.py -v` — all pass.
- `python scripts/run.py --dry-run baseline benchmark_cross_device` prints a line containing `num-supernodes=6040 mode=benchmark_cross_device`.
- `python scripts/run.py --dry-run pfedrec cross_silo_legacy` prints a line containing `num-supernodes=5`.
- `python -c "from fedrec_foundation.mode import resolve_mode_defaults, assert_benchmark_one_user_per_client; p = resolve_mode_defaults('benchmark_cross_device'); print(p.num_supernodes)"` prints `6040`.
</verification>

<success_criteria>
- `resolve_mode_defaults(mode)` exposes three complete profiles per D-06..D-11.
- `log_mode_and_overrides` handles kebab→snake conversion (Pitfall 6) and prints loud `[MODE OVERRIDE]` warnings.
- `scripts/run.py` launcher is the canonical entry point for cross-device and paper-compat runs; it sets `num-supernodes` at the flwr-run level, outside the app (CR-2).
- `assert_benchmark_one_user_per_client` is ready for Phase 2-5 client_app.py integration; skips on `cross_silo_legacy` (Pitfall 8) and on visible override (D-10).
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation-contract/01-foundation-contract-05-SUMMARY.md` — document the three mode profiles' exact defaults, the `scripts/run.py` usage, and note that Phases 2-5's `server_app.py` must call `log_mode_and_overrides` at startup (to populate the manifest's `overrides` field) and `client_app.py` must call `assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)` inside `@app.train()`.
</output>
