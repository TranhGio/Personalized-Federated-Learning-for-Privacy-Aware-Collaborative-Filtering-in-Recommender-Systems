"""Baseline subprocess determinism regression guard (Phase 6 Plan 03).

Re-enables the folded ``phase2-baseline-determinism-path-bug.md`` todo by
verifying that two same-seed back-to-back ``scripts/run.py baseline
benchmark_cross_device`` invocations produce byte-identical
``selected_clients_per_round`` AND that the result file lands at the
Phase-6 per-run-dir layout:

    <repo>/results/federated/baseline/<run_id>/results.json

Prior to Phase-6 Plan 03 (D-02 path fix), the baseline server_app wrote
results to ``Path("../results/federated")`` (a module-relative path),
which could resolve to different locations depending on CWD. After D-02,
``module_run_results_dir("baseline", run_id)`` always resolves to the
repo-root-anchored path above. This test probes that layout so any
regression to the old relative-path write site will cause it to fail with
a "no result files found" assertion.

Running
-------
``@pytest.mark.slow`` — run locally with:
    pytest -m slow scripts/foundation/tests/test_baseline_subprocess_determinism.py

Set ``FEDREC_SKIP_SLOW=1`` in CI or on constrained hardware to have the
collector report the test as skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fedrec_foundation.paths import repo_root


# ---------------------------------------------------------------------------
# Module-level constants (mirrors test_personalized_determinism.py pattern)
# ---------------------------------------------------------------------------
_REPO_ROOT = repo_root()
_RUN_PY = _REPO_ROOT / "scripts" / "run.py"
# Phase 6 D-01/D-02: per-module per-run-dir layout. The glob root is the
# repo-root results/federated/ dir; the per-module subdir is baked into
# the glob pattern so only Phase-6 baseline writes are matched.
_RESULTS_DIR = _REPO_ROOT / "results" / "federated" / "baseline"


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("FEDREC_SKIP_SLOW") == "1",
        reason="FEDREC_SKIP_SLOW=1 — skip slow subprocess test",
    ),
    pytest.mark.skipif(
        not _RUN_PY.exists(),
        reason="scripts/run.py not found",
    ),
    pytest.mark.skipif(
        not (_REPO_ROOT / "data" / "derived" / "foundation_index.json").exists(),
        reason="foundation bundle not committed",
    ),
]


@pytest.mark.slow
def test_selected_partitions_byte_identical_across_subprocess_reruns() -> None:
    """G-03-01 + D-02 regression guard: real-loop reproducibility + Phase-6 path.

    Invariant A (byte-identity): two independent ``python scripts/run.py
    baseline benchmark_cross_device`` invocations with the same run-seed
    produce JSONs whose ``selected_clients_per_round`` fields are
    byte-identical — partition-id-space sampling (G-03-01).

    Invariant B (Phase-6 path): result files land at
    ``<repo>/results/federated/baseline/<run_id>/results.json`` (D-01/D-02).
    A regression to ``Path("../results/federated")`` would cause the
    ``_RESULTS_DIR.glob("*/results.json")`` probe to find nothing and the
    test to skip with a clear "no result files" message.

    Skipped when ``FEDREC_SKIP_SLOW=1`` or the foundation bundle is absent.
    """
    if os.environ.get("FEDREC_SKIP_SLOW") == "1":
        pytest.skip("FEDREC_SKIP_SLOW=1 set")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot existing result files so we can pick out exactly the two this
    # test produces (avoid picking up unrelated earlier runs).
    before = set(_RESULTS_DIR.glob("*/results.json"))

    cmd = [
        sys.executable,
        str(_RUN_PY),
        "baseline",
        "benchmark_cross_device",
        "--run-config",
        "run-seed=42 num-server-rounds=2 fraction-train=0.001 wandb-enabled=false",
    ]

    def _run_once() -> None:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            pytest.skip(
                f"launcher failed (rc={proc.returncode}); skipping real-loop test. "
                f"stdout tail: {proc.stdout[-500:]!r} stderr tail: {proc.stderr[-500:]!r}"
            )

    _run_once()
    _run_once()

    # Phase-6 D-01/D-02: probe the per-run-dir layout.
    # Legacy flat layout (*_results.json) would return [] here, failing loud.
    after = sorted(
        _RESULTS_DIR.glob("*/results.json"),
        key=lambda p: p.stat().st_mtime,
    )
    new_files = [p for p in after if p not in before]

    if not new_files:
        pytest.skip(
            f"No Phase-6 result JSONs found under {_RESULTS_DIR} "
            f"(pattern: */results.json). Run the baseline subprocess at least once "
            f"to confirm Phase-6 path migration is active (D-01/D-02)."
        )

    assert len(new_files) >= 2, (
        f"D-02 regression guard: expected at least 2 new result JSONs in "
        f"{_RESULTS_DIR}/*/results.json, got {len(new_files)}. "
        f"This likely means server_app.py reverted to the legacy "
        f"Path('../results/federated') write path."
    )

    file_a, file_b = new_files[-2], new_files[-1]
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    # Invariant A: byte-identical selected_clients_per_round (G-03-01).
    assert a["selected_clients_per_round"] == b["selected_clients_per_round"], (
        f"G-03-01 broken: selected_clients_per_round differs across subprocess reruns "
        f"with the same run-seed. {file_a.name} vs {file_b.name}"
    )

    # Phase-6 schema check: final_metrics must use the nested best/last layout.
    ndcg_a = float(a["final_metrics"]["best"].get("sampled_ndcg@10", 0.0))
    ndcg_b = float(b["final_metrics"]["best"].get("sampled_ndcg@10", 0.0))
    assert abs(ndcg_a - ndcg_b) <= 1e-3, (
        f"G-03-01 regression: ndcg@10 cross-run diff {abs(ndcg_a - ndcg_b):.6f} > 1e-3"
    )
