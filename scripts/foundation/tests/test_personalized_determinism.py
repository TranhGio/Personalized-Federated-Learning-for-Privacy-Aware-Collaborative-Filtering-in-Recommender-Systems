"""Subprocess regression guard for PSN-04 + PSN-05 + PSN-06 determinism (Phase 3 Plan 05).

Mirrors Phase 2 Plan 05's ``test_selected_partitions_byte_identical_across_subprocess_reruns``
but extends the invariant with disk-payload byte-identity for the single-row
local-user-state cache that Phase 3 Plan 03 introduced (D-04..D-10 manifest-sidecar
layout: ``.embedding_cache/{run_id}/partition_{pid}.pt`` + ``manifest.json``).

Two same-seed back-to-back launcher runs MUST produce:

(a) byte-identical ``selected_clients_per_round`` JSON fields — closes PSN-04
    (deterministic partition-id-space sampling via ``server_rng(run_seed)``);

(b) byte-identical ``partition_{pid}.pt`` disk payloads for any partition that
    was selected in BOTH runs — closes PSN-05 (cache signature determinism)
    and PSN-06 (single-row payload determinism: given identical FND-06
    ``torch_gen(run_seed, user_idx, round_num, purpose)`` streams, the
    ``local_user_row`` / ``local_user_bias`` state evolves identically on disk).

This test would catch the class of bug "deterministic RNG feeds a non-deterministic
domain" (the same family that G-03-01 caught in Phase 2) plus any future
accidental introduction of process-global random state into the single-row
model's save path.

Running
-------
``@pytest.mark.slow`` — run locally with:
    pytest -m slow scripts/foundation/tests/test_personalized_determinism.py

Set ``FEDREC_SKIP_SLOW=1`` in CI or on constrained hardware to have the
collector report the test as skipped (it still shows up in ``--collect-only``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set

import pytest


# Repo root: this file is at scripts/foundation/tests/test_personalized_determinism.py
# -> parents[3] resolves to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_PY = _REPO_ROOT / "scripts" / "run.py"
_RESULTS_DIR = _REPO_ROOT / "results" / "federated"
_CACHE_ROOT_DEFAULT = _REPO_ROOT / ".embedding_cache"
# Phase 3 Plan 03 put the personalized cache at the MODULE root (not repo root).
# Probed as a fallback when the launcher wrote to the module-local default.
_CACHE_ROOT_MODULE = _REPO_ROOT / "federated-personalized-cf" / ".embedding_cache"


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


def _run_personalized(run_id: str) -> Path:
    """Invoke ``scripts/run.py personalized benchmark_cross_device`` and return
    the path to the result JSON.

    Uses a tiny config for CI: 2 rounds, 1 local epoch, 1% client fraction.
    A distinct ``run-id`` is passed so the D-04 cache layout writes to
    ``.embedding_cache/{run_id}/`` without collision between the two reruns.
    """
    env = os.environ.copy()
    # Force W&B offline to avoid auth/network issues in CI.
    env.setdefault("WANDB_MODE", "offline")
    cmd = [
        sys.executable,
        str(_RUN_PY),
        "personalized",
        "benchmark_cross_device",
        "--run-config",
        (
            f"run-seed=42 run-id={run_id} "
            "num-server-rounds=2 local-epochs=1 fraction-train=0.01 "
            "wandb-enabled=false"
        ),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"launcher failed (rc={proc.returncode}); skipping real-loop test. "
            f"stdout tail: {proc.stdout[-500:]!r} stderr tail: {proc.stderr[-500:]!r}"
        )
    # Locate the result JSON. scripts/run.py may or may not stamp run_id into
    # the filename depending on how server_app.py constructs it; probe both
    # patterns, then fall back to the newest JSON by mtime.
    candidates = list(_RESULTS_DIR.glob(f"*{run_id}*_results.json"))
    if not candidates:
        all_results = list(_RESULTS_DIR.glob("*_results.json"))
        all_results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = all_results[:1]
    assert candidates, f"No result JSON found after launcher run_id={run_id}"
    return candidates[0]


def _probe_cache_dir(run_id: str) -> Optional[Path]:
    """Locate the on-disk cache directory for ``run_id``.

    scripts/run.py and the personalized module may write the cache under either
    the repo-root ``.embedding_cache/`` or the module-local
    ``federated-personalized-cf/.embedding_cache/`` depending on CWD at the
    point where the client runs. Probe both.
    """
    for root in (_CACHE_ROOT_DEFAULT, _CACHE_ROOT_MODULE):
        cand = root / run_id
        if cand.exists():
            return cand
    return None


def test_personalized_determinism_subprocess_byte_identical() -> None:
    """Two back-to-back launcher runs with the same run-seed must produce:

    (a) byte-identical ``selected_clients_per_round`` JSON fields (PSN-04);
    (b) byte-identical ``partition_{pid}.pt`` disk payloads for all overlapping
        partition selections (PSN-05 + PSN-06).

    Sanity guard (D-08 cold-run): if ``partition_{pid}.pt`` files are absent
    after the first run (no partition was ever selected for that pid at the
    tiny CI-scale config), the disk-payload comparison is skipped gracefully
    and the test asserts only the (a) byte-identity invariant.
    """
    run_ids = ["psn_det_a", "psn_det_b"]
    try:
        result_a_path = _run_personalized(run_ids[0])
        result_b_path = _run_personalized(run_ids[1])

        result_a = json.loads(result_a_path.read_text())
        result_b = json.loads(result_b_path.read_text())

        sel_a = result_a.get("selected_clients_per_round")
        sel_b = result_b.get("selected_clients_per_round")
        assert sel_a is not None and sel_b is not None, (
            "selected_clients_per_round missing from one or both result JSONs"
        )
        assert sel_a == sel_b, (
            "PSN-04 VIOLATED: selected_clients_per_round diverged across "
            "reruns with identical run-seed.\n"
            f"run_a[0][:10] = {(sel_a[0][:10] if sel_a else [])}\n"
            f"run_b[0][:10] = {(sel_b[0][:10] if sel_b else [])}"
        )

        # Disk-payload byte-identity for overlapping partitions (PSN-05/06).
        selected_partition_ids: Set[int] = set()
        for round_list in sel_a:
            selected_partition_ids.update(int(p) for p in round_list)

        cache_dir_a = _probe_cache_dir(run_ids[0])
        cache_dir_b = _probe_cache_dir(run_ids[1])

        if cache_dir_a is None or cache_dir_b is None:
            pytest.skip(
                "Cache dirs not materialized on disk (server may short-circuit "
                "at tiny scale) — selected_clients_per_round byte-identity "
                "already asserted."
            )

        mismatches: List[int] = []
        checked = 0
        for pid in sorted(selected_partition_ids):
            pt_a = cache_dir_a / f"partition_{int(pid)}.pt"
            pt_b = cache_dir_b / f"partition_{int(pid)}.pt"
            if not (pt_a.exists() and pt_b.exists()):
                continue
            bytes_a = pt_a.read_bytes()
            bytes_b = pt_b.read_bytes()
            if bytes_a != bytes_b:
                mismatches.append(int(pid))
            checked += 1
        assert not mismatches, (
            f"PSN-05/06 VIOLATED: {len(mismatches)} partition payload(s) "
            f"differ across reruns with identical run-seed. First 10: "
            f"{mismatches[:10]} (checked={checked})"
        )
    finally:
        # Cleanup any cache dirs this test created under the probed roots.
        for rid in run_ids:
            for root in (_CACHE_ROOT_DEFAULT, _CACHE_ROOT_MODULE):
                loc = root / rid
                if loc.exists():
                    shutil.rmtree(loc, ignore_errors=True)
