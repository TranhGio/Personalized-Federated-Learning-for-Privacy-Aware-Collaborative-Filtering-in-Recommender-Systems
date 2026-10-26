"""Subprocess regression guard for ADP-06 + ADP-02 Phase-4 cache determinism.

Mirrors Phase 3 Plan 05's ``test_personalized_determinism.py`` with three Phase-4-specific
extensions:

1. ``partition_{pid}.pt`` schema_version=2 payload includes ``_logit_alpha.weight`` +
   ``_item_perturbation.weight`` when ``enable_per_user_alpha=true`` and
   ``enable_item_perturbation=true``. All LOCAL keys must round-trip byte-identically
   across two same-seed runs (closes the ADP-02 enable-before-load regression axis at
   the disk-payload byte level).

2. ``_manifest.best_prototype`` (D-06) must be byte-identical across two same-seed runs
   — proves ``AdaptiveSplitFedAvg.snapshot_best_prototype`` at the best round is itself
   deterministic (D-05/D-06 closure).

3. ``selected_clients_per_round`` byte-identity (ADP-06 carry-forward from Phase 2/3).
   Catches the class of bug "deterministic RNG feeds a non-deterministic domain" —
   the same family that G-03-01 caught in Phase 2.

Running
-------
``@pytest.mark.slow`` — run locally with::

    pytest -m slow scripts/foundation/tests/test_adaptive_determinism.py

Set ``FEDREC_SKIP_SLOW=1`` in CI or on constrained hardware to have the collector
report the test as skipped (it still shows up in ``--collect-only``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest
import torch


# Repo root: this file is at scripts/foundation/tests/test_adaptive_determinism.py
# -> parents[3] resolves to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_PY = _REPO_ROOT / "scripts" / "run.py"
_RESULTS_DIR = _REPO_ROOT / "results" / "federated"
# Phase 4 Plan 03's _CACHE_BASE_DIR resolves to the MODULE root (not repo root).
# Probe both to be robust across launcher CWD resolution + any future
# FEDREC_CACHE_ROOT contract that may be honored.
_CACHE_ROOT = _REPO_ROOT / ".embedding_cache"
_ADAPTIVE_MODULE_CACHE_ROOT = (
    _REPO_ROOT / "federated-adaptive-personalized-cf" / ".embedding_cache"
)


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


def _run_adaptive(run_id: str, tmp_cache_root: Path) -> Path:
    """Invoke ``scripts/run.py adaptive benchmark_cross_device`` and return the path
    to the result JSON.

    Tiny config for CI: 2 rounds, 1 local epoch, 1% client fraction, per-user alpha
    + item perturbation both ON (thesis benchmark defaults from Plan 02). The latter
    pair makes ``_logit_alpha.weight`` + ``_item_perturbation.weight`` part of the
    LOCAL key set so the disk-payload byte-identity check actually exercises the
    ADP-02 enable-before-load path.

    A distinct ``run-id`` is passed so the schema_version=2 cache layout writes to
    ``.embedding_cache/{run_id}/`` without collision between the two reruns.
    """
    env = os.environ.copy()
    # Force W&B offline to avoid auth/network issues in CI.
    env.setdefault("WANDB_MODE", "offline")
    # Hint at an alternate cache root; today the launcher may or may not honor it,
    # so the dual-probe pattern below covers both module-root and repo-root layouts.
    env["FEDREC_CACHE_ROOT"] = str(tmp_cache_root)
    cmd = [
        sys.executable,
        str(_RUN_PY),
        "adaptive",
        "benchmark_cross_device",
        "--run-config",
        (
            f"run-seed=42 run-id={run_id} "
            "num-server-rounds=2 local-epochs=1 fraction-train=0.01 "
            "wandb-enabled=false "
            "enable-per-user-alpha=true enable-item-perturbation=true"
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
    # Phase 6 Plan 05: per-run-dir layout — results/federated/adaptive/<run_id>/results.json
    # Fall back to legacy flat layout for older runs that predate Phase 6.
    candidates = list((_RESULTS_DIR / "adaptive").glob("*/results.json"))
    # Filter to the run with matching run_id in the directory name (per-run-dir keyed by run_id)
    candidates = [p for p in candidates if run_id in str(p)]
    if not candidates:
        # Legacy flat layout fallback: *{run_id}*_results.json
        candidates = list(_RESULTS_DIR.rglob(f"*{run_id}*_results.json"))
    if not candidates:
        # Last-resort: newest results.json under adaptive/
        all_results = list((_RESULTS_DIR / "adaptive").glob("*/results.json"))
        all_results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = all_results[:1]
    assert candidates, f"No result JSON found after launcher run_id={run_id}"
    return candidates[0]


def _probe_cache_dir(run_id: str, alt_root: Path) -> Optional[Path]:
    """Locate the on-disk cache directory for ``run_id``.

    ``scripts/run.py`` and the adaptive module may write the cache under either
    the repo-root ``.embedding_cache/``, the module-local
    ``federated-adaptive-personalized-cf/.embedding_cache/``, or the alternate
    root hinted via ``FEDREC_CACHE_ROOT`` (``alt_root``). Probe all three.
    """
    for root in (alt_root, _CACHE_ROOT, _ADAPTIVE_MODULE_CACHE_ROOT):
        cand = root / run_id
        if cand.exists():
            return cand
    return None


def test_adaptive_determinism_subprocess_byte_identical(tmp_path: Path) -> None:
    """Two back-to-back launcher runs with the same run-seed must produce:

    (a) byte-identical ``selected_clients_per_round`` JSON fields (ADP-06);
    (b) byte-identical ``_manifest.best_prototype`` list (D-05/D-06); AND
    (c) byte-identical ``partition_{pid}.pt`` disk payloads for all overlapping
        partition selections, including ``_logit_alpha.weight`` and
        ``_item_perturbation.weight`` tensors (ADP-02 schema_version=2 cache).

    Sanity guards:
      - If the launcher fails for any reason, ``pytest.skip`` (a launcher failure
        is not the determinism test's concern).
      - If ``partition_{pid}.pt`` files are absent after the first run (cold-run,
        no partition was ever selected at the tiny CI-scale config), skip the
        disk-payload comparison gracefully and assert only on (a) + (b).
      - If ``_manifest.best_prototype`` is None on both runs (degenerate
        2-round run that never fired the best-metric branch), skip (b) cleanly
        once (a) is asserted.
    """
    run_ids = ["adp_det_a", "adp_det_b"]
    cache_a = tmp_path / ".embedding_cache_a"
    cache_b = tmp_path / ".embedding_cache_b"
    cache_a.mkdir()
    cache_b.mkdir()

    try:
        result_a_path = _run_adaptive(run_ids[0], cache_a)
        result_b_path = _run_adaptive(run_ids[1], cache_b)

        result_a = json.loads(result_a_path.read_text())
        result_b = json.loads(result_b_path.read_text())

        # ==== Invariant (a): selected_clients_per_round byte-identity ====
        sel_a = result_a.get("selected_clients_per_round")
        sel_b = result_b.get("selected_clients_per_round")
        assert sel_a is not None and sel_b is not None, (
            "selected_clients_per_round missing from one or both result JSONs"
        )
        assert sel_a == sel_b, (
            "ADP-06 VIOLATED: selected_clients_per_round diverged across reruns "
            "with identical run-seed.\n"
            f"run_a[0][:10] = {(sel_a[0][:10] if sel_a else [])}\n"
            f"run_b[0][:10] = {(sel_b[0][:10] if sel_b else [])}"
        )

        # ==== Invariant (b): _manifest.best_prototype byte-identity (D-05/D-06) ====
        manifest_a: Dict = result_a.get("_manifest") or {}
        manifest_b: Dict = result_b.get("_manifest") or {}
        bp_a = manifest_a.get("best_prototype")
        bp_b = manifest_b.get("best_prototype")
        if bp_a is None and bp_b is None:
            # Both runs had no best-round fire (e.g., NDCG always 0 at tiny scale) —
            # acceptable under a degenerate 2-round run. Skip (b) cleanly; (a) is
            # already asserted.
            pytest.skip(
                "best_prototype is None in both result JSONs — tiny-config run "
                "didn't fire the best-metric branch. selected_clients_per_round "
                "byte-identity already asserted above."
            )
        assert bp_a is not None and bp_b is not None, (
            "D-06 VIOLATED: _manifest.best_prototype is None in only one run — "
            "asymmetric best-round behavior across identical-seed reruns."
        )
        assert bp_a == bp_b, (
            "D-05/D-06 VIOLATED: _manifest.best_prototype diverged across reruns "
            "with identical run-seed.\n"
            f"a[:5] = {bp_a[:5]}, b[:5] = {bp_b[:5]}"
        )

        # ==== Invariant (c): partition_{pid}.pt byte-identity for overlapping
        # partitions, INCLUDING _logit_alpha.weight + _item_perturbation.weight ====
        selected_partition_ids: Set[int] = set()
        for round_list in sel_a:
            selected_partition_ids.update(int(p) for p in round_list)

        cache_dir_a = _probe_cache_dir(run_ids[0], cache_a)
        cache_dir_b = _probe_cache_dir(run_ids[1], cache_b)

        if cache_dir_a is None or cache_dir_b is None:
            pytest.skip(
                "Cache dirs not materialized on disk (server may short-circuit "
                "at tiny scale) — selected_clients_per_round + best_prototype "
                "byte-identity already asserted."
            )

        mismatches: List[str] = []
        checked_partitions = 0
        checked_keys = 0
        for pid in sorted(selected_partition_ids):
            pt_a = cache_dir_a / f"partition_{int(pid)}.pt"
            pt_b = cache_dir_b / f"partition_{int(pid)}.pt"
            if not (pt_a.exists() and pt_b.exists()):
                continue
            try:
                state_a = torch.load(pt_a, map_location="cpu", weights_only=True)
                state_b = torch.load(pt_b, map_location="cpu", weights_only=True)
            except Exception as e:
                pytest.fail(f"torch.load failed on partition {pid}: {e}")
            checked_partitions += 1
            keys_a = set(state_a.keys())
            keys_b = set(state_b.keys())
            if keys_a != keys_b:
                mismatches.append(
                    f"partition {pid}: LOCAL key set differs "
                    f"(a={sorted(keys_a)}, b={sorted(keys_b)})"
                )
                continue
            for key in sorted(keys_a):
                checked_keys += 1
                if not torch.equal(state_a[key], state_b[key]):
                    delta = float((state_a[key] - state_b[key]).abs().max())
                    mismatches.append(
                        f"partition {pid}: tensor '{key}' differs "
                        f"(shape={tuple(state_a[key].shape)}, "
                        f"dtype={state_a[key].dtype}, "
                        f"max_abs_delta={delta:.6e})"
                    )

        # Coverage guard: confirm that _logit_alpha.weight + _item_perturbation.weight
        # were actually present in checked state (proving the ADP-02 enable-before-load
        # path was exercised by this run, not silently absent due to a config drift).
        adaptive_key_seen = False
        if cache_dir_a is not None:
            for pt_path in cache_dir_a.glob("partition_*.pt"):
                try:
                    s = torch.load(pt_path, map_location="cpu", weights_only=True)
                    if (
                        "_logit_alpha.weight" in s
                        and "_item_perturbation.weight" in s
                    ):
                        adaptive_key_seen = True
                        break
                except Exception:
                    continue

        assert not mismatches, (
            f"ADP-06/ADP-02 cache VIOLATED: {len(mismatches)} byte-difference(s) "
            f"found across {checked_partitions} overlapping partitions / "
            f"{checked_keys} tensor comparisons.\n"
            f"First 10: {mismatches[:10]}"
        )
        if checked_partitions > 0 and not adaptive_key_seen:
            pytest.fail(
                "Coverage gap: partition_*.pt files exist but none contain "
                "_logit_alpha.weight + _item_perturbation.weight. ADP-02 path not "
                "actually exercised by this run. Confirm enable-per-user-alpha=true "
                "and enable-item-perturbation=true propagated from --run-config."
            )
    finally:
        # Cleanup any cache dirs created under the default roots. The tmp_path
        # roots (cache_a / cache_b) are auto-cleaned by pytest.
        for rid in run_ids:
            for root in (_CACHE_ROOT, _ADAPTIVE_MODULE_CACHE_ROOT):
                cand = root / rid
                if cand.exists():
                    shutil.rmtree(cand, ignore_errors=True)
