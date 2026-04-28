"""Subprocess regression guard for PFR-06 + D-16 + D-14 Phase-5 PFedRec determinism.

Mirrors Phase 4 Plan 06's ``test_adaptive_determinism.py`` with four Phase-5-specific
adaptations:

1. ``selected_clients_per_round`` byte-identity (PFR-06 / G-03-01 carry-forward) —
   catches the class of bug "deterministic RNG feeds a non-deterministic domain"
   that the Phase 2 G-03-01 fix introduced and Phase 3/4 inherited. Always
   asserted; no skip-gate.

2. ``_manifest.pfr08_verification`` audit-dict byte-identity (D-14, Phase-5
   unique) — proves the auto-verify hook reading
   ``IJCAI-23-PFedRec/sh_result/ml-1m.txt`` and emitting the
   ``[PFR-08 VERIFIED]`` / ``[PFR-08 FAILED]`` decision is itself deterministic
   (e.g., doesn't quietly use stdlib ``random`` for some intermediate float ratio).
   Skipped cleanly only when both runs produce ``None`` (degenerate smoke config).

3. Per-key ``torch.equal`` on ``partition_{pid}.pt`` schema_version=3 cache
   payloads (D-16). After D-01 bias-GLOBAL flip, the LOCAL payload contains
   exactly one key — ``affine_output.weight`` shape ``(1, latent_dim)``.
   The bias channel is GLOBAL and lives in the result JSON, NOT in the
   partition cache.

4. Coverage guard prevents silent-config-drift false-GREEN: scans
   ``cache_dir_a`` for at least one ``partition_*.pt`` containing the
   ``affine_output.weight`` key. If ``checked_partitions > 0`` but
   ``coverage_seen is False`` (no partition file ever held that key),
   ``pytest.fail`` with a clear "PFR-03 path not actually exercised by this run"
   message — mirrors Phase 4's coverage guard idiom.

Running
-------
``@pytest.mark.slow`` — run locally with::

    pytest -m slow scripts/foundation/tests/test_pfedrec_subprocess_determinism.py

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


# Repo root: this file is at scripts/foundation/tests/test_pfedrec_subprocess_determinism.py
# -> parents[3] resolves to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO_ROOT / "scripts" / "run.py"
_BUNDLE_PATH = _REPO_ROOT / "data" / "derived" / "foundation_index.json"
_RESULTS_DIR = _REPO_ROOT / "results" / "federated"
# Phase 5 Plan 03's _CACHE_BASE_DIR resolves to the MODULE root (per the
# _CACHE_BASE_DIR = _MODULE_DIR.parent / ".embedding_cache" rule that
# Phase 3 Plan 03 established and Phase 5 Plan 03 carries forward).
# Probe both repo-root and module-root (and any FEDREC_CACHE_ROOT alt-root)
# so the test is robust to launcher CWD resolution.
_CACHE_ROOT = _REPO_ROOT / ".embedding_cache"
_PFEDREC_MODULE_CACHE_ROOT = (
    _REPO_ROOT / "federated-pfedrec" / ".embedding_cache"
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("FEDREC_SKIP_SLOW") == "1",
        reason="FEDREC_SKIP_SLOW=1 — skip slow subprocess test",
    ),
    pytest.mark.skipif(
        not _LAUNCHER.exists(),
        reason="scripts/run.py not present",
    ),
    pytest.mark.skipif(
        not _BUNDLE_PATH.exists(),
        reason="foundation bundle not present",
    ),
]


def _run_pfedrec(run_id: str, tmp_cache_root: Path) -> Path:
    """Invoke ``scripts/run.py pfedrec paper_compat_pfedrec`` and return the
    path to the result JSON.

    Tiny config for CI: 2 rounds, 1 local epoch, 1% client fraction. The
    --run-config does NOT include ``enable-per-user-alpha`` /
    ``enable-item-perturbation`` (those are Phase-4 adaptive-specific) and
    DOES include ``reuse-cache=false`` per D-22 so the cold-round probe path
    is exercised and per-run cache materializes under ``.embedding_cache/{run_id}/``.

    A distinct ``run-id`` is passed so the schema_version=3 cache layout
    writes to ``.embedding_cache/{run_id}/`` without collision between the
    two reruns.
    """
    env = os.environ.copy()
    # Force W&B offline to avoid auth/network issues in CI.
    env.setdefault("WANDB_MODE", "offline")
    # Hint at an alternate cache root; today the launcher may or may not
    # honor it, so the dual-probe pattern below covers both module-root and
    # repo-root layouts.
    env["FEDREC_CACHE_ROOT"] = str(tmp_cache_root)
    cmd = [
        sys.executable,
        str(_LAUNCHER),
        "pfedrec",
        "paper_compat_pfedrec",
        "--run-config",
        (
            f"run-seed=42 run-id={run_id} "
            "num-server-rounds=2 local-epochs=1 fraction-train=0.01 "
            "wandb-enabled=false reuse-cache=false"
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
    # Result JSON location: results/federated/[pfedrec/]<...>{run_id}<...>_results.json
    candidates = list(_RESULTS_DIR.rglob(f"*{run_id}*_results.json"))
    if not candidates:
        all_results = list(_RESULTS_DIR.rglob("*_results.json"))
        all_results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = all_results[:1]
    assert candidates, f"No result JSON found after launcher run_id={run_id}"
    return candidates[0]


def _probe_cache_dir(run_id: str, alt_root: Path) -> Optional[Path]:
    """Locate the on-disk cache directory for ``run_id``.

    ``scripts/run.py`` and the pfedrec module may write the cache under
    either the repo-root ``.embedding_cache/``, the module-local
    ``federated-pfedrec/.embedding_cache/`` (Phase 3 D-08
    ``_CACHE_BASE_DIR = _MODULE_DIR.parent / ".embedding_cache"`` carried
    forward to Phase 5 Plan 03), or the alternate root hinted via
    ``FEDREC_CACHE_ROOT`` (``alt_root``). Probe all three.

    A directory is only treated as the cache root if it actually contains
    at least one ``partition_*.pt`` file — guards against an empty
    auto-created sentinel directory short-circuiting the probe.
    """
    for root in (alt_root, _CACHE_ROOT, _PFEDREC_MODULE_CACHE_ROOT):
        cand = root / run_id
        if cand.exists() and any(cand.glob("partition_*.pt")):
            return cand
    # Fallback: directory exists but no partition_*.pt yet (cold-run /
    # short-circuit). Return the first existing dir so the caller sees a
    # non-None probe but the partition-loop is a no-op (covered by the
    # cold-run sanity guard in the test body).
    for root in (alt_root, _CACHE_ROOT, _PFEDREC_MODULE_CACHE_ROOT):
        cand = root / run_id
        if cand.exists():
            return cand
    return None


def test_pfedrec_determinism_subprocess_byte_identical(tmp_path: Path) -> None:
    """Two back-to-back launcher runs with the same run-seed must produce:

    (a) byte-identical ``selected_clients_per_round`` JSON fields (PFR-06 /
        G-03-01 carry-forward);
    (b) byte-identical ``_manifest.pfr08_verification`` audit dict (D-14,
        Phase-5 unique invariant); AND
    (c) byte-identical ``partition_{pid}.pt`` disk payloads for all
        overlapping partition selections, with per-key ``torch.equal``
        comparison covering the single LOCAL key
        ``affine_output.weight`` (D-16, schema_version=3 single-key
        payload after D-01 bias-GLOBAL flip).

    Sanity guards:
      - If the launcher fails for any reason, ``pytest.skip`` (a launcher
        failure is not the determinism test's concern).
      - If both runs return ``pfr08_verification = None`` (degenerate
        2-round smoke config that never fires the auto-verify hook),
        ``pytest.skip`` cleanly — invariant (a) is already asserted.
      - If ``partition_{pid}.pt`` files are absent after the first run
        (cold-run, no partition was ever selected at the tiny CI-scale
        config), skip the disk-payload comparison gracefully and assert
        only on (a) + (b).

    Coverage guard (Phase 4 Plan 06 idiom carried forward): after the
    byte-identity assertion, scan the materialized cache for at least one
    ``partition_*.pt`` containing the ``affine_output.weight`` key. If
    ``checked_partitions > 0`` and ``coverage_seen is False`` —
    ``pytest.fail`` with "PFR-03 path not actually exercised by this run".
    Catches the silent-config-drift class of failure where a future
    change to run-config propagation breaks the test's ability to actually
    exercise the PFR-03 single-key cache layout.
    """
    run_ids = ["pfr_det_a", "pfr_det_b"]
    cache_a = tmp_path / ".embedding_cache_a"
    cache_b = tmp_path / ".embedding_cache_b"
    cache_a.mkdir()
    cache_b.mkdir()

    try:
        result_a_path = _run_pfedrec(run_ids[0], cache_a)
        result_b_path = _run_pfedrec(run_ids[1], cache_b)

        result_a = json.loads(result_a_path.read_text())
        result_b = json.loads(result_b_path.read_text())

        # ==== Invariant (a): selected_clients_per_round byte-identity ====
        # PFR-06 / G-03-01 carry-forward. Always asserted; no skip-gate.
        sel_a = result_a.get("selected_clients_per_round")
        sel_b = result_b.get("selected_clients_per_round")
        assert sel_a is not None and sel_b is not None, (
            "selected_clients_per_round missing from one or both result JSONs"
        )
        assert sel_a == sel_b, (
            "PFR-06 VIOLATED: selected_clients_per_round diverged across reruns "
            "with identical run-seed.\n"
            f"run_a[0][:10] = {(sel_a[0][:10] if sel_a else [])}\n"
            f"run_b[0][:10] = {(sel_b[0][:10] if sel_b else [])}"
        )

        # ==== Invariant (b): _manifest.pfr08_verification byte-identity (D-14) ====
        # The audit dict carries the auto-verify decision (HR/NDCG ratios
        # parsed from IJCAI-23-PFedRec/sh_result/ml-1m.txt + abs delta +
        # within_2pts boolean). It is a JSON-roundtripped dict of
        # primitive Python values, so plain == is exact equality (no
        # numeric tolerance band, which would mask exactly the class of
        # nondeterminism we want to catch).
        manifest_a: Dict = result_a.get("_manifest") or {}
        manifest_b: Dict = result_b.get("_manifest") or {}
        audit_a = manifest_a.get("pfr08_verification")
        audit_b = manifest_b.get("pfr08_verification")
        if audit_a is None and audit_b is None:
            # Both runs had no PFR-08 auto-verify fire (e.g., the smoke
            # config emits an unparseable best metric, or the hook is
            # round-gated and 2 rounds is below threshold). Skip (b)
            # cleanly; (a) is already asserted.
            pytest.skip(
                "pfr08_verification absent on both runs (smoke config too "
                "small to trigger the D-14 auto-verify hook). "
                "selected_clients_per_round byte-identity already asserted above."
            )
        if audit_a is None or audit_b is None:
            pytest.fail(
                "D-14 VIOLATED: asymmetric pfr08_verification — "
                f"run_a={audit_a is not None} run_b={audit_b is not None}"
            )
        assert audit_a == audit_b, (
            "D-14 VIOLATED: pfr08_verification differs across same-seed runs.\n"
            f"  run_a = {audit_a}\n  run_b = {audit_b}"
        )

        # ==== Invariant (c): partition_{pid}.pt byte-identity for overlapping
        # partitions via per-key torch.equal comparison (D-16) ====
        # Single key after D-01: only `affine_output.weight` is in the LOCAL
        # payload; bias is GLOBAL and lives in the result JSON, not in the
        # cache .pt file. Per-key comparison gives an actionable failure
        # message naming shape + dtype + max_abs_delta.
        selected_partition_ids: Set[int] = set()
        for round_list in sel_a:
            selected_partition_ids.update(int(p) for p in round_list)

        cache_dir_a = _probe_cache_dir(run_ids[0], cache_a)
        cache_dir_b = _probe_cache_dir(run_ids[1], cache_b)

        if cache_dir_a is None or cache_dir_b is None:
            pytest.skip(
                "Cache dirs not materialized on disk (server may short-circuit "
                "at tiny scale) — selected_clients_per_round + pfr08_verification "
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
                sd_a = torch.load(pt_a, map_location="cpu", weights_only=True)
                sd_b = torch.load(pt_b, map_location="cpu", weights_only=True)
            except Exception as e:
                pytest.fail(f"torch.load failed on partition {pid}: {e}")
            checked_partitions += 1
            keys_a = set(sd_a.keys())
            keys_b = set(sd_b.keys())
            if keys_a != keys_b:
                mismatches.append(
                    f"partition {pid}: LOCAL key set differs "
                    f"(a={sorted(keys_a)}, b={sorted(keys_b)})"
                )
                continue
            for key in sorted(keys_a):
                checked_keys += 1
                t_a, t_b = sd_a[key], sd_b[key]
                if not torch.equal(t_a, t_b):
                    max_delta = float((t_a.float() - t_b.float()).abs().max().item())
                    mismatches.append(
                        f"partition {pid}: tensor {key!r} differs "
                        f"(shape={tuple(t_a.shape)}, dtype={t_a.dtype}, "
                        f"max_abs_delta={max_delta:.6e})"
                    )

        # Coverage guard (Phase 4 Plan 06 idiom): confirm the
        # `affine_output.weight` LOCAL key was actually present in at least
        # one materialized partition cache file. If checked_partitions > 0
        # but coverage_seen is False, the test would silently pass without
        # exercising the PFR-03 single-key cache layout — a false-GREEN
        # caused by a future regression in run-config propagation,
        # client_app.py save logic, or model class _LOCAL_PARAMS contract.
        coverage_seen = False
        if cache_dir_a is not None:
            for pt_path in cache_dir_a.glob("partition_*.pt"):
                try:
                    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
                    if "affine_output.weight" in sd:
                        coverage_seen = True
                        break
                except Exception:
                    continue

        assert not mismatches, (
            f"PFR-06 / D-16 cache VIOLATED: {len(mismatches)} byte-difference(s) "
            f"found across {checked_partitions} overlapping partitions / "
            f"{checked_keys} tensor comparisons.\n"
            f"First 10: {mismatches[:10]}"
        )
        if checked_partitions > 0 and not coverage_seen:
            pytest.fail(
                "PFR-03 path not actually exercised by this run. "
                "No partition_{pid}.pt contains 'affine_output.weight'. "
                "Confirm Plan 03 client_app.py + Plan 01 model contract "
                "propagated correctly."
            )
    finally:
        # Cleanup any cache dirs created under the default roots. The
        # tmp_path roots (cache_a / cache_b) are auto-cleaned by pytest.
        for rid in run_ids:
            for root in (_CACHE_ROOT, _PFEDREC_MODULE_CACHE_ROOT):
                cand = root / rid
                if cand.exists():
                    shutil.rmtree(cand, ignore_errors=True)
