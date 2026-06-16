#!/usr/bin/env python
"""Eval-validity sidecar generator (run-id audit remediation, 2026-06-10).

Classifies every ``results/federated/<module>/<run_dir>/`` by whether its
canonical ``final_metrics.best`` block was produced by a WARM (run_id-stamped)
D-06 full-population eval, and emits a machine-readable sidecar next to each
results.json:

- ``EVAL_VALIDITY.json``  — always written: {status, reason, ...}
- ``INVALID_COLD_EVAL.md`` — human-readable marker, only for invalid runs

Statuses
--------
valid
    best block trustworthy for claim tables: module is structurally immune
    (baseline — no per-user local state), or the run's git commit contains the
    module's run_id eval fix.
invalid_cold_eval
    best_round-protocol run from BEFORE the module's fix: the D-06 eval read
    .embedding_cache/default/ (nonexistent) and scored every user with COLD
    local state. The best block understates (split-learning) or craters
    (pfedrec/dual) the true number. Poisoned paths: final_metrics.best,
    _manifest.metrics.best.
diagnostics_only
    Not a claim-table cell by protocol: last_round/coherent runs, non-canonical
    dirs (test_*), or known metric artifacts (BasicMF clamp NDCG==1.0).

The effective checkpoint rule is resolved OVERRIDES-FIRST (manifest top-level
fields record mode-profile defaults, not effective values — the "manifest
effective-value trap"): manifest.overrides > results.checkpoint.rule >
manifest.checkpoint_rule.

Manifests are NEVER edited (schema_version=3 is test-locked); sidecars are
separate files. ``aggregate_results.collect_thesis_results`` consumes the
sidecar: status != valid is skipped, and ``thesis_run_label_backfill`` /
``run_seed_backfill`` (set here only for designated provisional cells) override
the manifest label so pre-sweep VALID runs can appear in provisional tables.

Usage:
    python scripts/thesis/eval_validity.py            # classify + write sidecars
    python scripts/thesis/eval_validity.py --dry-run  # classify + print only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]

# Per-module run_id-eval fix commits (the run is valid iff its git_commit
# CONTAINS the fix, i.e. fix is an ancestor of the run commit).
# REBASE HAZARD: these short hashes break if the branch is squash-merged or
# rebased (merge-base can no longer prove ancestry). Failure direction is
# CONSERVATIVE — runs become invalid_cold_eval and claim tables empty out,
# which is loud, not silent. Long-term: record fix presence in the run
# manifest at run time instead of proving it via git ancestry.
FIX_COMMITS: Dict[str, Optional[str]] = {
    "pfedrec": "01d8b72",
    "personalized": "3a393fb",
    "adaptive": "70a92bd",
    "baseline": None,  # structurally immune — no per-user local state in eval
}

# Provisional label backfills for runs that are VALID but predate the
# thesis_run_label tagging convention. Single-seed provisional cells: they let
# pre-sweep tables render and are superseded by locked-sweep runs (3 seeds).
LABEL_BACKFILL: Dict[str, Dict[str, Any]] = {
    "20260608-071106-ef41ab": {"thesis_run_label_backfill": "main", "run_seed_backfill": 42},
    "20260504-203751-1bf513": {"thesis_run_label_backfill": "main", "run_seed_backfill": 42},
    # personalized warm re-run (D-06 fix 3a393fb): manifest run_seed is already 42,
    # so only the label needs backfilling.
    "20260610-064423-f18e64": {"thesis_run_label_backfill": "main"},
}

_SIDECAR = "EVAL_VALIDITY.json"
_MARKER = "INVALID_COLD_EVAL.md"
_POISONED_PATHS = ["final_metrics.best", "_manifest.metrics.best"]


def _commit_contains_fix(run_commit: str, fix_commit: str) -> Optional[bool]:
    """True iff ``fix_commit`` is an ancestor of ``run_commit`` (run has the fix).

    Returns None when either commit is unknown to git (dirty/unrecorded runs).
    """
    if not run_commit or not fix_commit:
        return None
    try:
        res = subprocess.run(
            ["git", "merge-base", "--is-ancestor", fix_commit, run_commit],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode == 0:
        return True
    if res.returncode == 1:
        return False
    return None  # unknown revision etc.


def _effective_checkpoint_rule(manifest: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Overrides-first resolution of the checkpoint rule (manifest-trap workaround)."""
    overrides = manifest.get("overrides") or {}
    for key in ("checkpoint_rule", "checkpoint-rule"):
        if overrides.get(key):
            return str(overrides[key])
    ckpt = results.get("checkpoint") or {}
    if ckpt.get("rule"):
        return str(ckpt["rule"])
    return str(manifest.get("checkpoint_rule", ""))


def classify_run(module: str, run_dir: Path) -> Dict[str, Any]:
    """Classify one run dir; returns the sidecar payload."""
    run_id = run_dir.name
    results_path = run_dir / "results.json"
    manifest_path = run_dir / "manifest.json"

    out: Dict[str, Any] = {
        "schema": "eval_validity/1",
        "module": module,
        "run_id": run_id,
        "fix_commit": FIX_COMMITS.get(module),
        "poisoned_paths_if_invalid": _POISONED_PATHS,
    }
    out.update(LABEL_BACKFILL.get(run_id, {}))

    if not results_path.exists():
        out.update(status="diagnostics_only", reason="no results.json (incomplete/aborted run)")
        return out
    try:
        results = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        out.update(status="diagnostics_only", reason=f"unreadable results.json ({e})")
        return out
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
    if not manifest:
        manifest = results.get("_manifest", {}) or {}

    rule = _effective_checkpoint_rule(manifest, results)
    out["effective_checkpoint_rule"] = rule
    run_commit = str(manifest.get("git_commit", "") or "")
    out["git_commit"] = run_commit[:9]

    # Non-canonical dirs (smoke tests etc.) never enter claim tables.
    if not run_id[:8].isdigit():
        out.update(status="diagnostics_only", reason="non-canonical run dir (test/smoke)")
        return out

    # Known metric artifact: saturated NDCG is never a real result. Name the
    # BasicMF rating-clamp cause only when the run actually used BasicMF.
    best = (results.get("final_metrics") or {}).get("best") or {}
    if float(best.get("sampled_ndcg@10", 0.0) or 0.0) >= 0.999:
        model_type = str((results.get("federated_config") or {}).get("model_type", ""))
        if model_type == "basic":
            out.update(status="diagnostics_only", reason="BasicMF clamp artifact (NDCG@10==1.0)")
        else:
            out.update(status="diagnostics_only", reason="saturated-metric artifact (NDCG@10>=0.999)")
        return out

    # last_round / coherent protocol runs are diagnostics by the protocol ruling.
    if rule and rule not in ("best_round_restore", "best_round"):
        out.update(status="diagnostics_only", reason=f"non-matrix protocol (checkpoint_rule={rule})")
        return out

    fix = FIX_COMMITS.get(module)
    if fix is None:
        out.update(
            status="valid",
            reason="module structurally immune (eval loads whole model from broadcast arrays; no per-user local cache)",
        )
        return out

    has_fix = _commit_contains_fix(run_commit, fix)
    if has_fix:
        out.update(status="valid", reason=f"run commit {run_commit[:9]} contains run_id eval fix {fix}")
    else:
        detail = "predates" if has_fix is False else "cannot be proven to contain"
        out.update(
            status="invalid_cold_eval",
            reason=(
                f"run commit {run_commit[:9] or '<missing>'} {detail} fix {fix}: D-06 eval read "
                ".embedding_cache/default/ -> all users scored with COLD local state"
            ),
        )
    return out


def _marker_text(payload: Dict[str, Any]) -> str:
    return (
        f"# INVALID — cold-eval artifact\n\n"
        f"Run `{payload['run_id']}` ({payload['module']}) was produced BEFORE the module's "
        f"run_id eval fix (`{payload['fix_commit']}`). Its D-06 full-population eval omitted "
        f"`run_id`, so the client read per-user local state from `.embedding_cache/default/` "
        f"(nonexistent) and scored every user with COLD init.\n\n"
        f"Poisoned: `final_metrics.best`, `_manifest.metrics.best` (and per-group variants).\n"
        f"Trustworthy: in-loop `final_metrics.last` / `eval_metrics_history` (in-loop eval stamped run_id).\n\n"
        f"Do NOT cite the best block in any claim table. See EVAL_VALIDITY.json.\n"
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="eval_validity.py")
    parser.add_argument("--results-root", type=Path, default=_REPO_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true", help="classify + print, write nothing")
    args = parser.parse_args(list(argv))

    # Staleness provenance: when the classifier was run + at which HEAD.
    from datetime import datetime, timezone
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        head = ""

    rows = []
    for module in ("baseline", "personalized", "adaptive", "pfedrec"):
        module_dir = args.results_root / "federated" / module
        if not module_dir.exists():
            continue
        for run_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):
            payload = classify_run(module, run_dir)
            payload["generated_at"] = generated_at
            payload["generator_head"] = head
            rows.append(payload)
            if not args.dry_run:
                (run_dir / _SIDECAR).write_text(json.dumps(payload, indent=2) + "\n")
                marker = run_dir / _MARKER
                if payload["status"] == "invalid_cold_eval":
                    marker.write_text(_marker_text(payload))
                elif marker.exists():
                    marker.unlink()

    width = max(len(r["run_id"]) for r in rows) if rows else 10
    print(f"{'module':<13}{'run':<{width + 2}}{'status':<20}reason")
    print("-" * (width + 70))
    for r in rows:
        print(f"{r['module']:<13}{r['run_id']:<{width + 2}}{r['status']:<20}{r['reason'][:90]}")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nTOTAL {len(rows)} runs: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.dry_run:
        print("(dry-run: no sidecars written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
