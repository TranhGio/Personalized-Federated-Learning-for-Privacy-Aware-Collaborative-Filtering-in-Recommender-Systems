#!/usr/bin/env python3
"""Analyze the bug-investigation diagnostic runs (#4 PFedRec + #1 Adaptive Basic).

Loads the manifest + results JSON for:
  - PFedRec paper reproduction (mode=paper_compat_pfedrec)
  - Adaptive basic ablation (mode=thesis_crossdevice_main, model-type=basic)
  - Adaptive Bug3 verify (existing reference: 20260526-082810-a9a08f)
  - Baseline reference (20260504-203751-1bf513)

Renders a comparison table and emits a verdict per hypothesis:
  H1: PFedRec framework healthy? (HR@10 ≈ 0.70 ± 0.02, NDCG@10 ≈ 0.38 ± 0.02)
  H2: PersonalMLP overkill?     (basic ≥ dual + basic clears thesis bar)
  H3: Joint diagnosis           (combines H1 and H2 into a root-cause statement)

Usage
-----
  python scripts/analyze_bug_investigation.py
  python scripts/analyze_bug_investigation.py --pfedrec PATH --basic PATH
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(ROOT, "results", "federated")

# Known reference runs (existing, pre-investigation)
BASELINE_RUN = "20260504-203751-1bf513"      # baseline BPR-MF cross-device, full pop
ADAPTIVE_BUG3_RUN = "20260526-082810-a9a08f"  # adaptive dual + Bug3 Alt-A verify

# Hypothesis thresholds
PFEDREC_HR_TARGET = 0.70
PFEDREC_NDCG_TARGET = 0.38
PFEDREC_TOL = 0.02  # +-2 points per CLAUDE.md
THESIS_BAR_NDCG = 0.20  # baseline cross-device


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_manifest(module: str, run_id: str) -> Optional[str]:
    p = os.path.join(RESULTS_ROOT, module, run_id, "manifest.json")
    return p if os.path.exists(p) else None


def _latest_pfedrec_run() -> Optional[str]:
    """Find newest PFedRec run with mode=paper_compat_pfedrec, full-pop eval."""
    candidates = []
    for p in glob.glob(os.path.join(RESULTS_ROOT, "pfedrec", "*", "manifest.json")):
        try:
            m = _load(p)
        except Exception:
            continue
        if m.get("mode") != "paper_compat_pfedrec":
            continue
        if m.get("metrics", {}).get("best", {}).get("evaluated_users", 0) < 1000:
            continue  # full-pop only
        candidates.append((os.path.getmtime(p), p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _latest_adaptive_basic_run() -> Optional[str]:
    """Find newest adaptive run with model_type=basic in manifest."""
    candidates = []
    for p in glob.glob(os.path.join(RESULTS_ROOT, "adaptive", "*", "manifest.json")):
        try:
            m = _load(p)
        except Exception:
            continue
        if m.get("mode") != "thesis_crossdevice_main":
            continue
        # model_type override or pyproject default — check both manifest.overrides and results.json
        overrides = m.get("overrides", {})
        model_type = overrides.get("model_type")
        if model_type is None:
            # Fall back to results.json model_name
            results_path = os.path.join(os.path.dirname(p), "results.json")
            if os.path.exists(results_path):
                try:
                    r = _load(results_path)
                    arch = r.get("architecture") or r.get("model_name", "")
                    if isinstance(arch, dict):
                        model_type = arch.get("model_type")
                    elif "BasicMF" in str(arch) and "Dual" not in str(arch):
                        model_type = "basic"
                except Exception:
                    pass
        if model_type != "basic":
            continue
        candidates.append((os.path.getmtime(p), p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _summarize(manifest_path: str, label: str) -> Dict[str, object]:
    m = _load(manifest_path)
    best = m.get("metrics", {}).get("best", {})
    overrides = m.get("overrides", {})
    return {
        "label": label,
        "manifest_path": manifest_path,
        "run_id": m.get("run_id", os.path.basename(os.path.dirname(manifest_path))),
        "module": m.get("module"),
        "mode": m.get("mode"),
        "num_supernodes": m.get("num_supernodes"),
        "best_round": m.get("metrics", {}).get("best_round"),
        "overrides": overrides,
        "ndcg": best.get("sampled_ndcg@10"),
        "hr": best.get("sampled_hr@10"),
        "ndcg_sparse": best.get("sampled_ndcg@10/sparse"),
        "ndcg_medium": best.get("sampled_ndcg@10/medium"),
        "ndcg_dense": best.get("sampled_ndcg@10/dense"),
        "evaluated_users": best.get("evaluated_users"),
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_table(rows: List[Dict[str, object]]) -> str:
    header = ("label", "module", "mode", "overrides", "NDCG@10", "HR@10",
              "sparse", "medium", "dense", "n_users", "best_r")
    widths = [22, 10, 26, 50, 9, 9, 9, 9, 9, 8, 7]
    lines = []
    lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("-" * (sum(widths) + 2 * len(widths)))
    for r in rows:
        ov = r.get("overrides") or {}
        ov_str = ", ".join(f"{k}={v}" for k, v in ov.items())[:50]
        row = (
            str(r.get("label", ""))[:22],
            str(r.get("module", ""))[:10],
            str(r.get("mode", ""))[:26],
            ov_str,
            _fmt(r.get("ndcg")),
            _fmt(r.get("hr")),
            _fmt(r.get("ndcg_sparse")),
            _fmt(r.get("ndcg_medium")),
            _fmt(r.get("ndcg_dense")),
            str(r.get("evaluated_users", ""))[:8],
            str(r.get("best_round", ""))[:7],
        )
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def verdict_h1_pfedrec(pf: Optional[Dict[str, object]]) -> Tuple[str, str]:
    """H1: PFedRec framework health vs paper target."""
    if pf is None:
        return "PENDING", "No PFedRec run available yet."
    hr, ndcg = pf.get("hr"), pf.get("ndcg")
    if hr is None or ndcg is None:
        return "PENDING", "PFedRec run incomplete (no best metrics)."
    hr_ok = abs(hr - PFEDREC_HR_TARGET) <= PFEDREC_TOL
    ndcg_ok = abs(ndcg - PFEDREC_NDCG_TARGET) <= PFEDREC_TOL
    if hr_ok and ndcg_ok:
        return "PASS", (
            f"HR@10={hr:.4f} (target {PFEDREC_HR_TARGET}±{PFEDREC_TOL}) ✓, "
            f"NDCG@10={ndcg:.4f} (target {PFEDREC_NDCG_TARGET}±{PFEDREC_TOL}) ✓. "
            "Framework healthy — Flower/eval/cache/protocol all reproduce paper."
        )
    return "FAIL", (
        f"HR@10={hr:.4f} (off by {hr - PFEDREC_HR_TARGET:+.4f}), "
        f"NDCG@10={ndcg:.4f} (off by {ndcg - PFEDREC_NDCG_TARGET:+.4f}). "
        "Framework misses paper target — audit 9 known Flower PFedRec bugs before trusting any cross-method comparison."
    )


def verdict_h2_personalmlp(basic: Optional[Dict[str, object]],
                            dual: Dict[str, object],
                            baseline: Dict[str, object]) -> Tuple[str, str]:
    """H2: PersonalMLP overkill? basic ≥ dual AND basic ≥ baseline thesis bar."""
    if basic is None:
        return "PENDING", "No adaptive-basic run available yet."
    b_ndcg = basic.get("ndcg")
    d_ndcg = dual.get("ndcg")
    base_ndcg = baseline.get("ndcg")
    if b_ndcg is None or d_ndcg is None or base_ndcg is None:
        return "PENDING", "One of the runs missing best metrics."
    delta_vs_dual = b_ndcg - d_ndcg
    delta_vs_baseline = b_ndcg - base_ndcg
    if b_ndcg >= base_ndcg - 0.01 and b_ndcg > d_ndcg + 0.02:
        return "CONFIRMED", (
            f"basic NDCG@10={b_ndcg:.4f} > dual {d_ndcg:.4f} (Δ={delta_vs_dual:+.4f}) AND "
            f"matches baseline {base_ndcg:.4f} (Δ={delta_vs_baseline:+.4f}). "
            "PersonalMLP IS hurting — strip it from the architecture."
        )
    if abs(b_ndcg - d_ndcg) <= 0.02 and b_ndcg < base_ndcg - 0.05:
        return "REJECTED", (
            f"basic NDCG@10={b_ndcg:.4f} ≈ dual {d_ndcg:.4f} (Δ={delta_vs_dual:+.4f}), "
            f"both far below baseline {base_ndcg:.4f}. "
            "PersonalMLP is NOT the bottleneck — issue is in split-learning core or framework."
        )
    if b_ndcg < d_ndcg - 0.02:
        return "INVERTED", (
            f"basic NDCG@10={b_ndcg:.4f} < dual {d_ndcg:.4f} (Δ={delta_vs_dual:+.4f}). "
            "Surprising: removing PersonalMLP makes things worse. "
            "PersonalMLP may be doing useful work despite the absolute metrics being low."
        )
    return "PARTIAL", (
        f"basic NDCG@10={b_ndcg:.4f} vs dual {d_ndcg:.4f} (Δ={delta_vs_dual:+.4f}), "
        f"vs baseline {base_ndcg:.4f} (Δ={delta_vs_baseline:+.4f}). "
        "Ambiguous result — needs more diagnostic work."
    )


def joint_diagnosis(h1: Tuple[str, str], h2: Tuple[str, str]) -> str:
    s1, _ = h1
    s2, _ = h2
    if s1 == "PENDING" or s2 == "PENDING":
        return "Joint diagnosis pending — wait for both runs."
    if s1 == "PASS" and s2 == "CONFIRMED":
        return ("ROOT CAUSE = PersonalMLP. Framework reproduces paper, basic split-learning matches baseline → "
                "the dual model's PersonalMLP overparameterizes locally and hurts. "
                "ACTION: ablate PersonalMLP; rebuild adaptive on basic + α only.")
    if s1 == "PASS" and s2 == "REJECTED":
        return ("ROOT CAUSE = split-learning core. Framework healthy, basic split-learning ALSO underperforms → "
                "bug in user-embedding caching, cold-start, or global-prototype EMA. "
                "ACTION: investigate split-learning mechanism (cold-start rate 24%, prototype norm 0.009 from prior run).")
    if s1 == "PASS" and s2 == "INVERTED":
        return ("Framework healthy but basic < dual. PersonalMLP is contributing despite low absolute numbers. "
                "ACTION: profile what dual is doing — maybe one of the next-gen techniques (per-user α, contrastive) "
                "is carrying the signal.")
    if s1 == "FAIL":
        return ("ROOT CAUSE = framework. PFedRec misses paper target → bug affects all modules. "
                "ACTION: re-audit 9 Flower PFedRec bugs (see [[project_config_comparison]]); "
                "fix before drawing any adaptive-vs-baseline conclusion.")
    return "Joint result is ambiguous — manual interpretation required."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfedrec", help="Path to PFedRec manifest.json or run dir", default=None)
    parser.add_argument("--basic", help="Path to adaptive-basic manifest.json or run dir", default=None)
    parser.add_argument("--baseline-run", default=BASELINE_RUN)
    parser.add_argument("--adaptive-bug3-run", default=ADAPTIVE_BUG3_RUN)
    args = parser.parse_args()

    def _resolve_arg(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        if os.path.isdir(p):
            return os.path.join(p, "manifest.json")
        return p

    pfedrec_path = _resolve_arg(args.pfedrec) or _latest_pfedrec_run()
    basic_path = _resolve_arg(args.basic) or _latest_adaptive_basic_run()
    baseline_path = _find_manifest("baseline", args.baseline_run)
    dual_path = _find_manifest("adaptive", args.adaptive_bug3_run)

    if baseline_path is None:
        print(f"ERROR: baseline reference run {args.baseline_run} not found", file=sys.stderr)
        return 1
    if dual_path is None:
        print(f"ERROR: adaptive Bug3 reference run {args.adaptive_bug3_run} not found", file=sys.stderr)
        return 1

    rows = []
    pf = bs = None
    if pfedrec_path:
        pf = _summarize(pfedrec_path, "PFedRec paper repro #4")
        rows.append(pf)
    else:
        rows.append({"label": "PFedRec paper repro #4", "module": "pfedrec",
                     "mode": "paper_compat_pfedrec", "overrides": {}, "ndcg": None})
    if basic_path:
        bs = _summarize(basic_path, "Adaptive basic abl #1")
        rows.append(bs)
    else:
        rows.append({"label": "Adaptive basic abl #1", "module": "adaptive",
                     "mode": "thesis_crossdevice_main", "overrides": {"model_type": "basic"}, "ndcg": None})
    dl = _summarize(dual_path, "Adaptive dual Bug3 ref")
    bl = _summarize(baseline_path, "Baseline ref")
    rows.append(dl)
    rows.append(bl)

    print("=" * 80)
    print("BUG INVESTIGATION — Comparison Table")
    print("=" * 80)
    print(render_table(rows))
    print()
    print("=" * 80)
    print("HYPOTHESIS VERDICTS")
    print("=" * 80)
    h1 = verdict_h1_pfedrec(pf)
    h2 = verdict_h2_personalmlp(bs, dl, bl)
    print(f"\nH1 — PFedRec framework healthy?  [{h1[0]}]")
    print(f"     {h1[1]}")
    print(f"\nH2 — PersonalMLP overkill?       [{h2[0]}]")
    print(f"     {h2[1]}")
    print()
    print("=" * 80)
    print("JOINT DIAGNOSIS")
    print("=" * 80)
    print(joint_diagnosis(h1, h2))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
