#!/usr/bin/env python3
"""Analyze C3 Run 1 (PFedRec FIX-A isolation) against the consensus decision rule.

Reads the newest pfedrec results.json, enforces the validity gates (run_id !=
default in the log, evaluated_users == 6040, no "no extra-eval responses"
warning), then scores full-pop best/sampled_ndcg@10 against the bands and prints
the per-group (sparse/medium/dense) recovery vs the cratered 0.0711.

Usage: python .bug-investigation/analyze_c3_run1.py
"""
import glob
import json
import os

REPO = "/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system"
LOG = os.path.join(REPO, ".bug-investigation", "c3-run1.log")
CRATER = 0.0711  # prior full-pop (run_id bug); in-loop was 0.3478


def newest_pfedrec_results():
    dirs = sorted(glob.glob(os.path.join(REPO, "results/federated/pfedrec/*/")))
    if not dirs:
        return None, None
    d = dirs[-1]
    rj = os.path.join(d, "results.json")
    return (rj if os.path.exists(rj) else None), os.path.basename(os.path.dirname(d + "/"))


def log_gate_checks():
    out = []
    if not os.path.exists(LOG):
        return ["(log not found — run not finished?)"]
    txt = open(LOG, errors="ignore").read()
    if "run_id=\"default\"" in txt or "run_id='default'" in txt or "base/default/" in txt:
        out.append("FAIL: log references run_id=default / base/default/ — INVALID run")
    else:
        out.append("ok: no base/default/ reference in log")
    if "no extra-eval responses" in txt:
        out.append("FAIL: '[D-06] WARNING: no extra-eval responses' — best==last, INVALID")
    else:
        out.append("ok: no missing-extra-eval warning")
    return out


def main():
    rj, run = newest_pfedrec_results()
    print("=" * 70)
    print(f"C3 RUN 1 — PFedRec FIX-A isolation   ({run})")
    print("=" * 70)
    if not rj:
        print("No pfedrec results.json found.")
        return
    d = json.load(open(rj))
    fm = d.get("final_metrics", {}) or {}
    best = fm.get("best", {}) or {}
    last = fm.get("last", {}) or {}

    print("\n-- VALIDITY GATES --")
    for c in log_gate_checks():
        print(" ", c)
    ev = best.get("evaluated_users", best.get("num_evaluated_users"))
    print(f"  best.evaluated_users = {ev}  (must be 6040 for a valid full-pop number)")

    fp = best.get("sampled_ndcg@10")
    il = last.get("sampled_ndcg@10")
    print("\n-- HEADLINE --")
    print(f"  full-pop best/sampled_ndcg@10 : {fp}   (was {CRATER} with the run_id bug)")
    print(f"  in-loop last/sampled_ndcg@10  : {il}   (this run's own ceiling)")
    print(f"  full-pop HR@10                : {best.get('sampled_hit_rate@10')}")

    print("\n-- PER-GROUP full-pop (all must lift; sparse is load-bearing) --")
    for g in ("sparse", "medium", "dense"):
        print(f"  sampled_ndcg@10/{g:7} : {best.get(f'sampled_ndcg@10/{g}')}")

    print("\n-- VERDICT (consensus bands) --")
    if fp is None:
        print("  full-pop best missing — cannot score.")
    elif fp >= 0.28:
        print(f"  {fp:.4f} >= 0.28  => STRONG: FIX A was the crater cause. PFedRec calibration")
        print("    baseline restored; cross-architecture eval-pitfall finding confirmed.")
    elif fp >= 0.20:
        print(f"  0.20 <= {fp:.4f} < 0.28  => PARTIAL: FIX A real, residual vintage-staleness")
        print("    remains -> run the gated calibration Run 2 (final-calibration-enabled=true).")
    elif fp <= 0.12:
        print(f"  {fp:.4f} <= 0.12  => with confirmed 6040 warm hits, FALSIFIES FIX-A-primary")
        print("    (genuine staleness/architecture) -> calibration mandatory; re-trace if ~0.07.")
    else:
        print(f"  {fp:.4f} in (0.12, 0.20) => weak/ambiguous; inspect per-group + best_round.")
    if il:
        print(f"  (full-pop is {100*fp/il:.0f}% of in-loop ceiling {il:.4f})" if fp else "")


if __name__ == "__main__":
    main()
