#!/usr/bin/env python3
"""Analyze the tuned-dual Path-A probe.

Reads the coherent full-pop curve from the [DIAG] lines in tuned-dual.log,
finds the PEAK full-pop NDCG@10 (last_round can catch an end-of-training dip,
so the peak over the diagnostic curve is the fair number), prints the
age-bucket staleness shape at peak and at end, then renders the verdict vs the
reference matrix.

Usage: python .bug-investigation/analyze_tuned_dual.py
"""
import glob
import json
import os
import re

REPO = "/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system"
LOG = os.path.join(REPO, ".bug-investigation", "tuned-dual.log")

# Reference matrix (full-pop sampled NDCG@10, cross-device N=6040) -------------
REF = {
    "split-learning personalized (best, n=6040)": 0.2246,
    "adaptive-BPR flags-off (best, n=6040)": 0.2256,
    "baseline BPR-MF (best, n=6040)": 0.2013,
    "discriminator dual lr=0.005 (coherent peak)": 0.2112,
    "discriminator dual lr=0.005 (coherent last)": 0.1989,
}
TARGET = 0.2246  # the bar to beat = best simpler comparator

DIAG_RE = re.compile(r"fullpop\(n=(\d+)\)\s+ndcg=([0-9.]+)")
AGE_RE = re.compile(r"(age\w+)\(n=\d+\)=([0-9.]+)")


def parse_diag(log_path):
    """Return list of (fullpop_n, fullpop_ndcg, {age_bucket: ndcg}) per DIAG line."""
    rows = []
    if not os.path.exists(log_path):
        return rows
    with open(log_path) as fh:
        for line in fh:
            if "[DIAG]" not in line:
                continue
            m = DIAG_RE.search(line)
            if not m:
                continue
            n = int(m.group(1))
            ndcg = float(m.group(2))
            ages = {k: float(v) for k, v in AGE_RE.findall(line)}
            rows.append((n, ndcg, ages))
    return rows


def latest_adaptive_results():
    dirs = sorted(glob.glob(os.path.join(REPO, "results/federated/adaptive/*/")))
    if not dirs:
        return None
    rj = os.path.join(dirs[-1], "results.json")
    return rj if os.path.exists(rj) else None


def main():
    rows = parse_diag(LOG)
    print("=" * 70)
    print("TUNED DUAL — coherent full-pop curve ([DIAG] lines)")
    print("=" * 70)
    if not rows:
        print("No [DIAG] lines found in", LOG, "(run not finished or flag off?)")
    else:
        for i, (n, ndcg, ages) in enumerate(rows):
            age_str = "  ".join(f"{k}={v:.4f}" for k, v in ages.items())
            print(f"  diag[{i:02d}] fullpop(n={n:5d}) ndcg={ndcg:.4f}   {age_str}")
        peak_i = max(range(len(rows)), key=lambda i: rows[i][1])
        pn, pndcg, pages = rows[peak_i]
        ln, lndcg, lages = rows[-1]
        print("-" * 70)
        print(f"  PEAK coherent full-pop : {pndcg:.4f}  (diag idx {peak_i}, n={pn})")
        print(f"    age buckets @ peak   : " + "  ".join(f"{k}={v:.4f}" for k, v in pages.items()))
        print(f"  LAST coherent full-pop : {lndcg:.4f}  (n={ln})")
        print(f"    age buckets @ last   : " + "  ".join(f"{k}={v:.4f}" for k, v in lages.items()))

    # In-loop final from results.json -----------------------------------------
    rj = latest_adaptive_results()
    inloop = None
    if rj:
        d = json.load(open(rj))
        fm = d.get("final_metrics", {})
        last = fm.get("last", {})
        inloop = last.get("sampled_ndcg@10")
        print("\n" + "=" * 70)
        print("In-loop (sampled, n~604) from", os.path.basename(os.path.dirname(rj)))
        print("=" * 70)
        for k in ["sampled_ndcg@10", "sampled_ndcg@10/sparse",
                  "sampled_ndcg@10/medium", "sampled_ndcg@10/dense", "sampled_hit_rate@10"]:
            if k in last:
                print(f"  {k:28} {last[k]:.4f}")

    # Verdict ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT — coherent full-pop PEAK vs reference matrix")
    print("=" * 70)
    peak = max((r[1] for r in rows), default=None)
    for name, val in sorted(REF.items(), key=lambda kv: -kv[1]):
        print(f"  {val:.4f}  {name}")
    if peak is not None:
        print(f"  ------")
        print(f"  {peak:.4f}  *** TUNED DUAL (this run, coherent peak) ***")
        delta = (peak - TARGET) / TARGET * 100
        print("-" * 70)
        if peak > TARGET + 0.005:
            print(f"  >>> CLEARS the bar: +{delta:.1f}% vs split-learning {TARGET}.")
            print("      -> Path A viable. Justify a fair 2-run head-to-head")
            print("         (re-measure split-learning under last_round + diagnostic).")
        elif peak >= TARGET - 0.005:
            print(f"  >>> TIES the bar ({delta:+.1f}% vs {TARGET}). Marginal —")
            print("      dual matches but does not beat the simpler model.")
        else:
            print(f"  >>> BELOW the bar ({delta:+.1f}% vs {TARGET}).")
            print("      -> Even healthy, dual does not win -> Path B confirmed, pivot.")
        print("  CAVEAT: [DIAG] excludes ~600 cold-init users the n=6040 number includes")
        print("          (this peak is slightly optimistic vs a true full-6040 eval).")


if __name__ == "__main__":
    main()
