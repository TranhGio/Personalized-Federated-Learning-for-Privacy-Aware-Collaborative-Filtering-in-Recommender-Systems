#!/usr/bin/env python3
"""
Analyze and compare FedProx hyperparameter sweep results.

Usage:
    python analyze_sweep_results.py <sweep_directory>

Example:
    python analyze_sweep_results.py ../results/federated/sweeps/fedprox_sweep_20251228_120000
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_sweep_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from sweep directory."""
    results = []
    for json_file in sorted(sweep_dir.glob("*_results.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["_filename"] = json_file.name
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return results


def extract_metrics(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract key metrics from results into a flat structure."""
    records = []
    for r in results:
        config = r.get("federated_config", {})
        metrics = r.get("final_metrics", {})

        records.append({
            # Hyperparameters
            "mu": config.get("proximal_mu", 0.0),
            "rounds": config.get("num_rounds", 0),
            "frac": config.get("fraction_train", 1.0),
            "strategy": config.get("strategy", "unknown"),

            # Rating prediction metrics
            "rmse": metrics.get("rmse", 0.0),
            "mae": metrics.get("mae", 0.0),

            # Ranking metrics @10
            "hit@10": metrics.get("hit_rate@10", 0.0),
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "prec@10": metrics.get("precision@10", 0.0),
            "recall@10": metrics.get("recall@10", 0.0),

            # Other ranking metrics
            "mrr": metrics.get("mrr", 0.0),
            "hit@20": metrics.get("hit_rate@20", 0.0),
            "ndcg@20": metrics.get("ndcg@20", 0.0),

            # Metadata
            "filename": r.get("_filename", ""),
        })

    return records


def print_comparison_table(records: List[Dict[str, Any]]):
    """Print formatted comparison table."""
    if not records:
        print("No results to display.")
        return

    # Sort by NDCG@10 (higher is better)
    records_sorted = sorted(records, key=lambda x: x.get("ndcg@10", 0), reverse=True)

    print("\n" + "=" * 90)
    print("FedProx Hyperparameter Sweep Results (sorted by NDCG@10)")
    print("=" * 90)

    # Header
    print(f"{'Rank':>4} {'mu':>6} {'rounds':>6} {'frac':>6} | "
          f"{'RMSE':>7} {'MAE':>7} | "
          f"{'Hit@10':>7} {'NDCG@10':>8} {'MRR':>7}")
    print("-" * 90)

    # Data rows
    for i, r in enumerate(records_sorted, 1):
        print(f"{i:>4} {r['mu']:>6} {r['rounds']:>6} {r['frac']:>6} | "
              f"{r['rmse']:>7.4f} {r['mae']:>7.4f} | "
              f"{r['hit@10']:>7.4f} {r['ndcg@10']:>8.4f} {r['mrr']:>7.4f}")

    # Best configuration summary
    best = records_sorted[0]
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION (by NDCG@10)")
    print("=" * 90)
    print(f"  proximal-mu:       {best['mu']}")
    print(f"  num-server-rounds: {best['rounds']}")
    print(f"  fraction-train:    {best['frac']}")
    print()
    print("  Performance Metrics:")
    print(f"    RMSE:        {best['rmse']:.4f}")
    print(f"    MAE:         {best['mae']:.4f}")
    print(f"    Hit Rate@10: {best['hit@10']:.4f} ({best['hit@10']*100:.2f}%)")
    print(f"    NDCG@10:     {best['ndcg@10']:.4f}")
    print(f"    MRR:         {best['mrr']:.4f}")

    # Comparison with baseline (if exists)
    if len(records_sorted) > 1:
        worst = records_sorted[-1]
        print()
        print("  Improvement over worst config:")
        if worst['ndcg@10'] > 0:
            ndcg_imp = (best['ndcg@10'] - worst['ndcg@10']) / worst['ndcg@10'] * 100
            print(f"    NDCG@10: +{ndcg_imp:.1f}%")
        if worst['hit@10'] > 0:
            hit_imp = (best['hit@10'] - worst['hit@10']) / worst['hit@10'] * 100
            print(f"    Hit@10:  +{hit_imp:.1f}%")


def save_summary_csv(records: List[Dict[str, Any]], output_path: Path):
    """Save results to CSV for further analysis."""
    records_sorted = sorted(records, key=lambda x: x.get("ndcg@10", 0), reverse=True)

    # Write CSV manually (no pandas dependency)
    headers = ["rank", "mu", "rounds", "frac", "rmse", "mae",
               "hit@10", "ndcg@10", "prec@10", "recall@10", "mrr",
               "hit@20", "ndcg@20", "filename"]

    with open(output_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for i, r in enumerate(records_sorted, 1):
            row = [str(i), str(r['mu']), str(r['rounds']), str(r['frac']),
                   f"{r['rmse']:.6f}", f"{r['mae']:.6f}",
                   f"{r['hit@10']:.6f}", f"{r['ndcg@10']:.6f}",
                   f"{r['prec@10']:.6f}", f"{r['recall@10']:.6f}",
                   f"{r['mrr']:.6f}", f"{r['hit@20']:.6f}", f"{r['ndcg@20']:.6f}",
                   r['filename']]
            f.write(",".join(row) + "\n")

    print(f"\nSummary CSV saved to: {output_path}")


def print_hyperparam_analysis(records: List[Dict[str, Any]]):
    """Print analysis of each hyperparameter's effect."""
    if len(records) < 4:
        return

    print("\n" + "=" * 90)
    print("Hyperparameter Impact Analysis")
    print("=" * 90)

    # Analyze mu effect
    mu_values = set(r['mu'] for r in records)
    if len(mu_values) > 1:
        print("\nEffect of proximal-mu (averaged across other params):")
        for mu in sorted(mu_values):
            subset = [r for r in records if r['mu'] == mu]
            avg_ndcg = sum(r['ndcg@10'] for r in subset) / len(subset)
            avg_hit = sum(r['hit@10'] for r in subset) / len(subset)
            print(f"  mu={mu:>4}: NDCG@10={avg_ndcg:.4f}, Hit@10={avg_hit:.4f}")

    # Analyze rounds effect
    rounds_values = set(r['rounds'] for r in records)
    if len(rounds_values) > 1:
        print("\nEffect of num-server-rounds (averaged):")
        for rounds in sorted(rounds_values):
            subset = [r for r in records if r['rounds'] == rounds]
            avg_ndcg = sum(r['ndcg@10'] for r in subset) / len(subset)
            avg_hit = sum(r['hit@10'] for r in subset) / len(subset)
            print(f"  rounds={rounds:>3}: NDCG@10={avg_ndcg:.4f}, Hit@10={avg_hit:.4f}")

    # Analyze fraction effect
    frac_values = set(r['frac'] for r in records)
    if len(frac_values) > 1:
        print("\nEffect of fraction-train (averaged):")
        for frac in sorted(frac_values):
            subset = [r for r in records if r['frac'] == frac]
            avg_ndcg = sum(r['ndcg@10'] for r in subset) / len(subset)
            avg_hit = sum(r['hit@10'] for r in subset) / len(subset)
            print(f"  frac={frac:>3}: NDCG@10={avg_ndcg:.4f}, Hit@10={avg_hit:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep_results.py <sweep_directory>")
        print()
        print("Example:")
        print("  python analyze_sweep_results.py ../results/federated/sweeps/fedprox_sweep_20251228_120000")
        sys.exit(1)

    sweep_dir = Path(sys.argv[1])
    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {sweep_dir}")
    results = load_sweep_results(sweep_dir)

    if not results:
        print(f"No result files found in {sweep_dir}")
        print("Looking for files matching pattern: *_results.json")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")

    # Extract and analyze
    records = extract_metrics(results)
    print_comparison_table(records)
    print_hyperparam_analysis(records)

    # Save summary CSV
    summary_path = sweep_dir / "sweep_summary.csv"
    save_summary_csv(records, summary_path)


if __name__ == "__main__":
    main()
