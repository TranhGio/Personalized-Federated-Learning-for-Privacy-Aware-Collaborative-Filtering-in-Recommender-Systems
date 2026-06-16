#!/usr/bin/env python3
"""
Compare all baseline and proposed approach results for thesis.

Reads JSON results from results/ directory and generates a markdown comparison
table with key metrics (HR@10, NDCG@10, MRR) across all methods.

Usage:
    python scripts/compare_all_results.py
    python scripts/compare_all_results.py --output results/comparison.md
    python scripts/compare_all_results.py --latest   # Only latest run per config type
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# Result loading
# =============================================================================

def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def classify_method(data: dict, filename: str) -> str:
    """Classify a result into a human-readable method name."""
    model_name = data.get("model_name", "")
    arch = data.get("architecture", "")
    fc = data.get("federated_config", {})
    ac = data.get("adaptive_config", {})

    model_type = fc.get("model_type", "")
    strategy = fc.get("strategy", "fedavg")
    split = fc.get("split_learning", False)
    fusion = fc.get("fusion_type")
    mlp = fc.get("mlp_hidden_dims")

    pua = ac.get("enable_per_user_alpha", False)
    ip = ac.get("enable_item_perturbation", False)
    cl = ac.get("contrastive_lambda", 0.0)

    # Centralized
    if "centralized" in filename.lower() or "Centralized" in model_name:
        return "BPR-MF (Centralized)"
    if model_name == "SVD" or "svd" in filename.lower():
        return "SVD (Centralized)"
    if model_name == "NCF" or "ncf" in filename.lower():
        return "NCF (Centralized)"

    # Federated adaptive (dual-level)
    if model_type == "dual" or "DUAL" in model_name or arch == "dual_level_personalization":
        techniques = []
        if pua:
            techniques.append("PUA")
        if ip:
            techniques.append("IP")
        if cl and cl > 0:
            techniques.append("CL")

        fusion_str = f" ({fusion})" if fusion else ""
        if techniques:
            return f"Dual-Level{fusion_str} + {'+'.join(techniques)}"
        return f"Dual-Level{fusion_str}"

    # Federated adaptive (single-level BPR with alpha)
    if "Adaptive" in model_name or arch == "split_learning_adaptive":
        strategy_str = strategy.upper().replace("FEDAVG", "FedAvg").replace("FEDPROX", "FedProx")
        return f"Adaptive BPR-MF + {strategy_str}"

    # Federated personalized (split learning)
    if split or "split" in filename or "Personalized" in model_name:
        strategy_str = strategy.upper().replace("FEDAVG", "FedAvg").replace("FEDPROX", "FedProx")
        return f"Split BPR-MF + {strategy_str}"

    # Federated baseline (all global)
    if "Federated" in model_name or "federated" in filename:
        strategy_str = strategy.upper().replace("FEDAVG", "FedAvg").replace("FEDPROX", "FedProx")
        model_str = "BPR-MF" if model_type == "bpr" else "Basic-MF"
        return f"Fed {model_str} + {strategy_str}"

    return filename.replace("_results.json", "")


def get_sort_key(method: str) -> tuple:
    """Sort methods in thesis progression order."""
    order = [
        "SVD (Centralized)",
        "NCF (Centralized)",
        "BPR-MF (Centralized)",
        "Fed Basic-MF",
        "Fed BPR-MF + FedAvg",
        "Fed BPR-MF + FedProx",
        "Split BPR-MF + FedAvg",
        "Split BPR-MF + FedProx",
        "Adaptive BPR-MF + FedAvg",
        "Adaptive BPR-MF + FedProx",
        "Adaptive BPR-MF",
        "Dual-Level",
    ]
    for i, prefix in enumerate(order):
        if method.startswith(prefix):
            return (i, method)
    return (len(order), method)


def extract_metrics(data: dict) -> dict[str, Any]:
    """Extract key metrics from a result JSON."""
    metrics = {}
    fm = data.get("final_metrics", {})
    fc = data.get("federated_config", {})
    es = data.get("early_stopping", {})

    # Centralized results have different structures
    if "metrics" in data:
        # Centralized BPR-MF
        m = data["metrics"]
        metrics["hr@10"] = m.get("hr@10")
        metrics["ndcg@10"] = m.get("ndcg@10")
        metrics["mrr"] = m.get("mrr")
        metrics["rounds"] = data.get("params", {}).get("n_epochs")
        return metrics

    if "svd_results" in data:
        # SVD
        m = data["svd_results"]
        metrics["rmse"] = m.get("rmse")
        metrics["mae"] = m.get("mae")
        return metrics

    if "rmse" in data and "model_name" in data and not fm:
        # NCF
        metrics["rmse"] = data.get("rmse")
        metrics["mae"] = data.get("mae")
        return metrics

    # Federated results
    metrics["hr@10"] = fm.get("hit_rate@10")
    metrics["ndcg@10"] = fm.get("ndcg@10")
    metrics["mrr"] = fm.get("mrr")
    metrics["hr@5"] = fm.get("hit_rate@5")
    metrics["ndcg@5"] = fm.get("ndcg@5")
    metrics["hr@20"] = fm.get("hit_rate@20")
    metrics["ndcg@20"] = fm.get("ndcg@20")

    # Sampled metrics (NCF protocol - for fair comparison with published baselines)
    metrics["s_hr@10"] = fm.get("sampled_hr@10")
    metrics["s_ndcg@10"] = fm.get("sampled_ndcg@10")
    metrics["s_mrr"] = fm.get("sampled_mrr")

    # Training info
    metrics["rounds"] = fc.get("actual_rounds") or fc.get("num_rounds") or data.get("training_rounds")
    metrics["strategy"] = fc.get("strategy", "")
    metrics["early_stopped"] = es.get("stopped_early", False)
    metrics["best_round"] = es.get("best_round")
    metrics["alpha_mean"] = data.get("alpha_analysis", {}).get("mean")

    # Timestamp for latest filtering
    metrics["timestamp"] = data.get("timestamp", "")

    return metrics


def load_all_results() -> list[dict]:
    """Load all result JSONs and return classified entries."""
    entries = []

    # Centralized
    for p in (RESULTS_DIR / "centralized").glob("*.json"):
        data = load_json(p)
        if data:
            entries.append({
                "file": str(p.relative_to(PROJECT_ROOT)),
                "method": classify_method(data, p.name),
                "category": "centralized",
                **extract_metrics(data),
            })

    # Federated baseline
    for p in (RESULTS_DIR / "federated").glob("*_results.json"):
        data = load_json(p)
        if data:
            entries.append({
                "file": str(p.relative_to(PROJECT_ROOT)),
                "method": classify_method(data, p.name),
                "category": "baseline",
                **extract_metrics(data),
            })

    # Federated personalized + adaptive
    personalized_dir = RESULTS_DIR / "federated" / "personalized"
    if personalized_dir.exists():
        for p in personalized_dir.glob("*_results.json"):
            data = load_json(p)
            if data:
                cat = "proposed" if ("dual" in p.name or "Adaptive" in data.get("model_name", "") or data.get("architecture") in ("split_learning_adaptive", "dual_level_personalization")) else "personalized"
                entries.append({
                    "file": str(p.relative_to(PROJECT_ROOT)),
                    "method": classify_method(data, p.name),
                    "category": cat,
                    **extract_metrics(data),
                })

    return entries


def filter_latest(entries: list[dict]) -> list[dict]:
    """Keep only the latest run per method name."""
    latest: dict[str, dict] = {}
    for e in entries:
        key = e["method"]
        ts = e.get("timestamp", "")
        if key not in latest or ts > latest[key].get("timestamp", ""):
            latest[key] = e
    return list(latest.values())


# =============================================================================
# Report generation
# =============================================================================

def fmt_pct(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val * 100:.2f}%"


def fmt_float(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.4f}"


def generate_report(entries: list[dict], published_baselines: bool = True) -> str:
    """Generate markdown comparison report."""
    lines = []
    lines.append("# Thesis Comparison: MovieLens 1M")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Sort entries
    entries.sort(key=lambda e: get_sort_key(e["method"]))

    # --- Centralized ---
    cent = [e for e in entries if e["category"] == "centralized"]
    if cent:
        lines.append("## Centralized Baselines (Upper Bound - Full Data Access)")
        lines.append("")
        lines.append("| Method | RMSE | MAE | HR@10 | NDCG@10 | MRR |")
        lines.append("|--------|------|-----|-------|---------|-----|")
        for e in cent:
            lines.append(
                f"| {e['method']} "
                f"| {fmt_float(e.get('rmse'))} "
                f"| {fmt_float(e.get('mae'))} "
                f"| {fmt_pct(e.get('hr@10'))} "
                f"| {fmt_pct(e.get('ndcg@10'))} "
                f"| {fmt_pct(e.get('mrr'))} |"
            )
        lines.append("")

    # --- Published Baselines (reference) ---
    if published_baselines:
        lines.append("## Published Federated Baselines (from IJCAI'23)")
        lines.append("")
        lines.append("| Method | HR@10 | NDCG@10 | Source |")
        lines.append("|--------|-------|---------|--------|")
        lines.append("| FedMF | 65.15% | 39.38% | IJCAI'23 |")
        lines.append("| FedNCF | 60.62% | 33.25% | IJCAI'23 |")
        lines.append("| FedRecon | 64.45% | 37.78% | TFF |")
        lines.append("| PFedRec (SOTA) | **73.26%** | **44.36%** | IJCAI'23 |")
        lines.append("")
        lines.append("*Note: Published baselines use leave-one-out + 99 negative samples (sampled evaluation).*")
        lines.append("")

    # --- Federated methods ---
    fed = [e for e in entries if e["category"] != "centralized"]
    if not fed:
        lines.append("*No federated results found.*")
        return "\n".join(lines)

    # Full-rank evaluation table
    lines.append("## Federated Methods - Full-Rank Evaluation")
    lines.append("")
    lines.append("Rankings computed over ALL ~3,700 items (harder, more realistic).")
    lines.append("")
    lines.append("| # | Method | Rounds | HR@10 | NDCG@10 | MRR | Category |")
    lines.append("|---|--------|--------|-------|---------|-----|----------|")
    for i, e in enumerate(fed, 1):
        rounds_str = str(e.get("rounds", "-"))
        if e.get("early_stopped"):
            rounds_str += f" (ES@{e.get('best_round', '?')})"
        cat_label = {"baseline": "Baseline", "personalized": "Personalized", "proposed": "**Proposed**"}.get(e["category"], e["category"])
        lines.append(
            f"| {i} "
            f"| {e['method']} "
            f"| {rounds_str} "
            f"| {fmt_pct(e.get('hr@10'))} "
            f"| {fmt_pct(e.get('ndcg@10'))} "
            f"| {fmt_pct(e.get('mrr'))} "
            f"| {cat_label} |"
        )
    lines.append("")

    # Sampled evaluation table (for methods that have it)
    sampled = [e for e in fed if e.get("s_ndcg@10") is not None]
    if sampled:
        lines.append("## Federated Methods - Sampled Evaluation (NCF Protocol)")
        lines.append("")
        lines.append("Leave-one-out + 99 negative samples. **Directly comparable to published baselines.**")
        lines.append("")
        lines.append("| # | Method | Rounds | s_HR@10 | s_NDCG@10 | s_MRR | vs PFedRec |")
        lines.append("|---|--------|--------|---------|-----------|-------|------------|")
        pfedrec_ndcg = 0.4436
        for i, e in enumerate(sampled, 1):
            rounds_str = str(e.get("rounds", "-"))
            if e.get("early_stopped"):
                rounds_str += f" (ES@{e.get('best_round', '?')})"
            s_ndcg = e.get("s_ndcg@10")
            gap = f"{(s_ndcg - pfedrec_ndcg) * 100:+.2f}pp" if s_ndcg else "-"
            lines.append(
                f"| {i} "
                f"| {e['method']} "
                f"| {rounds_str} "
                f"| {fmt_pct(e.get('s_hr@10'))} "
                f"| {fmt_pct(s_ndcg)} "
                f"| {fmt_pct(e.get('s_mrr'))} "
                f"| {gap} |"
            )
        lines.append("")

    # --- Best method summary ---
    lines.append("## Summary")
    lines.append("")

    # Best by full-rank NDCG@10
    valid_ndcg = [e for e in fed if e.get("ndcg@10") is not None]
    if valid_ndcg:
        best_full = max(valid_ndcg, key=lambda e: e["ndcg@10"])
        lines.append(f"**Best full-rank NDCG@10**: {fmt_pct(best_full['ndcg@10'])} ({best_full['method']})")

    # Best by sampled NDCG@10
    if sampled:
        best_sampled = max(sampled, key=lambda e: e["s_ndcg@10"])
        lines.append(f"**Best sampled NDCG@10**: {fmt_pct(best_sampled['s_ndcg@10'])} ({best_sampled['method']})")

    lines.append("")

    # --- Improvement analysis ---
    baselines_fed = [e for e in fed if e["category"] == "baseline" and e.get("ndcg@10")]
    proposed_fed = [e for e in fed if e["category"] == "proposed" and e.get("ndcg@10")]

    if baselines_fed and proposed_fed:
        best_baseline = max(baselines_fed, key=lambda e: e["ndcg@10"])
        best_proposed = max(proposed_fed, key=lambda e: e["ndcg@10"])
        improvement = (best_proposed["ndcg@10"] - best_baseline["ndcg@10"]) / best_baseline["ndcg@10"] * 100

        lines.append("### Improvement Over Best Baseline (full-rank)")
        lines.append("")
        lines.append(f"| Metric | Best Baseline | Best Proposed | Improvement |")
        lines.append(f"|--------|---------------|---------------|-------------|")
        for metric in ["hr@10", "ndcg@10", "mrr"]:
            bl_val = best_baseline.get(metric)
            pr_val = best_proposed.get(metric)
            if bl_val and pr_val:
                imp = (pr_val - bl_val) / bl_val * 100
                lines.append(f"| {metric.upper()} | {fmt_pct(bl_val)} | {fmt_pct(pr_val)} | {imp:+.1f}% |")
        lines.append("")

    # Sampled improvement
    baselines_sampled = [e for e in fed if e["category"] in ("baseline", "personalized") and e.get("s_ndcg@10")]
    proposed_sampled = [e for e in fed if e["category"] == "proposed" and e.get("s_ndcg@10")]

    if baselines_sampled and proposed_sampled:
        best_bl_s = max(baselines_sampled, key=lambda e: e["s_ndcg@10"])
        best_pr_s = max(proposed_sampled, key=lambda e: e["s_ndcg@10"])

        lines.append("### Improvement Over Best Baseline (sampled, NCF protocol)")
        lines.append("")
        lines.append(f"| Metric | Best Baseline | Best Proposed | Improvement |")
        lines.append(f"|--------|---------------|---------------|-------------|")
        for metric in ["s_hr@10", "s_ndcg@10", "s_mrr"]:
            bl_val = best_bl_s.get(metric)
            pr_val = best_pr_s.get(metric)
            if bl_val and pr_val:
                imp = (pr_val - bl_val) / bl_val * 100
                lines.append(f"| {metric} | {fmt_pct(bl_val)} | {fmt_pct(pr_val)} | {imp:+.1f}% |")
        lines.append("")

    # --- File index ---
    lines.append("## Result Files")
    lines.append("")
    for e in entries:
        lines.append(f"- `{e['file']}` -> {e['method']}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare all thesis experiment results")
    parser.add_argument("--output", "-o", type=str, help="Output markdown file path")
    parser.add_argument("--latest", action="store_true", help="Only show latest run per method")
    parser.add_argument("--no-published", action="store_true", help="Skip published baseline references")
    args = parser.parse_args()

    entries = load_all_results()
    if not entries:
        print("No results found in results/ directory.")
        sys.exit(1)

    if args.latest:
        entries = filter_latest(entries)

    report = generate_report(entries, published_baselines=not args.no_published)

    # Output
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
