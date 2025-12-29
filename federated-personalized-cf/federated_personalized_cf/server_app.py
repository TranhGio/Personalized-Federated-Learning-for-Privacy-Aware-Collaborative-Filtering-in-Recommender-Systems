"""federated-personalized-cf: Split Learning for Personalized Collaborative Filtering.

This server implements SPLIT ARCHITECTURE where:
- GLOBAL params (item embeddings) are sent to clients and aggregated
- LOCAL params (user embeddings) stay on clients (server never sees them)

NOTE: Centralized evaluation is NOT possible in split learning since the server
only has global parameters. Final metrics come from federated evaluation.
"""

import torch
import json
import wandb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from federated_personalized_cf.task import get_model
from federated_personalized_cf.strategy import SplitFedAvg, SplitFedProx, GLOBAL_PARAM_KEYS

# Create ServerApp
app = ServerApp()


def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Aggregate evaluation metrics from multiple clients using weighted average.

    NOTE: This function is available for custom metric aggregation but is not
    currently used. Flower's new ServerApp API handles metric aggregation
    automatically based on num-examples.

    This function aggregates both rating prediction metrics (RMSE, MAE) and
    ranking metrics (Hit Rate, Precision, Recall, NDCG, MRR) across clients.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client

    Returns:
        Dictionary of aggregated metrics
    """
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Aggregate metrics using weighted average
    aggregated = {}

    # Get all metric keys from first client (assumes all clients report same metrics)
    if metrics:
        metric_keys = metrics[0][1].keys()

        for key in metric_keys:
            if key == "num-examples":
                continue

            # Weighted average: sum(metric * num_examples) / total_examples
            weighted_sum = sum(
                metrics_dict.get(key, 0.0) * num_examples
                for num_examples, metrics_dict in metrics
            )
            aggregated[key] = weighted_sum / total_examples

    return aggregated


def print_evaluation_metrics(round_num: int, metrics: Dict[str, float], context: Context):
    """
    Pretty print evaluation metrics for a federated round.

    Args:
        round_num: Current federated learning round
        metrics: Aggregated metrics dictionary
        context: Flower context with configuration
    """
    print(f"\n{'='*70}")
    print(f"Evaluation Results - Round {round_num}")
    print(f"{'='*70}")

    # Rating prediction metrics
    if "rmse" in metrics or "mae" in metrics:
        print("\n📊 Rating Prediction Metrics:")
        if "eval_loss" in metrics:
            print(f"  Loss:      {metrics['eval_loss']:.4f}")
        if "rmse" in metrics:
            print(f"  RMSE:      {metrics['rmse']:.4f}")
        if "mae" in metrics:
            print(f"  MAE:       {metrics['mae']:.4f}")

    # Ranking metrics
    enable_ranking = context.run_config.get("enable-ranking-eval", True)
    if enable_ranking:
        # Parse K values from comma-separated string
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        # Check if we have any ranking metrics
        has_ranking = any(f"hit_rate@{k}" in metrics for k in k_values)

        if has_ranking:
            print("\n🎯 Ranking Metrics:")

            # MRR (not K-dependent)
            if "mrr" in metrics:
                print(f"  MRR:       {metrics['mrr']:.4f}")

            # Metrics for each K value
            for k in sorted(k_values):
                print(f"\n  @ K={k}:")
                if f"hit_rate@{k}" in metrics:
                    print(f"    Hit Rate:   {metrics[f'hit_rate@{k}']:.4f}")
                if f"precision@{k}" in metrics:
                    print(f"    Precision:  {metrics[f'precision@{k}']:.4f}")
                if f"recall@{k}" in metrics:
                    print(f"    Recall:     {metrics[f'recall@{k}']:.4f}")
                if f"f1@{k}" in metrics:
                    print(f"    F1:         {metrics[f'f1@{k}']:.4f}")
                if f"ndcg@{k}" in metrics:
                    print(f"    NDCG:       {metrics[f'ndcg@{k}']:.4f}")
                if f"map@{k}" in metrics:
                    print(f"    MAP:        {metrics[f'map@{k}']:.4f}")

            # Diversity/Popularity metrics (only for first K value to avoid repetition)
            k = sorted(k_values)[0]
            has_diversity = any(f"{m}@{k}" in metrics for m in ['coverage', 'novelty'])
            if has_diversity:
                print("\n📈 Diversity/Popularity Metrics:")
                for k in sorted(k_values):
                    print(f"\n  @ K={k}:")
                    if f"coverage@{k}" in metrics:
                        print(f"    Coverage:   {metrics[f'coverage@{k}']:.4f}")
                    if f"novelty@{k}" in metrics:
                        print(f"    Novelty:    {metrics[f'novelty@{k}']:.4f}")

    print(f"\n{'='*70}\n")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    model_type: str = context.run_config.get("model-type", "bpr")
    embedding_dim: int = context.run_config.get("embedding-dim", 64)
    dropout: float = context.run_config.get("dropout", 0.1)

    # FedProx configuration
    strategy_name: str = context.run_config.get("strategy", "fedavg").lower()
    proximal_mu: float = context.run_config.get("proximal-mu", 0.0)

    # Initialize Weights & Biases if enabled
    wandb_enabled = context.run_config.get("wandb-enabled", False)
    if wandb_enabled:
        wandb_config = {
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "local_epochs": context.run_config.get("local-epochs", 5),
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": context.run_config.get("weight-decay", 1e-5),
            "alpha": context.run_config.get("alpha", 0.5),
        }
        wandb_project = context.run_config.get("wandb-project", "federated-cf")
        wandb_entity = context.run_config.get("wandb-entity", "")
        wandb_run_name = context.run_config.get("wandb-run-name", "")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            name=wandb_run_name if wandb_run_name else None,
            config=wandb_config,
        )
        print("  Weights & Biases: Enabled")

    # Load global Matrix Factorization model
    print(f"\nInitializing {model_type.upper()} Matrix Factorization model...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Dropout: {dropout}")

    global_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Only send global params to clients (item embeddings, item bias, global bias)
    # User embeddings stay on clients (split architecture)
    arrays = ArrayRecord(global_model.get_global_parameters())

    # Initialize strategy based on configuration
    # Note: Flower automatically does weighted averaging of metrics based on num-examples
    # Using Split strategies that only aggregate global params (item embeddings)
    if strategy_name == "fedprox":
        strategy = SplitFedProx(
            fraction_train=fraction_train,
            proximal_mu=proximal_mu,
        )
        print(f"  Strategy: SplitFedProx (proximal_mu={proximal_mu})")
    else:
        strategy = SplitFedAvg(
            fraction_train=fraction_train,
        )
        print(f"  Strategy: SplitFedAvg")

    # Start strategy, run FedAvg for `num_rounds`
    print(f"\nStarting Federated Learning with {num_rounds} rounds...")
    print(f"  Clients per round: {fraction_train * 100:.0f}%")
    print(f"  Ranking evaluation: {'Enabled' if context.run_config.get('enable-ranking-eval', True) else 'Disabled'}")
    if context.run_config.get('enable-ranking-eval', True):
        k_values_str = context.run_config.get('ranking-k-values', "5,10,20")
        print(f"  K values: {k_values_str}")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "proximal_mu": proximal_mu}),
        num_rounds=num_rounds,
    )

    # Log per-round metrics to wandb
    if wandb_enabled:
        # Combine training and evaluation metrics for each round into a single log call
        # (wandb requires monotonically increasing steps)
        all_rounds = set(result.train_metrics_clientapp.keys()) | set(result.evaluate_metrics_clientapp.keys())
        for round_num in sorted(all_rounds):
            round_metrics = {"round": round_num}

            # Add training metrics for this round
            if round_num in result.train_metrics_clientapp:
                for key, value in result.train_metrics_clientapp[round_num].items():
                    round_metrics[f"train/{key}"] = value

            # Add evaluation metrics for this round
            if round_num in result.evaluate_metrics_clientapp:
                for key, value in result.evaluate_metrics_clientapp[round_num].items():
                    round_metrics[f"eval/{key}"] = value

            wandb.log(round_metrics, step=round_num)

    # Print training complete message
    print("\n" + "="*70)
    print("FEDERATED TRAINING COMPLETE")
    print("="*70)
    print(f"Total rounds completed: {num_rounds}")
    print("="*70)

    # =========================================================================
    # FEDERATED EVALUATION: Use aggregated metrics from final round
    # =========================================================================
    # NOTE: Centralized evaluation is NOT possible in split learning since the
    # server only has global parameters (item embeddings). User embeddings
    # remain on clients and are never sent to the server.
    print("\n📊 Using federated evaluation metrics from final round...")
    print("  (Centralized evaluation not possible in split learning)")

    # Get final round metrics from federated evaluation
    final_round_metrics = result.evaluate_metrics_clientapp.get(num_rounds, {})

    if not final_round_metrics:
        print("  Warning: No evaluation metrics from final round")
        final_metrics = {}
    else:
        # Use federated metrics (already aggregated by Flower)
        final_metrics = dict(final_round_metrics)

    # Print evaluation results
    print_evaluation_metrics(num_rounds, final_metrics, context)

    # Log final metrics to wandb
    if wandb_enabled:
        # Log final metrics at step num_rounds + 1 (after all round metrics)
        final_log = {"round": num_rounds + 1}
        for key, value in final_metrics.items():
            final_log[f"final/{key}"] = value
        wandb.log(final_log, step=num_rounds + 1)

        # Also add to summary for easy comparison in W&B dashboard
        for key, value in final_metrics.items():
            wandb.run.summary[f"final/{key}"] = value

    # Create results JSON structure similar to centralized results
    results_data = {
        "model_name": f"{model_type.upper()}_MF_Personalized_Split_{strategy_name.upper()}",
        "dataset": "ml-1m",
        "architecture": "split_learning",
        "federated_config": {
            "num_rounds": num_rounds,
            "num_clients": 10,  # Adjust based on your config
            "fraction_train": fraction_train,
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "learning_rate": lr,
            "split_learning": True,
            "global_params": ["item_embeddings", "item_bias", "global_bias"],
            "local_params": ["user_embeddings", "user_bias"],
        },
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": num_rounds,
    }

    # Save results to JSON file (in personalized subfolder)
    print("\nSaving evaluation results...")
    results_dir = Path("../results/federated/personalized")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_filename = results_dir / f"{model_type}_mf_split_{strategy_name}_mu{proximal_mu}_r{num_rounds}_f{fraction_train}_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to: {results_filename.resolve()}")

    # Optionally save model weights (commented out by default)
    # print("\nSaving final model weights...")
    # state_dict = result.arrays.to_torch_state_dict()
    # model_filename = results_dir / f"final_model_{model_type}_d{embedding_dim}.pt"
    # torch.save(state_dict, model_filename)
    # print(f"Model saved to: {model_filename.resolve()}")

    # Finish wandb run
    if wandb_enabled:
        wandb.finish()
        print("  Weights & Biases run completed")
