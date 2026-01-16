"""federated-personalized-cf: Split Learning for Personalized Collaborative Filtering.

This server implements SPLIT ARCHITECTURE where:
- GLOBAL params (item embeddings) are sent to clients and aggregated
- LOCAL params (user embeddings) stay on clients (server never sees them)

Supports Adaptive Personalization (α):
- Aggregates user prototypes from clients into global prototype
- Logs alpha statistics and correlation with metrics
- Tracks per-round prototype norm for stability monitoring

NOTE: Centralized evaluation is NOT possible in split learning since the server
only has global parameters. Final metrics come from federated evaluation.

Uses Flower's Grid message-passing API for federated orchestration.
"""

import torch
import json
import wandb
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common.record import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common.context import Context
from flwr.serverapp import Grid, ServerApp

from federated_adaptive_personalized_cf.task import get_model
from federated_adaptive_personalized_cf.strategy import SplitFedAvg, SplitFedProx, USER_PROTOTYPE_KEY
from federated_adaptive_personalized_cf.evaluation import AlphaAnalyzer

# Create ServerApp
app = ServerApp()


class DummyClientProxy(ClientProxy):
    """Minimal ClientProxy for strategy compatibility."""

    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, ins, timeout, group_id):
        return None

    def get_parameters(self, ins, timeout, group_id):
        return None

    def fit(self, ins, timeout, group_id):
        return None

    def evaluate(self, ins, timeout, group_id):
        return None

    def reconnect(self, ins, timeout, group_id):
        return None


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

            # Only aggregate numeric values (skip lists, dicts, etc.)
            first_value = metrics[0][1].get(key)
            if not isinstance(first_value, (int, float)):
                continue

            # Weighted average: sum(metric * num_examples) / total_examples
            weighted_sum = sum(
                metrics_dict.get(key, 0.0) * num_examples
                for num_examples, metrics_dict in metrics
                if isinstance(metrics_dict.get(key, 0.0), (int, float))
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

        # Sampled ranking metrics (leave-one-out with N negatives)
        # For fair comparison with published baselines (NCF, FedMF, PFedRec)
        has_sampled = any(f"sampled_hr@{k}" in metrics for k in k_values)
        if has_sampled:
            num_neg = int(metrics.get('sampled_num_negatives', 99))
            print(f"\n🔬 Sampled Ranking Metrics (leave-one-out + {num_neg} negatives):")
            print("  (For fair comparison with NCF, FedMF, PFedRec baselines)")

            if "sampled_mrr" in metrics:
                print(f"\n  MRR:       {metrics['sampled_mrr']:.4f}")

            for k in sorted(k_values):
                print(f"\n  @ K={k}:")
                if f"sampled_hr@{k}" in metrics:
                    print(f"    Hit Rate:   {metrics[f'sampled_hr@{k}']:.4f}")
                if f"sampled_ndcg@{k}" in metrics:
                    print(f"    NDCG:       {metrics[f'sampled_ndcg@{k}']:.4f}")

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

    # Get adaptive alpha configuration
    alpha_min = context.run_config.get("alpha-min", 0.1)
    alpha_max = context.run_config.get("alpha-max", 0.95)
    alpha_quantity_threshold = context.run_config.get("alpha-quantity-threshold", 50)
    alpha_quantity_temperature = context.run_config.get("alpha-quantity-temperature", 0.1)
    prototype_momentum = context.run_config.get("prototype-momentum", 0.9)

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
            "dirichlet_alpha": context.run_config.get("alpha", 0.5),
            # Adaptive personalization config
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_quantity_threshold": alpha_quantity_threshold,
            "alpha_quantity_temperature": alpha_quantity_temperature,
            "prototype_momentum": prototype_momentum,
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

    # Dual model specific configuration (Level 2: PersonalMLP)
    mlp_hidden_dims = None
    fusion_type = "add"
    if model_type == "dual":
        mlp_dims_str = context.run_config.get("mlp-hidden-dims", "128,64")
        mlp_hidden_dims = [int(d.strip()) for d in mlp_dims_str.split(",")]
        fusion_type = context.run_config.get("fusion-type", "add")

    # Load global Matrix Factorization model
    print(f"\nInitializing {model_type.upper()} Matrix Factorization model...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Dropout: {dropout}")
    if model_type == "dual":
        print(f"  MLP hidden dims: {mlp_hidden_dims}")
        print(f"  Fusion type: {fusion_type}")

    global_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        fusion_type=fusion_type,
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
            fraction_fit=fraction_train,
            proximal_mu=proximal_mu,
            prototype_momentum=prototype_momentum,
        )
        print(f"  Strategy: SplitFedProx (proximal_mu={proximal_mu}, prototype_momentum={prototype_momentum})")
    else:
        strategy = SplitFedAvg(
            fraction_fit=fraction_train,
            prototype_momentum=prototype_momentum,
        )
        print(f"  Strategy: SplitFedAvg (prototype_momentum={prototype_momentum})")

    # Start strategy, run FedAvg for `num_rounds`
    print(f"\nStarting Federated Learning with {num_rounds} rounds...")
    print(f"  Clients per round: {fraction_train * 100:.0f}%")
    print(f"  Ranking evaluation: {'Enabled' if context.run_config.get('enable-ranking-eval', True) else 'Disabled'}")
    if context.run_config.get('enable-ranking-eval', True):
        k_values_str = context.run_config.get('ranking-k-values', "5,10,20")
        print(f"  K values: {k_values_str}")

    # =========================================================================
    # FEDERATED TRAINING LOOP using Grid's message-passing API
    # =========================================================================
    train_metrics_history: Dict[int, Dict] = {}
    eval_metrics_history: Dict[int, Dict] = {}
    per_client_metrics_history: Dict[int, List[Tuple[str, Dict]]] = {}  # Per-client metrics for alpha analysis

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*50}")

        # Build train config with global prototype (if available)
        train_config_dict = {"lr": lr, "proximal_mu": proximal_mu}

        # Add global prototype to config if available
        global_prototype = strategy.get_global_prototype()
        if global_prototype is not None:
            train_config_dict["global_prototype"] = global_prototype.tolist()

        # Get all node IDs and select a fraction for this round
        node_ids = list(grid.get_node_ids())
        num_selected = max(1, int(len(node_ids) * fraction_train))
        selected_node_ids = node_ids[:num_selected]

        print(f"  Selected {num_selected}/{len(node_ids)} clients for training")

        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        train_messages = []
        for node_id in selected_node_ids:
            content = RecordDict({
                "arrays": arrays,
                "config": ConfigRecord(train_config_dict),
            })
            msg = grid.create_message(
                content=content,
                message_type="train",
                dst_node_id=node_id,
                group_id=f"train_round_{round_num}",
            )
            train_messages.append(msg)

        # Send training messages and receive responses
        train_responses = list(grid.send_and_receive(train_messages))

        # Parse training responses into FitRes format for strategy
        fit_results = []
        round_train_metrics = []
        per_client_metrics_history[round_num] = []  # Initialize per-client list for this round

        for response in train_responses:
            if response.has_error():
                print(f"  Warning: Client {response.metadata.src_node_id} returned error")
                continue

            # Extract arrays and metrics from response
            resp_arrays = response.content.get("arrays", ArrayRecord())
            resp_metrics = response.content.get("metrics", MetricRecord())

            # Convert to dict
            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get("num-examples", 1))

            # Create FitRes for strategy aggregation
            parameters = ndarrays_to_parameters(list(resp_arrays.to_torch_state_dict().values()))
            fit_res = FitRes(
                status=None,
                parameters=parameters,
                num_examples=num_examples,
                metrics=metrics_dict,
            )

            # Create dummy client proxy
            client_id = str(response.metadata.src_node_id)
            client_proxy = DummyClientProxy(client_id)
            fit_results.append((client_proxy, fit_res))

            # Collect metrics for aggregation
            round_train_metrics.append((num_examples, metrics_dict))

            # Store per-client metrics for alpha analysis (before aggregation)
            per_client_metrics_history[round_num].append((client_id, metrics_dict))

        # Aggregate training results using strategy
        if fit_results:
            aggregated_params, agg_metrics = strategy.aggregate_fit(
                server_round=round_num,
                results=fit_results,
                failures=[],
            )

            # Update global parameters for next round
            if aggregated_params is not None:
                param_ndarrays = parameters_to_ndarrays(aggregated_params)
                # Rebuild state dict from ndarrays
                param_keys = list(arrays.to_torch_state_dict().keys())
                new_state_dict = {k: torch.from_numpy(v) for k, v in zip(param_keys, param_ndarrays)}
                arrays = ArrayRecord(new_state_dict)

            # Aggregate training metrics
            train_metrics_history[round_num] = weighted_average_metrics(round_train_metrics)
            train_metrics_history[round_num].update(agg_metrics)  # Add strategy metrics

            print(f"  Training loss: {train_metrics_history[round_num].get('train_loss', 'N/A'):.4f}")

        # =====================================================================
        # EVALUATION PHASE
        # =====================================================================
        eval_messages = []
        for node_id in selected_node_ids:
            eval_config_dict = {"lr": lr}
            if global_prototype is not None:
                eval_config_dict["global_prototype"] = global_prototype.tolist()

            content = RecordDict({
                "arrays": arrays,
                "config": ConfigRecord(eval_config_dict),
            })
            msg = grid.create_message(
                content=content,
                message_type="evaluate",
                dst_node_id=node_id,
                group_id=f"eval_round_{round_num}",
            )
            eval_messages.append(msg)

        # Send evaluation messages and receive responses
        eval_responses = list(grid.send_and_receive(eval_messages))

        # Parse evaluation responses
        round_eval_metrics = []
        for response in eval_responses:
            if response.has_error():
                continue

            resp_metrics = response.content.get("metrics", MetricRecord())
            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get("num-examples", 1))
            round_eval_metrics.append((num_examples, metrics_dict))

        # Aggregate evaluation metrics
        if round_eval_metrics:
            eval_metrics_history[round_num] = weighted_average_metrics(round_eval_metrics)

            # Print key metrics
            rmse = eval_metrics_history[round_num].get('rmse', 'N/A')
            ndcg10 = eval_metrics_history[round_num].get('ndcg@10', 'N/A')
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            ndcg10_str = f"{ndcg10:.4f}" if isinstance(ndcg10, (int, float)) else str(ndcg10)
            print(f"  RMSE: {rmse_str}")
            print(f"  NDCG@10: {ndcg10_str}")

        # Log to wandb
        if wandb_enabled:
            round_metrics = {"round": round_num}
            for key, value in train_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_metrics[f"train/{key}"] = value
            for key, value in eval_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
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
    final_round_metrics = eval_metrics_history.get(num_rounds, {})

    if not final_round_metrics:
        print("  Warning: No evaluation metrics from final round")
        final_metrics = {}
    else:
        # Use federated metrics (already aggregated)
        final_metrics = dict(final_round_metrics)

    # Print evaluation results
    print_evaluation_metrics(num_rounds, final_metrics, context)

    # Log final metrics to wandb
    if wandb_enabled:
        # Log final metrics at step num_rounds + 1 (after all round metrics)
        final_log = {"round": num_rounds + 1}
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                final_log[f"final/{key}"] = value
        wandb.log(final_log, step=num_rounds + 1)

        # Also add to summary for easy comparison in W&B dashboard
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"final/{key}"] = value

    # Analyze alpha values from per-client metrics (not aggregated per-round values)
    alpha_analyzer = AlphaAnalyzer()
    alpha_entry_id = 0
    for round_num, client_metrics_list in per_client_metrics_history.items():
        for client_id, metrics_dict in client_metrics_list:
            if "client_alpha" in metrics_dict:
                alpha_analyzer.add_client_data(
                    client_id=alpha_entry_id,  # Unique integer ID for each client-round pair
                    alpha=metrics_dict["client_alpha"],
                    metrics={k: v for k, v in metrics_dict.items() if k not in ["client_alpha", "num-examples"] and isinstance(v, (int, float))}
                )
                alpha_entry_id += 1

    # Log alpha statistics
    alpha_stats = alpha_analyzer.compute_statistics()
    if alpha_stats.count > 0:
        print("\n📊 Adaptive Alpha Analysis:")
        print(f"  Mean alpha: {alpha_stats.mean:.4f} (std: {alpha_stats.std:.4f})")
        print(f"  Range: [{alpha_stats.min:.4f}, {alpha_stats.max:.4f}]")
        print(f"  Quartiles: Q25={alpha_stats.q25:.4f}, Median={alpha_stats.median:.4f}, Q75={alpha_stats.q75:.4f}")

        if wandb_enabled:
            wandb.run.summary["alpha/mean"] = alpha_stats.mean
            wandb.run.summary["alpha/std"] = alpha_stats.std
            wandb.run.summary["alpha/min"] = alpha_stats.min
            wandb.run.summary["alpha/max"] = alpha_stats.max

    # Get global prototype info from strategy
    global_prototype = strategy.get_global_prototype()
    prototype_norm = float(np.linalg.norm(global_prototype)) if global_prototype is not None else None

    if prototype_norm is not None:
        print(f"\n🔮 Global Prototype:")
        print(f"  Final norm: {prototype_norm:.4f}")
        if wandb_enabled:
            wandb.run.summary["prototype/final_norm"] = prototype_norm

    # Create results JSON structure similar to centralized results
    # Determine local params based on model type
    local_params_list = ["user_embeddings", "user_bias"]
    if model_type == "dual":
        local_params_list.extend(["personal_mlp", "fusion_gate" if fusion_type == "gate" else "fusion_layer" if fusion_type == "concat" else None])
        local_params_list = [p for p in local_params_list if p is not None]

    results_data = {
        "model_name": f"{model_type.upper()}_MF_Personalized_Split_{strategy_name.upper()}_Adaptive",
        "dataset": "ml-1m",
        "architecture": "split_learning_adaptive" if model_type != "dual" else "dual_level_personalization",
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
            "local_params": local_params_list,
            # Dual model specific config
            "mlp_hidden_dims": mlp_hidden_dims if model_type == "dual" else None,
            "fusion_type": fusion_type if model_type == "dual" else None,
        },
        "adaptive_config": {
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "quantity_threshold": alpha_quantity_threshold,
            "quantity_temperature": alpha_quantity_temperature,
            "prototype_momentum": prototype_momentum,
        },
        "alpha_analysis": {
            "mean": alpha_stats.mean,
            "std": alpha_stats.std,
            "min": alpha_stats.min,
            "max": alpha_stats.max,
            "median": alpha_stats.median,
        } if alpha_stats.count > 0 else None,
        "global_prototype_norm": prototype_norm,
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": num_rounds,
    }

    # Save results to JSON file (in personalized subfolder)
    print("\nSaving evaluation results...")
    results_dir = Path("../results/federated/personalized")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename with dual model specific suffix
    if model_type == "dual" and mlp_hidden_dims is not None:
        mlp_dims_str = "-".join(str(d) for d in mlp_hidden_dims)
        results_filename = results_dir / f"{model_type}_mf_split_{strategy_name}_mu{proximal_mu}_r{num_rounds}_f{fraction_train}_{fusion_type}_mlp{mlp_dims_str}_results.json"
    else:
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
