"""federated-baseline-cf: A Flower / PyTorch app for Matrix Factorization.

Uses Grid message-passing API for federated orchestration with support
for early stopping and sampled evaluation metrics.
"""

import torch
import json
import wandb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common.record import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common.context import Context
from flwr.serverapp import Grid, ServerApp
from flwr.server.strategy import FedAvg, FedProx

from federated_baseline_cf.task import (
    get_model, test, evaluate_ranking, evaluate_ranking_sampled,
)
from federated_baseline_cf.dataset import load_full_data
from federated_baseline_cf.early_stopping import EarlyStopping

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

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client

    Returns:
        Dictionary of aggregated metrics
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    aggregated = {}

    if metrics:
        metric_keys = metrics[0][1].keys()

        for key in metric_keys:
            if key == "num-examples":
                continue

            # Only aggregate numeric values
            first_value = metrics[0][1].get(key)
            if not isinstance(first_value, (int, float)):
                continue

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
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        has_ranking = any(f"hit_rate@{k}" in metrics for k in k_values)

        if has_ranking:
            print("\n🎯 Ranking Metrics:")

            if "mrr" in metrics:
                print(f"  MRR:       {metrics['mrr']:.4f}")

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

            # Diversity/Popularity metrics
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

    # Early stopping configuration
    early_stopping_enabled = context.run_config.get("early-stopping-enabled", False)
    early_stopping_patience = context.run_config.get("early-stopping-patience", 10)
    early_stopping_metric = context.run_config.get("early-stopping-metric", "sampled_ndcg@10")
    early_stopping_mode = context.run_config.get("early-stopping-mode", "max")
    early_stopping_min_delta = context.run_config.get("early-stopping-min-delta", 0.001)

    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            metric_name=early_stopping_metric,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
            verbose=True,
        )
        print(f"  Early stopping: Enabled (patience={early_stopping_patience}, metric={early_stopping_metric})")

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
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_metric": early_stopping_metric,
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

    arrays = ArrayRecord(global_model.state_dict())

    # Initialize strategy based on configuration
    if strategy_name == "fedprox":
        strategy = FedProx(
            fraction_fit=fraction_train,
            proximal_mu=proximal_mu,
        )
        print(f"  Strategy: FedProx (proximal_mu={proximal_mu})")
    else:
        strategy = FedAvg(
            fraction_fit=fraction_train,
        )
        print(f"  Strategy: FedAvg")

    # Start federated learning
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

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*50}")

        train_config = ConfigRecord({"lr": lr, "proximal_mu": proximal_mu})

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
                "config": train_config,
            })
            msg = grid.create_message(
                content=content,
                message_type="train",
                dst_node_id=node_id,
                group_id=f"train_round_{round_num}",
            )
            train_messages.append(msg)

        train_responses = list(grid.send_and_receive(train_messages))

        # Parse training responses and aggregate
        fit_results = []
        round_train_metrics = []

        for response in train_responses:
            if response.has_error():
                print(f"  Warning: Client {response.metadata.src_node_id} returned error")
                continue

            resp_arrays = response.content.get("arrays", ArrayRecord())
            resp_metrics = response.content.get("metrics", MetricRecord())

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

            client_id = str(response.metadata.src_node_id)
            client_proxy = DummyClientProxy(client_id)
            fit_results.append((client_proxy, fit_res))

            round_train_metrics.append((num_examples, metrics_dict))

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
                param_keys = list(arrays.to_torch_state_dict().keys())
                new_state_dict = {k: torch.from_numpy(v) for k, v in zip(param_keys, param_ndarrays)}
                arrays = ArrayRecord(new_state_dict)

            # Aggregate training metrics
            train_metrics_history[round_num] = weighted_average_metrics(round_train_metrics)

            train_loss = train_metrics_history[round_num].get('train_loss', 'N/A')
            if isinstance(train_loss, (int, float)):
                print(f"  Training loss: {train_loss:.4f}")

        # =====================================================================
        # EVALUATION PHASE
        # =====================================================================
        eval_messages = []
        for node_id in selected_node_ids:
            eval_config = ConfigRecord({"lr": lr})
            content = RecordDict({
                "arrays": arrays,
                "config": eval_config,
            })
            msg = grid.create_message(
                content=content,
                message_type="evaluate",
                dst_node_id=node_id,
                group_id=f"eval_round_{round_num}",
            )
            eval_messages.append(msg)

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
            s_ndcg10 = eval_metrics_history[round_num].get('sampled_ndcg@10', 'N/A')
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            ndcg10_str = f"{ndcg10:.4f}" if isinstance(ndcg10, (int, float)) else str(ndcg10)
            s_ndcg10_str = f"{s_ndcg10:.4f}" if isinstance(s_ndcg10, (int, float)) else str(s_ndcg10)
            print(f"  RMSE: {rmse_str}")
            print(f"  NDCG@10: {ndcg10_str}")
            print(f"  Sampled NDCG@10: {s_ndcg10_str}")

        # Log to wandb
        if wandb_enabled:
            round_log = {"round": round_num}
            for key, value in train_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_log[f"train/{key}"] = value
            for key, value in eval_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_log[f"eval/{key}"] = value
            wandb.log(round_log, step=round_num)

        # Check early stopping
        if early_stopper is not None and round_eval_metrics:
            current_eval_metrics = eval_metrics_history.get(round_num, {})
            if early_stopper.step(round_num, current_eval_metrics):
                print(f"\n⏹ Training stopped early at round {round_num}")
                if wandb_enabled:
                    wandb.log({
                        "early_stopped": True,
                        "early_stopped_round": round_num,
                        "best_round": early_stopper.best_round,
                        f"best_{early_stopping_metric}": early_stopper.best_metric,
                    }, step=round_num)
                break

    # Determine actual rounds completed
    actual_rounds = round_num if early_stopper and early_stopper.state.should_stop else num_rounds

    # Print training complete message
    print("\n" + "="*70)
    print("FEDERATED TRAINING COMPLETE")
    print("="*70)
    print(f"Total rounds completed: {actual_rounds}/{num_rounds}")
    if early_stopper and early_stopper.state.should_stop:
        print(f"Early stopping: Triggered at round {actual_rounds}")
        print(f"Best {early_stopping_metric}: {early_stopper.best_metric:.4f} at round {early_stopper.best_round}")
    print("="*70)

    # =========================================================================
    # CENTRALIZED EVALUATION: Run evaluation on server with final model
    # =========================================================================
    print("\n📊 Running centralized evaluation with final model...")

    # Load final model weights
    final_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    final_state_dict = arrays.to_torch_state_dict()
    final_model.load_state_dict(final_state_dict)

    # Auto-detect device
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            device = torch.device("cuda:0")
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"  CUDA available but not compatible: {e}")
            print(f"  Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"  Using CPU")
    final_model.to(device)

    # Load full test data for evaluation
    trainloader, testloader, _, _, _, _ = load_full_data(
        test_ratio=0.2,
        batch_size=256,
    )

    # Compute rating prediction metrics (RMSE, MAE)
    print("  Computing rating prediction metrics...")
    eval_loss, rating_metrics = test(
        model=final_model,
        testloader=testloader,
        device=str(device),
        model_type=model_type,
    )

    # Compute ranking metrics
    print("  Computing ranking metrics...")
    k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
    k_values = [int(k.strip()) for k in k_values_str.split(",")]

    ranking_metrics = evaluate_ranking(
        model=final_model,
        testloader=testloader,
        device=str(device),
        k_values=k_values,
        trainloader=trainloader,
    )

    # Compute sampled ranking metrics
    print("  Computing sampled ranking metrics...")
    num_negatives = context.run_config.get("eval-num-negatives", 99)
    sampled_metrics = evaluate_ranking_sampled(
        model=final_model,
        testloader=testloader,
        trainloader=trainloader,
        device=str(device),
        k_values=k_values,
        num_negatives=num_negatives,
    )

    # Combine all metrics
    final_metrics = {
        "eval_loss": float(eval_loss),
        **rating_metrics,
        **ranking_metrics,
        **sampled_metrics,
    }

    # Print evaluation results
    print_evaluation_metrics(actual_rounds, final_metrics, context)

    # Log final metrics to wandb
    if wandb_enabled:
        final_log = {"round": actual_rounds + 1}
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                final_log[f"final/{key}"] = value
        wandb.log(final_log, step=actual_rounds + 1)

        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"final/{key}"] = value

    # Create results JSON
    results_data = {
        "model_name": f"{model_type.upper()}_MF_Federated_{strategy_name.upper()}",
        "dataset": "ml-1m",
        "federated_config": {
            "num_rounds": num_rounds,
            "actual_rounds": actual_rounds,
            "num_clients": len(list(grid.get_node_ids())),
            "fraction_train": fraction_train,
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "learning_rate": lr,
        },
        "early_stopping": early_stopper.get_summary() if early_stopper else None,
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": actual_rounds,
    }

    # Save results to JSON file
    print("\nSaving evaluation results...")
    results_dir = Path("../results/federated")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_filename = results_dir / f"{model_type}_mf_{strategy_name}_mu{proximal_mu}_r{num_rounds}_f{fraction_train}_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to: {results_filename.resolve()}")

    # Finish wandb run
    if wandb_enabled:
        wandb.finish()
        print("  Weights & Biases run completed")
