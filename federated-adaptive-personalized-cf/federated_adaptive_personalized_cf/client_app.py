"""federated-adaptive-personalized-cf: Split Learning for Personalized Collaborative Filtering.

This client implements SPLIT ARCHITECTURE where:
- GLOBAL params (item embeddings) are received from server and sent back
- LOCAL params (user embeddings) are persisted locally between rounds

Supports Adaptive Personalization (α):
- Computes per-user alpha based on interaction count
- Uses global prototype for sparse users to improve recommendations
- Sends user prototype to server for aggregation
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_adaptive_personalized_cf.task import (
    get_model, load_data, compute_client_alpha, get_user_stats
)
from federated_adaptive_personalized_cf.task import test as test_fn
from federated_adaptive_personalized_cf.task import train as train_fn
from federated_adaptive_personalized_cf.task import evaluate_ranking, evaluate_ranking_sampled
from federated_adaptive_personalized_cf.models import AlphaConfig
from federated_adaptive_personalized_cf.strategy import USER_PROTOTYPE_KEY

# Flower ClientApp
app = ClientApp()

# Cache for device detection (avoid repeated CUDA tests)
_device_cache = None

# Module directory for cache path
_MODULE_DIR = Path(__file__).parent


# =============================================================================
# User Embedding Persistence Functions
# =============================================================================

def get_cache_dir(partition_id: int, cache_dir: str = None) -> Path:
    """
    Get the cache directory for a specific partition's user embeddings.

    Args:
        partition_id: Client partition ID (0 to num_partitions-1)
        cache_dir: Override cache directory (default: project/.embedding_cache)

    Returns:
        Path to partition's cache directory
    """
    if cache_dir is None:
        # Default: project_root/.embedding_cache/partition_{id}
        cache_dir = _MODULE_DIR.parent / ".embedding_cache"
    else:
        cache_dir = Path(cache_dir)

    partition_dir = cache_dir / f"partition_{partition_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    return partition_dir


def save_local_user_embeddings(
    model,
    partition_id: int,
    cache_dir: str = None,
    round_num: int = None,
) -> None:
    """
    Save user embeddings to local cache.

    Called AFTER training completes, BEFORE sending updates to server.

    Args:
        model: The trained model (BasicMF or BPRMF)
        partition_id: Client partition ID
        cache_dir: Optional override for cache directory
        round_num: Optional round number for debugging
    """
    cache_path = get_cache_dir(partition_id, cache_dir)
    filepath = cache_path / "user_embeddings.pt"

    # Get local parameters from model
    local_params = model.get_local_parameters()

    # Add metadata
    local_params['_round'] = round_num
    local_params['_timestamp'] = datetime.now().isoformat()

    # Atomic save (write to temp file, then rename)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(cache_path),
            suffix='.tmp'
        )
        os.close(fd)
        torch.save(local_params, tmp_path)
        os.replace(tmp_path, str(filepath))
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise RuntimeError(f"Failed to save user embeddings: {e}")


def load_local_user_embeddings(
    model,
    partition_id: int,
    cache_dir: str = None,
) -> bool:
    """
    Load user embeddings from local cache.

    Called AFTER loading global params from server.

    Args:
        model: Model to load embeddings into (already has global params)
        partition_id: Client partition ID
        cache_dir: Optional override for cache directory

    Returns:
        True if embeddings were loaded, False if this is first round
    """
    cache_path = get_cache_dir(partition_id, cache_dir)
    filepath = cache_path / "user_embeddings.pt"

    if not filepath.exists():
        # First round - use model's default initialization
        return False

    try:
        local_state = torch.load(filepath, map_location='cpu', weights_only=False)

        # Remove metadata before loading
        local_state.pop('_round', None)
        local_state.pop('_timestamp', None)

        # Load local parameters into model
        loaded_keys, missing_keys = model.set_local_parameters(local_state, strict=False)

        if missing_keys:
            print(f"  Warning: Missing local param keys: {missing_keys}")

        return True

    except Exception as e:
        # Log warning but don't fail - use default initialization
        print(f"  Warning: Failed to load user embeddings for partition {partition_id}: {e}")
        return False


def clear_embedding_cache(partition_id: int = None, cache_dir: str = None) -> None:
    """
    Clear cached user embeddings.

    Args:
        partition_id: Specific partition to clear, or None for all
        cache_dir: Override cache directory
    """
    import shutil

    if cache_dir is None:
        cache_dir = _MODULE_DIR.parent / ".embedding_cache"
    else:
        cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return

    if partition_id is not None:
        # Clear specific partition
        partition_dir = cache_dir / f"partition_{partition_id}"
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
    else:
        # Clear all partitions
        shutil.rmtree(cache_dir)


# =============================================================================
# Device Detection
# =============================================================================

def get_device():
    """Get device with safe CUDA detection (handles incompatible GPU architectures)."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by creating a small tensor
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            _device_cache = torch.device("cuda:0")
        except RuntimeError:
            # CUDA available but not compatible (e.g., RTX 5090 with old PyTorch)
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")

    return _device_cache


# =============================================================================
# Training Function (Split Architecture)
# =============================================================================

@app.train()
def train(msg: Message, context: Context):
    """
    Train the Matrix Factorization model with SPLIT ARCHITECTURE.

    Split Learning Flow:
    1. Create model with default initialization
    2. Load GLOBAL params from server message (item embeddings)
    3. Load LOCAL params from cache (user embeddings, if exists)
    4. Set adaptive alpha and global prototype (if available)
    5. Train on local data
    6. Save LOCAL params to cache
    7. Return GLOBAL params and user prototype to server
    """
    # Get partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Get model configuration
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = context.run_config.get("embedding-dim", 64)
    dropout = context.run_config.get("dropout", 0.1)

    # Dual model specific configuration (Level 2: PersonalMLP)
    mlp_hidden_dims = None
    fusion_type = "add"
    if model_type == "dual":
        mlp_dims_str = context.run_config.get("mlp-hidden-dims", "128,64")
        mlp_hidden_dims = [int(d.strip()) for d in mlp_dims_str.split(",")]
        fusion_type = context.run_config.get("fusion-type", "add")

    # Step 1: Create model with default initialization
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        fusion_type=fusion_type,
    )

    # Step 2: Load GLOBAL parameters from server message
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # Step 3: Load LOCAL parameters from cache (if exists)
    loaded = load_local_user_embeddings(model, partition_id)
    if loaded:
        print(f"  Client {partition_id}: Loaded cached user embeddings")
    else:
        print(f"  Client {partition_id}: First round - using initialized user embeddings")

    # Move to device
    device = get_device()
    model.to(device)

    # === Adaptive Personalization: Set alpha and global prototype ===
    # Get alpha configuration from run config
    alpha_method = context.run_config.get("alpha-method", "data_quantity")

    # Build factor weights dict for multi-factor method
    factor_weights = {
        'quantity': context.run_config.get("alpha-weight-quantity", 0.40),
        'diversity': context.run_config.get("alpha-weight-diversity", 0.25),
        'coverage': context.run_config.get("alpha-weight-coverage", 0.20),
        'consistency': context.run_config.get("alpha-weight-consistency", 0.15),
    }

    alpha_config = AlphaConfig(
        method=alpha_method,
        min_alpha=context.run_config.get("alpha-min", 0.1),
        max_alpha=context.run_config.get("alpha-max", 0.95),
        quantity_threshold=context.run_config.get("alpha-quantity-threshold", 100),
        quantity_temperature=context.run_config.get("alpha-quantity-temperature", 0.05),
        factor_weights=factor_weights,
        max_entropy=context.run_config.get("alpha-max-entropy", 3.0),
        coverage_threshold=context.run_config.get("alpha-coverage-threshold", 100),
        max_rating_std=context.run_config.get("alpha-max-rating-std", 1.5),
    )

    # Load data with user stats for alpha computation
    dirichlet_alpha = context.run_config.get("alpha", 0.5)
    trainloader, _, user_stats = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=dirichlet_alpha,
        compute_stats=True,
    )

    # Compute client alpha (weighted average of per-user alphas)
    client_alpha = compute_client_alpha(user_stats, alpha_config)

    # Set alpha on model (only for BPRMF with adaptive support)
    if hasattr(model, 'set_alpha'):
        model.set_alpha(client_alpha)
        print(f"  Client {partition_id}: Set alpha = {client_alpha:.4f}")

    # Set global prototype from server (if available in config)
    global_prototype_list = msg.content["config"].get("global_prototype", None)
    if global_prototype_list is not None and hasattr(model, 'set_global_prototype'):
        global_prototype = torch.tensor(global_prototype_list, dtype=torch.float32)
        model.set_global_prototype(global_prototype)
        print(f"  Client {partition_id}: Set global prototype from server")

    # === FedProx: Save ONLY global parameters for proximal term ===
    proximal_mu = msg.content["config"].get("proximal_mu", 0.0)
    global_params_for_prox = None
    global_param_names = None

    if proximal_mu > 0:
        # Only save global params for proximal term
        # User embeddings should NOT be regularized toward server
        global_param_names = model.get_global_parameter_names()
        global_params_for_prox = []
        for name, p in model.named_parameters():
            if name in global_param_names:
                global_params_for_prox.append(p.detach().clone())

    # Step 5: Train the model
    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
        device=device,
        model_type=model_type,
        weight_decay=context.run_config.get("weight-decay", 1e-5),
        num_negatives=context.run_config.get("num-negatives", 1),
        # FedProx parameters (only for global params)
        proximal_mu=proximal_mu,
        global_params=global_params_for_prox,
        global_param_names=global_param_names,
    )

    # Step 6: Save LOCAL parameters to cache for next round
    save_local_user_embeddings(model, partition_id)

    # Step 7: Return GLOBAL parameters and user prototype to server
    global_params = model.get_global_parameters()
    model_record = ArrayRecord(global_params)

    # Compute user prototype for server aggregation
    user_prototype = None
    if hasattr(model, 'compute_user_prototype'):
        user_prototype = model.compute_user_prototype().detach().cpu().numpy().tolist()

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "client_alpha": float(client_alpha),
    }

    # Add user prototype to metrics for server aggregation
    if user_prototype is not None:
        metrics[USER_PROTOTYPE_KEY] = user_prototype

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


# =============================================================================
# Evaluation Function (Split Architecture)
# =============================================================================

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Evaluate the Matrix Factorization model with SPLIT ARCHITECTURE.

    Evaluation requires both global and local params:
    1. Load GLOBAL params from server message
    2. Load LOCAL params from cache (required for personalized evaluation)
    3. Set adaptive alpha and global prototype (matching training)
    4. Evaluate on local test data
    """
    # Get partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Get model configuration
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = context.run_config.get("embedding-dim", 64)
    dropout = context.run_config.get("dropout", 0.1)

    # Dual model specific configuration (Level 2: PersonalMLP)
    mlp_hidden_dims = None
    fusion_type = "add"
    if model_type == "dual":
        mlp_dims_str = context.run_config.get("mlp-hidden-dims", "128,64")
        mlp_hidden_dims = [int(d.strip()) for d in mlp_dims_str.split(",")]
        fusion_type = context.run_config.get("fusion-type", "add")

    # Step 1: Create model with default initialization
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        fusion_type=fusion_type,
    )

    # Step 2: Load GLOBAL parameters from server message
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # Step 3: Load LOCAL parameters from cache (required for evaluation)
    loaded = load_local_user_embeddings(model, partition_id)
    if not loaded:
        # This should only happen if evaluate is called before first train
        print(f"  Warning: No cached user embeddings for partition {partition_id}")

    # Move to device
    device = get_device()
    model.to(device)

    # === Adaptive Personalization: Set alpha and global prototype ===
    # Get alpha configuration from run config
    alpha_method = context.run_config.get("alpha-method", "data_quantity")

    # Build factor weights dict for multi-factor method
    factor_weights = {
        'quantity': context.run_config.get("alpha-weight-quantity", 0.40),
        'diversity': context.run_config.get("alpha-weight-diversity", 0.25),
        'coverage': context.run_config.get("alpha-weight-coverage", 0.20),
        'consistency': context.run_config.get("alpha-weight-consistency", 0.15),
    }

    alpha_config = AlphaConfig(
        method=alpha_method,
        min_alpha=context.run_config.get("alpha-min", 0.1),
        max_alpha=context.run_config.get("alpha-max", 0.95),
        quantity_threshold=context.run_config.get("alpha-quantity-threshold", 100),
        quantity_temperature=context.run_config.get("alpha-quantity-temperature", 0.05),
        factor_weights=factor_weights,
        max_entropy=context.run_config.get("alpha-max-entropy", 3.0),
        coverage_threshold=context.run_config.get("alpha-coverage-threshold", 100),
        max_rating_std=context.run_config.get("alpha-max-rating-std", 1.5),
    )

    # Load the data (both train and test for item popularity computation)
    dirichlet_alpha = context.run_config.get("alpha", 0.5)
    trainloader, testloader, user_stats = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=dirichlet_alpha,
        compute_stats=True,
    )

    # Compute client alpha (weighted average of per-user alphas)
    client_alpha = compute_client_alpha(user_stats, alpha_config)

    # Set alpha on model (only for BPRMF with adaptive support)
    if hasattr(model, 'set_alpha'):
        model.set_alpha(client_alpha)

    # Set global prototype from server (if available in config)
    global_prototype_list = msg.content.get("config", {}).get("global_prototype", None)
    if global_prototype_list is not None and hasattr(model, 'set_global_prototype'):
        global_prototype = torch.tensor(global_prototype_list, dtype=torch.float32)
        model.set_global_prototype(global_prototype)

    # Call the evaluation function (rating prediction metrics)
    eval_loss, metrics = test_fn(
        model=model,
        testloader=testloader,
        device=device,
        model_type=model_type,
    )

    # Construct result metrics
    result_metrics = {
        "eval_loss": eval_loss,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "num-examples": len(testloader.dataset),
        "client_alpha": float(client_alpha),
    }

    # Add ranking metrics if enabled
    enable_ranking_eval = context.run_config.get("enable-ranking-eval", True)
    if enable_ranking_eval:
        # Get K values from config (parse comma-separated string)
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        # Compute FULL-RANK ranking metrics (ranking among all items)
        ranking_metrics = evaluate_ranking(
            model=model,
            testloader=testloader,
            device=device,
            k_values=k_values,
            trainloader=trainloader,
        )

        # Add full-rank ranking metrics to results
        result_metrics.update(ranking_metrics)

        # Compute SAMPLED ranking metrics (leave-one-out with N negatives)
        # This follows the evaluation protocol used in NCF, FedMF, PFedRec papers
        num_negatives = context.run_config.get("eval-num-negatives", 99)
        sampled_metrics = evaluate_ranking_sampled(
            model=model,
            testloader=testloader,
            trainloader=trainloader,
            device=device,
            k_values=k_values,
            num_negatives=num_negatives,
        )

        # Add sampled ranking metrics to results
        result_metrics.update(sampled_metrics)

    metric_record = MetricRecord(result_metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
