#!/usr/bin/env python3
"""
Weights & Biases Sweep Runner for Federated Adaptive Personalized CF.

This script is called by wandb agent during hyperparameter sweeps.
It reads hyperparameters from WANDB_CONFIG environment variable and runs
the federated training, letting server_app.py handle wandb initialization.

Usage:
    # Create a sweep (run once)
    wandb sweep sweep.yaml

    # Run agents (can run multiple in parallel)
    wandb agent <YOUR_ENTITY>/<PROJECT>/<SWEEP_ID>

    # Or use this script directly with wandb initialized
    python scripts/run_wandb_sweep.py

For manual testing without wandb:
    python scripts/run_wandb_sweep.py --test
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_default_config() -> Dict[str, Any]:
    """Default configuration for testing without wandb."""
    return {
        "model_type": "bpr",
        "embedding_dim": 128,
        "mlp_hidden_dims": "128,64",
        "fusion_type": "add",
        "lr": 0.005,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "local_epochs": 12,
        "num_negatives": 1,
        "num_server_rounds": 50,
        "fraction_train": 1.0,
        "strategy": "fedavg",
        "proximal_mu": 0.01,
        "alpha_method": "hierarchical_conditional",
        "alpha_min": 0.1,
        "alpha_max": 0.95,
        "alpha_quantity_threshold": 100,
        "alpha_quantity_temperature": 0.05,
        # Multi-factor weights (used when alpha_method = "multi_factor")
        "alpha_weight_quantity": 0.25,
        "alpha_weight_diversity": 0.35,
        "alpha_weight_coverage": 0.20,
        "alpha_weight_consistency": 0.20,
        # Hierarchical Conditional parameters (used when alpha_method = "hierarchical_conditional")
        "alpha_hc_data_volume_weight": 0.55,
        "alpha_hc_preference_weight": 0.45,
        "alpha_hc_sparse_threshold": 20,
        "alpha_hc_sparse_penalty_max": 0.5,
        "alpha_hc_niche_diversity_threshold": 0.25,
        "alpha_hc_niche_quantity_threshold": 0.6,
        "alpha_hc_niche_bonus": 0.15,
        "alpha_hc_inconsistent_threshold": 0.3,
        "alpha_hc_inconsistent_penalty": 0.3,
        "alpha_hc_completionist_coverage": 0.7,
        "alpha_hc_completionist_diversity": 0.3,
        "alpha_hc_completionist_bonus": 0.1,
        "prototype_momentum": 0.9,
        "early_stopping_patience": 10,
        "early_stopping_metric": "sampled_ndcg@10",
        "early_stopping_min_delta": 0.001,
    }


def get_config_from_env() -> Optional[Dict[str, Any]]:
    """
    Read config from WANDB_CONFIG environment variable.

    wandb agent sets this as a JSON string with all hyperparameters.
    This avoids calling wandb.init() in this script, letting server_app.py
    handle the wandb run initialization and logging.
    """
    wandb_config_str = os.environ.get("WANDB_CONFIG")
    if not wandb_config_str:
        return None

    try:
        # WANDB_CONFIG is a JSON string
        config = json.loads(wandb_config_str)
        # The config might be nested under different keys depending on wandb version
        # Usually it's a flat dict with parameter values
        if isinstance(config, dict):
            # Extract just the values if it's in the format {"param": {"value": x}}
            flat_config = {}
            for key, value in config.items():
                if isinstance(value, dict) and "value" in value:
                    flat_config[key] = value["value"]
                else:
                    flat_config[key] = value
            return flat_config
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse WANDB_CONFIG: {e}")
        return None

    return None


def normalize_weights(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize alpha weights to sum to 1.0.

    wandb sweeps may sample weights independently, so we need to normalize.
    """
    # Normalize multi-factor weights
    mf_weight_keys = [
        "alpha_weight_quantity",
        "alpha_weight_diversity",
        "alpha_weight_coverage",
        "alpha_weight_consistency",
    ]

    mf_total = sum(config.get(k, 0.25) for k in mf_weight_keys)

    if mf_total > 0:
        for k in mf_weight_keys:
            if k in config:
                config[k] = config[k] / mf_total

    # Normalize hierarchical conditional weights
    hc_weight_keys = [
        "alpha_hc_data_volume_weight",
        "alpha_hc_preference_weight",
    ]

    hc_total = sum(config.get(k, 0.5) for k in hc_weight_keys)

    if hc_total > 0:
        for k in hc_weight_keys:
            if k in config:
                config[k] = config[k] / hc_total

    return config


def build_run_config(config: Dict[str, Any], enable_wandb: bool = True) -> str:
    """
    Convert wandb config to flwr run --run-config string.

    Args:
        config: Dictionary of hyperparameters from wandb
        enable_wandb: Whether to enable wandb logging in server_app.py

    Returns:
        String formatted for flwr run --run-config
    """
    # Map from wandb config keys to flwr config keys
    key_mapping = {
        "model_type": "model-type",
        "embedding_dim": "embedding-dim",
        "mlp_hidden_dims": "mlp-hidden-dims",
        "fusion_type": "fusion-type",
        "lr": "lr",
        "weight_decay": "weight-decay",
        "dropout": "dropout",
        "local_epochs": "local-epochs",
        "num_negatives": "num-negatives",
        "num_server_rounds": "num-server-rounds",
        "fraction_train": "fraction-train",
        "strategy": "strategy",
        "proximal_mu": "proximal-mu",
        "alpha_method": "alpha-method",
        "alpha_min": "alpha-min",
        "alpha_max": "alpha-max",
        "alpha_quantity_threshold": "alpha-quantity-threshold",
        "alpha_quantity_temperature": "alpha-quantity-temperature",
        # Multi-factor weights
        "alpha_weight_quantity": "alpha-weight-quantity",
        "alpha_weight_diversity": "alpha-weight-diversity",
        "alpha_weight_coverage": "alpha-weight-coverage",
        "alpha_weight_consistency": "alpha-weight-consistency",
        # Hierarchical Conditional parameters
        "alpha_hc_data_volume_weight": "alpha-hc-data-volume-weight",
        "alpha_hc_preference_weight": "alpha-hc-preference-weight",
        "alpha_hc_sparse_threshold": "alpha-hc-sparse-threshold",
        "alpha_hc_sparse_penalty_max": "alpha-hc-sparse-penalty-max",
        "alpha_hc_niche_diversity_threshold": "alpha-hc-niche-diversity-threshold",
        "alpha_hc_niche_quantity_threshold": "alpha-hc-niche-quantity-threshold",
        "alpha_hc_niche_bonus": "alpha-hc-niche-bonus",
        "alpha_hc_inconsistent_threshold": "alpha-hc-inconsistent-threshold",
        "alpha_hc_inconsistent_penalty": "alpha-hc-inconsistent-penalty",
        "alpha_hc_completionist_coverage": "alpha-hc-completionist-coverage",
        "alpha_hc_completionist_diversity": "alpha-hc-completionist-diversity",
        "alpha_hc_completionist_bonus": "alpha-hc-completionist-bonus",
        "prototype_momentum": "prototype-momentum",
        "early_stopping_patience": "early-stopping-patience",
        "early_stopping_metric": "early-stopping-metric",
        "early_stopping_min_delta": "early-stopping-min-delta",
    }

    # Build config parts
    config_parts = []

    # Always enable early stopping for sweeps
    config_parts.append("early-stopping-enabled=true")

    # Enable wandb if requested (and running under sweep agent)
    if enable_wandb:
        config_parts.append("wandb-enabled=true")

        # Generate run name from key hyperparameters
        model = config.get("model_type", "bpr")
        emb = config.get("embedding_dim", 128)
        lr = config.get("lr", 0.005)
        strategy = config.get("strategy", "fedavg")
        run_name = f"sweep_{model}_e{emb}_lr{lr:.4f}_{strategy}"
        config_parts.append(f'wandb-run-name="{run_name}"')

    # Add all config values
    for wandb_key, flwr_key in key_mapping.items():
        if wandb_key in config:
            value = config[wandb_key]

            # Handle string values (need quotes)
            if isinstance(value, str):
                config_parts.append(f'{flwr_key}="{value}"')
            # Handle numeric values
            elif isinstance(value, float):
                config_parts.append(f"{flwr_key}={value:.6f}")
            else:
                config_parts.append(f"{flwr_key}={value}")

    return " ".join(config_parts)


def run_training(config: Dict[str, Any], dry_run: bool = False, enable_wandb: bool = True) -> Optional[int]:
    """
    Run federated training with given configuration.

    Args:
        config: Hyperparameter configuration
        dry_run: If True, print command without executing
        enable_wandb: Whether to enable wandb in server_app.py

    Returns:
        Return code from flwr run, or None if dry_run
    """
    # Normalize weights
    config = normalize_weights(config)

    # Build run config string
    run_config = build_run_config(config, enable_wandb=enable_wandb)

    # Build command
    cmd = ["flwr", "run", ".", "--run-config", run_config]

    print("\n" + "=" * 70)
    print("Running Federated Training")
    print("=" * 70)
    print(f"Config: {run_config[:100]}...")
    print("=" * 70 + "\n")

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return None

    # Execute training
    # Pass through environment variables so server_app.py can use wandb sweep context
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,  # Don't raise on non-zero exit
            env=os.environ.copy(),  # Pass all env vars including WANDB_*
        )
        return result.returncode
    except Exception as e:
        print(f"Error running training: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run wandb sweep training")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with default config (no wandb)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )
    args = parser.parse_args()

    if args.test:
        # Test mode: use default config, no wandb
        print("Running in test mode with default configuration")
        config = get_default_config()
        return run_training(config, dry_run=args.dry_run, enable_wandb=False)

    # Try to get config from environment (set by wandb agent)
    config = get_config_from_env()

    if config:
        print(f"Received config from WANDB_CONFIG env: {config}")
        # Let server_app.py handle wandb initialization
        # It will automatically connect to the sweep via WANDB_* env vars
        return_code = run_training(config, dry_run=args.dry_run, enable_wandb=True)
        return return_code
    else:
        # Fallback: try using wandb.init() if env var not available
        print("WANDB_CONFIG not found, trying wandb.init()...")
        try:
            import wandb
            wandb.init()
            config = dict(wandb.config)
            print(f"Received config from wandb.config: {config}")

            # IMPORTANT: Disable wandb in server_app.py to avoid duplicate runs
            # The sweep will track this run via wandb.init() above
            return_code = run_training(config, dry_run=args.dry_run, enable_wandb=False)

            # Log completion status to the sweep run
            if return_code == 0:
                wandb.log({"training_completed": True})
            else:
                wandb.log({"training_completed": False, "exit_code": return_code})

            wandb.finish(exit_code=return_code or 0)
            return return_code
        except ImportError:
            print("Warning: wandb not installed. Running in test mode.")
            config = get_default_config()
            return run_training(config, dry_run=args.dry_run, enable_wandb=False)
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            print("Falling back to test mode")
            config = get_default_config()
            return run_training(config, dry_run=args.dry_run, enable_wandb=False)


if __name__ == "__main__":
    sys.exit(main() or 0)
