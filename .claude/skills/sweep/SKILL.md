---
name: sweep
description: Launch and manage Weights & Biases hyperparameter sweeps for federated experiments.
user-invocable: true
argument-hint: "[create|status|analyze] [sweep-id]"
allowed-tools: Bash, Read, Glob, Grep
---

# W&B Hyperparameter Sweep Management

Manage wandb sweeps for federated learning experiments.

## Subcommands

### `create` (default)
Validate `sweep.yaml` configuration and create a new sweep:
1. Read and validate `federated-adaptive-personalized-cf/sweep.yaml`
2. Check wandb authentication (`wandb whoami`)
3. Create sweep: `cd federated-adaptive-personalized-cf && wandb sweep sweep.yaml`
4. Report the sweep ID and agent command

### `status <sweep-id>`
Check status of a running sweep:
1. Report number of completed/running/pending runs
2. Show best run so far (highest NDCG@10)

### `analyze <sweep-id>`
Analyze completed sweep results:
1. Run `python scripts/analyze_sweep_results.py` if available
2. Report best hyperparameter configuration
3. Summarize key findings (best lr, embedding_dim, fusion_type, etc.)

## Notes

- Sweep config uses Bayesian optimization with Hyperband early termination
- Primary metric: maximize `final_ndcg@10`
- Early stopping is automatically enabled during sweeps
