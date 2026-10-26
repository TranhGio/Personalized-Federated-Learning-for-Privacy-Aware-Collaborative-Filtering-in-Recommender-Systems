---
name: analyze-results
description: Analyze federated learning experiment results, compare metrics, and provide insights.
user-invocable: true
argument-hint: "[result-file-or-latest]"
allowed-tools: Bash, Read, Glob, Grep, Task
---

# Analyze Federated Learning Results

Analyze experiment results from the `results/` directory.

## Arguments

- `$ARGUMENTS[0]`: Path to a specific result JSON file, or "latest" to analyze the most recent result (default: "latest")

## Analysis Steps

1. **Locate results**: Search `results/federated/` and `results/centralized/` for result JSON files
2. **Parse metrics**: Extract key metrics (NDCG@10, Hit Rate@10, Precision@10, RMSE, MAE)
3. **Compare**: If multiple results exist, create a comparison table
4. **Baseline comparison**: Compare federated results against centralized baselines in `results/centralized/`
5. **User group analysis**: Break down performance by sparse/medium/dense user groups if available
6. **Recommendations**: Suggest next experiments based on the findings

Use the `federated-results-analyzer` agent for deep analysis when needed.
