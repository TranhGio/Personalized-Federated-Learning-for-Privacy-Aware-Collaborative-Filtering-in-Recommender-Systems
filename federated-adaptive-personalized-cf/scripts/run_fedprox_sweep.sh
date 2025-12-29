#!/bin/bash
# FedProx Hyperparameter Sweep Script
# Systematically tunes proximal-mu, num-server-rounds, and fraction-train

set -e  # Exit on error

# Get script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE_DIR="$PROJECT_DIR/../results/federated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="$RESULTS_BASE_DIR/sweeps/fedprox_sweep_$TIMESTAMP"

# Create results directory
mkdir -p "$SWEEP_DIR"

# Hyperparameter grid
MU_VALUES=(0.05 0.1 0.5)
ROUNDS_VALUES=(50 100)
FRACTION_VALUES=(0.3 0.5)

# Fixed parameters
MODEL_TYPE="basic"
STRATEGY="fedprox"

# Log file
LOG_FILE="$SWEEP_DIR/sweep_log.txt"

echo "========================================" | tee "$LOG_FILE"
echo "FedProx Hyperparameter Sweep" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Results directory: $SWEEP_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Hyperparameter Grid:" | tee -a "$LOG_FILE"
echo "  proximal-mu: ${MU_VALUES[*]}" | tee -a "$LOG_FILE"
echo "  num-server-rounds: ${ROUNDS_VALUES[*]}" | tee -a "$LOG_FILE"
echo "  fraction-train: ${FRACTION_VALUES[*]}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Counter for experiments
exp_count=0
total_experiments=$((${#MU_VALUES[@]} * ${#ROUNDS_VALUES[@]} * ${#FRACTION_VALUES[@]}))

# Change to project directory
cd "$PROJECT_DIR"

for mu in "${MU_VALUES[@]}"; do
    for rounds in "${ROUNDS_VALUES[@]}"; do
        for fraction in "${FRACTION_VALUES[@]}"; do
            ((++exp_count))

            # Generate unique run name for WandB
            RUN_NAME="fedprox_mu${mu}_r${rounds}_f${fraction}"

            echo "" | tee -a "$LOG_FILE"
            echo "========================================" | tee -a "$LOG_FILE"
            echo "[$exp_count/$total_experiments] Running: $RUN_NAME" | tee -a "$LOG_FILE"
            echo "  proximal-mu=$mu" | tee -a "$LOG_FILE"
            echo "  num-server-rounds=$rounds" | tee -a "$LOG_FILE"
            echo "  fraction-train=$fraction" | tee -a "$LOG_FILE"
            echo "  Started: $(date)" | tee -a "$LOG_FILE"
            echo "========================================" | tee -a "$LOG_FILE"

            # Run experiment with CLI overrides
            # Note: All config must be in single --run-config, strings must be quoted
            flwr run . --run-config "strategy=\"fedprox\" proximal-mu=$mu num-server-rounds=$rounds fraction-train=$fraction wandb-run-name=\"$RUN_NAME\"" \
                       2>&1 | tee -a "$LOG_FILE"

            # Copy results to sweep directory with unique name
            RESULT_FILE="$RESULTS_BASE_DIR/${MODEL_TYPE}_mf_${STRATEGY}_mu${mu}_r${rounds}_f${fraction}_results.json"
            if [ -f "$RESULT_FILE" ]; then
                cp "$RESULT_FILE" "$SWEEP_DIR/${RUN_NAME}_results.json"
                echo "  Results saved to: ${RUN_NAME}_results.json" | tee -a "$LOG_FILE"
            else
                echo "  WARNING: Result file not found: $RESULT_FILE" | tee -a "$LOG_FILE"
            fi

            echo "  Completed: $(date)" | tee -a "$LOG_FILE"
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Sweep Complete!" | tee -a "$LOG_FILE"
echo "Total experiments: $exp_count" | tee -a "$LOG_FILE"
echo "Results directory: $SWEEP_DIR" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Generate summary
echo "" | tee -a "$LOG_FILE"
echo "To analyze results, run:" | tee -a "$LOG_FILE"
echo "  python $SCRIPT_DIR/analyze_sweep_results.py $SWEEP_DIR" | tee -a "$LOG_FILE"
