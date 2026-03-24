#!/bin/bash
# =============================================================================
# Re-run Baselines with Leave-One-Out Evaluation Protocol
# =============================================================================
# Runs all baseline and personalized experiments with the corrected NCF
# evaluation protocol (leave-one-out split + 99 negative samples) at 50 rounds.
#
# This replaces older results that used random 80/20 splits which are NOT
# comparable to published baselines (PFedRec, FedMF, FedNCF).
#
# Experiments (6 total):
#   Federated Baseline (all params global):
#     1. BPR-MF + FedAvg, 50 rounds
#     2. BPR-MF + FedProx, 50 rounds
#     3. Basic-MF + FedAvg, 50 rounds (MSE baseline)
#   Federated Personalized (split learning):
#     4. BPR-MF + SplitFedAvg, 50 rounds
#     5. BPR-MF + SplitFedProx, 50 rounds
#     6. Basic-MF + SplitFedAvg, 50 rounds (MSE baseline)
#
# Usage:
#   bash scripts/run_baseline_sweep_loo.sh              # Run all
#   bash scripts/run_baseline_sweep_loo.sh --dry-run    # Print commands only
#   bash scripts/run_baseline_sweep_loo.sh --cpu        # Use CPU federation
#   bash scripts/run_baseline_sweep_loo.sh --quick      # 10 rounds for testing
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Common hyperparameters — matched to run_all_baselines.sh for fair comparison
# =============================================================================
ROUNDS=50
EPOCHS=10
EMB_DIM=128
LR=0.005
WEIGHT_DECAY=1e-5
DROPOUT=0.1
DIRICHLET_ALPHA=0.5
FRACTION=1.0
FEDERATION="local-sim-gpu"
EVAL_SPLIT="leave-one-out"
EVAL_NEG=99

# =============================================================================
# Parse arguments
# =============================================================================
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --cpu) FEDERATION="local-simulation" ;;
        --quick) ROUNDS=10; EPOCHS=5 ;;
        --help)
            head -28 "$0" | tail -25
            exit 0
            ;;
    esac
done

# =============================================================================
# Helpers
# =============================================================================
TOTAL=6
CURRENT=0
FAILED=0
SUCCEEDED=0

run_experiment() {
    local name="$1"
    local dir="$2"
    local config="$3"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "=============================================="
    echo "  [$CURRENT/$TOTAL] $name"
    echo "=============================================="
    echo "  Directory: $dir"
    echo "  Federation: $FEDERATION"
    echo "  Eval: $EVAL_SPLIT + $EVAL_NEG negatives"
    echo "----------------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] cd $PROJECT_ROOT/$dir && flwr run . $FEDERATION --run-config \"$config\""
        return 0
    fi

    cd "$PROJECT_ROOT/$dir"

    # Clean embedding cache for split learning (fresh start)
    if [ -d ".embedding_cache" ]; then
        echo "  Cleaning embedding cache..."
        rm -rf .embedding_cache
    fi

    local start_time=$(date +%s)

    if flwr run . "$FEDERATION" --run-config "$config"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "  Done in ${duration}s"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi

    cd "$PROJECT_ROOT"
}

# =============================================================================
# Common config strings
# =============================================================================
# Shared across all experiments
COMMON="num-server-rounds=$ROUNDS local-epochs=$EPOCHS embedding-dim=$EMB_DIM lr=$LR weight-decay=$WEIGHT_DECAY dropout=$DROPOUT alpha=$DIRICHLET_ALPHA fraction-train=$FRACTION"

# Evaluation protocol — leave-one-out (NCF protocol)
EVAL="eval-split-mode=\"$EVAL_SPLIT\" eval-num-negatives=$EVAL_NEG"

# Early stopping (metric set per experiment — BPR uses sampled_ndcg@10, Basic uses rmse)
ES_BASE="early-stopping-enabled=true early-stopping-patience=10"
ES_BPR="$ES_BASE early-stopping-metric=\"sampled_ndcg@10\""
ES_BASIC="$ES_BASE early-stopping-metric=\"rmse\" early-stopping-mode=\"min\""

# =============================================================================
# Main
# =============================================================================
echo "=============================================="
echo "  Baseline Sweep — Leave-One-Out Protocol"
echo "  Experiments: $TOTAL"
echo "  Timestamp:   $TIMESTAMP"
echo "  Federation:  $FEDERATION"
echo "  Rounds: $ROUNDS | Epochs: $EPOCHS | Emb: $EMB_DIM"
echo "  Eval: $EVAL_SPLIT + $EVAL_NEG negatives"
echo "=============================================="

# =====================================================
# FEDERATED BASELINE (all params global)
# =====================================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "  FEDERATED BASELINE (all params global)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# 1. BPR-MF + FedAvg
run_experiment \
    "Fed BPR-MF + FedAvg (all global)" \
    "federated-baseline-cf" \
    "model-type=\"bpr\" strategy=\"fedavg\" $COMMON $EVAL $ES_BPR wandb-run-name=\"loo_baseline_bpr_fedavg_${TIMESTAMP}\""

# 2. BPR-MF + FedProx
run_experiment \
    "Fed BPR-MF + FedProx (all global)" \
    "federated-baseline-cf" \
    "model-type=\"bpr\" strategy=\"fedprox\" proximal-mu=0.01 $COMMON $EVAL $ES_BPR wandb-run-name=\"loo_baseline_bpr_fedprox_${TIMESTAMP}\""

# 3. Basic-MF + FedAvg (MSE baseline for RMSE comparison)
run_experiment \
    "Fed Basic-MF + FedAvg (all global)" \
    "federated-baseline-cf" \
    "model-type=\"basic\" strategy=\"fedavg\" $COMMON $EVAL $ES_BASIC wandb-run-name=\"loo_baseline_basic_fedavg_${TIMESTAMP}\""

# =====================================================
# FEDERATED PERSONALIZED (split learning)
# =====================================================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "  FEDERATED PERSONALIZED (split learning)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# 4. BPR-MF + SplitFedAvg
run_experiment \
    "Split BPR-MF + FedAvg (local user emb)" \
    "federated-personalized-cf" \
    "model-type=\"bpr\" strategy=\"fedavg\" $COMMON $EVAL $ES_BPR wandb-run-name=\"loo_split_bpr_fedavg_${TIMESTAMP}\""

# 5. BPR-MF + SplitFedProx
run_experiment \
    "Split BPR-MF + FedProx (local user emb)" \
    "federated-personalized-cf" \
    "model-type=\"bpr\" strategy=\"fedprox\" proximal-mu=0.01 $COMMON $EVAL $ES_BPR wandb-run-name=\"loo_split_bpr_fedprox_${TIMESTAMP}\""

# 6. Basic-MF + SplitFedAvg (MSE baseline)
run_experiment \
    "Split Basic-MF + FedAvg (local user emb)" \
    "federated-personalized-cf" \
    "model-type=\"basic\" strategy=\"fedavg\" $COMMON $EVAL $ES_BASIC wandb-run-name=\"loo_split_basic_fedavg_${TIMESTAMP}\""

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Baseline Sweep Complete"
echo "  Succeeded: $SUCCEEDED / $TOTAL"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failed: $FAILED"
fi
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Generating comparison table..."
    if python3 "$PROJECT_ROOT/scripts/compare_all_results.py" --output "$PROJECT_ROOT/results/comparison_loo_${TIMESTAMP}.md" 2>/dev/null; then
        echo "Comparison saved to: results/comparison_loo_${TIMESTAMP}.md"
    else
        echo "Warning: Could not generate comparison table (script may need updating)"
    fi
    echo ""
    echo "Results saved to:"
    echo "  results/federated/  (baseline)"
    echo "  results/federated/personalized/  (split learning)"
fi
