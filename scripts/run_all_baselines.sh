#!/bin/bash
# =============================================================================
# Thesis Baseline Comparison Runner
# =============================================================================
# Runs all baselines and proposed approach experiments with consistent settings
# for fair comparison on MovieLens 1M.
#
# Experiments (12 total):
#   Baselines:
#     1. Fed-BPR-MF + FedAvg       (all params global)
#     2. Fed-BPR-MF + FedProx      (all params global)
#     3. Split-BPR-MF + FedAvg     (local user embeddings)
#     4. Split-BPR-MF + FedProx    (local user embeddings)
#   Proposed (FedProx):
#     5. Adaptive BPR-MF + HC Alpha + FedProx
#     6. Dual-Level + Concat + HC Alpha + FedProx
#     7. Dual-Level + Per-User Alpha + FedProx
#     8. Dual-Level + All Techniques + FedProx
#   Proposed (FedAvg):
#     9. Adaptive BPR-MF + HC Alpha + FedAvg
#    10. Dual-Level + Concat + HC Alpha + FedAvg
#    11. Dual-Level + Per-User Alpha + FedAvg
#    12. Dual-Level + All Techniques + FedAvg
#
# Usage:
#   bash scripts/run_all_baselines.sh              # Run all
#   bash scripts/run_all_baselines.sh --dry-run     # Print commands only
#   bash scripts/run_all_baselines.sh --skip-baselines  # Only proposed
#   bash scripts/run_all_baselines.sh --skip-proposed   # Only baselines
#   bash scripts/run_all_baselines.sh --cpu             # Use CPU federation
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Common hyperparameters for FAIR comparison
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

# =============================================================================
# Parse arguments
# =============================================================================
DRY_RUN=false
SKIP_BASELINES=false
SKIP_PROPOSED=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --skip-baselines) SKIP_BASELINES=true ;;
        --skip-proposed) SKIP_PROPOSED=true ;;
        --cpu) FEDERATION="local-simulation" ;;
        --help)
            head -34 "$0" | tail -31
            exit 0
            ;;
    esac
done

# =============================================================================
# Helpers
# =============================================================================
TOTAL=0
CURRENT=0
FAILED=0
SUCCEEDED=0

count_experiments() {
    if [ "$SKIP_BASELINES" = false ]; then
        TOTAL=$((TOTAL + 4))
    fi
    if [ "$SKIP_PROPOSED" = false ]; then
        TOTAL=$((TOTAL + 8))
    fi
}

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
    echo "----------------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] flwr run . $FEDERATION --run-config \"$config\""
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
# Main
# =============================================================================
count_experiments

echo "=============================================="
echo "  Thesis Baseline Comparison Runner"
echo "  Experiments: $TOTAL"
echo "  Timestamp:   $TIMESTAMP"
echo "  Federation:  $FEDERATION"
echo "  Rounds: $ROUNDS | Epochs: $EPOCHS | Emb: $EMB_DIM"
echo "=============================================="

# Common config string shared by all experiments
COMMON="num-server-rounds=$ROUNDS local-epochs=$EPOCHS embedding-dim=$EMB_DIM lr=$LR weight-decay=$WEIGHT_DECAY dropout=$DROPOUT alpha=$DIRICHLET_ALPHA fraction-train=$FRACTION"

# Early stopping config shared by all experiments
ES="early-stopping-enabled=true early-stopping-patience=10"

# --- BASELINES ---

if [ "$SKIP_BASELINES" = false ]; then
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  BASELINES"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    # 1. Federated Baseline - FedAvg (all params global)
    run_experiment \
        "Fed-BPR-MF + FedAvg (all global)" \
        "federated-baseline-cf" \
        "model-type=\"bpr\" strategy=\"fedavg\" $COMMON $ES wandb-run-name=\"cmp_baseline_fedavg_${TIMESTAMP}\""

    # 2. Federated Baseline - FedProx (all params global)
    run_experiment \
        "Fed-BPR-MF + FedProx (all global)" \
        "federated-baseline-cf" \
        "model-type=\"bpr\" strategy=\"fedprox\" proximal-mu=0.01 $COMMON $ES wandb-run-name=\"cmp_baseline_fedprox_${TIMESTAMP}\""

    # 3. Federated Personalized - SplitFedAvg (local user embeddings)
    run_experiment \
        "Split-BPR-MF + FedAvg (local user emb)" \
        "federated-personalized-cf" \
        "model-type=\"bpr\" strategy=\"fedavg\" $COMMON $ES wandb-run-name=\"cmp_split_fedavg_${TIMESTAMP}\""

    # 4. Federated Personalized - SplitFedProx (local user embeddings)
    run_experiment \
        "Split-BPR-MF + FedProx (local user emb)" \
        "federated-personalized-cf" \
        "model-type=\"bpr\" strategy=\"fedprox\" proximal-mu=0.01 $COMMON $ES wandb-run-name=\"cmp_split_fedprox_${TIMESTAMP}\""
fi

# --- PROPOSED APPROACH ---

if [ "$SKIP_PROPOSED" = false ]; then
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  PROPOSED APPROACH (FedProx)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    ADAPTIVE_FEDPROX="strategy=\"fedprox\" proximal-mu=0.01 alpha-method=\"hierarchical_conditional\" $COMMON $ES"

    # 5. Adaptive BPR-MF (single-level, HC alpha) + FedProx
    run_experiment \
        "Adaptive BPR-MF + HC Alpha + FedProx" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"bpr\" $ADAPTIVE_FEDPROX wandb-run-name=\"cmp_adaptive_bpr_fedprox_${TIMESTAMP}\""

    # 6. Dual-Level + Concat fusion + FedProx
    run_experiment \
        "Dual-Level + Concat + HC Alpha + FedProx" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"dual\" fusion-type=\"concat\" mlp-hidden-dims=\"512,256,128\" $ADAPTIVE_FEDPROX wandb-run-name=\"cmp_dual_concat_fedprox_${TIMESTAMP}\""

    # 7. Dual-Level + Per-User Alpha + FedProx
    run_experiment \
        "Dual-Level + Per-User Alpha + FedProx" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"dual\" fusion-type=\"concat\" mlp-hidden-dims=\"512,256,128\" enable-per-user-alpha=true $ADAPTIVE_FEDPROX wandb-run-name=\"cmp_dual_pua_fedprox_${TIMESTAMP}\""

    # 8. Dual-Level + All Techniques (Full Combo) + FedProx
    run_experiment \
        "Dual-Level + Full Combo + FedProx" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"dual\" fusion-type=\"concat\" mlp-hidden-dims=\"512,256,128\" enable-per-user-alpha=true enable-item-perturbation=true item-perturbation-reg=0.01 contrastive-lambda=0.1 contrastive-tau=0.1 $ADAPTIVE_FEDPROX wandb-run-name=\"cmp_dual_full_fedprox_${TIMESTAMP}\""

    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  PROPOSED APPROACH (FedAvg)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    ADAPTIVE_FEDAVG="strategy=\"fedavg\" alpha-method=\"hierarchical_conditional\" $COMMON $ES"

    # 9. Adaptive BPR-MF (single-level, HC alpha) + FedAvg
    run_experiment \
        "Adaptive BPR-MF + HC Alpha + FedAvg" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"bpr\" $ADAPTIVE_FEDAVG wandb-run-name=\"cmp_adaptive_bpr_fedavg_${TIMESTAMP}\""

    # 10. Dual-Level + Concat fusion + FedAvg
    run_experiment \
        "Dual-Level + Concat + HC Alpha + FedAvg" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"dual\" fusion-type=\"concat\" mlp-hidden-dims=\"512,256,128\" $ADAPTIVE_FEDAVG wandb-run-name=\"cmp_dual_concat_fedavg_${TIMESTAMP}\""

    # 11. Dual-Level + Per-User Alpha + FedAvg
    run_experiment \
        "Dual-Level + Per-User Alpha + FedAvg" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"dual\" fusion-type=\"concat\" mlp-hidden-dims=\"512,256,128\" enable-per-user-alpha=true $ADAPTIVE_FEDAVG wandb-run-name=\"cmp_dual_pua_fedavg_${TIMESTAMP}\""

    # 12. Dual-Level + All Techniques (Full Combo) + FedAvg
    run_experiment \
        "Dual-Level + Full Combo + FedAvg" \
        "federated-adaptive-personalized-cf" \
        "model-type=\"dual\" fusion-type=\"concat\" mlp-hidden-dims=\"512,256,128\" enable-per-user-alpha=true enable-item-perturbation=true item-perturbation-reg=0.01 contrastive-lambda=0.1 contrastive-tau=0.1 $ADAPTIVE_FEDAVG wandb-run-name=\"cmp_dual_full_fedavg_${TIMESTAMP}\""
fi

# =============================================================================
# Summary & Comparison
# =============================================================================
echo ""
echo "=============================================="
echo "  Experiment Run Complete"
echo "  Succeeded: $SUCCEEDED / $TOTAL"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failed: $FAILED"
fi
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Generating comparison table..."
    python3 "$PROJECT_ROOT/scripts/compare_all_results.py" --output "$PROJECT_ROOT/results/comparison_${TIMESTAMP}.md"
    echo ""
    echo "Comparison saved to: results/comparison_${TIMESTAMP}.md"
fi
