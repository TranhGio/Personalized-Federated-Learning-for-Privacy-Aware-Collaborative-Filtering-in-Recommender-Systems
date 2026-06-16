#!/usr/bin/env bash
# ============================================================================
# DECOMPOSITION RUN C — dual base, BOTH next-gen flags OFF (missing cell)
# ============================================================================
# Purpose: Complete the 2x2 factorial over the dual base. Tests the dual
# PersonalMLP + fusion + α-heuristic architecture with NEITHER next-gen
# add-on (no per-user learnable α, no item perturbation).
#
# Prediction (from 2026-05-31 factorial findings):
#   • in-loop ~0.24 (IP off → training not destroyed; matches Decomp A 0.236)
#   • full-pop ~0.22 (per-user-α off → no full-pop staleness crater)
#   → If confirmed, proves the dual PersonalMLP architecture is HARMLESS and
#     pins BOTH failures on the two next-gen flags:
#       - item-perturbation = training-stability bug (in-loop collapse)
#       - per-user-α        = full-pop eval/calibration bug (staleness crater)
#
# Factorial completed by this run (NDCG@10 full-pop):
#                       item_perturb=ON     item_perturb=OFF
#   per_user_alpha=ON     0.0842 (Bug3)       0.0727 (Decomp A)
#   per_user_alpha=OFF    0.0636 (Decomp B)   ???    (THIS RUN)
#
# Pre-flight: with both flags off, dual model's required LOCAL state is
# {user_embeddings.weight} + PersonalMLP/fusion weights (always present for
# dual) — neither _logit_alpha.weight nor _item_perturbation.weight required
# (D-01/D-03 contract). model-type=dual is the canonical thesis architecture.
# ============================================================================

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/decomp-C-dual-both-off.log

cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DECOMP RUN C (PID $$)" | tee -a "$LOG"
echo "  model-type:               dual  (PersonalMLP + fusion + α-heuristic)" | tee -a "$LOG"
echo "  enable-per-user-alpha:    false" | tee -a "$LOG"
echo "  enable-item-perturbation: false   ← both next-gen add-ons OFF" | tee -a "$LOG"
echo "  mode: thesis_crossdevice_main (N=6040, canonical)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

python scripts/run.py adaptive thesis_crossdevice_main --federation local-sim-gpu \
    --run-config "model-type=dual" \
    --run-config "enable-per-user-alpha=false" \
    --run-config "enable-item-perturbation=false" \
    --run-config "alpha-method=hierarchical_conditional" \
    --run-config "fusion-type=concat" \
    --run-config "lr=0.005" \
    --run-config "checkpoint-rule=best_round_restore" \
    --run-config "fraction-train=0.1" \
    --run-config "local-epochs=1" \
    --run-config "num-server-rounds=100" \
    --run-config "embedding-dim=64" \
    --run-config "final-calibration-enabled=true" \
    --run-config "wandb-enabled=false" 2>&1 | tee -a "$LOG"

RC=$?
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DECOMP_C FINISHED rc=$RC" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
