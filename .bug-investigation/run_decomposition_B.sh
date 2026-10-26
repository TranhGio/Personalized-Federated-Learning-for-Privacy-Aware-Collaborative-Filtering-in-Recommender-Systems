#!/usr/bin/env bash
# ============================================================================
# DECOMPOSITION RUN B — item perturbation isolation
# ============================================================================
# Purpose: Within the dual model (PersonalMLP + fusion + α-heuristic kept ON),
# turn ONLY item-perturbation ON and per-user learnable α OFF. Tells us how
# much per-user-α specifically contributes to the dual model's failure.
#
# Comparison points:
#   • Run B (this) vs Bug3 (both flags on, 0.0842):
#       Δ → effect of REMOVING per-user-α while keeping item-perturbation
#   • Run B (this) vs Run A (per-user-α on, item-perturb off):
#       direct A/B contrast of the two next-gen flags
#   • Run B (this) vs H2 (BPRMF + both off, 0.2256):
#       gap from item-perturb + dual machinery vs no machinery at all
#
# Pre-flight verified 2026-05-29:
#   - dual + item-perturbation=true requires _item_perturbation.weight in
#     LOCAL state (D-01/D-03 contract)
#   - DualPersonalizedBPRMF.enable_item_perturbation() creates the tensor,
#     called unconditionally before cache restore (ADP-02)
#   - All 3 source fixes from commit b6a6a9b are present
# ============================================================================

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/decomp-B-item-perturbation.log

cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DECOMP RUN B (PID $$)" | tee -a "$LOG"
echo "  model-type:              dual" | tee -a "$LOG"
echo "  enable-per-user-alpha:   false" | tee -a "$LOG"
echo "  enable-item-perturbation: true   ← only this ON" | tee -a "$LOG"
echo "  mode: thesis_crossdevice_main (N=6040, canonical)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

python scripts/run.py adaptive thesis_crossdevice_main --federation local-sim-gpu \
    --run-config "model-type=dual" \
    --run-config "enable-per-user-alpha=false" \
    --run-config "enable-item-perturbation=true" \
    --run-config "item-perturbation-reg=0.01" \
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DECOMP_B FINISHED rc=$RC" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
