#!/usr/bin/env bash
# ============================================================================
# BUG 2 DIAGNOSTIC — does MORE calibration recover the dual full-pop crater?
# ============================================================================
# Discriminates the two live hypotheses for the dual model's full-pop crater
# (in-loop ~0.236 → full-pop ~0.055, 4.3x gap):
#   H-epochs:     deep PersonalMLP needs >1 calibration epoch to realign the
#                 ~5400 users not freshly trained at best_round. This run uses
#                 final-calibration-epochs=5 (was 1 in Run C, which cratered).
#   H-structural: per-user MLP cannot generalize full-pop at ~165 ratings/user
#                 regardless of calibration.
#
# VERIFIED PRECONDITIONS (code + Run C log, 2026-06-01):
#   - Chain is correct: D-06.5 restore -> D-06.7 calibrate ALL 6040 (trains
#     full model incl. PersonalMLP) -> D-06 full-pop eval loads calibrated cache.
#   - Run C (decomp-C, calib-epochs=1) fired calibration on 6040/6040 clients
#     and STILL cratered to full-pop 0.0554. So 1 epoch is insufficient; this
#     tests whether 5 is enough.
#
# CONFIG: dual + both next-gen flags OFF (item-perturbation now defaults OFF
# after Bug 1 fix d0cffd4; per-user-alpha explicitly off to isolate the MLP
# staleness from the alpha local-state). Identical to Run C except calib-epochs.
#
# READ RESULT:
#   full-pop NDCG@10 -> ~0.18-0.22  => H-epochs confirmed, Bug 2 fixable (lock
#                                       calib-epochs>=5 for dual best_round_restore)
#   full-pop NDCG@10 -> ~0.08-0.14  => partial; retry calib-epochs=10
#   full-pop NDCG@10 -> still ~0.05 => H-structural; Path A cannot clear the bar
# ============================================================================

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/bug2-calib-epochs5.log

cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] BUG 2 DIAGNOSTIC (calib-epochs=5) PID $$" | tee -a "$LOG"
echo "  model-type=dual  per-user-alpha=OFF  item-perturbation=OFF" | tee -a "$LOG"
echo "  final-calibration-enabled=true  final-calibration-epochs=5" | tee -a "$LOG"
echo "  baseline to beat (Run C, calib-epochs=1): full-pop NDCG@10=0.0554" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

python scripts/run.py adaptive thesis_crossdevice_main --federation local-sim-gpu \
    --run-config "model-type=dual" \
    --run-config "enable-per-user-alpha=false" \
    --run-config "enable-item-perturbation=false" \
    --run-config "final-calibration-enabled=true" \
    --run-config "final-calibration-epochs=5" \
    --run-config "alpha-method=hierarchical_conditional" \
    --run-config "fusion-type=concat" \
    --run-config "lr=0.005" \
    --run-config "checkpoint-rule=best_round_restore" \
    --run-config "fraction-train=0.1" \
    --run-config "local-epochs=1" \
    --run-config "num-server-rounds=100" \
    --run-config "embedding-dim=64" \
    --run-config "wandb-enabled=false" 2>&1 | tee -a "$LOG"

RC=$?
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] BUG2_CALIB_EPOCHS5 FINISHED rc=$RC" | tee -a "$LOG"
echo "  Compare full-pop NDCG@10 in manifest vs Run C's 0.0554." | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
