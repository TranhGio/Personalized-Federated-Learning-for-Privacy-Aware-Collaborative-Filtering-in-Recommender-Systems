#!/usr/bin/env bash
# ============================================================================
# BUG 2 DISCRIMINATOR (corrected, post 4-reviewer pass)
# ============================================================================
# Separates the two live hypotheses for the dual model's full-pop crater
# (in-loop ~0.236 vs full-pop ~0.055) WITHIN one run, avoiding every confound
# that sank the earlier 17h design (cold-init, best_round-semantics, cross-run,
# convergence):
#   - checkpoint-rule=last_round  → no best_round_restore rollback, no D-06.7
#     calibration (both gated on best_round*). Pure during-training signal.
#   - diagnostic-fullpop-eval=true → each strided round (1-3, then every 5th)
#     evaluate users trained >=1 time but NOT this round (cold-init excluded),
#     age-bucketed by rounds-since-last-trained. Emits [DIAG] lines + diag/* keys.
#   - 90 rounds + early-stopping OFF → trained-subset reaches its plateau (Run C
#     peaked @83). NOTE: the first attempt (run 20260604-141550) early-stopped at
#     round 12 because the noisy early metric never beat round-2's 0.0543; with
#     early-stopping enabled the run dies before plateau and the probe is moot.
#   - flags off (Bug 1 default) → isolates the dual PersonalMLP+fusion head.
#
# READ THE RESULT (age-bucket shape in the [DIAG] lines):
#   NDCG decays monotonically with staleness age (age1_2 >> age31plus), and the
#     gap to the trained-subset closes as coverage grows → FIXABLE staleness
#     (points to joint, not frozen-discard, calibration).
#   NDCG flat-low even at age 1-2 (recently-trained users also crater) →
#     STRUCTURAL / per-user-head co-adaptation → Path A is a dead end → pivot.
#
# Pre-flight verified 2026-06-01: GPU free; 89 adaptive tests pass; dry-run
# clean; aggregate_evaluate is side-effect-free; all 3 source fixes present.
# ============================================================================

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/bug2-discriminator.log

cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] BUG 2 DISCRIMINATOR (PID $$)" | tee -a "$LOG"
echo "  dual + flags off + last_round + diagnostic-fullpop-eval + 90 rounds" | tee -a "$LOG"
echo "  watch [DIAG] lines: age-bucket NDCG decay = staleness; flat-low = structural" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

python scripts/run.py adaptive thesis_crossdevice_main --federation local-sim-gpu \
    --run-config "model-type=dual" \
    --run-config "enable-per-user-alpha=false" \
    --run-config "enable-item-perturbation=false" \
    --run-config "checkpoint-rule=last_round" \
    --run-config "diagnostic-fullpop-eval=true" \
    --run-config "early-stopping-enabled=false" \
    --run-config "alpha-method=hierarchical_conditional" \
    --run-config "fusion-type=concat" \
    --run-config "fraction-train=0.1" \
    --run-config "local-epochs=1" \
    --run-config "num-server-rounds=90" \
    --run-config "embedding-dim=64" \
    --run-config "lr=0.005" \
    --run-config "wandb-enabled=false" 2>&1 | tee -a "$LOG"

RC=$?
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] BUG2_DISCRIMINATOR FINISHED rc=$RC" | tee -a "$LOG"
echo "  Analyze: grep '\\[DIAG\\]' $LOG  → plot age-bucket NDCG vs round." | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
