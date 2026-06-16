#!/usr/bin/env bash
# ============================================================================
# TUNED DUAL (Path-A decisive probe, post-discriminator-verdict 2026-06-06)
# ============================================================================
# Discriminator (20260604-164934-8cad16) proved the dual full-pop "crater"
# (0.0554) was a best_round_restore + 1-epoch-calibration eval-path artifact:
# measured coherently (last_round, cold-init excluded) the SAME dual+flags-off
# model reaches full-pop NDCG@10 = 0.1989 (peak 0.2112 @ round 85) ~ in-loop
# 0.1906. Staleness is mild+fixable (age 0.224 -> 0.153), NOT structural.
#
# BUT at lr=0.005 the dual only matches baseline (0.20) and trails split-learning
# (personalized 0.2246 / adaptive-BPR 0.2256). This run asks the decisive Path-A
# question: with the crater removed (coherent eval) and the obvious tuning lever
# applied, can the dual's COHERENT full-pop clear 0.2246?
#
# Tuning vs the discriminator: lr 0.005 -> 0.01 (2x; Run C at lr=0.005 was still
# climbing in-loop at round 100 -> headroom on both lr and rounds). Everything
# else held identical for clean attribution: dual, both next-gen flags OFF,
# emb=64, frac=0.1, local-epochs=1, checkpoint-rule=last_round (coherent),
# diagnostic-fullpop-eval=true (reads the coherent full-pop curve directly),
# early-stopping OFF (so it reaches plateau; v1 died at round 12).
#
# READ THE RESULT: grep '[DIAG]' for the coherent full-pop curve. Report the
#   PEAK fullpop ndcg (last_round can catch an end dip; the diagnostic gives the
#   whole curve). Compare PEAK vs split-learning 0.2246:
#     PEAK >> 0.2246  -> Path A can win; justify a fair 2-run head-to-head
#                        (re-measure split-learning under last_round too).
#     PEAK ~ 0.20     -> dual doesn't win even healthy -> Path B confirmed, pivot.
#   CAVEAT: [DIAG] fullpop excludes ~600 never-trained cold-init users that the
#   n=6040 split-learning number includes (slightly optimistic by ~10% of pop).
#
# This config is NEW (matrix scan 2026-06-06): no dual+flags-off+emb64 run exists
# at lr>0.005 with last_round. Pre-flight: 89 adaptive tests, GPU free (32GB),
# all 14 run-config keys declared, committed fixes 65924d0+4ff4a4f present.
# ============================================================================

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/tuned-dual.log

cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TUNED DUAL (PID $$)" | tee -a "$LOG"
echo "  dual + flags off + last_round + diagnostic + lr=0.01 + 100 rounds" | tee -a "$LOG"
echo "  Q: can coherent full-pop clear split-learning 0.2246?" | tee -a "$LOG"
echo "  READ: grep '[DIAG]' -> PEAK fullpop ndcg vs 0.2246" | tee -a "$LOG"
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
    --run-config "num-server-rounds=100" \
    --run-config "embedding-dim=64" \
    --run-config "lr=0.01" \
    --run-config "wandb-enabled=false" 2>&1 | tee -a "$LOG"

RC=$?
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TUNED_DUAL FINISHED rc=$RC" | tee -a "$LOG"
echo "  Analyze: python $INVEST/analyze_tuned_dual.py" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
