#!/usr/bin/env bash
# ============================================================================
# JOB 2 — ADAPTIVE DUAL CANONICAL WARM RE-RUN (Path-A decisive run)
# ============================================================================
# Exact mirror of Run C 20260601-024243-7a5429 (dual + both next-gen flags off
# + best_round_restore + final-calibration-enabled=true + lr=0.005 + 100 rds,
# emb=64, frac=0.1; pyproject defaults keep fedprox mu=0.01 + contrastive 0.1),
# now executed at HEAD >= 70a92bd where the D-06 full-pop eval stamps
# run_id/reuse_cache/round_num — the ONLY behavioral delta vs Run C is the
# WARM eval: D-06 reads each user's calibrated PersonalMLP/fusion/user-emb
# from .embedding_cache/{run_id}/ instead of the nonexistent default/ dir.
#
# Run C (cold D-06): full-pop best = 0.0554 despite healthy in-loop 0.236 and
# a calibration pass that wrote warm heads the eval never read. This run
# answers: with calibration + warm eval, where does the dual really land?
#   >= ~0.2275 (warm personalized, Job 1) -> Path A alive at protocol parity.
#   ~0.20-0.22                            -> dual ~ baseline/parity, Path B.
#   << 0.20                               -> dual genuinely trails.
# Note Job 1's lesson: an EARLY best_round inflates local-vs-global vintage
# mismatch; calibration (enabled here) realigns local heads to the restored
# globals, so this cell is calibrated-warm — record best_round for context.
#
# VALIDITY GATES (post-run): manifest git_commit >= 70a92bd; D-06 log shows
# run_id stamped (zero base/default reads); best.evaluated_users == 6040;
# [D-06.7] calibration-complete line present (6040/6040).
# ETA ~14-16h.
# ============================================================================

set -uo pipefail
REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
LOG=$REPO/.bug-investigation/rerun-adaptive-dual-warm.log
cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] JOB 2 — ADAPTIVE DUAL WARM RE-RUN (PID $$)" | tee -a "$LOG"
echo "  mirror of Run C 7a5429 @ HEAD>=70a92bd (warm D-06 + calibration)" | tee -a "$LOG"
echo "  decision: >=0.2275 Path A alive | ~0.20-0.22 parity | <<0.20 trails" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"

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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] RERUN_ADAPTIVE_DUAL_WARM FINISHED rc=$RC" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
