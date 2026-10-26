#!/usr/bin/env bash
# ============================================================================
# C3 RUN 1 — PFedRec FIX-A isolation (consensus GO: 3 pre-flight + 3 panel + codex)
# ============================================================================
# Re-runs the EXACT cratered config (paper_compat_pfedrec + fraction-train=0.1 +
# best_round_restore; latent=32, lr=0.1, lr_eta=80, 100 rounds) with the
# now-committed run_id FIX A active (commit 01d8b72) and calibration explicitly
# OFF — so any full-pop delta is attributable to FIX A alone.
#
# WHY FIX A: the cratered run 20260527-085237-573e19 cached 6040/6040 warm
# affine_output heads, yet full-pop best/sampled_ndcg@10 = 0.0711 vs in-loop
# 0.3478 (5.19x). The D-06 eval broadcast omitted run_id → client fell back to
# base/default/ (nonexistent) → every user scored with a COLD head. FIX A stamps
# run_id/reuse_cache/round_num on the D-06 eval → reads base/{run_id}/ (warm).
#
# DECISION RULE (vs THIS run's own in-loop 'last', not prior 0.3478):
#   full-pop best >= ~0.28  -> STRONG: FIX A was the crater cause.
#   0.20 - 0.28             -> PARTIAL: FIX A real, residual vintage-staleness
#                              -> justifies the gated calibration Run 2.
#   <= ~0.12 w/ 6040 warm hits -> FALSIFIES FIX-A-primary (genuine staleness).
#   ~0.07                   -> fix did NOT engage; re-trace cache path.
#
# VALIDITY GATES (run is INVALID if any fail — see analyze_c3_run1.py):
#   - D-06 eval must show run_id=<current YYYYMMDD-HHMMSS-hex>, NOT "default".
#   - best block evaluated_users must == 6040 (else clients silently errored).
#   - NOT "[D-06] WARNING: no extra-eval responses" (would make best==last).
#   - per-group full-pop (sparse/medium/dense) must ALL lift; sparse is the
#     load-bearing thesis subgroup.
# ============================================================================

set -uo pipefail
REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/c3-run1.log
cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] C3 RUN 1 — PFedRec FIX-A isolation (PID $$)" | tee -a "$LOG"
echo "  paper_compat + frac=0.1 + best_round_restore + calib OFF; run_id FIX A active" | tee -a "$LOG"
echo "  READ: full-pop best/sampled_ndcg@10 vs in-loop last; >=0.28 strong, 0.20-0.28 partial" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

python scripts/run.py pfedrec paper_compat_pfedrec --federation local-sim-gpu \
    --run-config "fraction-train=0.1" \
    --run-config "checkpoint-rule=best_round_restore" \
    --run-config "final-calibration-enabled=false" \
    --run-config "wandb-enabled=false" 2>&1 | tee -a "$LOG"

RC=$?
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] C3_RUN1 FINISHED rc=$RC" | tee -a "$LOG"
echo "  Analyze: python $INVEST/analyze_c3_run1.py" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
