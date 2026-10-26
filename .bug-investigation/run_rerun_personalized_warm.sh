#!/usr/bin/env bash
# ============================================================================
# JOB 1 — PERSONALIZED CANONICAL WARM RE-RUN (run-id audit remediation)
# ============================================================================
# Exact mirror of the invalidated canonical run 20260527-085558-5ea2ef
# (mode=thesis_crossdevice_main, overrides: lr=0.005,
# checkpoint-rule=best_round_restore; emb=64 = mode default), now executed at
# HEAD >= 3a393fb where the D-06 full-pop eval stamps run_id/reuse_cache/
# round_num — so every user's cached LOCAL user_embeddings/user_bias load
# WARM at the final eval. The ONLY behavioral delta vs the original is the
# warm eval; any change in final_metrics.best is attributable to the fix.
#
# Original (cold) numbers: best/sampled_ndcg@10 = 0.2246 (n=6040), in-loop
# best 0.2530 @ r83. Expected warm: ~0.24-0.25 (PFedRec precedent: recovered
# 94% of in-loop). Per-group: expect medium/dense to recover most (they were
# crushed by cold state), sparse roughly unchanged.
#
# POST-RUN VALIDITY GATES (check before trusting the number):
#   - manifest git_commit >= 3a393fb, mode=thesis_crossdevice_main
#   - final_metrics.best.evaluated_users == 6040
#   - log: NO "no extra-eval responses", NO base/default/ reads
# ETA: 20-30h solo GPU (original took 36h44m under 2-way contention).
# ============================================================================

set -uo pipefail
REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
LOG=$REPO/.bug-investigation/rerun-personalized-warm.log
cd "$REPO"
: > "$LOG"

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] JOB 1 — PERSONALIZED WARM RE-RUN (PID $$)" | tee -a "$LOG"
echo "  mirror of 20260527-085558-5ea2ef @ HEAD>=3a393fb (warm D-06)" | tee -a "$LOG"
echo "  expect best/sampled_ndcg@10 ~0.24-0.25 (was 0.2246 cold)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"

python scripts/run.py personalized thesis_crossdevice_main --federation local-sim-gpu \
    --run-config "embedding-dim=64" \
    --run-config "lr=0.005" \
    --run-config "checkpoint-rule=best_round_restore" \
    --run-config "wandb-enabled=false" 2>&1 | tee -a "$LOG"

RC=$?
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] RERUN_PERSONALIZED_WARM FINISHED rc=$RC" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
