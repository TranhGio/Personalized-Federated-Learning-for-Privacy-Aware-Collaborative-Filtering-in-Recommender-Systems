#!/usr/bin/env bash
# Phase 6: launch PFedRec paper repro AFTER orchestrator pipeline (PID 4165241)
# exits. The orchestrator's Phase 2 failed because scripts/run.py used
# --federation FLAG which flwr 1.24 doesn't accept. Fix applied in commit-pending
# scripts/run.py change (positional federation arg).
#
# Outputs:
#   .bug-investigation/phase6.log            — milestones
#   .bug-investigation/pfedrec-paper-repro.log  — PFedRec full output (overwrites failed v1)
#   .bug-investigation/analyze-final-v2.txt — re-run analyze with PFedRec + adaptive basic data

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
ORCH_PID=4165241
LOG=$INVEST/phase6.log

cd "$REPO"
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo
echo "================================================================"
echo "[$(ts)] Phase 6 started (PID $$)"
echo "================================================================"

echo "[$(ts)] Waiting for orchestrator PID $ORCH_PID to exit..."
while kill -0 $ORCH_PID 2>/dev/null; do sleep 60; done
echo "[$(ts)] Orchestrator done. Sleeping 30s for GPU release..."
sleep 30
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo
echo "[$(ts)] Launching PFedRec paper repro (with fixed launcher)..."
python scripts/run.py pfedrec paper_compat_pfedrec --federation local-sim-gpu \
    --run-config "wandb-enabled=false" \
    > "$INVEST/pfedrec-paper-repro.log" 2>&1
PF_RC=$?
echo "[$(ts)] PFedRec exit code: $PF_RC"

echo
echo "[$(ts)] Running FULL analyze (both PFedRec + adaptive basic)..."
python scripts/analyze_bug_investigation.py > "$INVEST/analyze-final-v2.txt" 2>&1 || true
echo "--- Tail of final v2 analyze ---"
tail -25 "$INVEST/analyze-final-v2.txt"

echo
echo "[$(ts)] PHASE 6 COMPLETE. PFedRec rc=$PF_RC"
echo "Final verdict: $INVEST/analyze-final-v2.txt"
