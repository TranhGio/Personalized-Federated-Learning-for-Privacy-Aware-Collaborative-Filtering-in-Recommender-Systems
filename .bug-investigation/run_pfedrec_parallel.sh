#!/usr/bin/env bash
# Launch PFedRec paper repro in PARALLEL with the already-running adaptive basic.
# scripts/run.py has been fixed to emit positional federation (commit-pending).
# After PFedRec completes, run the final analyze script.

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/pfedrec-parallel.log

cd "$REPO"
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo
echo "================================================================"
echo "[$(ts)] PFedRec parallel launch (PID $$)"
echo "================================================================"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo
echo "[$(ts)] Launching PFedRec paper repro..."
python scripts/run.py pfedrec paper_compat_pfedrec --federation local-sim-gpu \
    --run-config "wandb-enabled=false" \
    > "$INVEST/pfedrec-paper-repro.log" 2>&1
PF_RC=$?
echo "[$(ts)] PFedRec exit code: $PF_RC"

echo
echo "[$(ts)] PFedRec done. Running analyze (combined with whatever's available)..."
python scripts/analyze_bug_investigation.py > "$INVEST/analyze-after-pfedrec-parallel.txt" 2>&1 || true
tail -25 "$INVEST/analyze-after-pfedrec-parallel.txt"

echo
echo "[$(ts)] PARALLEL PFEDREC COMPLETE."
