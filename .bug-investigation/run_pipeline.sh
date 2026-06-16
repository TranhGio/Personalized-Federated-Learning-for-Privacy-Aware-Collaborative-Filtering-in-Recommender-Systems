#!/usr/bin/env bash
# Bug investigation pipeline orchestrator.
# Phase 1: wait for XTTS PID 4081700 to free GPU
# Phase 2: launch PFedRec paper repro (~3h)
# Phase 3: partial analyze
# Phase 4: launch adaptive basic ablation (~5h)
# Phase 5: final analyze
#
# Logs:
#   .bug-investigation/orchestrator.log         — milestone log
#   .bug-investigation/pfedrec-paper-repro.log  — PFedRec full output
#   .bug-investigation/adaptive-basic-ablation.log — adaptive full output
#   .bug-investigation/analyze-after-pfedrec.txt — partial verdict (H1 ready)
#   .bug-investigation/analyze-final.txt         — full verdict (H1 + H2 + joint)

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
XTTS_PID=4081700
LOG=$INVEST/orchestrator.log

cd "$REPO"
mkdir -p "$INVEST"
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo
echo "================================================================"
echo "[$(ts)] Pipeline started (PID $$)"
echo "================================================================"

# -------------------------------- Phase 1 --------------------------------
echo "[$(ts)] Phase 1: waiting for XTTS PID $XTTS_PID to exit..."
while kill -0 $XTTS_PID 2>/dev/null; do sleep 60; done
echo "[$(ts)] XTTS finished. Sleeping 30s for GPU memory release..."
sleep 30
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# -------------------------------- Phase 2 --------------------------------
echo
echo "[$(ts)] Phase 2: launching PFedRec paper repro..."
python scripts/run.py pfedrec paper_compat_pfedrec \
    --federation local-sim-gpu \
    --run-config "wandb-enabled=false" \
    > "$INVEST/pfedrec-paper-repro.log" 2>&1
PF_RC=$?
echo "[$(ts)] PFedRec exit code: $PF_RC"
if [ $PF_RC -ne 0 ]; then
    echo "[$(ts)] WARNING: PFedRec returned non-zero. Continuing to analyze + adaptive anyway."
fi

# -------------------------------- Phase 3 --------------------------------
echo
echo "[$(ts)] Phase 3: running partial analyze..."
python scripts/analyze_bug_investigation.py > "$INVEST/analyze-after-pfedrec.txt" 2>&1 || true
tail -5 "$INVEST/analyze-after-pfedrec.txt"

# Brief pause to let Ray actors fully release GPU between flwr runs
sleep 30

# -------------------------------- Phase 4 --------------------------------
echo
echo "[$(ts)] Phase 4: launching adaptive basic ablation..."
python scripts/run.py adaptive thesis_crossdevice_main \
    --run-config "model-type=basic" \
    --run-config "lr=0.005" \
    --run-config "checkpoint-rule=best_round_restore" \
    --run-config "wandb-enabled=false" \
    > "$INVEST/adaptive-basic-ablation.log" 2>&1
AB_RC=$?
echo "[$(ts)] Adaptive basic exit code: $AB_RC"

# -------------------------------- Phase 5 --------------------------------
echo
echo "[$(ts)] Phase 5: running final analyze..."
python scripts/analyze_bug_investigation.py > "$INVEST/analyze-final.txt" 2>&1 || true
tail -20 "$INVEST/analyze-final.txt"

echo
echo "[$(ts)] PIPELINE COMPLETE. PFedRec rc=$PF_RC, AdaptiveBasic rc=$AB_RC"
echo "Final verdict: $INVEST/analyze-final.txt"
