#!/usr/bin/env bash
# Adaptive Basic ablation — V2 launch with:
#   1. Clamp-bug fix applied to adaptive task.py (mirrored from baseline fd4450b)
#   2. Explicit config matching adaptive Bug3 verify (20260526-082810):
#      emb=64, lr=0.005, fraction-train=0.1, local-epochs=1, num-server-rounds=100,
#      checkpoint-rule=best_round_restore
#   3. Runs in parallel with PFedRec (already going on PID 81480)

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/adaptive-basic-v2-wrapper.log

cd "$REPO"
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo
echo "================================================================"
echo "[$(ts)] Adaptive Basic V2 launch (PID $$)"
echo "================================================================"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo
echo "[$(ts)] Launching adaptive basic (config matched to adaptive Bug3 verify)..."
python scripts/run.py adaptive thesis_crossdevice_main \
    --run-config "model-type=basic" \
    --run-config "lr=0.005" \
    --run-config "checkpoint-rule=best_round_restore" \
    --run-config "fraction-train=0.1" \
    --run-config "local-epochs=1" \
    --run-config "num-server-rounds=100" \
    --run-config "embedding-dim=64" \
    --run-config "wandb-enabled=false" \
    > "$INVEST/adaptive-basic-ablation.log" 2>&1
AB_RC=$?
echo "[$(ts)] Adaptive basic exit code: $AB_RC"

echo
echo "[$(ts)] Running analyze..."
python scripts/analyze_bug_investigation.py > "$INVEST/analyze-final-v2.txt" 2>&1 || true
tail -25 "$INVEST/analyze-final-v2.txt"

echo
echo "[$(ts)] ADAPTIVE BASIC V2 COMPLETE."
