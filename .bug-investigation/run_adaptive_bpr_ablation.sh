#!/usr/bin/env bash
# ============================================================================
# H2 SAME-MODULE CONFIRMATION — Codex review concern #2
# ============================================================================
# Purpose: Confirm "PersonalMLP + α machinery hurts" finding within ONE
# code path (kills cross-module confound from comparing adaptive vs personalized).
#
# Config: adaptive module with BPRMF (no PersonalMLP, no fusion) and both
# next-gen flags OFF. Apples-to-apples vs adaptive dual Bug3 reference
# (20260526-082810-a9a08f) — same module, same mode, same hyperparams,
# only personalization machinery removed.
#
# Expected outcomes:
#   • If NDCG@10 lands ~0.22 (close to personalized 0.2246):
#     → PersonalMLP/α IS the bottleneck. H2 CONFIRMED within same code path.
#   • If NDCG@10 lands ~0.08 (close to adaptive dual 0.0842):
#     → Something else in adaptive infrastructure is broken; PersonalMLP not
#       the root cause. Look at split-learning state handling.
#   • Crash with `D-01/D-03 violated`:
#     → Design bug: adaptive's local-state contract assumes next-gen tensors
#       even when flags are off. Worth reporting separately.
#
# Pre-flight verified 2026-05-29:
#   - Matrix scan: zero existing runs at this exact config.
#   - Fix #1 (federation positional), Fix #2 (raw_data_hash), Fix #3 (BasicMF
#     clamp bypass) all present in working tree.
#   - D-01/D-03 contract: with both flags off, required set is
#     {user_embeddings.weight} only — BPRMF satisfies this.
#   - Dry-run command emits correct positional federation.
#   - Sanity import: BPRMF has all 5 expected local keys.
# ============================================================================

set -uo pipefail

REPO=/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
INVEST=$REPO/.bug-investigation
LOG=$INVEST/adaptive-bpr-ablation.log

cd "$REPO"
: > "$LOG"   # truncate previous attempt if any

echo "================================================================" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] H2 same-module ablation (PID $$)" | tee -a "$LOG"
echo "  module:     adaptive (federated-adaptive-personalized-cf)" | tee -a "$LOG"
echo "  model-type: bpr (BPRMF — no PersonalMLP, no fusion, no α)" | tee -a "$LOG"
echo "  next-gen:   enable-per-user-alpha=false, enable-item-perturbation=false" | tee -a "$LOG"
echo "  mode:       thesis_crossdevice_main (N=6040, canonical)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

python scripts/run.py adaptive thesis_crossdevice_main --federation local-sim-gpu \
    --run-config "model-type=bpr" \
    --run-config "enable-per-user-alpha=false" \
    --run-config "enable-item-perturbation=false" \
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ADAPTIVE_BPR_ABLATION FINISHED rc=$RC" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "Next step: python scripts/analyze_bug_investigation.py (after results.json is written)"
