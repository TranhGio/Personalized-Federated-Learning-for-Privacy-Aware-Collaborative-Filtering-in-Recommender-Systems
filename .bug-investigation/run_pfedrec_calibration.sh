#!/usr/bin/env bash
# Queue exp #2 — PFedRec + final-calibration (capstone headroom gate).
# Mirrors credited run 20260608-071106-ef41ab (paper_compat_pfedrec, C=0.1,
# 100 rounds, seed 42, best_round_restore, ES off) EXCEPT:
#   final-calibration-enabled=true  (+freeze-items=true, lr_eta=0 during calib)
#   strict-d06-cache=true           (fail-closed cold-eval guard, protocol-neutral)
#   thesis tagging labels           (ablation_final_calibration=true)
# Pre-flight: workflow wf_956c54c5-3a8 (5 agents, GO-WITH-CHANGES, 2026-07-26).
# Expected wall-clock ~42-46h (ef41ab 41.2h + calibration pass ~6040x1 epoch).
set -uo pipefail
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
exec /home/bes/miniconda3/bin/python3 scripts/run.py pfedrec paper_compat_pfedrec --run-config 'fraction-train=0.1 strategy="fedavg" early-stopping-enabled=false final-calibration-enabled=true final-calibration-freeze-items=true strict-d06-cache=true run-seed=42 thesis-run-label="ablation_final_calibration=true" ablation-dimension="final_calibration" ablation-value="true" wandb-run-name="thesis-ablation-pfedrec-seed42-final_calibration=true"'
