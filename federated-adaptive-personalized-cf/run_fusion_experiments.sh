#!/bin/bash
# =============================================================================
# Run Dual Model Experiments with Different Fusion Types
# =============================================================================
# This script runs the dual personalized model with:
# 1. fusion-type = "gate" (learnable gate)
# 2. fusion-type = "concat" (concatenate and project)
#
# Results will be saved separately:
# - dual_mf_split_fedprox_mu0.01_r50_f1.0_gate_mlp128-64_results.json
# - dual_mf_split_fedprox_mu0.01_r50_f1.0_concat_mlp128-64_results.json
# =============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Dual Model Fusion Type Experiments${NC}"
echo -e "${BLUE}=============================================${NC}"

# Clear local user embeddings cache to start fresh
echo -e "\n${GREEN}Clearing local user embeddings cache...${NC}"
rm -rf local_user_embeddings/
echo "Cache cleared."

# =============================================================================
# Experiment 1: Gate Fusion
# =============================================================================
echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}  Experiment 1: fusion-type = gate${NC}"
echo -e "${GREEN}=============================================${NC}"
echo "Formula: score = σ(g) * cf_score + (1-σ(g)) * mlp_score"
echo "Starting at: $(date)"

flwr run . --run-config "model-type='dual' fusion-type='concat' mlp-hidden-dims = '512,256,128' "

echo -e "\n${GREEN}Gate fusion experiment completed at: $(date)${NC}"

# Clear cache between experiments
echo -e "\n${GREEN}Clearing local user embeddings cache for next experiment...${NC}"
rm -rf local_user_embeddings/

# =============================================================================
# Experiment 2: Concat Fusion
# =============================================================================
echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}  Experiment 2: fusion-type = concat${NC}"
echo -e "${GREEN}=============================================${NC}"
echo "Formula: score = Linear([cf_score, mlp_score])"
echo "Starting at: $(date)"

flwr run . --run-config "model-type='dual' fusion-type='concat'"

echo -e "\n${GREEN}Concat fusion experiment completed at: $(date)${NC}"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${BLUE}=============================================${NC}"
echo -e "${BLUE}  All Experiments Completed!${NC}"
echo -e "${BLUE}=============================================${NC}"
echo "Results saved to: ../results/federated/personalized/"
echo ""
echo "Files created:"
ls -la ../results/federated/personalized/dual_mf_split_*_gate_*.json 2>/dev/null || echo "  (gate results not found)"
ls -la ../results/federated/personalized/dual_mf_split_*_concat_*.json 2>/dev/null || echo "  (concat results not found)"
