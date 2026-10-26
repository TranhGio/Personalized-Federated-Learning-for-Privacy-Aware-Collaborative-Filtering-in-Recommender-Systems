#!/bin/bash
# Ablation study: isolate the effect of each next-gen personalization technique
# Base: dual concat mlp512-256-128, FedProx, hierarchical_conditional alpha
# Previous best (no techniques): sampled_NDCG@10 = 0.4270
# Full combo (all 3): sampled_NDCG@10 = 0.4079

set -e
cd "$(dirname "$0")/.."

# Common config: string values must be quoted for TOML parsing
BASE='model-type="dual" fusion-type="concat"'
ES='early-stopping-enabled=true early-stopping-patience=10'

echo "=============================================="
echo "  Ablation Study: Next-Gen Techniques"
echo "=============================================="

# 1. Per-user alpha ONLY
echo ""
echo "[1/5] Per-user alpha ONLY"
echo "----------------------------------------------"
flwr run . --run-config "$BASE enable-per-user-alpha=true enable-item-perturbation=false contrastive-lambda=0.0 num-server-rounds=50 $ES"

# 2. Item perturbation ONLY
echo ""
echo "[2/5] Item perturbation ONLY"
echo "----------------------------------------------"
flwr run . --run-config "$BASE enable-per-user-alpha=false enable-item-perturbation=true item-perturbation-reg=0.01 contrastive-lambda=0.0 num-server-rounds=50 $ES"

# 3. Contrastive ONLY
echo ""
echo "[3/5] Contrastive ONLY"
echo "----------------------------------------------"
flwr run . --run-config "$BASE enable-per-user-alpha=false enable-item-perturbation=false contrastive-lambda=0.1 contrastive-tau=0.1 num-server-rounds=50 $ES"

# 4. Alpha + perturbation (no contrastive)
echo ""
echo "[4/5] Alpha + perturbation (no contrastive)"
echo "----------------------------------------------"
flwr run . --run-config "$BASE enable-per-user-alpha=true enable-item-perturbation=true item-perturbation-reg=0.01 contrastive-lambda=0.0 num-server-rounds=50 $ES"

# 5. Alpha + contrastive (no perturbation)
echo ""
echo "[5/5] Alpha + contrastive (no perturbation)"
echo "----------------------------------------------"
flwr run . --run-config "$BASE enable-per-user-alpha=true enable-item-perturbation=false contrastive-lambda=0.1 contrastive-tau=0.1 num-server-rounds=50 $ES"

echo ""
echo "=============================================="
echo "  Ablation complete! Results saved to:"
echo "  ../results/federated/personalized/"
echo "=============================================="
