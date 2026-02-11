#!/bin/bash
# Convenience commands for running wandb sweeps
#
# Usage:
#   source scripts/sweep_commands.sh
#   create_sweep          # Create a new sweep
#   run_sweep_agent       # Run an agent (requires SWEEP_ID)
#   test_sweep_config     # Test the sweep runner locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

create_sweep() {
    echo -e "${GREEN}Creating wandb sweep...${NC}"
    cd "$PROJECT_DIR"
    wandb sweep sweep.yaml
    echo -e "${YELLOW}Copy the SWEEP_ID from above and run:${NC}"
    echo "  run_sweep_agent <SWEEP_ID>"
}

run_sweep_agent() {
    if [ -z "$1" ]; then
        echo "Usage: run_sweep_agent <SWEEP_ID> [COUNT]"
        echo "  SWEEP_ID: Just the ID (e.g., bys9r8hj) - entity/project will be added automatically"
        echo "  COUNT: Number of experiments to run (default: 10)"
        echo ""
        echo "  Get SWEEP_ID from 'create_sweep' or wandb dashboard"
        return 1
    fi

    SWEEP_ID="$1"
    COUNT="${2:-10}"  # Default: run 10 experiments

    # Full sweep path with entity and project
    ENTITY="vinh-federated-learning"
    PROJECT="federated-adaptive-personalized-cf"
    FULL_SWEEP_PATH="$ENTITY/$PROJECT/$SWEEP_ID"

    echo -e "${GREEN}Starting wandb agent for sweep: $FULL_SWEEP_PATH${NC}"
    echo "  Will run $COUNT experiments"
    cd "$PROJECT_DIR"
    wandb agent --count "$COUNT" "$FULL_SWEEP_PATH"
}

test_sweep_config() {
    echo -e "${GREEN}Testing sweep configuration locally...${NC}"
    cd "$PROJECT_DIR"
    python scripts/run_wandb_sweep.py --test --dry-run
}

run_single_test() {
    echo -e "${GREEN}Running single training with default config...${NC}"
    cd "$PROJECT_DIR"
    python scripts/run_wandb_sweep.py --test
}

# Export functions so they can be used after sourcing
export -f create_sweep
export -f run_sweep_agent
export -f test_sweep_config
export -f run_single_test

echo -e "${GREEN}Sweep commands loaded!${NC}"
echo "Available commands:"
echo "  create_sweep          - Create a new wandb sweep from sweep.yaml"
echo "  run_sweep_agent <ID>  - Run sweep agent with given SWEEP_ID"
echo "  test_sweep_config     - Test sweep config locally (dry run)"
echo "  run_single_test       - Run single training with defaults"
