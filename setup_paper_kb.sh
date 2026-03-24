#!/bin/bash
# ============================================================
# setup_paper_kb.sh
# Initialize the Paper Knowledge Base structure for thesis project
# 
# Usage: bash setup_paper_kb.sh [project_root]
#   project_root: path to your thesis project (default: current directory)
# ============================================================

set -euo pipefail

PROJECT_ROOT="${1:-.}"

echo "=== Setting up Paper Knowledge Base ==="
echo "Project root: $(realpath "$PROJECT_ROOT")"

# Create directory structure
mkdir -p "$PROJECT_ROOT/Papers/raw"
mkdir -p "$PROJECT_ROOT/Papers/digested"
mkdir -p "$PROJECT_ROOT/.claude/commands"

# Create .gitkeep for raw (so git tracks the empty folder)
touch "$PROJECT_ROOT/Papers/raw/.gitkeep"

echo ""
echo "=== Directory structure ==="
echo "
$PROJECT_ROOT/
├── CLAUDE.md                          # Project instructions (loaded every session)
├── .claude/
│   └── commands/
│       ├── digest-paper.md            # /digest-paper <path>  → digest one paper
│       └── batch-digest.md            # /batch-digest          → digest all unprocessed
├── Papers/
│   ├── raw/                           # Drop PDF files here
│   │   └── .gitkeep
│   └── digested/                      # AI-generated structured summaries
│       ├── _INDEX.md                  # Master index (read this first)
│       └── _EXAMPLE_mcmahan_2017_fedavg.md  # Reference example
"

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy your PDF papers into Papers/raw/"
echo "  2. In Claude Code, run: /digest-paper Papers/raw/<filename>.pdf"
echo "  3. Or run: /batch-digest to process all papers at once"
echo "  4. Review and refine the generated digests"
echo ""
echo "Tip: The _EXAMPLE file in Papers/digested/ shows the expected quality."
echo "     Delete it once you have real digests, or keep it as reference."
