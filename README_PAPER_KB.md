# Paper Knowledge Base System for Thesis Research

## Overview

This system transforms raw academic PDFs into structured, AI-agent-optimized knowledge files. It follows a **lazy-loading** architecture: instead of forcing every paper into context, the agent reads only what it needs, when it needs it.

## Architecture

```
                    ┌─────────────┐
                    │  CLAUDE.md  │  ← Loaded EVERY session (~30 lines)
                    │  (concise)  │     Points agent to Paper KB when needed
                    └──────┬──────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  _INDEX.md     │  ← Agent reads this to FIND relevant papers
                  │  (categorized) │     Organized by topic + relevance tier
                  └────────┬───────┘
                           │
                  ┌────────▼────────┐
                  │ <paper_id>.md   │  ← Agent reads SPECIFIC digest for details
                  │ (structured)    │     Equations, algorithms, implementation notes
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ Papers/raw/*.pdf │  ← ONLY when agent needs to verify exact details
                  │ (full papers)   │     Rare — digests should be sufficient 95% of time
                  └─────────────────┘
```

## How It Works

### Digesting a Paper (one at a time)

```bash
# In Claude Code:
/digest-paper Papers/raw/pfedme_2020.pdf
```

The agent will:
1. Read the PDF using optimal extraction strategy (text + rasterize key figures)
2. Extract all technical details following the structured template
3. Assess relevance to your specific thesis
4. Write the digest to `Papers/digested/<paper_id>.md`
5. Update `_INDEX.md` with the new entry

### Batch Processing

```bash
# In Claude Code:
/batch-digest
```

Processes all undigested PDFs in `Papers/raw/`, prioritized by relevance to your thesis topic.

### During Implementation

When you ask the agent to implement something (e.g., "implement pFedMe with BPR-MF"), it will:
1. Read `CLAUDE.md` → sees "check Paper KB for prior work"
2. Read `_INDEX.md` → finds `pfedme_2020.md` under "Personalization"
3. Read `pfedme_2020.md` → gets exact equations, algorithm steps, hyperparameters
4. Implement with full knowledge of the method

## Why This Design?

| Design Choice | Rationale |
|---|---|
| **Slash commands** (not CLAUDE.md instructions) | Digest workflow is on-demand, not every-session |
| **Separate digest per paper** | Agent loads only the papers it needs |
| **_INDEX.md as router** | Cheap to read (~50 lines), helps agent find the right digest |
| **Structured format with sections** | Agent can jump to "3.1 Objective Function" without reading motivation |
| **"Connections to My Thesis" section** | Most valuable section — pre-computed relevance assessment |
| **Example digest included** | Few-shot reference for consistent quality across papers |
| **Consistent notation convention** | $w$ for global, $\theta_i$ for personal — reduces confusion across papers |

## Maintenance Tips

1. **Review digests after generation** — especially "Section 6: Connections to My Thesis". The agent's assessment may miss nuances you see.
2. **Keep _INDEX.md updated** — this is the agent's "table of contents". If it's stale, the agent won't find papers.
3. **Delete the _EXAMPLE file** once you have 2-3 real digests. It's a bootstrap reference.
4. **Update CLAUDE.md's "Current Phase"** as your thesis progresses. This helps the agent prioritize relevance.
