---
name: train-remote
description: Prepare a federated training run on the LAPTOP and hand it to the training workstation (3-machine flow). Use when the user wants to run/launch/re-run an experiment, start a training run, or "chạy train"; also for finishing a run (commit results) and pulling results back for analysis. Replaces the old run-experiment skill.
user-invocable: true
argument-hint: "<module> <what to change vs. a reference run>"
allowed-tools: Bash, Read, Glob, Grep, Edit, Write
---

# Train on the workstation, develop on the laptop

Three machines (see CLAUDE.md > "Three-machine workflow"):
- **Laptop (this Claude)** — code, debug, smoke tests, analysis. All Claude work happens here.
- **Workstation** (RTX 5090, shared, runs someone else's Claude account) — TRAIN ONLY. Never edit code there; it only runs `scripts/remote/*.sh`.
- **MacBook** — just a terminal that ssh-es into the other two.

Channels: **git** (code + small result JSONs), **W&B/Forge** (live metrics + end-of-run email/Slack alert). No rsync, no heavy files on the laptop.

## Step 1 — Pre-flight on the laptop (all 5, never skip)

1. **Matrix scan** — does this cell already exist? `ls results/federated/<module>/` and read each `manifest.json` / `EVAL_VALIDITY.json`; never re-run an existing cell or launch on an n=1 hunch.
2. **Fix-present** — the module's run_id eval fix and any fix the run depends on are in the current branch (`git log --oneline -- <file>`; grep the live code, not commit messages).
3. **Contract-check** — every `--run-config` key is declared in `<module>/pyproject.toml [tool.flwr.app.config]` (flwr rejects undeclared keys at startup). Read the defaults there instead of guessing.
4. **Sanity import** — `python3 -c "import federated_<pkg>, fedrec_foundation"`.
5. **Dry-run** — `python3 scripts/run.py --dry-run <module> <mode> --run-config '...'` prints the exact `flwr run` line.

Protocol pins for claim runs: `fraction-train=0.1`, 100 rounds, `run-seed=42`, `checkpoint-rule="best_round_restore"`, **`early-stopping-enabled=false`** (ES killed a 15h run once), `strict-d06-cache=true` for modules with per-user state. Thesis tags: `thesis-run-label="ablation_<dim>=<value>"` (keep `"main"` for credited cells), `ablation-dimension`, `ablation-value`, `wandb-run-name`.

**Quoting rule:** single quotes outside, double quotes around string values inside: `'key="string" n=42 flag=true'`.

Optional: a smoke run on the laptop GPU to prove the code path works (`num-server-rounds=2 fraction-train=0.01`). Its numbers are NOT comparable — official numbers only come from the workstation.

## Step 2 — Write a portable launch wrapper and push it

Create `.bug-investigation/run_<short_name>.sh` (portable — no absolute paths):

```bash
#!/usr/bin/env bash
# <What this run tests, in one line.>
# Mirrors <reference run_id> EXCEPT: <the one knob that changes>.
# Pre-flight: <date>, 5/5 checks passed. Expected wall-clock: <h>.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
exec python3 scripts/run.py <module> <mode> --run-config '<pinned knobs + the change + thesis tags>'
```

Then `chmod +x`, commit (`chore(investigation): add launch wrapper for <name>`) and `git push`.
Expected wall-clock (from past runs): pfedrec ~8h, baseline ~15h, personalized/adaptive ~15h.

## Step 3 — Give the user the exact workstation commands

The user pastes these into an ssh session (from the MacBook). Print them filled in:

```bash
cd <repo on workstation> && git checkout <branch>
```
```bash
scripts/remote/start_run.sh .bug-investigation/run_<short_name>.sh <tmux-name>
```

`start_run.sh` pulls `--ff-only`, refuses to start if: the wrapper is not committed, code is dirty, <40GB disk free, another federation is running, or the tmux name is taken. It prints `tmux attach` / `tail -f` commands. Suggest `--dry-run` first if anything changed in the scripts.

While it runs: metrics on W&B (project `federated-cf-cross-device`, entity `vinh-federated-learning`); an email/Slack alert arrives at the end ([OK] or [CHECK] = cold-eval / population warning).

## Step 4 — After the alert: record the run (only if it will be cited)

```bash
scripts/remote/finish_run.sh <module> <run_id> --archive-cache
```

Runs eval_validity, commits ONLY that run's `results.json` / `manifest.json` / `EVAL_VALIDITY.json` (+ `reeval_*.json`, `per_user_metrics.json` if present), pushes, and (with `--archive-cache`) moves the run's per-user cache to the HDD archive after a per-file SHA-256 check, leaving a symlink. Smoke/debug runs: skip this, W&B is enough.

## Step 5 — Back on the laptop

`git pull`, then analyze (`/analyze-results`). Before citing a number: `EVAL_VALIDITY.json` status `valid`, `evaluated_users == 6040`, and `d06_cache_misses == 0` for per-user-state modules. Never compare in-loop best with full-pop numbers.

## Offline re-evaluation (reeval)

Needs the run's `global_state_best.pt` + per-user cache, which stay on the workstation. Write the reeval command on the laptop, commit a wrapper the same way, run it on the workstation with `start_run.sh` (CPU, minutes), then `finish_run.sh` picks up the `reeval_*.json` files.
