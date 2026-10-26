---
name: train-remote
description: Prepare, hand off, and close out a federated training run under the thesis's three-machine flow (laptop = code + Claude, shared RTX 5090 workstation = training only, MacBook = terminal). Use this whenever the user wants to run, re-run, launch, queue, or schedule a training run, ablation, extra seed, or offline reeval on the workstation — including Vietnamese phrasings like "chạy train", "chạy lại run", "chạy thêm seed", "launch lên máy trạm" — and when a run just finished (W&B alert arrived, [OK]/[CHECK]) and its results need recording. Do not use it for analyzing results that are already pulled (use analyze-results), for W&B hyperparameter sweeps (use sweep), or for pure code edits/unit tests with no run.
user-invocable: true
argument-hint: "<module> <what changes vs. a reference run>"
allowed-tools: Bash, Read, Glob, Grep, Edit, Write
---

# Train on the workstation, develop on the laptop

The workstation is shared and runs someone else's Claude account, so **nothing Claude-driven happens there**. It only executes two reviewed scripts. You (on the laptop) do all the thinking and leave the user a short, copy-pasteable handoff. Information moves through **git** (code, launch wrappers, small result JSONs) and **W&B/Forge** (live metrics + an email/Slack alert at the end). Heavy artifacts (`global_state_*.pt`, `.embedding_cache/`) never leave the workstation.

The reason this flow is strict: past runs were lost to unchecked configs (an undeclared key crashing at startup, early stopping killing a 15h run at round 34, a missing `run_id` silently scoring every user cold). Each step below exists because of one of those.

Read `references/protocol.md` when choosing a mode, pinning knobs, or picking a reference run to mirror.

## 1. Pre-flight (on the laptop)

Do all five; report each as a one-line ✅/❌ in your answer.

1. **Matrix scan** — list `results/federated/<module>/*/` and read `manifest.json` / `EVAL_VALIDITY.json`. If the exact cell already exists, say so instead of launching. A new run should change exactly one thing versus a named reference run — knobs *and* `--federation` backend — so its result is interpretable.
2. **Fix-present** — confirm in the *live code* (grep, not commit messages) that the fixes the run depends on are on the branch, especially the module's D-06 `run_id` eval fix.
3. **Contract-check** — every `--run-config` key must be declared in `<module>/pyproject.toml` under `[tool.flwr.app.config]`; flwr rejects undeclared keys at startup, hours after the user went to bed. Read defaults there rather than guessing.
4. **Sanity import** — `python3 -c "import <package>, fedrec_foundation"`.
5. **Dry-run** — `python3 scripts/run.py --dry-run <module> <mode> --run-config '...'` and show the printed `flwr run` line.

Quoting: single quotes outside, double quotes around string values inside — `'run-seed=42 thesis-run-label="main"'`. Barewords fail with `TOMLDecodeError`.

A laptop smoke run (`num-server-rounds=2 fraction-train=0.01`) is optional and only proves the code path; its numbers are never cited.

## 2. Write a portable launch wrapper, commit, push

One wrapper per run in `.bug-investigation/run_<short_name>.sh`. It must work on any machine: no absolute paths, no hard-coded interpreter. Template in `references/protocol.md`. Header comments state what the run tests, which reference run it mirrors, the single knob that changes, and the expected wall-clock — the user reads this on the workstation before launching.

`chmod +x`, commit (`chore(investigation): add launch wrapper for <name>`), `git push`. If the push fails or the branch has no upstream, stop and tell the user; the workstation can only run what is on GitHub.

Several runs (e.g. extra seeds) → one wrapper each, launched one after another: there is a single GPU, and `start_run.sh` refuses to start while a federation is alive.

## 3. Hand off to the user

End your answer with this block, filled in (the user pastes it into an ssh session from the MacBook):

```
### Handoff — workstation
cd ~/Desktop/vinh/federated-learning/movie-recommendation-system
git fetch && git checkout <branch>
scripts/remote/start_run.sh .bug-investigation/run_<short_name>.sh <tmux-name> --dry-run
scripts/remote/start_run.sh .bug-investigation/run_<short_name>.sh <tmux-name>
# expected: ~<h>h · W&B project federated-cf-cross-device · alert by email/Slack at the end
# afterwards (only if this run will be cited):
scripts/remote/finish_run.sh <module> <run_id> --archive-cache
```

`start_run.sh` pulls `--ff-only` and refuses on: uncommitted wrapper, dirty code tree, < 40 GB free disk, a live federation, or a taken tmux name. `<run_id>` is unknown until the run starts — it appears in the W&B run and in `results/federated/<module>/`.

Do not run these commands yourself or ssh into the workstation: the user wants the workstation under their own hands.

## 4. When the run finishes

The alert title starts with `[OK]` or `[CHECK]`.

- **[OK]** and the run will be cited (thesis label, number in a report) → `finish_run.sh <module> <run_id> --archive-cache`, then `git pull` on the laptop and analyze. Smoke/debug runs → W&B is enough, skip `finish_run`.
- **[CHECK]** (`d06_cache_misses > 0` or `evaluated_users ≠ 6040`) → the best block is not citable yet. Do not tell the user to commit it as a result. Have them grep the run log for cache misses / the D-06 eval config, and compare against the in-loop numbers; cold-eval artifacts look like a ~5× crater versus in-loop. Recording it with `finish_run.sh` is fine only as an explicitly flagged invalid run.

Before citing any number: `EVAL_VALIDITY.json` status `valid`, `evaluated_users == 6040`, `d06_cache_misses == 0` for modules with per-user state, and compare full-pop with full-pop (never in-loop best vs full-pop).

## 5. Offline re-evaluation

`reeval.py` needs the run's `global_state_best.pt` + per-user cache, which live on the workstation (possibly archived to the HDD behind a symlink — that still works). Write the reeval command into a wrapper the same way and hand it off via `start_run.sh`; it runs on CPU in minutes, and `finish_run.sh` then commits the `reeval_*.json` files.
