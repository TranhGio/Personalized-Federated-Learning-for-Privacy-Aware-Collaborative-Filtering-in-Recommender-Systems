# Codebase Concerns

**Analysis Date:** 2026-04-17

Thesis research codebase. Correctness for reported metrics matters more than production hardening. This document prioritizes concerns that could **invalidate reported results** over pure code-quality smells.

## Tech Debt

**Federation setting (cross-silo) is methodologically unsupported:**
- Every published federated recommendation paper (PFedRec, FedMF, FedNCF, FedRecon, MetaMF, FedPerGNN, FedRAP, CoFedRec, P2FedRec, GPFedRec) uses cross-device (1 user = 1 client). Only one 4-page workshop paper uses cross-silo for recommendation.
- Current default in all four modules is cross-silo via `options.num-supernodes = 5`:
  - `federated-baseline-cf/pyproject.toml:92,97`
  - `federated-personalized-cf/pyproject.toml:92,97`
  - `federated-adaptive-personalized-cf/pyproject.toml:207,212`
  - `federated-pfedrec/pyproject.toml:85,88`
- The `partition-mode = "natural"` default (e.g., `federated-baseline-cf/pyproject.toml:62`) implies cross-device intent, but with only 5 supernodes the module collapses 6040 users onto 5 clients anyway. The two configs are inconsistent.
- Migration plan exists at `docs/superpowers/plans/2026-04-04-cross-device-migration.md` but is not yet executed.
- Impact: A thesis reviewer familiar with FedRec would flag this as a fundamental methodological concern. All thesis contributions (hierarchical conditional alpha, per-user alpha, dual-level personalization, global prototype) are designed around **per-user** statistics and are better fits for cross-device.
- Fix approach: Execute the cross-device migration plan. Set `num-supernodes = 6040`, `fraction-train = 0.1`, align embedding dim / local epochs with published baselines.

**PFedRec reference vs Flower configs diverge in six category-A dimensions:**
- Reference: `IJCAI-23-PFedRec/` reproduces paper numbers (HR@10=72.86%, NDCG@10=44.07% at round 89) with dim=32, BCE, SGD(lr=0.1), 1 local epoch, 4 training negatives, 6040 cross-device clients.
- Flower PFedRec: same defaults were adopted (`federated-pfedrec/pyproject.toml:38-54`) EXCEPT `num-supernodes = 5` — so cross-device is silently undone at run time.
- Flower baseline / personalized / adaptive modules still use dim=128, BPR, Adam, 5 local epochs, 1 training negative. These are valid thesis variants, but a direct HR/NDCG comparison against PFedRec reference is apples-to-oranges.
- Fix approach: Adopt a single standardized config (dim=64, 4 negatives, 1 local epoch, 100 rounds, early-stopping patience=10, LOO + 99 neg eval) across all four Flower modules for the main comparison. Keep PFedRec reference numbers in a separate "context" table, not in the main results.

**Eight known bugs in `federated-pfedrec` (Codex review, not all yet fixed):**
1. Training negatives can include test positives (`federated-pfedrec/federated_pfedrec/task.py:134-166`). `prepare_user_train_data` builds `user_positives` from the trainloader only (`task.py:137-142`) and samples negatives from `all_items - user_positives` (`task.py:150`), but the leave-one-out test positive is held out of the trainloader and therefore can be drawn as a "negative" in training. Leaks test info into train.
2. No validation split — early stopping monitors `sampled_ndcg@10` directly on the test set (`federated-pfedrec/federated_pfedrec/dataset.py:356` only has train/test, no val; `pyproject.toml:68-72`). Any comparison-relevant early stopping decision uses test info.
3. Cross-silo aggregation weighting historically wrong (`client_app.py:286-293`): `item_embedding_accum` is averaged by `num_trained_users` (plain mean), not by sample count. Fine for cross-device (1 user = 1 sample count) but incorrect if cross-silo is ever used.
4. Per-user cache contaminates new experiments (`federated-pfedrec/federated_pfedrec/client_app.py:60-66`). `cache_dir = _MODULE_DIR.parent / ".embedding_cache"` persists across runs. Rerunning with different hyperparameters silently loads stale `affine_output.pt` per user.
5. Same training negatives every round — RNG re-seeded per call (`federated-pfedrec/federated_pfedrec/task.py:134` `rng = random.Random(seed)` with default seed=42). Each round produces identical negatives.
6. Partial participation: server uses `random.sample` (`federated-pfedrec/federated_pfedrec/server_app.py:250`) but no seed is set, so selection is non-deterministic across runs — yet within a run the same seed is used everywhere else. Limits reproducibility.
7. Early stopping doesn't checkpoint best model: `federated-*/early_stopping.py` records the best metric but does not snapshot `arrays`; final reported metrics are from the last round, not the best round (four duplicated copies of the class confirm this).
8. Eval BCE loss computed on positives only (`federated-pfedrec/federated_pfedrec/task.py:420` region via `test_pfedrec`). Loss value is not directly comparable to training loss, which includes negatives.
9. `affine_output.bias` classified LOCAL in Flower (`federated-pfedrec/federated_pfedrec/strategy.py:19-22`) but reference aggregates the bias globally (`IJCAI-23-PFedRec/engine.py:143` deletes only `affine_output.weight` before aggregation, so `affine_output.bias` is sent to the server). Flower PFedRec therefore never shares the bias — a semantic divergence from the reference algorithm.
- Files: `federated-pfedrec/federated_pfedrec/task.py`, `federated-pfedrec/federated_pfedrec/client_app.py`, `federated-pfedrec/federated_pfedrec/server_app.py`, `federated-pfedrec/federated_pfedrec/strategy.py`, `federated-pfedrec/federated_pfedrec/dataset.py`, `federated-pfedrec/federated_pfedrec/early_stopping.py`
- Impact: These must be fixed before Flower PFedRec can claim to reproduce the paper's numbers.
- Fix approach: Track each as a discrete phase. Start with the test-positive-in-training-negatives leak and the validation-split gap — those directly invalidate metrics.

**Training-negatives can include test positive in other modules too:**
- `federated-baseline-cf/federated_baseline_cf/task.py:238-246` builds `user_rated_items` from `trainloader` only, then `model.sample_negatives(..., user_rated_items=user_rated_items)` at `task.py:259-265` can draw the held-out test item as a "negative" positive.
- Same pattern in `federated-personalized-cf/federated_personalized_cf/task.py` (train-negative pipeline) and `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py`.
- Impact: All four modules leak the held-out test positive into training as a negative. Under leave-one-out evaluation this inflates training difficulty in the wrong direction (trains the model to push the test positive down) — silently hurts the very metric being reported.
- Fix approach: Include test items in the `user_rated_items` exclusion set when building training negatives; same bug pattern must be fixed in every `task.py`.

**Per-user learned alpha does not accumulate across rounds:**
- Flow in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py`: 
  1. `get_model(...)` (line 247) — no `_logit_alpha` yet.
  2. `load_local_user_embeddings(model, partition_id)` (line 260) — iterates `self._LOCAL_PARAMS`, but `_LOCAL_PARAMS` (`models/dual_personalized_bpr_mf.py:542-569`) includes `_logit_alpha.weight` only if `_per_user_alpha_enabled` is true, which it isn't yet.
  3. `model.enable_per_user_alpha(...)` (line 351) — creates a *fresh* `nn.Embedding` and overwrites it from the heuristic every round (`models/dual_personalized_bpr_mf.py:440-452`).
- Result: Cached `_logit_alpha.weight` from the previous round is never loaded; gradient-refined alpha is thrown away every round and re-initialized from the heuristic. The "per-user learned alpha" feature does not actually learn across rounds.
- Same concern for `_item_perturbation` (`client_app.py:360-363` and `models/dual_personalized_bpr_mf.py:467-479`) — loaded behavior depends on enable ordering.
- CLAUDE.md claims "Per-user alpha is initialized from heuristic only on first call; subsequent rounds load from cache" — this is the intent, but the code order contradicts it.
- Fix approach: Call `enable_per_user_alpha` / `enable_item_perturbation` BEFORE `load_local_user_embeddings` so the embedding modules exist and their keys are in `_LOCAL_PARAMS` when loading.

**Code duplicated across four modules:**
- `early_stopping.py` is byte-identical across `federated-baseline-cf`, `federated-personalized-cf`, `federated-adaptive-personalized-cf`, `federated-pfedrec` (diff returns no differences).
- `dataset.py` is ~95% shared (the files differ in whether `compute_stats` is returned and minor partition-mode handling) but at 577-692 lines each, 4 copies = ~2400 lines of near-duplicate code.
- Model files (`basic_mf.py`, `bpr_mf.py`) are also heavily duplicated.
- Impact: A fix to negative-sampling logic (see above) must be applied four times; it is easy to fix one and miss another. The PFedRec reference implementation already drifted because of this.
- Fix approach: Extract shared `dataset.py`, `early_stopping.py`, and loss/model base classes into a single top-level package (e.g., `fedrec_common/`) and import from there. Not urgent for results, critical for maintenance before new experiments.

## Known Bugs

**Non-deterministic per-round client selection:**
- `federated-baseline-cf/federated_baseline_cf/server_app.py:297`, `federated-personalized-cf/federated_personalized_cf/server_app.py:303`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:371`, `federated-pfedrec/federated_pfedrec/server_app.py:250` all call `random.sample(node_ids, num_selected)` without setting a server-level seed.
- Symptoms: Two runs with identical hyperparameters select different clients each round → different aggregate metrics. Blocks exact reproducibility.
- Workaround: Export `PYTHONHASHSEED`, seed `random` and `numpy.random` at top of `@app.main`.

**`sampled_ndcg@10` uses globally re-seeded RNG:**
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py:952-953`: `import random; random.seed(seed)` at the top of `evaluate_ranking_sampled`. Same pattern in the baseline and personalized modules' sampled eval functions.
- Symptom: Every round the 99 negatives sampled for each user are identical. Evaluation variance is artificially low, and if any of the 99 negatives happens to be an "easy" or "hard" case for the held-out user, that bias is baked into every reported round.
- Fix approach: Use a local `random.Random(seed + round_num + user_id)` rather than globally re-seeding.

**`torch.load(..., weights_only=False)` used for embedding cache:**
- `federated-pfedrec/federated_pfedrec/client_app.py:147`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:138`, `federated-personalized-cf/federated_personalized_cf/client_app.py:127`.
- Impact: Deserializes arbitrary Python via pickle. If `.embedding_cache/` ever comes from an untrusted source (e.g., shared between researchers), this is arbitrary code execution. Research-grade risk, but still a code smell.
- Fix approach: `weights_only=True` is PyTorch 2.6+ default-safe. Save as plain tensors and load with `weights_only=True`.

**No version/schema check on cached `.pt` files:**
- `federated-*/client_app.py` `load_*` functions strip `_round` and `_timestamp` (e.g., `federated-adaptive-personalized-cf/.../client_app.py:141-142`) but do not validate the embedding dimensionality, model type, or run hash. If a user reruns with a different `embedding-dim`, shape-mismatch falls through to `strict=False` + partial-load (`set_local_parameters` in `models/dual_personalized_bpr_mf.py:632-637` silently truncates or zero-pads).
- Impact: Silent contamination between experiments. The PFedRec memory already calls this out as "Per-user cache contaminates new experiments".
- Fix approach: Save a sidecar JSON with `{embedding_dim, model_type, num_items, num_users, config_hash}`; refuse to load if any differ.

## Security Considerations

**W&B auth token not leaked in committed code, but:**
- `results/federated/sweeps/fedprox_sweep_20251228_224812/sweep_log.txt:30,5183` contains `Currently logged in as: tranhgio (vinh-federated-learning)` — identity disclosure, not a key, but exposes the W&B entity. Harmless.
- No W&B API keys hard-coded in Python files (grep confirms).
- No `.env` file present at repo root.

**`wandb/` directories committed / not in gitignore:**
- `du -sh` shows 20MB in `federated-baseline-cf/wandb/`, 20MB in `federated-personalized-cf/wandb/`, 8.9MB in `federated-adaptive-personalized-cf/wandb/`. 44 `wandb-metadata.json` files present.
- `.gitignore` at repo root (`.gitignore:1-58`) excludes `*.pt`, `__pycache__`, etc., but `wandb/` is NOT listed. Per-module `.gitignore` (e.g., `federated-adaptive-personalized-cf/.gitignore:1-164`) also does not exclude `wandb/`.
- Impact: Run logs, system metadata, and user/host names leak into the repo on commit. Increases repo size unnecessarily.
- Fix approach: Add `wandb/` to `.gitignore` (root-level) and `git rm --cached -r */wandb/` after confirming no critical logs are kept only there.

**`.embedding_cache/` not in root `.gitignore`:**
- Caught only by the `*.pt` rule, which excludes files but not directory entries. The empty `partition_{i}` directories can still be tracked. Minor.
- Per-module `.gitignore` (`federated-pfedrec/.gitignore`) may handle it locally — not verified here.

## Performance Bottlenecks

**`.embedding_cache/` grows with per-user files (PFedRec):**
- `federated-pfedrec/federated_pfedrec/client_app.py:60-67` creates `partition_{id}/user_{user_idx}/affine_output.pt` per user. With 6040 users and cross-device mode, that's 6040 files per partition; with cross-silo and 5 partitions × 1200 users, ~6040 files total.
- Current observed size: 20MB in `federated-adaptive-personalized-cf/.embedding_cache`, 15MB in `federated-personalized-cf/.embedding_cache`. Small today, but grows linearly with `embedding-dim` and number of experiment reruns (never cleaned automatically).
- Fix approach: Either consolidate all per-user tensors into a single sharded file, or implement `clear_embedding_cache(partition_id=None)` invocation at the start of every `@app.main` (clearing is wired in `federated-adaptive-personalized-cf/.../client_app.py:158-183` but not called).

**Large item-embedding learning rate in PFedRec is an intentional trick, not a bug:**
- `federated-pfedrec/federated_pfedrec/task.py:234-238`: `optimizer_item = SGD(model.embedding_item.parameters(), lr=lr * num_items * lr_eta)`. With `lr=0.1`, `num_items=3706`, `lr_eta=80` → effective LR = 29,648.
- This matches the reference (`IJCAI-23-PFedRec/engine.py:117-119`) and is required because item gradients are sparse across the full embedding table.
- Concern: The high LR is fragile. Any change to `num_items` (e.g., dataset swap), `batch_size`, or negative-sampling count could destabilize training silently. There is no learning-rate guardrail and no LR warm-up.
- Fix approach: None needed for reproduction. When generalizing to other datasets, add a sanity check that `lr * num_items * lr_eta` stays within a known-stable band.

**Communication cost of split architectures (scaling concern):**
- `federated-baseline-cf` transmits 874K params/round/client; `federated-personalized-cf` transmits 485K params/round/client (-44%); `federated-adaptive-personalized-cf` transmits ~38% of params per round. Fine for simulation, relevant for any on-device deployment claim.
- Not a bug, but should be surfaced in thesis discussion alongside communication-cost baselines.

## Fragile Areas

**Xavier init sensitivity (RecSys 2024):**
- `federated-adaptive-personalized-cf/.../models/bpr_mf.py:142-143`, `basic_mf.py:90-91`, `dual_personalized_bpr_mf.py:202-203` — all use `init.xavier_uniform_`. Same pattern in other modules (baseline, personalized).
- Biases use `init.normal_(mean=0.0, std=0.01)` (`bpr_mf.py:147-148`).
- CLAUDE.md flags this: "Xavier initialization is critical (50% performance variance with poor init per RecSys 2024)".
- Concern: No unit test verifies that `reset_parameters()` is actually called during `get_model()`. Any refactor that drops this init will cause a silent ~50% performance regression. No seed is set before the init, so run-to-run variance can be large.
- Fix approach: Assert post-init weight statistics in `test_models.py`; seed before model creation at the top of `client_app.train`.

**Alpha clipping to `[0.1, 0.95]` can mask real signal:**
- `federated-adaptive-personalized-cf/.../models/adaptive_alpha.py:208, 306, 339, 486` all apply `np.clip(alpha_raw, config.min_alpha, config.max_alpha)` with defaults 0.1 / 0.95.
- For `DataQuantityAlpha` with threshold=100, users with n=50 produce `α ≈ 0.076` → clipped to 0.1 (docstring example, line 189). A large fraction of sparse users in ML-1M will hit this floor.
- Concern: Clip-floor effects can make sparse users look like they have more personalization than the formula intends, erasing the very "more global for sparse users" behavior the method advertises.
- Fix approach: Log the clip-hit rate per round; ensure alpha distribution plots (`evaluation/alpha_analysis.py`) report the proportion of users at `min_alpha` and `max_alpha`.

**Dual-level fusion (`fusion-type=gate`) uses a single scalar gate:**
- `federated-adaptive-personalized-cf/.../models/dual_personalized_bpr_mf.py` — the gate is learnable via sigmoid but applies uniformly across dimensions and users. No per-user fusion learning unless `concat` is used.
- Fragile because: If the gate collapses to 0 or 1, the CF branch or MLP branch is silently ignored. No assertion or monitoring.
- Fix approach: Log `sigmoid(fusion_gate)` per round alongside per-user alpha.

**BPR "RMSE ~2.2" is expected but only documented in CLAUDE.md:**
- No assertion in code. A future dev who "fixes" the RMSE by clamping or rescaling could break the ranking performance without noticing.
- Fix approach: Add a comment in the BPR loss implementation explaining why raw scores are unbounded.

**FedProx proximal term scope differs by module — easy to confuse:**
- Baseline (`federated-baseline-cf/federated_baseline_cf/task.py:176-180`) regularizes all params because all are global.
- Personalized (`federated-personalized-cf/federated_personalized_cf/task.py:182-196`) uses `global_param_names` to filter.
- PFedRec (`federated-pfedrec/federated_pfedrec/task.py:267-271`) only regularizes `model.embedding_item.weight`.
- The three implementations use three different control flows. A future change to any one of them is unlikely to be mirrored in the others.

## Scaling Limits

**Flower simulation at `num-supernodes = 6040`:**
- Memory suggests this is supported, but not yet tested in this repo. Current runs are all at `num-supernodes = 5`.
- Risk: Any shared in-memory state (e.g., `_dataset_cache` in `federated-pfedrec/federated_pfedrec/task.py:23`, `federated-adaptive-personalized-cf/.../task.py`) assumes partition IDs are small (0..4) and may not scale, but looks dict-keyed so should be fine.
- Per-partition `.embedding_cache/partition_{id}/` at 6040 partitions = 6040 directories. Filesystem overhead non-trivial on some FSes.

**`data/ml-1m/` download (`dataset.py:79`):** Hard-coded URL `https://files.grouplens.org/datasets/movielens/ml-1m.zip`. If the host goes down, all four modules break. Not a bug today; worth mirroring.

## Dependencies at Risk

**`flwr>=1.22.0` API in flux:**
- All four modules use the "new" `flwr.serverapp.ServerApp`, `flwr.clientapp.ClientApp`, `Grid.send_and_receive(...)` API. The older Flower API (pre-1.20) used `start_server` / `start_client` and `Strategy.aggregate_fit`. The newer API is still evolving.
- Risk: Version bump could break the `Grid` message-passing flow in `federated-adaptive-personalized-cf/.../server_app.py:393` (`list(grid.send_and_receive(train_messages))`).
- Fix approach: Pin `flwr==1.22.x` in `pyproject.toml` until the new API stabilizes.

**`torch>=2.7.1`, `torchvision>=0.22.1`:** Very recent. The PFedRec reference had to be patched for Python 3.13 / pandas 2.x compatibility (noted in config_comparison memory). Future minor torch updates may bring more churn.

**`IJCAI-23-PFedRec/` reference added as untracked directory:** `git ls-files --others` shows the whole tree. Not yet committed. If that directory ever desyncs from the original repo, reproduction claims become unverifiable.

## Missing Critical Features

**No validation split — early stopping monitors test set:**
- `federated-*/federated_*/dataset.py` provides only `create_train_test_split` and `create_leave_one_out_split`. No `val` split.
- `early-stopping-metric = "sampled_ndcg@10"` (`pyproject.toml:76` across modules) is computed on the test set.
- Impact: Reported numbers are post-selection on the same data they're reported on. Standard information leak.
- Fix approach: Hold out a second per-user interaction (penultimate timestamp) as validation. Use validation for early stopping; use test only for final numbers.

**No automated test / CI runner:**
- `.github/workflows/claude.yml` is the only CI job, and it only spins up Claude on PR events, not pytest.
- `federated-baseline-cf/test_dataset.py`, `test_models.py`, `federated-personalized-cf/test_*.py` exist but are not byte-identical to other modules' counterparts, and there is no `pytest.ini` / `conftest.py` / tox / CI hook.
- Impact: Every code change ships without any regression coverage. For a codebase targeting thesis results, this is risky — a silent numerical regression is exactly the failure mode that invalidates everything.
- Fix approach: Add a minimum CI job that runs `pytest` in each module and an end-to-end smoke test (1 round, 1 local epoch, 5 supernodes) — fail if HR@10 drops to 0.

**No best-model checkpoint in early stopping:**
- `EarlyStopping.step(...)` (`federated-baseline-cf/federated_baseline_cf/early_stopping.py:68-161`) tracks the best metric but the four duplicated `step` methods do not persist `arrays` at the best round. Final `result.arrays` reflects the LAST round, not the BEST round.
- Impact: Reported final numbers may be from a round where performance has already started to degrade past the best.
- Fix approach: Extend `EarlyStoppingState` with a `best_arrays` field; restore before final evaluation.

## Test Coverage Gaps

**Ranking-metric evaluation has no unit test:**
- `evaluate_ranking_sampled` (`federated-adaptive-personalized-cf/.../task.py:925-1074`, ~150 lines of ranking logic) has no test. Equivalents in other modules (`federated-baseline-cf/.../task.py:748+`, `federated-personalized-cf/.../task.py:763+`) are also untested.
- Risk: Off-by-one in rank calculation, wrong denominator in NDCG, or silent clamp of scores could distort metrics by 10-20% without anyone noticing.
- Priority: **High** — these metrics are the primary thesis numbers.

**Adaptive alpha has docstring examples but no integration test:**
- `federated-adaptive-personalized-cf/.../models/adaptive_alpha.py:226-237, 440-487` contain numerical examples in docstrings but no pytest verification that `HierarchicalConditionalAlpha.compute(...)` returns the documented values.
- Priority: **Medium** — easy to add, directly protects the thesis contribution.

**Split-learning `get/set_local_parameters` is not tested under shape mismatch:**
- `federated-adaptive-personalized-cf/.../models/dual_personalized_bpr_mf.py:603-640` has a three-way branch (exact shape, partial load for new users, truncate) — comment at line 624-637 flags it as handled but no test exercises the truncation path.
- Priority: Medium — easy to produce silent corruption when `num-users` changes between runs.

**`test_models.py` only covers baseline and personalized:**
- `federated-adaptive-personalized-cf/` has no top-level `test_models.py`. `federated-pfedrec/` has no tests at all. The most complex modules (the ones actually used for thesis results) have the least coverage.

## In-Flight Work

**Active branch `feat/try_to_run_the_baseline`:**
- Commit history is noisy: "have no idea but sweep", "try to run the baseline", "dump", "feat: result of run_baseline_sweep_loo.sh". Not production-stable state.
- 18 files modified (`git status` before analysis): `CLAUDE.md`, all four modules' `client_app.py`/`dataset.py`/`server_app.py`/`task.py`/`pyproject.toml`, plus `Papers/digested/_INDEX.md`.
- 12+ untracked files: entire `federated-pfedrec/` module (recently added, not yet committed), entire `IJCAI-23-PFedRec/` reference, 10 new digested paper markdown files, new `docs/superpowers/plans/2026-04-04-cross-device-migration.md`, and `results/federated/pfedrec/pfedrec_mlp_fedavg_dim32_lr0.1_eta80_r2_f1.0_results.json`.
- Impact: A large, uncommitted surface makes rollback impossible if something goes wrong. Mixing dataset changes with reference-code addition with a paper-KB update in a single branch is hard to review.
- Fix approach: Split into focused commits — (a) commit the `IJCAI-23-PFedRec/` reference as a separate commit or submodule, (b) commit `federated-pfedrec/` module as a single "add PFedRec calibration baseline" commit, (c) commit the four-module `partition-mode` additions as "add natural partitioning (cross-device option)", (d) commit paper-KB updates separately.

## Research vs Production Gaps

**Expected at this stage, worth acknowledging:**
- No typing enforcement (`mypy` not configured) despite CLAUDE.md mandate "Type hints on all function signatures".
- No formatter config (`.prettierrc` / `black` / `ruff` not configured at root). Code style varies across the four module copies.
- No pinned dependency lockfile — only `>=` constraints in `pyproject.toml`.
- Logging goes through `print()` (883 occurrences across 31 files); no structured logger. `task.py` and `server_app.py` mix progress prints, warnings, and debug output.
- `_device_cache` module-global in each `client_app.py` (`federated-adaptive-personalized-cf/.../client_app.py:37`) is a race hazard under Flower's process-based simulation, though in practice each supernode is a separate process so this is fine.
- Four copies of `early_stopping.py` (byte-identical), four mostly-identical `dataset.py`, and four very similar `bpr_mf.py` — addressed above under "Code duplicated across four modules".
- Centralized baselines (`centralize_baseline_ncf.py`, `centralized_baseline_svd.ipynb`) are in the repo root, not part of any module, and are not referenced by any other file. Fine as reference scripts; document their status.

---

*Concerns audit: 2026-04-17*
