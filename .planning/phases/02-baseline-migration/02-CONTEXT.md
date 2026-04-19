# Phase 2: Baseline Migration — Context

**Gathered:** 2026-04-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Migrate `federated-baseline-cf` to consume the `fedrec-foundation` package shipped in Phase 1, so it runs as a correct cross-device benchmark (1 user = 1 client, N=6040) with seeded sampling, sufficient-statistic metrics, and protocol fingerprint logged. This is the **lower-bound comparison baseline** for the thesis: all parameters global, no personalization. In scope:

- Wire `fedrec_foundation.{mapping, split, exclusion, rng, manifest, mode, evaluator, weight_policy, fit_metrics}` into `federated-baseline-cf/`.
- Replace the module's internal `random.seed`/`random.sample` calls with foundation RNGs (FND-06).
- Refactor the client to use the exclusion set for training-negative sampling (FND-03).
- Refactor the server to select clients via the seeded `server_rng` (BSL-04) and aggregate sufficient stats instead of averaging per-client ratios (BSL-06).
- Add the benchmark-mode `num_users_in_client == 1` assertion (BSL-02, D-11 from Phase 1).
- Emit the run manifest / protocol fingerprint into every result JSON (BSL-08, FND-07).

Out of scope for this phase:

- Any modifications to `federated-pfedrec/`, `federated-personalized-cf/`, `federated-adaptive-personalized-cf/` (Phases 3, 4, 5 own those).
- Any model-architecture change (BPR-MF / Basic-MF classes stay as-is).
- New evaluator implementations beyond consuming `fedrec_foundation.evaluator.get_primary_evaluator()` (selector-only; the existing `evaluate_ranking_sampled` function stays).
- Running the thesis comparison table (Phase 7).
- Extracting shared code into `fedrec_common/` (PROJECT.md: deferred to v2).

</domain>

<decisions>
## Implementation Decisions

### Migration shape

- **D-17:** **Rip-and-replace** the baseline's existing dataset.py helpers (`create_global_mappings`, `create_leave_one_out_split`, `load_movielens_1m`) with `fedrec_foundation` loaders. Baseline's `dataset.py` becomes a thin adapter that calls `fedrec_foundation.mapping.load_mapping()`, `.split.load_split_manifest()`, `.exclusion.load_exclusion()`. Single source of truth; prevents drift between foundation and baseline.

- **D-18:** **Surgical migration.** Phase 2 touches ONLY functions / lines directly implementing BSL-01..08. Pre-existing uncommitted edits in `federated-baseline-cf/{dataset,client_app,server_app,task}.py` (visible in `git status`) stay untouched. Planner receives explicit "do not touch" ranges per file. Accepts that the branch carries work-in-progress; Phase 2 is NOT a tidy-up pass.

- **D-19:** **CLI overrides allowed + loud warning + manifest capture.** Matches Phase 1 D-10. `flwr run . --run-config "num-supernodes=10"` is applied, prints `⚠ OVERRIDE: num-supernodes=10 (mode default=6040)` at run start, and is recorded in `manifest.overrides`. Never block; always surface.

### Sufficient-stat aggregation (BSL-06)

- **D-20:** **Custom `BaselineFedAvg(FedAvg)` subclass** owns aggregation. Override `aggregate_evaluate()` to sum sufficient stats (`hit_count@10`, `ndcg_sum@10`, `evaluated_users`) across all client `EvaluateRes` responses and compute the ratio ONCE server-side. Mirrors what `SplitFedAvg` does in the personalized/adaptive modules; survives future evaluator changes.

- **D-21:** **Strict `FitMetricsContract` keys on the wire.** Client returns a dict containing exactly `FIT_METRICS_REQUIRED_KEYS` (from Phase 1 CR-4). Server calls `validate_fit_metrics(fit_res.metrics)` and raises on malformed clients. No free-form extras in the metrics dict.

- **D-22:** **Per-group sufficient stats are client-side.** Each client knows its user's group (from `split_manifest.user_groups`) and returns per-group sufficient stats as part of its payload. **Implication:** `FitMetricsContract` is extended in Phase 2 Plan 01 with per-group keys (`hit_count_sparse@10`, `evaluated_users_sparse`, `hit_count_medium@10`, …) so `validate_fit_metrics()` continues to enforce schema — NOT free-form extras. Server aggregates per-group the same way it aggregates overall.

### Per-client memory model

- **D-23:** **Keep full 6040×d user embedding matrix on every client.** Preserves the baseline's defining invariant: "all params GLOBAL." Each client receives the full matrix from the server, trains only its user's row via D-24, returns the full (updated) matrix. ~3 MB/client at d=128 is fine at simulation scale.

- **D-24:** **Gradient masking via zeroing non-user rows.** After `loss.backward()`, zero out all rows of `user_embeddings.weight.grad` except the client's assigned user. Optimizer step then only moves the one row. One-line fix, optimizer-agnostic (works for Adam and SGD), no architecture change. The same pattern is also applied to `user_bias.weight.grad` if the model has user biases.

- **D-25:** **`resolve_mode_defaults(mode)` owns canonical hyperparams.** `fedrec_foundation.mode.resolve_mode_defaults("benchmark_cross_device")` returns the locked dict including `embedding_dim`, `lr`, `local_epochs`, `num_server_rounds`, `num_train_negatives`. The baseline's `pyproject.toml` values are fallback only (consulted when mode is unset — never, in practice). Server_app at startup calls `resolve_mode_defaults(mode)` and overrides `context.run_config` values, with any overrides captured per D-19.

### Result artifact shape

- **D-26:** **Selected client IDs per round live in both the result JSON AND the W&B step log.** Result JSON grows a `selected_clients_per_round: [[c1, c2, ...], [c1, c3, ...], ...]` top-level field (or under `_run_log.selected_clients`). Server also calls `wandb.log({"round/selected_clients": ids_list}, step=round)`. Reproducible (JSON is committed) + queryable (W&B). Disk cost: ~25 KB for a 50-round run at C=0.01.

- **D-27:** **Default `checkpoint-rule = "best_round_restore"` tracking `sampled_ndcg@10`.** Server tracks per-round `sampled_ndcg@10` from the sufficient-stat aggregator (D-20), saves global params at the best round, and at training end restores those params before producing the final result JSON. Matches what Phase 5 PFedRec reproduction needs for fair comparison against IJCAI-23 paper numbers (which report best-round, not final-round). Captured in `manifest.checkpoint_rule = "best_round_restore"`.

- **D-28:** **Flat `results/federated/` directory; `_manifest.mode` differentiates runs.** All FL runs share `results/federated/<run_id>.json`. Queries and comparison plots filter by `_manifest.mode ∈ {"benchmark_cross_device", "cross_silo_legacy"}`. No directory churn; matches Phase 1 D-15 one-file-per-run pattern; cross-silo historical runs remain colocated and comparable.

### Claude's Discretion

The following were NOT explicitly discussed; planner may decide at planning time within reasonable principles:

- **RNG purpose names for client sampling and batch shuffling.** Phase 1 D-14 defined three purposes (`train_neg`, `eval_neg`, `model_init`). BSL-04 needs server-side client sampling; plausibly `client_sampling` is a new purpose that uses `server_rng(seed)` directly (not per-user-scoped). Batch-shuffle determinism during local training may need another purpose (`batch_shuffle`) or may be skippable if the `DataLoader` is constructed with a dedicated `torch.Generator`. Planner picks the minimal set that closes BSL-04 and BSL-05.

- **Test coverage strategy for Phase 2.** Phase 1 introduced pytest-based TDD inside `scripts/foundation/tests/`. Baseline module currently has ad-hoc `test_dataset.py` / `test_models.py` scripts. Default recommendation: add a `federated-baseline-cf/tests/` directory with pytest-style tests mirroring Phase 1 layout, cover BSL-01..08 with at least one RED/GREEN pair each, and extend `scripts/foundation/tests/test_integration.py` with a baseline-specific smoke test. Planner may compress or expand as appropriate per task count.

- **Where the sufficient-stat aggregator class lives.** D-20 says "custom FedAvg subclass"; `BaselineFedAvg` could live in `federated_baseline_cf/strategy.py` (matches the pattern in `federated-personalized-cf/federated_personalized_cf/strategy.py` and `federated-adaptive-personalized-cf/.../strategy.py`). Planner places it there unless cause.

- **Manifest `git-commit` value when the repo has a dirty working tree** (pre-existing uncommitted edits, D-18). Two reasonable options: (a) record the current HEAD sha and add `dirty: true` flag when working tree is dirty; (b) record `HEAD-dirty-<short-hash-of-diff>`. Planner picks; either keeps provenance honest.

- **Loud-warning wording for D-19.** One suggestion: `"⚠ OVERRIDE: <key>=<value> (mode default=<default>). Run is NOT comparable to benchmark thesis table."` Planner may tighten.

- **Best-round-restore checkpoint storage location.** Two options: (a) in-memory only (restore then report, no disk write); (b) save to `.checkpoints/<run_id>-best.pt` so a crashed run can resume. Default recommendation: (a) — simpler, and crash-resilience is not a requirement for single-machine simulation.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner, executor) MUST read these before planning or implementing.**

### Phase 1 Foundation (consumed directly)
- `scripts/foundation/fedrec_foundation/mapping.py` — `build_mapping()`, `load_mapping()` (FND-01)
- `scripts/foundation/fedrec_foundation/split.py` — `build_split()`, `load_split_manifest()`, `SplitManifest` dataclass (FND-02)
- `scripts/foundation/fedrec_foundation/exclusion.py` — `build_exclusion()`, `load_exclusion()`, flat items+indptr layout (FND-03)
- `scripts/foundation/fedrec_foundation/evaluator.py` — `EvalProtocol` enum, `get_primary_evaluator()` (FND-04)
- `scripts/foundation/fedrec_foundation/weight_policy.py` — `WeightPolicy` enum, `compute_aggregation_weight()` (FND-05)
- `scripts/foundation/fedrec_foundation/fit_metrics.py` — `FitMetricsContract`, `FIT_METRICS_REQUIRED_KEYS`, `validate_fit_metrics()` (CR-4)
- `scripts/foundation/fedrec_foundation/rng.py` — `server_rng()`, `py_rng()`, `np_rng()`, `torch_gen()` factories (FND-06, CR-3)
- `scripts/foundation/fedrec_foundation/manifest.py` — `RunManifest` (24 fields), `build_run_manifest()`, `write_manifest_sibling()`, `embed_manifest_in_result()` (FND-07, D-15, D-16)
- `scripts/foundation/fedrec_foundation/mode.py` — `ModeProfile`, `resolve_mode_defaults()`, `log_mode_and_overrides()`, `assert_benchmark_one_user_per_client()` (D-06..D-11)
- `scripts/run.py` — Federation-level launcher (CR-2) that sets `num-supernodes` at Flower federation init time
- `scripts/foundation/fedrec_foundation/user_groups.py` — `classify_user_group()`, user-group buckets (sparse ≤ 30, 30 < medium ≤ 100, dense > 100)

### Phase 1 artifacts (on disk, consumed read-only)
- `data/derived/mapping.json` — canonical `user2idx` / `item2idx` (6040 users, 3706 items)
- `data/derived/split_manifest.json` — deterministic LOO split, `split_hash=5685bed7e4b650807e58a49e25ea611cdef82444a034bdf105e46aa3755284d6`
- `data/derived/exclusion_items.npz` — per-user `train_pos ∪ test_pos` (flat items+indptr)
- `data/derived/foundation_index.json` — atomic bundle sentinel, `foundation_contract_sha256=fe181dafe6f791d6679b08fdf6a4ad88f17ea3a7cce98df7f491d1c96a14233a`

### Phase 1 discussion and plans (context on locked decisions)
- `.planning/phases/01-foundation-contract/01-CONTEXT.md` — D-01..D-16 and rationale
- `.planning/phases/01-foundation-contract/01-foundation-contract-01-SUMMARY.md` — scaffold + Wave-0 TDD harness
- `.planning/phases/01-foundation-contract/01-foundation-contract-02-SUMMARY.md` — FND-01/02/03 delivery
- `.planning/phases/01-foundation-contract/01-foundation-contract-03-SUMMARY.md` — FND-04/05 + CR-4 delivery
- `.planning/phases/01-foundation-contract/01-foundation-contract-04-SUMMARY.md` — FND-06/07 delivery (RNG + manifest)
- `.planning/phases/01-foundation-contract/01-foundation-contract-05-SUMMARY.md` — mode resolver + CR-2 launcher
- `.planning/phases/01-foundation-contract/01-foundation-contract-06-SUMMARY.md` — integration wiring across 4 modules
- `.planning/phases/01-foundation-contract/01-foundation-contract-VERIFICATION.md` — 4/4 must-haves verified

### Project-level
- `.planning/PROJECT.md` — Core value, out-of-scope list, key decisions (no `fedrec_common/` this cycle, cross-silo preserved, per-group reporting first-class)
- `.planning/REQUIREMENTS.md` §BSL — BSL-01..08 (the requirements this phase must satisfy)
- `.planning/ROADMAP.md` §Phase 2 — Success criteria (4 observable criteria phrased in terms of artifact existence + determinism)
- `CLAUDE.md` — Project-wide conventions (notation: `w` global, `theta_i` local; code standards: type hints, NumPy docstrings, dataclasses for config, seed+config reproducibility)
- `federated-baseline-cf/claude.md` — Module-specific architecture notes (all params global, BPR-MF canonical, ~874K params transmitted per round)

### Codebase map (brownfield context)
- `.planning/codebase/ARCHITECTURE.md` §"Personalization Boundary Matrix" — baseline = all global invariant we must preserve (D-23)
- `.planning/codebase/CONCERNS.md` — bugs to NOT re-introduce: global `random.seed()` in evaluators, unseeded `random.sample(node_ids, …)`, shared unscoped `.embedding_cache/`
- `.planning/codebase/STACK.md` — Flower ≥ 1.22, PyTorch ≥ 2.7, ML-1M schema
- `.planning/codebase/CONVENTIONS.md` — notation, type hint style (`typing.Dict`, `typing.List` — pre-3.10), docstring conventions

### Research outputs (from Phase 1 but still relevant)
- `.planning/research/PITFALLS.md` §1, §2, §12, §13, §14 — num-supernodes survival, eval-protocol drift, unseeded sampling, client RNG collisions, evaluator reseeding
- `.planning/research/FEATURES.md` §P0, §P1 — table-stakes features for a correct cross-device FedRec protocol

### Existing baseline code (the migration target)
- `federated-baseline-cf/federated_baseline_cf/dataset.py` (579 LOC) — `create_global_mappings`, `create_leave_one_out_split`, Dirichlet/natural partitioning; replaced by foundation loaders per D-17
- `federated-baseline-cf/federated_baseline_cf/client_app.py` (198 LOC) — `@app.train`, `@app.evaluate`; gets one-user assertion (BSL-02), exclusion-set-aware training (BSL-03), contract-compliant metrics payload (D-21)
- `federated-baseline-cf/federated_baseline_cf/server_app.py` (587 LOC) — main FL loop; gets seeded client sampling (BSL-04, line 297 `random.sample(node_ids, …)`), sufficient-stat aggregation via `BaselineFedAvg` (D-20), best-round checkpoint (D-27), manifest emit (D-15, BSL-08)
- `federated-baseline-cf/federated_baseline_cf/task.py` (883 LOC) — `train_bpr_mf`, `train_basic_mf`, `evaluate_ranking_sampled`; line 758 `random.seed(seed)` stripped (BSL-05), line 819 `random.sample(negative_candidates, …)` replaced with foundation RNG instance
- `federated-baseline-cf/federated_baseline_cf/models/` — BPR-MF / Basic-MF; get gradient-mask hook (D-24) but no architecture change
- `federated-baseline-cf/pyproject.toml` — `fedrec-foundation` already wired as local-path dep (Phase 1 Plan 06); gets the mode-selector line + potentially an `extras` block for dev tests

### External specs (for thesis comparability)
- NCF WWW-17 paper (referenced via `.planning/research/SUMMARY.md`) — `sampled_loo_99` protocol definition (1 positive + 99 negatives per held-out test item), the convention D-27 best-round is compared against.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`BPRMF` and `BasicMF` classes** (`federated-baseline-cf/federated_baseline_cf/models/`) — architecture is untouched; just get a gradient-mask hook per D-24
- **`evaluate_ranking_sampled()` in `task.py`** — already implements `sampled_loo_99`; gets parameters strip (remove `random.seed(seed)`, accept `rng: np.random.Generator`)
- **Existing `@app.train()` / `@app.evaluate()` handlers in `client_app.py`** — structure remains; payloads and assertions change per D-21, D-22, BSL-02
- **`FedAvg` / `FedProx` strategies (imported from `flwr.server.strategy`)** — `BaselineFedAvg` subclasses these, adds sufficient-stat aggregation per D-20
- **`_partition_cache` module-level dict in `dataset.py`** — obsolete once foundation loaders own mapping/split/exclusion; remove per D-17

### Established Patterns (extend, do not break)
- **Flower `Message`-passing in server_app** — `grid.send_and_receive(messages)` loop continues as-is; `BaselineFedAvg` plugs into the response aggregation step, not the message-sending step
- **W&B logging from `server_app.py`** — `wandb.init()` still called once at start; adds a `wandb.config.update({"_manifest": run_manifest.to_dict()})` call immediately after init (D-26, D-15)
- **Result JSON shape** — `{model_name, dataset, federated_config, early_stopping, timestamp, final_metrics, training_rounds}` grows a top-level `_manifest` (D-15) and `selected_clients_per_round` (D-26) field
- **Module-level `_device_cache` and `_dataset_cache` in `task.py`** — kept; same pattern, just the dataset cache keys become `foundation_contract_sha256` instead of ad-hoc string
- **Atomic write pattern** — tempfile + `os.replace` — extended from embedding cache (current) to manifest writes (new, per D-15)

### Integration Points
- `client_app.py::@app.train()`: call `assert_benchmark_one_user_per_client(profile, num_users_in_partition, overrides)` at handler entry (BSL-02); call `np_rng(seed, user_id, round, "train_neg")` for negative sampling (BSL-03 + BSL-05); return `FitMetricsContract.to_dict()` (D-21)
- `client_app.py::@app.evaluate()`: call `np_rng(seed, user_id, round, "eval_neg")` for 99-negative sampling (BSL-05); return overall + per-group sufficient stats (D-22)
- `server_app.py::@app.main()`: call `log_mode_and_overrides(mode, profile, context.run_config)` at startup (Phase 1 D-10/D-19); instantiate `BaselineFedAvg` strategy (D-20); call `server_rng(seed)` for client selection replacing line 297 `random.sample(node_ids, ...)` (BSL-04); call `build_run_manifest(...)` once, embed in result JSON + write sibling file (BSL-08, D-15)
- `strategy.py` (new file): define `BaselineFedAvg(FedAvg)` with overridden `aggregate_evaluate()` for D-20; `BaselineFedProx(FedProx)` sibling for mode=fedprox path
- `task.py::train_bpr_mf()` and `train_basic_mf()`: add gradient-masking step after `loss.backward()` (D-24); pass RNG through negative-sampling call sites
- `task.py::evaluate_ranking_sampled()`: drop `random.seed(seed)` (line 758); accept `rng: np.random.Generator` parameter threaded through the 99-negative sampling loop

### Known Anti-Patterns to Retire
- Line 297 in `server_app.py`: `selected_node_ids = random.sample(node_ids, num_selected)` — replace with `server_rng(run_seed)(round).sample(node_ids, num_selected)` (or equivalent foundation API)
- Line 758 in `task.py`: `random.seed(seed)` — strip entirely; inject RNG instance
- Line 819 in `task.py`: `negative_items = random.sample(negative_candidates, num_negatives)` — replace with `rng.choice(negative_candidates, size=num_negatives, replace=False)` using the `np_rng` instance from the client
- Unscoped `.embedding_cache/` at project root — baseline has no user cache (all params global), so this doesn't apply here; foundation's `run_id`-scoped cache path is a Phase 3/4 concern

</code_context>

<specifics>
## Specific Ideas

- User picked **all four recommended defaults** in every sub-question. Interpretation: strong preference for the conservative, contract-preserving path — minimize deviation from Phase 1 patterns, don't introduce new architectural invariants just to migrate one module. Planner should default to "same approach for Phases 3, 4, 5 as for Phase 2" unless a module's personalization boundary makes that impossible.
- User picked **rip-and-replace for dataset.py helpers (D-17)** AND **surgical migration for pre-existing edits (D-18)**. Interpretation: replace OUR helpers but DON'T touch the user's WIP. Planner should call out which functions/lines are Phase-2 territory vs. WIP in each plan's task list.
- User picked **extend FitMetricsContract with per-group keys (D-22)** instead of free-form extras. Interpretation: the contract is the source of truth for everything that crosses the client/server boundary. If it's in metrics, it's in the contract. Planner should add per-group keys to the contract in Phase 2 Plan 01 (not defer to Phase 6).
- User picked **best_round_restore as default (D-27)**. Interpretation: this is a **defensibility choice**, not a performance choice — the thesis reproduction target (PFedRec IJCAI-23 paper) reports best-round numbers, and apples-to-apples comparison requires the baseline to report best-round too. Planner should propagate this default to Phases 3, 4, 5 unless a module has an explicit reason to differ.
- User picked **flat `results/federated/` directory (D-28)**. Interpretation: results layout does NOT change this cycle. All cross-device runs and all historical cross-silo runs remain colocated. The cross-device vs cross-silo split is a manifest filter, not a directory split.

</specifics>

<deferred>
## Deferred Ideas

- **Full `fedrec_common/` refactor** — Out of scope per PROJECT.md. The thin-adapter pattern from D-17 makes it easier to extract shared code in v2, but Phase 2 does not pursue it.
- **Per-group evaluator dashboards in W&B** — The per-group sufficient stats (D-22) enable live per-group NDCG/HR plots. Wiring the W&B dashboards is a Phase 6 concern; Phase 2 only produces the numbers.
- **Best-round checkpoint crash-resilience** — D-27 best-round is in-memory-only by default; disk-backed checkpointing is a future concern (Claude's Discretion during planning), not Phase 2.
- **`cross_silo_legacy` mode regression tests** — D-19 preserves the legacy path; formally testing that it still produces the same numbers as pre-migration is a nice-to-have but not a must-have for this phase. If the user wants it, it can become a dedicated Phase-2 plan.
- **Sparse-update `SparseFedAvg` aggregator** — Discussed as the "best comm-savings without breaking baseline contract" option but rejected in favor of the simpler full-matrix approach. Revisit post-thesis if comm cost becomes a paper-worthy result.
- **Single-user fast-path model** — Discussed as an alternative to full-matrix + gradient-mask but rejected (violates "baseline = all global" invariant). Could be a Phase 3/4 topic if the split-learning modules discover they need tighter memory bounds.
- **Unified test runner across all four modules** — Phase 1 introduced pytest inside `scripts/foundation/tests/`; Phase 2 tests may live inside `federated-baseline-cf/tests/`. A meta-runner at project root (`pytest scripts/foundation/tests/ federated-*-cf/tests/`) is a post-Phase-2 nice-to-have.

</deferred>

---

*Phase: 02-baseline-migration*
*Context gathered: 2026-04-19*
