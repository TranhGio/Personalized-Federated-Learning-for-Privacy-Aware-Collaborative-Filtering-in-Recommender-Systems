---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Phase 3 context gathered
last_updated: "2026-04-19T15:04:50.070Z"
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 11
  completed_plans: 11
---

# STATE: Federated Movie Recommendation — Cross-Device Migration & Thesis Evaluation

**Last updated:** 2026-04-19 after roadmap creation

## Project Reference

**Core value:** Under a correct cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method beats all three baselines on NDCG@10 — including on sparse users — while the Flower PFedRec reproduces the IJCAI-23 reference within ±2 points.

**Current focus:** Phase 02 — baseline-migration

**Branch:** `feat/try_to_run_the_baseline` (existing; thesis work continues on this branch until milestone boundary is reached).

## Current Position

Phase: 3
Plan: Not started

## Performance Metrics

Populated as phases complete. Primary thesis metric: `sampled_ndcg@10` (leave-one-out + 99 negatives).

| Module | Protocol | NDCG@10 | HR@10 | Sparse NDCG@10 | Notes |
|--------|----------|---------|-------|-----------------|-------|
| baseline | — | — | — | — | cross-device run pending |
| personalized | — | — | — | — | cross-device run pending |
| adaptive | — | — | — | — | cross-device run pending |
| pfedrec (paper_compat) | — | — | — | — | target: HR@10 ≈ 0.729 ± 2pts, NDCG@10 ≈ 0.441 ± 2pts |
| Phase 01-foundation-contract P01 | 5min | 2 tasks | 19 files |
| Phase 01-foundation-contract P02 | 8min | 3 tasks | 15 files |
| Phase 01-foundation-contract P03 | 3min | 2 tasks | 5 files |
| Phase 01-foundation-contract P04 | 4min | 2 tasks | 4 files |
| Phase 01-foundation-contract P06 | 6min | 2 tasks | 6 files |
| Phase 02-baseline-migration P02 | 5min | 2 tasks | 3 files |
| Phase 02-baseline-migration P01 | 6min | 2 tasks | 7 files |
| Phase 02-baseline-migration P04 | 7min | 2 tasks | 3 files |
| Phase 02-baseline-migration P03 | 11min | 2 tasks | 4 files |
| Phase 02-baseline-migration P05 | 11min | 5 tasks | 6 files |

## Accumulated Context

### Decisions

- **Migrate to cross-device** (1 user = 1 client, N=6040) for all four modules; cross-silo kept as explicit opt-in only.
- **Do NOT extract `fedrec_common/`** during this cycle — refactor risks invalidating the codebase map and bug audit mid-experiment.
- **Re-audit PFedRec bugs from scratch** against `IJCAI-23-PFedRec/` (don't trust the prior note list at face value — validate each against the reference).
- **Per-round sampling fraction `C`** is a swept hyperparameter, not fixed; defaults per module match the paper each module calibrates against.
- **New W&B project** for cross-device runs — keep cross-silo dashboards untouched.
- **Per-user-group (sparse/medium/dense) metrics** are first-class reported fields, not an afterthought.
- **Centralized baselines (SVD, NCF)** remain as-is — not re-evaluated under LOO+99neg.
- **DP / privacy quantification** deferred to v2.
- **Primary evaluator:** `sampled_loo_99` (NCF protocol). `allrank_*` is a namespaced secondary, never mixed into thesis tables.
- **Canonical reported metric:** `best_*` (best-round restored), not `last_*`.
- [Phase 01-foundation-contract]: Foundation package lives at scripts/foundation/ (not inside any federated-*-cf/ module and not at repo root) — neutral shared location avoids namespace collision and duplication while respecting PROJECT.md decision to defer fedrec_common/ extraction.
- [Phase 01-foundation-contract]: Plan 01 uses skip-stub TDD handoff: downstream plans un-skip by deleting pytestmark and replacing NotImplementedError bodies — enumerates all 31 expected FND-01..07 tests at pytest --collect-only while keeping every run green (2 passed, 31 skipped, 0 failed).
- [Phase 01-foundation-contract]: compute_raw_data_hash concatenation order is LOCKED to ratings.dat || movies.dat || users.dat — every FND-02/FND-07 downstream fingerprint depends on it; any future change invalidates committed split manifests and run manifests.
- [Phase 01-foundation-contract]: Plan 03 closed-enum whitelist on get_primary_evaluator: typos in run config's mode string must fail loud (ValueError 'Unknown mode'); matches CONVENTIONS.md factory-function-on-unknown-enum rule used by create_alpha_computer and get_model.
- [Phase 01-foundation-contract]: Plan 03 FitMetricsContract.from_dict wraps dataclass TypeError as ValueError('...missing required field: ...') — Codex CR-4 polish; callers now get a clear error surface with field names instead of cryptic 'FitMetricsContract.__init__() missing N required positional arguments' TypeError.
- [Phase 01-foundation-contract]: Plan 03 FND-04 + FND-05 + CR-4 ship as 3 foundation modules (evaluator.py, weight_policy.py, fit_metrics.py) with 15 GREEN tests. Cross-phase contract: every Phase 2-5 client_app.py @app.train() handler MUST build its return dict via FitMetricsContract.to_dict() so server-side compute_aggregation_weight has the inputs it needs.
- [Phase 01-foundation-contract]: Plan 04 FND-06 uses hashlib.sha256 with FULL 256-bit digest (not truncated — Codex N-1) and namespace prefixes py/np/torch inside the payload, so py_rng/np_rng/torch_gen produce INDEPENDENT streams for identical (run_seed, user_idx, round_num, purpose) tuples while remaining cross-process stable under PYTHONHASHSEED=0/1/random (CR-3 anchor test).
- [Phase 01-foundation-contract]: Plan 04 cross-phase contract for Phases 2-5: every DataLoader(..., shuffle=True) MUST pass `generator=torch_gen(run_seed, user_idx, round_num, 'dataloader')` — CR-3's fourth reproducibility assertion. Without this, DataLoader worker shuffling is non-deterministic even with all three RNG factories seeded. `dataloader` is pre-declared in `_ALLOWED_PURPOSES` for this reason.
- [Phase 01-foundation-contract]: Plan 04 FND-07 RunManifest carries all four IMP-2 fingerprints (mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256) so a single-byte mutation to any foundation input is detectable at the run-manifest level; D-15 double-write (embedded _manifest key in result JSON + sibling <run_id>-manifest.json) guarantees at least one artifact survives partial failure.
- [Phase 01-foundation-contract]: Plan 04 uses duck-typed `mode_profile: Any` in build_run_manifest — avoids circular import with Plan 05's mode.py while documenting the required attribute surface in the docstring; test_manifest.py's _StubProfile demonstrates the minimal implementation.
- [Phase 01-foundation-contract]: Plan 02 SplitManifest stores BOTH fingerprints (raw_data_hash + mapping_sha256) as top-level dataclass fields per IMP-2 — consumers (publish_bundle, RunManifest) read directly from the manifest, no side-channel, no post-hoc assignment. `publish_bundle` is LOCKED at 4-param (derived_dir, mapping, split_manifest, exclusion); `build_split` is LOCKED at 5-param with explicit mapping_sha256 + raw_data_hash args. Changing either signature requires a replan.
- [Phase 01-foundation-contract]: Plan 02 CR-5 train-only user stats: PerUserStats (n_interactions, genre_entropy, n_unique_items, rating_std, user_group) is computed on train rows AFTER removing the LOO test item — prevents Phase 4's adaptive-alpha heuristic from seeing the test item's genre and underestimating its own improvement on sparse users.
- [Phase 01-foundation-contract]: Plan 02 IMP-3 flat NPZ layout (items + indptr, CSR-style) for data/derived/exclusion_items.npz — 1.5x smaller than keyed-dict at 6040 users, O(1) per-user slice, single np.load call, atomic tempfile + os.replace write that handles np.savez's .npz suffix-append behavior.
- [Phase 01-foundation-contract]: Plan 02 N-3 atomic bundle: foundation_index.json written LAST by publish_bundle; verify_bundle() re-computes mapping_sha256/exclusion_sha256/foundation_contract_sha256 on every load and raises RuntimeError with "incomplete" sentinel on missing payload or sha mismatch — readers never see a partially-published bundle.
- [Phase 01-foundation-contract]: Plan 02 empirical CR-1 anchor CONFIRMED on real data/ml-1m/: build_mapping produces 6040 users and 3706 items (NOT 3883 from movies.dat). test_ml1m_counts_6040_3706 in test_integration.py pins this as a regression test.
- [Phase 01-foundation-contract]: Plan 02 D-04 lock-forever COMMITTED: data/derived/mapping.json + split_manifest.json + exclusion_items.npz + foundation_index.json are committed artifacts (split_hash 5685bed7e4b6, foundation_contract_sha256 fe181dafe6f7, builder_version 1.0.0). save_split_or_verify refuses to overwrite on divergent hash with the sentinel "invalidate all cached results"; the commit IS the lock.
- [Phase 01-foundation-contract]: Plan 06 PEP 440 plain-name dep (not direct reference): fedrec-foundation appears as a bare dep name in each federated-*-cf/pyproject.toml without a URL. Hatchling does not resolve local file URIs at build time; direct-reference breaks editable-install semantics. Plain-name costs a load-bearing install-order step (foundation first) but works with every module's existing pip install -e . flow and matches docs/setup.md.
- [Phase 01-foundation-contract]: Plan 06 closes Phase 1: after wiring foundation as a local-path dep into all 4 modules' pyproject.toml, every federated module can now 'from fedrec_foundation.X import ...' freely — no sys.path hacks, no fallback imports, no conditional guards. Phases 2-5 inherit this contract via editable install. The one-time user-setup rule is 'pip install -e scripts/foundation/' BEFORE 'pip install -e federated-*-cf/'; that order is enforced by docs/setup.md and by the comment inside each pyproject.toml.
- [Phase 01-foundation-contract]: Plan 06 test design — subprocess-based cross-module import smoke test: 'test_cross_module_imports' parametrized across 4 modules uses subprocess.run(cwd=<module_dir>) to mirror 'flwr run .' execution semantics rather than in-process importlib. 'test_pyproject_declares_foundation_dep' is a cheap textual regression guard. Both land in scripts/foundation/tests/test_integration.py; full foundation suite goes from 65 to 70 GREEN tests.
- [Phase 02-baseline-migration]: [Phase 02-baseline-migration Plan 02]: BSL-01 closed fully in-file — federated-baseline-cf/pyproject.toml declares partition-mode=natural + num-supernodes=6040 in BOTH local-simulation and local-sim-gpu federation blocks; flwr run . now resolves cross-device without relying on scripts/run.py launcher. Pre-existing partition-mode='natural' WIP from prior work preserved via surgical edit discipline.
- [Phase 02-baseline-migration]: [Phase 02-baseline-migration Plan 02]: pytest dev dep exclusively owned by Plan 02 Task 1 — [project.optional-dependencies] dev = ['pytest>=7.0'] declared in baseline/pyproject.toml to eliminate iter-1 BLOCKER 1 Wave-1 write-race with Plan 01. Test plans 01/02/03/04 install pytest via pip install -e '.[dev]'.
- [Phase 02-baseline-migration]: [Phase 02-baseline-migration Plan 02]: D-17 rip-and-replace completed in federated-baseline-cf/dataset.py — 5 module-local helpers removed (create_global_mappings, create_leave_one_out_split, compute_user_genre_distribution, dirichlet_partition_users, create_train_test_split) + _partition_cache. load_partition_data and load_full_data keep signatures but delegate mapping/split/exclusion to fedrec_foundation. partition_mode='dirichlet' now raises NotImplementedError; split_mode='random' raises ValueError. Cross-phase contract: dataset.py is a thin (~440 LOC) foundation adapter; Plans 03/04/05 follow this shape.
- [Phase 02-baseline-migration]: [Phase 02-baseline-migration Plan 02]: D-18 surgical-edit discipline enforced — pre-existing uncommitted WIP in MovieLensDataset, download_movielens_1m, load_movielens_1m, natural_partition_users preserved verbatim in dataset.py. Pre-existing WIP in client_app.py, server_app.py, task.py UNTOUCHED (Plans 03 and 04 own those). git diff --stat verified delta shape consistent with D-17 scope only. Duplicate eval-num-negatives declaration auto-fixed to avoid TOML duplicate-key error.
- [Phase 02-baseline-migration]: Plan 01 unified _at10 suffix lock-in: FitMetricsContract fields + EvaluateMetricsContract fields + strategy _sum_sufficient_stats reader + client wire payload all share the SAME _at10 (no _at_10 drift) — iteration 2 fix. Prior iteration's drift would have silently zeroed all evaluate-side aggregation because wire-side dict keys wouldn't match the server reader's dict keys (tests bypassed via strategy-shape-dict construction).
- [Phase 02-baseline-migration]: Plan 01 EvaluateMetricsContract sibling (not subclass of FitMetricsContract): evaluate wire carries OPTIONAL diagnostics (eval_loss / sampled_hr_at10 / sampled_ndcg_at10) that are NOT required on fit side. Subclassing would over-constrain evaluate or let free-form extras slip through validate_fit_metrics. validate_evaluate_metrics enforces both required-keys AND no-free-form-extras (D-21). Diagnostics are cached client-side for logs only — server aggregator IGNORES them and re-computes headlines from summed sufficient stats.
- [Phase 02-baseline-migration]: Plan 01 aggregate_fit INHERITED UNCHANGED: BaselineFedAvg.aggregate_fit is FedAvg.aggregate_fit (identity check in test_aggregate_fit_inherited_unchanged). D-23 preserved — baseline = all params global. BaselineFedProx.aggregate_evaluate is an EXACT COPY of BaselineFedAvg's (not super() call) to avoid diamond-inheritance MRO; 4-line duplication + shared module-level _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics helpers keep logic DRY.
- [Phase 02-baseline-migration]: Plan 01 pyproject.toml UNTOUCHED + D-18 surgical migration preserved: Wave-1 write race avoided by exclusive file ownership (Plan 01 owns strategy.py + tests + fit_metrics.py + foundation tests; Plan 02 owns pyproject.toml + dataset.py rip-and-replace). Pre-existing uncommitted hunks in federated_baseline_cf/{client_app,dataset,server_app,task}.py left untouched for Plan 03 Task 2 to consume during client-side sufficient-stat population.
- [Phase 02-baseline-migration]: Plan 04 server_app.py migration: mode resolver owns canonical hyperparams (D-25) — every hyperparameter read is int(context.run_config.get(key, profile.field)) so pyproject values are only the override surface; seeded client sampling uses a SINGLE _server_sampler = server_rng(run_seed) instance instantiated before the FL loop (deterministic sequence across rounds); BaselineFedAvg/BaselineFedProx replaces raw FedAvg/FedProx and thesis metrics flow from strategy.aggregate_evaluate (sum-based sufficient stats) while RMSE/MAE preserved via legacy weighted_average_metrics fallback on D-18 scope-out; D-27 in-memory best-round restore snapshots ArrayRecord on current_ndcg > best_metric and assigns arrays = best_arrays before centralized eval; D-15 double-write via embed_manifest_in_result + write_manifest_sibling; default W&B project federated-cf-cross-device for cross-device modes per PROJECT.md; checkpoint_rule branch accepts both 'best_round_restore' (pyproject) and 'best_round' (ModeProfile) spellings to avoid bikeshed.
- [Phase 02-baseline-migration]: Plan 03 D-24 gradient isolation: gradient-only mask INSUFFICIENT under Adam weight-decay + momentum (RED-step regression caught row 1 moving by 0.3965 L2 norm). Fix = bracket optimizer.step() with _snapshot_non_user_rows / _restore_non_user_rows (snapshot marks user-idx row NaN so restore never overwrites legitimate update). Optimizer-agnostic; works for Adam + SGD. 3 new task.py module-level private helpers — near-duplicate will land in Plans 3/4/5 sibling modules.
- [Phase 02-baseline-migration]: Plan 03 BSL-05 _sample_negatives_seeded chosen over patching models/bpr_mf.py: BPRMF.sample_negatives uses process-global np.random.randint; extending its signature would have touched models/ (outside D-18 surgical scope) and created asymmetry vs personalized/adaptive modules' own sample_negatives. Inline helper is distribution-equivalent (rejection-uniform) from an np.random.Generator; confines determinism fix to task.py.
- [Phase 02-baseline-migration]: Plan 03 evaluate_ranking_sampled legacy seed param IGNORED: signature backward-compatible (seed:int=42 still accepted) but docstring explicitly documents it ignored. Seeds derive from (run_seed, user_idx, round_num, 'eval_neg') per BSL-05. Any pre-Phase-2 caller gets new deterministic behavior without code change; silent semantic break is intentional.
- [Phase 02-baseline-migration]: Plan 05 G-03-01 closure: selected_clients_per_round now stores stable partition_ids (0..N-1), not Flower ephemeral node_ids — server runs a one-shot discovery round broadcasting discover_only=true to build partition_to_node_id before the main loop; _server_sampler.sample(range(N), k) samples in partition-id space. Optional partition_id field added to FitMetricsContract + EvaluateMetricsContract (auto-whitelisted via dataclass fields()).
- [Phase 02-baseline-migration]: Plan 05 regression guard: test_selected_partitions_byte_identical_across_subprocess_reruns runs scripts/run.py twice and asserts byte-identity of selected_clients_per_round JSON field — catches deterministic-RNG-over-non-deterministic-domain regressions the Plan-04 pure-RNG test could not. @pytest.mark.slow + FEDREC_SKIP_SLOW=1 escape hatch.

### Todos

*(Rolling list of actionable items surfaced during work, carried across sessions.)*

- None yet — phase planning will populate this as concrete tasks materialize.

### Blockers

- None.

### Open Questions

- What exact value of `fraction-train` should each module default to after migration? (To be decided during Phase 2–5 planning; treated as a swept hyperparameter.)
- Should the canonical ID-mapping / split-manifest artifact live under `.planning/artifacts/` or under a new `data/artifacts/` path? (Decided during Phase 1 planning.)

## Session Continuity

**Last session summary (2026-04-19):** Phase 01 (foundation-contract) CLOSED. All 6 plans shipped: fedrec-foundation package + 10 submodules (paths, atomic, hashing, mapping, split, exclusion, bundle, build, user_groups, evaluator, weight_policy, fit_metrics, rng, manifest, mode) + scripts/run.py launcher. All 70 foundation tests GREEN. data/derived/ bundle committed (split_hash=5685bed7e4b6, foundation_contract_sha256=fe181dafe6f7). Plan 06 wired fedrec-foundation as plain-name local-path dep into all 4 federated-*-cf/pyproject.toml files + cross-module subprocess smoke test (test_cross_module_imports parametrized x4 + test_pyproject_declares_foundation_dep).

**Next session entry point:** Run the Phase 1 verifier / transition gate, then `/gsd:plan-phase 2` to decompose the baseline-migration phase (BSL-01..08) into plans. Phases 2-5 are parallelizable after Phase 1.

**Key files to reread on session resume:**

- `.planning/ROADMAP.md` — phase structure and success criteria (Phase 1 progress: 6/6 Complete)
- `.planning/REQUIREMENTS.md` — traceability table (FND-01..07 complete)
- `.planning/phases/01-foundation-contract/01-foundation-contract-06-SUMMARY.md` — Phase 1 closure summary + Phases 2-5 integration contract
- `docs/setup.md` — install order (pip install -e scripts/foundation/ BEFORE any module)
- `scripts/run.py` — CR-2 launcher for cross-device runs
- `.planning/research/ARCHITECTURE.md` — migration deltas and build-order implications
- `.planning/codebase/CONCERNS.md` — known bugs to re-verify during migration

**Stopped at:** Phase 3 context gathered

---
*State initialized: 2026-04-19 alongside roadmap creation.*
*Phase 01 complete: 2026-04-19 after Plan 06 landed.*
