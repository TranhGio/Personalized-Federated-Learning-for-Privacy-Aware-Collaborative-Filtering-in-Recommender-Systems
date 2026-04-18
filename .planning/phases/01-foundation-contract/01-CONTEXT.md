# Phase 1: Foundation Contract — Context

**Gathered:** 2026-04-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the shared cross-device protocol contract that all four downstream migrations will import. In scope:
- Canonical `raw_user_id → user_idx` and `raw_item_id → item_idx` mapping artifact.
- Deterministic leave-one-out split manifest with `split_hash`.
- Per-user training-negative exclusion set (`train_pos ∪ test_pos`).
- Primary evaluator selector (fixes `sampled_loo_99` as the thesis-table protocol).
- Aggregation `weight-policy` config abstraction.
- Run-scoped seeding discipline (four-tier RNG derivation).
- Run manifest / protocol fingerprint attached to every result.

Out of scope for this phase:
- Modifying any of the four module packages (that's Phases 2–5).
- Extracting code into `fedrec_common/` (PROJECT.md decision: deferred to v2).
- Adding evaluator implementations beyond the selector plumbing (evaluator bodies already exist in each module's `task.py`).
- Running experiments or producing thesis results (Phases 2–7).

</domain>

<decisions>
## Implementation Decisions

### Artifact storage & format
- **D-01:** Foundation artifacts live in a committed `data/derived/` folder at the project root. Ground-truth, bit-exact reproducibility. ~1–5 MB footprint is acceptable for a thesis repo.
- **D-02:** Format is a **mix** — `mapping.json` (tiny, human-readable), `split_manifest.json` (tiny, human-readable), `exclusion_items.npz` (binary int32 arrays per user, fast load for the hot path).
- **D-03:** `split_manifest.json` carries ALL of: `split_hash` (mandatory), builder metadata (builder version, creation timestamp, raw-data hash of `ratings.dat`), per-user stats (`n_interactions`, `genre_entropy`, `n_unique_items`, `rating_std` — precomputed once here so Phase 4 doesn't recompute), and user-group classification (sparse/medium/dense bucket per user — Phase 6 reporting reads it directly).
- **D-04:** Canonical split is **locked forever** after first generation. Re-running the builder recomputes the hash; if it diverges from the committed manifest, the builder errors with "a new split would invalidate all cached results" and refuses to overwrite. Immutable by policy.
- **D-05:** NPZ loads use `numpy.load(..., allow_pickle=False)` and JSON loads are plain `json.load`. No pickle anywhere in the foundation layer. (Follow-through on `.planning/codebase/CONCERNS.md` — `weights_only=False` risk also applies to pickled metadata.)

### Mode-selector interface
- **D-06:** Each `pyproject.toml` exposes a **single top-level `mode` selector** under `[tool.flwr.app.config]`: `mode = "benchmark_cross_device"` / `"paper_compat_pfedrec"` / `"cross_silo_legacy"` (PFedRec-compat mode only defined where relevant; baseline / personalized / adaptive expose `benchmark_cross_device` and `cross_silo_legacy` only).
- **D-07:** The mode value fully locks the downstream experiment: `num-supernodes`, `partition-mode`, `weight-policy`, `eval-protocol` (primary evaluator), AND the training hyperparameters (`embedding-dim`, optimizer, lr, local epochs, training negatives, num-server-rounds). A mode IS a complete experiment profile, not just a cross-silo/cross-device toggle.
- **D-08:** The mode-to-defaults mapping lives in a shared Python module (not scattered in pyproject.toml). Each client/server reads `mode` from the Flower config, then calls a `resolve_mode_defaults(mode: str) -> dict` helper that returns the locked values. Per-module overrides allowed where a module's paper-compat setting legitimately differs (e.g., PFedRec's `weight-policy`).
- **D-09:** Cross-silo (`num-supernodes=5`) is **kept reachable** as `mode = "cross_silo_legacy"`, not deleted. PROJECT.md constraint: "We override defaults, we do not delete the code paths." Historical W&B runs remain reproducible; new runs never hit cross-silo by accident because the default is always `benchmark_cross_device`.
- **D-10:** Mode-locked settings **can be overridden** at the CLI (`flwr run . --run-config "num-supernodes=3"`) for debugging / ad-hoc experiments, but every override is captured in the run manifest's `overrides` field AND prints a loud warning at run start. Overrides are visible; drift is never silent.
- **D-11:** Benchmark-mode startup assertion: when `mode = "benchmark_cross_device"` and no overrides are in play, the client asserts `num_users_in_client == 1` and fails loudly otherwise. Assertion is skipped for `cross_silo_legacy` and relaxed for `paper_compat_pfedrec` (which also expects single-user clients, so the assertion actually stays on there too — overrides aside).

### Canonical evaluator and weight-policy (carried forward from research / prior decisions)
- **D-12 (locked):** Primary evaluator is `sampled_loo_99` (leave-one-out + 99 negatives, NCF protocol). `allrank_*` is kept as a namespaced secondary and explicitly excluded from the thesis comparison table.
- **D-13 (locked):** Per-user exclusion set equals `train_pos ∪ test_pos`; there is no val set at the foundation layer (validation-split strategy is a Phase 5/6 decision if adopted at all).
- **D-14 (locked):** Four-tier RNG derivation: `run_seed` → `server_rng = Random(run_seed)` → `per_user_rng(user_id, round, purpose) = Random(hash((run_seed, user_id, round, purpose)))`. Purposes include `train_neg`, `eval_neg`, `model_init`. No module-level or evaluator-level `random.seed(...)` / `np.random.seed(...)` calls are permitted.

### Run manifest / protocol fingerprint
- **D-15:** The run manifest is **embedded in every result JSON** under a top-level `_manifest` key AND written as a sibling `<run_id>-manifest.json` next to the result file. Belt-and-suspenders; both paths are cheap.
- **D-16:** Manifest fields (minimum): `mode`, `num-supernodes`, `partition-mode`, `fraction-train`, `fraction-eval`, `weight-policy`, `primary-evaluator`, `num-train-negatives`, `num-eval-negatives` (99), `run-seed`, `checkpoint-rule` (last vs best-round-restore), `split_hash` (from D-03), `raw-data-hash`, `builder-version`, `overrides` (dict of any CLI overrides applied), `module` (which of the four FL modules), `flwr-version`, `torch-version`, `git-commit`.

### Claude's Discretion
The following were NOT explicitly discussed; Claude may decide at planning time within reasonable principles:

- **Shared code placement** — Given PROJECT.md rules out `fedrec_common/` this cycle, default recommendation (planner may override with cause): place the foundation module as a standalone `scripts/foundation/` package at the project root with a clear `__init__.py` that each of the four FL modules imports via a relative import (`sys.path.insert(0, "..")` pattern in each `dataset.py`). Alternative acceptable: duplicate the foundation source into each module's package (mirrors the current duplication pattern; four-file sync burden). Planner picks the cleaner of the two; document the choice in PLAN.md.
- **Weight-policy defaults per module** — Not explicitly discussed. Recommendation: baseline / personalized / adaptive default to `num_positives` (research recommendation; aligns with the BPR pairwise-training convention). PFedRec's paper-compat mode defers to the Phase 5 reference audit (PFR-02); PFedRec's `benchmark_cross_device` mode uses `num_positives`. Planner may adjust based on additional analysis.
- **Directory layout inside `data/derived/`** — e.g., flat (`data/derived/mapping.json`, `.../split_manifest.json`, `.../exclusion_items.npz`) vs subdirectory per artifact version. Flat is fine for a locked single-version artifact; planner picks.
- **Atomic write pattern** — Tempfile + `os.replace()` for manifest writes so a crash mid-write doesn't corrupt the committed artifact. Safe default; planner enforces.
- **Per-user-group bucket boundaries** — `sparse ≤ 30`, `30 < medium ≤ 100`, `dense > 100` already in the codebase; keep unchanged unless planner has cause.
- **`run_id` generation** — ULID or short UUID or timestamp-slug. Claude picks at planning time; whatever it is, it's propagated into W&B run names.
- **Validation split** — Not adding a per-user val split at the foundation layer. If Phase 5 (PFedRec reproduction) or Phase 6 (eval harness) decides it needs one for early stopping, that phase handles the split derivation and updates `exclude_items` accordingly.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner, executor) MUST read these before planning or implementing.**

### Project-level
- `.planning/PROJECT.md` — Project scope, core value, out-of-scope list, key decisions (no `fedrec_common/` this cycle, cross-silo preserved, re-audit PFedRec from reference, per-group reporting first-class).
- `.planning/REQUIREMENTS.md` §Foundation — FND-01..07 (the actual requirements this phase must satisfy).
- `.planning/ROADMAP.md` §Phase 1 — Success criteria (4 observable criteria phrased in terms of artifact existence + determinism).

### Research outputs
- `.planning/research/SUMMARY.md` — Top 10 pitfalls, build-order implications.
- `.planning/research/FEATURES.md` — Table-stakes features for a correct cross-device FedRec protocol (especially P0 and P1 tables).
- `.planning/research/PITFALLS.md` §1, §2, §3, §12, §13, §14, §21, §23, §24 — Directly relevant to this phase (num-supernodes survival, eval-protocol drift, test-leak, unseeded sampling, client RNG collisions, evaluator reseeding, ID-mapping drift, split-timestamp nondeterminism, paper deviations).
- `.planning/research/ARCHITECTURE.md` — Migration deltas especially §"Data Layer (dataset.py)" and §"Orchestration Layer (server_app.py)".

### Codebase map (brownfield context)
- `.planning/codebase/ARCHITECTURE.md` §"Personalization Boundary Matrix" and §"Data Flow" — how the existing four modules are structured.
- `.planning/codebase/CONCERNS.md` — already-catalogued bugs (cross-silo methodological problem, 9 PFedRec bugs, test-positive leak pattern, `weights_only=False` risk).
- `.planning/codebase/STACK.md` — Flower ≥1.22, PyTorch ≥2.7, ML-1M details.
- `.planning/codebase/CONVENTIONS.md` — notation convention (`w` global, `theta_i` local, `D_i`, `K`, `R`, `N`, `C`), code standards.

### Prior work (brownfield)
- `docs/superpowers/plans/2026-04-04-cross-device-migration.md` — Earlier draft plan that covers Phase 2 territory (baseline module migration). Useful reference for the partition-mode config pattern; superseded here by the mode-selector decision in D-06/D-07.
- `IJCAI-23-PFedRec/` — Reference implementation used for Phase 5 PFedRec audit (NOT Phase 1, but downstream planner may cross-reference for weight-policy defaults).

### External specs
- PFedRec IJCAI-23 paper (see SUMMARY.md references) — Paper numbers (HR@10 ≈ 0.729, NDCG@10 ≈ 0.441) are the reproduction target consumed by Phase 5, not Phase 1.
- NCF WWW-17 paper (see SUMMARY.md references) — `sampled_loo_99` protocol definition (1 positive + 99 negatives per held-out test item).
- No project-specific ADRs or feature docs exist — research-quality decisions are captured in PROJECT.md / this CONTEXT.md.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable assets (already in the codebase)
- **`create_global_mappings(ratings_df)`** in each `dataset.py` (4 copies) — already builds `(user2idx, idx2user, item2idx, idx2item)` over the union of user_ids and movie_ids (6040 users, 3706 movies). Foundation's mapping artifact is a persisted version of this.
- **`create_leave_one_out_split()`** in each `dataset.py` (4 copies) — already does LOO splitting. Foundation's split manifest is a persisted, deterministic-sorted version.
- **`load_movielens_1m()`** in each `dataset.py` — reads `ratings.dat`, `movies.dat`, `users.dat`. Foundation's `raw-data-hash` field is computed over these files.
- **`compute_user_genre_distribution()`** in each `dataset.py` — already builds per-user genre proportions. Adaptive's `genre_entropy` field in `split_manifest.json` is derived from this.
- **`evaluate_ranking_sampled()`** in each `task.py` — already implements the `sampled_loo_99` protocol. Foundation doesn't replace this; it only defines `primary-evaluator = "sampled_loo_99"` as the config-level selector.
- **`UserGroupConfig` + `classify_user_group()` + `aggregate_metrics_by_group()`** in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/user_groups.py` — sparse/medium/dense classifier. Foundation's user-group classification field in `split_manifest.json` is produced by this (or the same logic lifted into the foundation module).
- **Module-level `_partition_cache`** in each `dataset.py` keyed by `f"{num_partitions}_{alpha}_{split_mode}_{partition_mode}"` — foundation's `split_hash` is a deterministic replacement for this ad-hoc cache key.

### Established patterns the foundation should extend (not break)
- **Python package layout**: each of `federated-baseline-cf/`, `federated-pfedrec/`, `federated-personalized-cf/`, `federated-adaptive-personalized-cf/` is an installable Python package (`pip install -e .` per CLAUDE.md). Foundation code imported by these packages must be importable from each module's `dataset.py` without editing each `pyproject.toml` beyond the ergonomic minimum.
- **Flower config consumption**: clients read config via the `Context` object in `@app.train()` / `@app.evaluate()` hooks; servers via `grid` + `config`. Foundation's `mode` value is passed through this existing pipe — no new config mechanism.
- **W&B logging**: already wired in every `server_app.py`. Foundation's manifest fields are added to the W&B config dict there (no new wandb.init call).
- **Type hints + NumPy-style docstrings** per CLAUDE.md `Code Standards`. Foundation code follows the same.
- **Seeded RNG pattern**: today, most modules call `random.seed(seed)` globally inside `evaluate_ranking_sampled()`. Foundation REPLACES this global-reseed pattern with explicit RNG-object passing (the four-tier hierarchy in D-14); downstream migrations inherit the new pattern.

### Integration points (where new foundation code connects to existing modules)
- Each module's `dataset.py` gains imports from `scripts/foundation/` (or wherever D-Discretion places the foundation module): the canonical mapping loader, the canonical split loader, the exclusion-set loader.
- Each module's `server_app.py` gains imports for the seeded server RNG factory and the run-manifest builder.
- Each module's `client_app.py` gains imports for the per-user RNG factory and the benchmark-mode assertion.
- Each module's `task.py` (esp. `evaluate_ranking_sampled`) is refactored to accept an RNG object parameter and drop internal `random.seed(...)` calls. This is a surgical change — the signature gets one new parameter; behavior otherwise unchanged.
- No changes to any `models/` subpackage in Phase 1.

### Known anti-patterns to retire
- Global `random.seed(seed)` inside evaluators (3 of 4 modules do this today — see `.planning/codebase/CONCERNS.md` "non-deterministic" + "globally re-seeded RNG" entries).
- Unseeded `random.sample(node_ids, num_selected)` in server-side client selection (all 4 modules — see same document).
- Shared project-root `.embedding_cache/` without run scoping (3 of 4 modules — same document).
- `weights_only=False` `torch.load(...)` calls (see CONCERNS.md) — foundation forbids pickle in its own artifacts; modules' embedding-cache fix is Phase 2/3/4's job.

</code_context>

<specifics>
## Specific Ideas

- User wants a **single mode selector** (D-06) — explicitly picked this over granular flags because "one obvious knob; hard to misconfigure." Interpretation: the preference extends to any other Phase 1 knob that could collapse into a mode; planner should prefer collapse-into-mode over exposing yet another flag.
- User picked "mode locks EVERYTHING including training hyperparams" (D-07). Interpretation: a `mode` value IS a fully-specified experiment profile. The manifest fingerprint (D-16) therefore tells you the whole experiment just by reporting the mode + overrides dict.
- User picked "Lock forever, refuse overwrites" for the split manifest (D-04). Strong preference for deterministic/immutable artifacts — planner should extend this mental model to any other locked file (e.g., raw-data hash, mapping).
- User picked "Yes, but log loudly" for overrides (D-10). Interpretation: never block the user, but make drift visible. Planner should apply this "allow + annotate" principle wherever a similar tension arises.
- User picked ALL four optional split-manifest fields including user-group classification (D-03). Interpretation: strong preference for precomputing anything that downstream phases will need, rather than leaving recomputation scattered.

</specifics>

<deferred>
## Deferred Ideas

- **Shared code refactor into `fedrec_common/`** — Out of scope per PROJECT.md. Revisit post-thesis (see REQUIREMENTS.md §v2 REF-01, REF-02).
- **Validation split** — Not introduced at the foundation layer. If Phase 5 (PFedRec reproduction) needs one for early stopping to avoid tuning on test, that phase handles it locally and extends the exclusion set format.
- **DP / privacy accounting** — Out of scope per PROJECT.md (v2 DP-01, DP-02).
- **ML-10M / ML-20M generalization** — Out of scope per PROJECT.md (v2 EXT-01). The foundation manifest format is extensible (new `raw-data-hash` would just be a different value), but the builder is written to ML-1M schema assumptions.
- **Profile-based config mechanism** — User picked single `mode` selector over named profiles (D-06). Profiles could be revisited if the number of paper-compat modes grows, but for now the flat enum is sufficient.
- **Cross-silo deletion** — User kept cross-silo reachable as `cross_silo_legacy` mode (D-09). Full removal could happen in a far-future cleanup cycle once all historical appendix results are regenerated under cross-device.
- **Atomic multi-writer semantics for `data/derived/`** — Single-writer pattern assumed (one researcher running one build). If the thesis setup ever adds CI-building the artifacts, revisit file locking.

</deferred>

---

*Phase: 01-foundation-contract*
*Context gathered: 2026-04-19*
