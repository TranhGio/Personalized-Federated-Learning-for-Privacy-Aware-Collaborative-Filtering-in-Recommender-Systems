# Phase 3: Personalized Migration - Context

**Gathered:** 2026-04-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Migrate `federated-personalized-cf/` to a correct cross-device split-learning benchmark:
6040 supernodes (one user per client), single-user local model with the local user row
kept on-disk in a run-scoped cache, sufficient-stat metrics with the Phase-2-style
server aggregator, and Phase-1 foundation contract fingerprints logged per run. The
thesis role of this phase is the **middle rung** of the comparison ladder
(baseline → personalized → adaptive): it isolates the effect of keeping user state
LOCAL without introducing any prototype / alpha / perturbation machinery — those
belong to Phase 4.

**In scope:** PSN-01..07 requirements, matching the Phase-2 baseline migration's
template (mode profile resolver, FND-06 RNG wiring, BaselineFedAvg-style
sufficient-stat aggregator, G-03-01 discovery-round + partition-id-space sampling,
D-25/D-26/D-27 result-artifact shape, D-15 double-write manifest).

**Out of scope:** server prototype EMA, per-user learned alpha, item perturbation,
contrastive loss, fusion layers, PersonalMLP — all inherited from `federated-adaptive-personalized-cf/`
and belong to Phase 4. Cross-silo (Dirichlet) runs for this module are frozen —
pre-Phase-3 commits remain the authoritative artifact if anyone needs to re-run them.

</domain>

<decisions>
## Implementation Decisions

### Local user-row representation (PSN-06)
- **D-01:** Replace `nn.Embedding(num_users, d)` with a single-row `nn.Parameter(shape=(d,))`
  plus a scalar `user_bias` parameter. The "user" is the client; `forward()` no longer
  accepts `user_ids`. New class `PersonalizedBPRMF` (or an inline refactor of the
  existing `BPRMF`) drops the ghost 6040×d table entirely.
- **D-02:** **Benchmark mode only.** Under `mode="cross_silo_legacy"` /
  `partition_mode="dirichlet"`, the module raises `NotImplementedError("Personalized
  cross-device migration removed multi-user support; check out pre-Phase-3 commit
  to reproduce legacy cross-silo numbers")`. Cross-silo numbers for this module are
  frozen in git history and not re-derived.
- **D-03:** `_LOCAL_PARAMS = ('local_user_row', 'local_user_bias')` in the get/set API.
  `get_local_parameters()` returns `OrderedDict([('local_user_row', tensor(d,)),
  ('local_user_bias', tensor(1,))])`; disk payload is the same shape — never the old
  `(num_users, d)` blob. `_GLOBAL_PARAMS` unchanged: `('item_embeddings.weight',
  'item_bias.weight', 'global_bias')`.

### Embedding cache signature layout (PSN-05)
- **D-04:** Manifest-sidecar layout:
    - `.embedding_cache/{run_id}/manifest.json` — six-field signature
    - `.embedding_cache/{run_id}/partition_{pid}.pt` — flat `torch.save`d state-dict
      (keys: `local_user_row`, `local_user_bias`) per selected partition.
  On every load, `manifest.json` is read and compared to the current run's
  signature BEFORE any `.pt` is touched.
- **D-05:** Signature mismatch behavior: raise `RuntimeError` with the per-field
  delta (e.g. `"cache signature mismatch: dim cached=64, current=128. Run:
  rm -rf .embedding_cache/{run_id}/ to reset, or check --run-config"`). The error
  message includes the literal `rm -rf` command for the specific `{run_id}` path.
  No auto-deletion under any condition.
- **D-06:** `manifest.json` includes `schema_version: 1` and five payload fields:
  `run_id`, `method` (`"bpr"` | `"basic"`), `num_users`, `num_items`, `dim`,
  `split_hash`. Phase 4 will bump `schema_version` to `2` and add fusion/alpha
  fields; schema_version mismatch is a signature mismatch per D-05.
- **D-07:** Manifest format is JSON — matches `data/derived/foundation_index.json`
  and `results/federated/*-manifest.json`. Written via the existing atomic
  `tempfile.mkstemp + os.replace` helper used in
  `fedrec_foundation.atomic.atomic_write_json`.

### Cross-run cache reuse policy
- **D-08:** Default is "never reuse across `run_id`". Every `flwr run .` / `python
  scripts/run.py personalized benchmark_cross_device` creates a fresh
  `.embedding_cache/{new_run_id}/` directory. This is the thesis-reproducibility
  default.
- **D-09:** Opt-in reuse via `--run-config "reuse-cache=true"` (new run_config key,
  default `false`). When set, the path drops `run_id` entirely and becomes
  `.embedding_cache/sig_{sha256(signature_fields)[:16]}/partition_{pid}.pt` plus a
  sibling `manifest.json`. Two runs with identical signature share cache silently.
- **D-10:** No auto-cleanup. Accumulation is the user's problem. Ship
  `scripts/clean_cache.py --keep N` (default `N=5`) as a manual helper — glob
  `.embedding_cache/{run_id}/` dirs, sort by mtime, delete all but the newest N.
  Content-hash (`sig_*`) dirs are NEVER touched by the helper.

### User-row initialization strategy
- **D-11:** Xavier-uniform on first use, persist on disk thereafter. **No**
  server-side warm-start from a population mean in Phase 3. Phase 4's
  `server._global_prototype` EMA is the canonical warm-start mechanism and stays
  in that phase to keep the comparison ladder clean (personalized's lift over
  baseline must come from local private rows ALONE, not from a prototype).
- **D-12:** A selected client with as little as 1 positive trains for
  `local_epochs` epochs with whatever negatives its RNG draws. No
  "too-few-positives" skip logic — matches Phase 2 baseline behavior exactly.
- **D-13:** Server logs a per-round scalar `cold_starts_per_round` = count of
  selected partitions whose on-disk cache did not exist BEFORE this round
  (i.e. truly-cold clients). Added to `eval_metrics_history[round_num]` and
  W&B (`round/cold_starts`). Reported at training end as
  `total_cold_starts` and `cold_start_rate = total_cold_starts / total_client_selections`.
  Thesis artifact: cite the fraction of the 50-round eval that happened on rows
  trained exactly once.

### Carried forward from Phase 02 (locked — do not re-discuss)
- Mode profile resolver + `benchmark_cross_device` default
- `num-supernodes=6040`, `partition-mode="natural"`, `run-seed=42`, `weight-policy="num_positives"`
- FND-06 RNG factories (`np_rng` / `torch_gen` / `py_rng`) wired into training,
  evaluation, and every `DataLoader(..., generator=...)` call
- `FitMetricsContract` + `EvaluateMetricsContract` wire payloads with optional
  `partition_id` field (auto-whitelisted via dataclass `fields()`)
- G-03-01 discovery-round protocol — one-shot `@app.evaluate(discover_only=true)`
  BEFORE the main loop builds `partition_to_node_id`; `_server_sampler.sample(range(N), k)`
  samples in partition-id space; `selected_clients_per_round` stores stable
  partition_ids (0..N-1)
- Per-group (sparse/medium/dense) sufficient-stat fields: `hit_count_*_at10`,
  `ndcg_sum_*_at10`, `evaluated_users_*`
- FedProx proximal term applied **only to GLOBAL params** (item embeddings +
  item bias + global bias); the local user row is NEVER touched by the proximal
  term — this is an architectural property of split learning, not a discussion point
- FND-07 manifest + D-15 double-write (embedded `_manifest` in result JSON +
  sibling `{run_id}-manifest.json`); `module="personalized"` in the manifest
- D-27 in-memory best-round restore before centralized evaluation
- Default W&B project `federated-cf-cross-device` for benchmark modes; legacy
  `federated-cf` for cross-silo (frozen)

### Claude's Discretion
- Exact name of the new model class (e.g. `PersonalizedBPRMF` vs refactor of
  existing `BPRMF` — planner decides based on diff size)
- Whether `PersonalizedSplitFedAvg` / `PersonalizedSplitFedProx` are new classes
  or a thin rename of `BaselineFedAvg` / `BaselineFedProx` (the aggregation
  logic is identical — D-23 preserved, only `_GLOBAL_PARAMS` key set differs)
- Exact partition between Plan 01/02/03/... — planner decides based on
  file-ownership race-free Wave-1 split, following Phase 2's precedent
- Whether `scripts/clean_cache.py` lives at repo root `scripts/` or under
  `federated-personalized-cf/scripts/` — planner picks
- Whether `reuse-cache` is one flag or split into `reuse-cache` (bool) +
  `cache-signature-override` (advanced) — planner picks based on test burden

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 3 requirements and boundary
- `.planning/ROADMAP.md` — Phase 3: Personalized Migration goal + success criteria
- `.planning/REQUIREMENTS.md` §PSN-01..07 — acceptance criteria
- `.planning/PROJECT.md` — project-level constraints (cross-silo as opt-in only,
  new W&B project for cross-device, tech stack locked)

### Phase 2 template to mirror
- `.planning/phases/02-baseline-migration/02-CONTEXT.md` — Phase 2 context for
  decision precedents (D-19..D-28)
- `.planning/phases/02-baseline-migration/02-baseline-migration-01-SUMMARY.md` —
  `BaselineFedAvg` / `BaselineFedProx` sufficient-stat aggregator pattern to clone
- `.planning/phases/02-baseline-migration/02-baseline-migration-02-SUMMARY.md` —
  pyproject + dataset rip-and-replace template (PSN-01 mirrors BSL-01)
- `.planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md` —
  `client_app.py` + `task.py` contract-wire migration (PSN-02..04 mirrors BSL-02..05)
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` —
  `server_app.py` main-loop migration + D-25/D-26/D-27 (PSN-04/PSN-07 mirrors BSL-04..08)
- `.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md` —
  G-03-01 closure: discovery-round protocol + partition-id-space sampling
  (this pattern is REQUIRED for Phase 3 — no rediscussion)
- `.planning/phases/02-baseline-migration/02-UAT.md` — Phase 2 UAT results including
  the determinism-regression pattern that Phase 3 must guard against

### Phase 1 foundation surface
- `.planning/phases/01-foundation-contract/01-foundation-contract-04-SUMMARY.md` —
  FND-06 RNG factories + FND-07 manifest schema
- `.planning/phases/01-foundation-contract/01-foundation-contract-06-SUMMARY.md` —
  `fedrec-foundation` editable-dep wiring (already installed; no new install steps)
- `scripts/foundation/fedrec_foundation/rng.py` — `py_rng` / `np_rng` /
  `torch_gen` / `server_rng` call sites
- `scripts/foundation/fedrec_foundation/mode.py` — `resolve_mode_defaults` +
  `assert_benchmark_one_user_per_client`
- `scripts/foundation/fedrec_foundation/fit_metrics.py` — `FitMetricsContract`
  + `EvaluateMetricsContract` (with optional `partition_id` field)
- `scripts/foundation/fedrec_foundation/manifest.py` —
  `build_run_manifest` / `embed_manifest_in_result` / `write_manifest_sibling`
- `scripts/foundation/fedrec_foundation/atomic.py` — `atomic_write_json` used by
  the new `manifest.json` sidecar (D-04)

### Existing personalized-cf module (code to refactor)
- `federated-personalized-cf/claude.md` — current module architecture + split-learning
  parameter protocol (this doc WILL need updating after Phase 3 lands)
- `federated-personalized-cf/pyproject.toml` — currently `num-supernodes=5`; PSN-01 flips
- `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` — current
  `BPRMF` with `nn.Embedding(num_users, d)` that D-01 collapses
- `federated-personalized-cf/federated_personalized_cf/models/basic_mf.py` — same refactor
  template for `BasicMF` (if kept — planner decides scope)
- `federated-personalized-cf/federated_personalized_cf/strategy.py` — existing
  `SplitFedAvg` / `SplitFedProx` that pre-date Phase 2's sufficient-stat pattern
- `federated-personalized-cf/federated_personalized_cf/client_app.py` —
  `save_local_user_embeddings` / `load_local_user_embeddings` (to be rewritten for
  single-row payload + manifest.json sidecar)
- `federated-personalized-cf/federated_personalized_cf/dataset.py` — needs same
  rip-and-replace as Phase 2 Plan 02 (delegate mapping/split/exclusion to
  `fedrec_foundation`)

### Launcher + tooling
- `scripts/run.py` — canonical launcher; `python scripts/run.py personalized
  benchmark_cross_device` must work at Phase 3 end (no pyproject-only entry point)
- `federated-baseline-cf/federated_baseline_cf/server_app.py` — server loop
  reference implementation to clone (NOT a canonical ref for code — it's the
  reference shape)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`BaselineFedAvg` / `BaselineFedProx`** (`federated-baseline-cf/federated_baseline_cf/strategy.py`):
  the sufficient-stat `aggregate_evaluate` is identical to what personalized needs.
  Per-group (sparse/medium/dense) sum-of-hits / sum-of-NDCG logic reusable verbatim.
  `PersonalizedSplitFedAvg` can be a thin subclass (override `_GLOBAL_PARAM_KEYS` +
  `_LOCAL_PARAM_KEYS` frozensets; inherit aggregate_evaluate unchanged).

- **Discovery-round + partition-id sampling** (`federated-baseline-cf/federated_baseline_cf/server_app.py`
  post-Plan-05): the `discover_only=true` handshake + `_server_sampler.sample(range(N), k)`
  + `partition_to_node_id` map is cut-paste reusable. Copy into
  `federated-personalized-cf/federated_personalized_cf/server_app.py` unchanged.

- **`FitMetricsContract` / `EvaluateMetricsContract`** (`scripts/foundation/fedrec_foundation/fit_metrics.py`):
  already supports optional `partition_id`. No foundation changes needed for Phase 3.

- **`atomic_write_json` + `tempfile.mkstemp + os.replace`** patterns: the new
  `manifest.json` sidecar (D-04) uses the foundation's existing atomic writer.

- **`server_rng(run_seed)` + `np_rng` / `torch_gen` / `py_rng`** factories: zero
  new RNG code required — Phase 3 consumes the Phase 1 factories.

- **FND-07 manifest builder** (`fedrec_foundation.manifest.build_run_manifest`):
  `module="personalized"` flag in the call, otherwise identical to baseline.

### Established Patterns

- **Split-learning API on models**: every split-aware `nn.Module` exposes
  `get_global_parameters()`, `set_global_parameters(state_dict)`,
  `get_local_parameters()`, `set_local_parameters(state_dict, strict=False)`.
  D-03 keeps this contract — only the payload shape changes.

- **Surgical-edit discipline (D-18)**: pre-existing uncommitted WIP in any touched
  file is preserved verbatim. `git diff --stat` in PR review asserts scope.

- **Wave-1 write-race avoidance**: Plan 01 owns one exclusive file set, Plan 02
  owns a disjoint set, etc. Phase 2 precedent: Plan 01 owned strategy.py +
  fit_metrics.py; Plan 02 owned pyproject.toml + dataset.py. Phase 3 mirrors.

- **Atomic cache writes**: `tempfile.mkstemp` + `torch.save` + `os.replace` under
  the `.embedding_cache/{run_id}/` dir. Never write `partition_{pid}.pt` directly.

- **`fedrec_foundation.atomic.atomic_write_json(path, data)`** for the new
  `manifest.json` sidecar — consistent with how `foundation_index.json` is written.

- **D-25 mode-resolver override surface**: every hyperparameter read is
  `int(context.run_config.get(key, profile.field))`; pyproject values ARE the
  override surface only.

- **D-21 strict-extras wire payload**: `FitMetricsContract.to_dict()` +
  `validate_fit_metrics()` before return. Planner applies the same contract at
  both `@app.train` and `@app.evaluate` edges.

- **D-15 double-write**: `embed_manifest_in_result(manifest, results_data)` +
  `write_manifest_sibling(manifest, json_path)`.

- **D-26 `selected_clients_per_round` stores partition_ids (0..N-1)**, not
  ephemeral Flower node_ids (G-03-01 fix from Plan 05).

- **D-27 in-memory best-round restore**: snapshot `ArrayRecord` when
  `current_ndcg > best_metric`; assign before centralized evaluation. Exactly
  as in baseline `server_app.py`.

### Integration Points

- **Installation order**: `pip install -e scripts/foundation/` before
  `pip install -e federated-personalized-cf/`. Docs already in `docs/setup.md`.
  No changes needed.

- **Launcher**: `scripts/run.py personalized benchmark_cross_device` (module alias
  already wired in `scripts/run.py MODULE_DIR` dict). Phase 3 just needs the
  server+client+model side to consume `mode=benchmark_cross_device` correctly.

- **Test location**: `scripts/foundation/tests/test_*.py` already has Phase 2
  precedent for subprocess-based real-loop regression tests
  (`test_selected_partitions_byte_identical_across_subprocess_reruns`). Phase 3
  adds a personalized-specific sibling: assert same-seed reruns produce
  byte-identical `selected_clients_per_round` AND identical `local_user_row`
  for each partition (the disk payload IS deterministic given FND-06 + natural
  partition-id-space sampling).

- **W&B project**: `federated-cf-cross-device` (same bucket as baseline).
  New run_config key `reuse-cache` is logged to the run config alongside
  existing keys.

- **Result file location**: `results/federated/{run_id}_results.json` (flat,
  no per-module subdir — D-28 locked in Phase 2). Sibling manifest
  `{run_id}-manifest.json` beside it.

</code_context>

<specifics>
## Specific Ideas

- "The thesis role of Phase 3 is the middle rung of the comparison ladder —
  baseline → personalized → adaptive. Lift of personalized over baseline must
  come from LOCAL PRIVATE USER ROWS alone, not from prototype warm-start or
  anything Phase 4 introduces. That's why D-11 defers warm-start."

- Single-row representation (D-01) is thesis-honest: "1 client = 1 user" should
  be reflected in the model, not papered over with a 6040-row ghost table.

- Cold-start logging (D-13) gives the thesis a concrete number to cite:
  "what fraction of 50-round eval happened on user rows trained exactly once?"
  Virtually free to compute; not valuable enough to expose per-user (C was rejected).

- `reuse-cache` flag (D-09) is opt-in specifically because cache-staleness bugs
  are the kind that silently corrupt thesis numbers. Default off protects the
  thesis; on exists for debugging cycles.

</specifics>

<deferred>
## Deferred Ideas

### Belongs in Phase 4 (adaptive)
- Server-side `_global_prototype` EMA for warm-starting cold user rows
- Per-user learned `logit_alpha`
- Dual-side item perturbation (`item_perturbation` embedding)
- Contrastive `InfoNCE` local-vs-blended auxiliary loss
- `PersonalMLP` score head + fusion layer
- `schema_version=2` cache manifest (adds `fusion_type`, `alpha_method`, etc.)

### Belongs in Phase 5 (PFedRec)
- Per-user `affine_output` Linear layer (PFedRec's personalization mechanism —
  different from both personalized and adaptive)
- Dual-LR alternating optimization (not relevant to split-learning MF)

### Frozen (not re-derived under cross-device)
- Cross-silo (`partition_mode="dirichlet"`) results for personalized module.
  D-02 raises `NotImplementedError` in that mode. Pre-Phase-3 commits remain
  the authoritative artifact if anyone needs those numbers.

### Out of this cycle
- DP / privacy quantification (PROJECT.md lists this as v2)
- `fedrec_common/` shared-code extraction (PROJECT.md locks this out of this cycle)

</deferred>

---

*Phase: 03-personalized-migration*
*Context gathered: 2026-04-19*
