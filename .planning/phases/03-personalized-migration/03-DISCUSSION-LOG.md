# Phase 3: Personalized Migration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-19
**Phase:** 03-personalized-migration
**Areas discussed:** Local user-row representation, Embedding cache signature layout, Cross-run cache reuse policy, User-row initialization strategy

---

## Area 1: Local user-row representation (PSN-06)

### Q1.1 — Which shape for the local user row?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Keep `nn.Embedding(6040, d)`, only row `partition_id` ever touched. ~3 MB/client memory waste but zero model refactor. | |
| B | Collapse to `nn.Parameter(shape=(d,))` — one row, no embedding table. `forward()` no longer takes `user_ids`; the user IS the client. | ✓ |
| C | Keyed dict on disk `{partition_id: tensor}`, in-memory `nn.Embedding(1, d)` rebuilt on load. | |

**User's choice:** B — single-row `nn.Parameter`
**Notes:** Thesis-honest ("1 client = 1 user" reflected in the model, not papered over with a 6040-row ghost table). Refactor is local to `models/bpr_mf.py` + forward signature. Recommended by Claude.

### Q1.2 — Cross-silo compatibility?

| Option | Description | Selected |
|--------|-------------|----------|
| B-i | Benchmark-mode only. `cross_silo_legacy` / `partition_mode="dirichlet"` raises `NotImplementedError`. Legacy cross-silo numbers frozen in git history. | ✓ |
| B-ii | Keep a `PersonalizedBPRMFMultiUser` for cross-silo + `PersonalizedBPRMFSingleUser` for cross-device. Factory picks based on mode. Double maintenance. | |
| B-iii | Pick A instead; live with the ghost table to keep one code path. | |

**User's choice:** B-i — benchmark-only; cross-silo frozen in git history
**Notes:** PROJECT.md lists cross-silo as explicit opt-in; no one is expected to rerun this module's cross-silo numbers. Recommended by Claude.

### Q1.3 — Expose local row via `_LOCAL_PARAMS` API, or inline?

| Option | Description | Selected |
|--------|-------------|----------|
| B-local | Keep `_LOCAL_PARAMS = ('local_user_row',)` + get/set/save/load API. Disk shape changes from `(6040,d)` to `(d,)`; contract symmetry preserved. | ✓ |
| B-inline | Inline `self.local_user_row = nn.Parameter(...)` without exposing via `_LOCAL_PARAMS`. Cache via `torch.save(model.state_dict(), ...)` with name-based filter. | |

**User's choice:** B-local — keep API contract
**Notes:** Symmetric with Phase 2 baseline strategy + G-03-01 mapping pattern. Only disk payload shape changes. Recommended by Claude.

---

## Area 2: `.embedding_cache/` signature layout (PSN-05)

### Q2.1 — Path scheme?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Directory-encoded: `.embedding_cache/{run_id}/{method}_u{N}_i{M}_d{D}_s{split[:8]}/partition_{pid}.pt`. Browsable, grep-friendly, long paths, fragile if signature fields change. | |
| B | Manifest-sidecar: `.embedding_cache/{run_id}/manifest.json` + `partition_{pid}.pt`. One JSON read validates all six fields on load. | ✓ |
| C | Content-hash: `.embedding_cache/{sha256(signature)[:16]}/partition_{pid}.pt`. Compact, opaque, conflates with Area 3's reuse policy. | |

**User's choice:** B — manifest-sidecar
**Notes:** Strict-contract enforcement via single `json.load + compare`. Avoids fragile path encoding. Matches `foundation_index.json` pattern. Recommended by Claude.

### Q2.2 — Mismatch behavior?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Raise `RuntimeError` with the delta ("dim mismatch: cache=64, current=128; delete {run_id}/ to reset"). Loud, manual cleanup. | |
| B | Raise but auto-delete the stale cache directory. Convenient, risk of silent data loss. | |
| C | Raise + print a one-liner `rm -rf` command. A's safety + user recovery. | ✓ |

**User's choice:** C — raise + print rm hint
**Notes:** No auto-deletion anywhere. Matches `save_split_or_verify` pattern from Phase 2. Recommended by Claude.

### Q2.3 — Scope of `method` field?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Just `model-type` (bpr/basic/dual). Simple. | |
| B | `model-type` + `fusion-type` + `alpha-method`. Adaptive-specific fields default to `"na"` here. | |
| C | Just `model-type` in Phase 3 with `schema_version=1`; Phase 4 bumps schema and adds fusion/alpha. | ✓ |

**User's choice:** C — schema_version=1 in Phase 3, v2 in Phase 4
**Notes:** Keeps Phase 3 focused. Schema version check doubles as a sanity mismatch under D-05. Recommended by Claude.

### Q2.4 — Manifest file format?

| Option | Description | Selected |
|--------|-------------|----------|
| A | JSON — matches `foundation_index.json`, human-readable, stable. | ✓ |
| B | TOML — matches `pyproject.toml` but adds a dep / inconsistency. | |
| C | Plain text key=value — trivial parse, no schema validation. | |

**User's choice:** A — JSON
**Notes:** Consistency with `foundation_index.json` + `*-manifest.json`. Atomic writer already exists in `fedrec_foundation.atomic`. Recommended by Claude.

---

## Area 3: Cross-run cache reuse policy

### Q3.1 — Default reuse behavior?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Never reuse. Every `flwr run .` gets a fresh `{run_id}/` dir. | |
| B | Content-addressed default. Drop `run_id`; two runs with same signature share cache silently. | |
| C | Opt-in reuse. Default A; `reuse-cache=true` flag switches to B. | ✓ |

**User's choice:** C — opt-in flag, default off
**Notes:** Default protects thesis reproducibility; opt-in helps iteration. Recommended by Claude.

### Q3.2 — Cleanup policy?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Never auto-delete; user runs `rm -rf .embedding_cache/` manually. | |
| B | Keep last N runs, auto-delete older at startup. Default N=5. | |
| C | Time-based (older than 7 days). Cron-style. | |
| D | No auto-delete + a helper script `scripts/clean_cache.py --keep N`. | ✓ |

**User's choice:** D — manual helper script, no auto-cleanup
**Notes:** Auto-deletion surprises people. Helper gives control without hidden state. Matches `results/federated/` pattern. Recommended by Claude.

### Q3.3 — On `reuse-cache=true` with signature match, how?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Ignore `run_id` entirely — path is `sig_{sha256(signature)[:16]}/`. Perfect reuse. | ✓ |
| B | Hard-link fresh `{run_id}/` to the content-hash dir. run_id visible, no double disk. Breaks with rsync. | |
| C | Copy on reuse — fresh dir with files copied from prior run. 2× disk. | |

**User's choice:** A — content-hash path, ignore run_id on reuse
**Notes:** Hard-links break with backup workflows; copy wastes disk. Users who opted in want speed, not cosmetics. Recommended by Claude.

---

## Area 4: User-row initialization strategy

### Q4.1 — First-round init?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Xavier-uniform on first use, persist thereafter. Current pattern. | |
| B | Server ships population-mean user embedding with `global_params`; cold clients copy it. Warmer start. | |
| C | Defer to Phase 4 — Phase 4 adaptive has `_global_prototype` EMA with `prototype_momentum=0.9`. Phase 3 keeps Xavier. | ✓ |

**User's choice:** C — defer warm-start to Phase 4
**Notes:** Keeps the comparison ladder clean (personalized lift = local rows alone). Thesis-relevant boundary. Recommended by Claude.

### Q4.2 — Cold client with few positives — what happens?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Xavier init + immediate local training. One epoch adjusts the row. Matches Phase 2 baseline. | ✓ |
| B | Xavier init + skip-if-train-positives<K (still eval). Asymmetric behavior. | |
| C | Xavier init + always train if selected, even with 1 positive. No threshold. | |

**User's choice:** A — matches baseline exactly
**Notes:** Preserves baseline comparability. C is equivalent to A in practice but under-specified (no K). Recommended by Claude.

### Q4.3 — Log cold-start prevalence?

| Option | Description | Selected |
|--------|-------------|----------|
| A | Nothing extra — keep Phase 3 minimal. | |
| B | Server logs `cold_starts_per_round` = count of selected clients with no prior disk cache. One int/round. Trivial. | ✓ |
| C | Client self-reports `was_cold_start: bool` in `FitMetricsContract.extras`. Per-user granularity. | |

**User's choice:** B — per-round int, log to W&B + eval_metrics_history
**Notes:** Thesis-valuable (one number, cite "fraction of evals on truly-cold rows"). C is overkill for now. Recommended by Claude.

---

## Claude's Discretion

Areas where the user delegated the decision to Claude or the planner:

- Exact name of the new single-row model class (e.g. `PersonalizedBPRMF` vs refactor of
  existing `BPRMF`) — planner decides based on diff size
- Whether `PersonalizedSplitFedAvg` / `PersonalizedSplitFedProx` are new classes or
  thin renames of `BaselineFedAvg` / `BaselineFedProx`
- Plan 01/02/03/... Wave-1 file-ownership partition — planner decides to avoid
  write-races, mirroring Phase 2 precedent
- Location of `scripts/clean_cache.py` (repo root `scripts/` vs
  `federated-personalized-cf/scripts/`)
- Whether `reuse-cache` is a single boolean flag or split into advanced override
  variants

## Deferred Ideas

Captured in CONTEXT.md `<deferred>` section:

- Phase 4: server prototype EMA, per-user alpha, item perturbation, contrastive loss,
  PersonalMLP, fusion layers, cache `schema_version=2`
- Phase 5: PFedRec per-user `affine_output`, dual-LR alternating optimization
- Frozen: cross-silo runs for this module (D-02 raises NotImplementedError)
- Out of cycle: DP / privacy, `fedrec_common/` extraction
