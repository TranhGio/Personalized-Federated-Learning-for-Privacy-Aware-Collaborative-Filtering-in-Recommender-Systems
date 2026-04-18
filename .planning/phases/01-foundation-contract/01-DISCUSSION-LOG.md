# Phase 1: Foundation Contract — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `01-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-19
**Phase:** 01-foundation-contract
**Areas discussed:** Artifact storage format & location · Benchmark vs paper-compat mode interface
**Areas NOT discussed (Claude's Discretion):** Shared code placement · Weight-policy defaults per module

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Shared code placement | Where foundation code lives (copy, source-of-truth module, project-root folder) | |
| Artifact storage format & location | JSON/NPZ/Parquet; committed vs gitignored; directory choice | ✓ |
| Benchmark vs paper-compat mode interface | How the cross-device switch surfaces in each pyproject.toml | ✓ |
| Weight-policy defaults per module | `uniform` / `num_positives` / `num_training_examples` per module | |

**User's choice:** Artifact storage format & location, Benchmark vs paper-compat mode interface.
**Notes:** The two skipped areas have strong priors (PROJECT.md rules out `fedrec_common/`; research recommends `num_positives` for BPR modules and per-paper value for PFedRec). Captured as Claude's Discretion in CONTEXT.md.

---

## Artifact Storage & Location

### Q1 — Location

| Option | Description | Selected |
|--------|-------------|----------|
| `data/derived/` (committed) | Built once from raw ML-1M, committed as ground-truth | ✓ |
| `.planning/artifacts/` (committed) | Lives alongside ROADMAP.md/REQUIREMENTS.md | |
| `.crossdevice_cache/` (gitignored, regenerated) | Rebuild from raw data on first use | |
| `data/derived/` committed + hash-verified on load | Committed AND re-hashed on every load | |

**User's choice:** `data/derived/` (committed).
**Notes:** Prefers bit-exact reproducibility over repo-footprint minimization. Hash-verify-on-load not selected but is a natural extension Claude can add at planning time if cheap.

### Q2 — Format

| Option | Description | Selected |
|--------|-------------|----------|
| JSON everything | All three artifacts as .json | |
| Mixed: JSON metadata + NPZ arrays (Recommended) | Tiny JSON for metadata, NPZ for the exclusion set tensors | ✓ |
| Parquet for everything | Columnar, analytics-friendly | |
| Pickle | Fast but unsafe (arbitrary-code-execution on load) | |

**User's choice:** Mixed: JSON metadata + NPZ arrays.
**Notes:** Keeps the exclusion set fast to load and the manifest human-readable/diffable. Pickle explicitly rejected — aligns with CONCERNS.md's existing `weights_only=False` flag.

### Q3 — Manifest contents (multiSelect)

| Option | Description | Selected |
|--------|-------------|----------|
| `split_hash` (mandatory) | Deterministic hash of `(user_id, timestamp, movie_id)` tuples | ✓ |
| Builder metadata | Builder version, timestamp, raw-data hash | ✓ |
| Per-user stats | `n_interactions`, `genre_entropy`, `n_unique_items`, `rating_std` | ✓ |
| User-group classification | Precomputed sparse/medium/dense bucket per user | ✓ |

**User's choice:** All four selected.
**Notes:** Strong preference for precomputing derived state in one canonical place rather than recomputing in downstream phases. Phase 4 (adaptive) no longer recomputes user stats; Phase 6 (eval harness) no longer recomputes user groups.

### Q4 — Versioning

| Option | Description | Selected |
|--------|-------------|----------|
| Lock forever, refuse overwrites (Recommended) | Once committed, canonical split is immutable; rebuild errors on hash mismatch | ✓ |
| Versioned (v1, v2, ...) | New runs write v2 alongside v1; experiments declare which version | |
| Rebuild every run, hash-verify | Builder runs at startup, verifies match, fails on drift | |
| Single file, silently overwrite | Not recommended, listed for completeness | |

**User's choice:** Lock forever, refuse overwrites.
**Notes:** Reinforces the immutability preference. Hash-verify-on-load (Q1's option D) is still a reasonable complement at planning time.

---

## Benchmark vs Paper-Compat Mode Interface

### Q1 — Interface style

| Option | Description | Selected |
|--------|-------------|----------|
| Single top-level `mode` selector (Recommended) | One key that locks downstream defaults | ✓ |
| Granular flags only | Every knob independently configured | |
| Named profiles in pyproject.toml | Multiple `[tool.flwr.app.config.profiles.NAME]` sections | |
| Base = benchmark, overrides via config file | Paper-compat lives in a separate file | |

**User's choice:** Single top-level `mode` selector.
**Notes:** Preference for one obvious knob that is hard to misconfigure. The mode-to-defaults mapping lives in code (Python module), not scattered across pyproject.toml.

### Q2 — Cross-silo legacy handling

| Option | Description | Selected |
|--------|-------------|----------|
| Keep reachable as `cross_silo_legacy` mode (Recommended) | Preserve code paths, default to cross-device | ✓ |
| Remove cross-silo entirely | Delete code paths | |
| Keep code, add deprecation warning | Still runs but prints warnings | |

**User's choice:** Keep reachable as `cross_silo_legacy` mode.
**Notes:** Consistent with PROJECT.md constraint "we override defaults, we do not delete the code paths." Historical W&B runs reproducible; cross-silo never hit accidentally.

### Q3 — Mode scope (multiSelect)

| Option | Description | Selected |
|--------|-------------|----------|
| `num-supernodes` + `partition-mode` | Core cross-silo-vs-cross-device axis | ✓ |
| `weight-policy` | `num_positives` for benchmark, paper-value for paper-compat | ✓ |
| `eval-protocol` (primary evaluator) | Belt-and-suspenders lock on `sampled_loo_99` | ✓ |
| Training hyperparams (dim, optimizer, lr, negatives, epochs) | Mode fully defines the experiment | ✓ |

**User's choice:** ALL four — mode fully locks the experiment.
**Notes:** A `mode` IS a complete experiment profile. This simplifies the manifest (just record `mode` + `overrides` and the full setup is derivable). Per-module customization for paper-compat (e.g., PFedRec's weight-policy) is allowed via module-specific overrides of the mode's defaults, documented in code.

### Q4 — Override behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, but log loudly (Recommended) | Overrides allowed; captured in manifest `overrides` field + console warning | ✓ |
| Yes, silent | Normal Flower override behavior | |
| No — modes are sealed | Overrides error out; must change mode to change settings | |

**User's choice:** Yes, but log loudly.
**Notes:** Never block iteration; always make drift visible. Implies the manifest has a dedicated `overrides` field and console output prints a prominent warning at run start when overrides are present.

---

## Claude's Discretion

**Shared code placement** — Not discussed explicitly. CONTEXT.md D-Discretion captures the recommendation: `scripts/foundation/` at project root (with each module's `dataset.py` importing via a relative path addition), or duplicate into each of the four module packages. Planner picks at plan time.

**Weight-policy defaults per module** — Not discussed explicitly. CONTEXT.md D-Discretion captures the recommendation: `num_positives` for baseline / personalized / adaptive; PFedRec defers to the Phase 5 reference audit (PFR-02) for its `paper_compat_pfedrec` mode, and uses `num_positives` in `benchmark_cross_device` mode.

**Directory layout inside `data/derived/`** — Flat, since the split is locked single-version (D-04).

**Atomic write pattern** — Tempfile + `os.replace()` default.

**Per-user-group bucket boundaries** — Keep the existing sparse ≤ 30 / medium 30–100 / dense > 100 boundaries from `federated-adaptive-personalized-cf/.../evaluation/user_groups.py`.

**`run_id` format** — Claude picks (ULID or short UUID or timestamp-slug).

**Validation split** — Not added at foundation layer; deferred to Phase 5/6 if needed.

## Deferred Ideas

(None arose during discussion. Any ideas outside the Phase 1 boundary were either already captured in PROJECT.md's Out-of-Scope list or routed to the v2 sections of REQUIREMENTS.md.)
