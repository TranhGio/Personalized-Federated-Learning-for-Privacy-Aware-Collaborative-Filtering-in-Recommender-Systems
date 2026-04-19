# Phase 2: Baseline Migration — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `02-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-19
**Phase:** 02-baseline-migration
**Areas discussed:** Migration shape, Sufficient-stat aggregation (BSL-06), Per-client memory model, Result artifact shape

---

## Gray-Area Selection (multi-select, 4 of 4 selected)

| Option | Description | Selected |
|--------|-------------|----------|
| Migration shape | Rip-and-replace vs wrap of dataset.py helpers; handling of pre-existing uncommitted edits | ✓ |
| Sufficient-stat aggregation (BSL-06) | Where and how the server computes the final NDCG/HR ratio | ✓ |
| Per-client memory model | Full 6040×d matrix vs single-user fast path vs sparse update | ✓ |
| Result artifact shape | Selected-client logging location + checkpoint-rule default + directory layout | ✓ |
| Test coverage strategy | (Not offered separately — folded into Claude's Discretion in CONTEXT.md) | — |
| Seed purpose namespace | (Not offered separately — folded into Claude's Discretion in CONTEXT.md) | — |

---

## Area 1: Migration shape

### Q1: How should the baseline's existing dataset.py helpers relate to the new fedrec_foundation loaders?

| Option | Description | Selected |
|--------|-------------|----------|
| Rip-and-replace (Recommended) | Delete existing helpers; dataset.py becomes a thin adapter around foundation loaders. Single source of truth. | ✓ |
| Wrap both behind mode | Keep existing helpers; call foundation loaders only when benchmark_cross_device. Two code paths. | |
| Delete legacy path entirely | Remove create_global_mappings AND cross_silo_legacy support. Violates PROJECT.md constraint. | |

**User's choice:** Rip-and-replace
**Notes:** Captured as D-17 in CONTEXT.md.

### Q2: Where should the cross-device migration edits live relative to pre-existing uncommitted edits?

| Option | Description | Selected |
|--------|-------------|----------|
| Surgical — leave pre-existing hunks untouched (Recommended) | Phase 2 only modifies BSL-01..08-relevant lines; pre-existing WIP stays. | ✓ |
| Integrate — rebase pre-existing work into Phase 2 plans | Fold compatible pre-existing edits into Phase 2 tasks. Scope-creep risk. | |
| Ask me — I'll decide per-file during planning | Defer annotation to user during plan review. | |

**User's choice:** Surgical
**Notes:** Captured as D-18 in CONTEXT.md. Planner to annotate "do not touch" ranges per file.

### Q3: For CLI overrides (e.g., `flwr run . --run-config 'num-supernodes=10'`), what behavior?

| Option | Description | Selected |
|--------|-------------|----------|
| Allow + loud warning + manifest capture (Recommended) | Matches Phase 1 D-10. Override applied, loud warning at run start, captured in manifest.overrides. | ✓ |
| Allow silently — just capture in manifest | No warning. Cleaner stdout; easier to miss. | |
| Block in benchmark mode; force cross_silo_legacy | Most strict; no accidental misconfig. | |

**User's choice:** Allow + loud warning + manifest capture
**Notes:** Captured as D-19 in CONTEXT.md.

---

## Area 2: Sufficient-stat aggregation (BSL-06)

### Q1: Where should the server compute the final NDCG@10 = sum(ndcg_sum@10) / sum(evaluated_users)?

| Option | Description | Selected |
|--------|-------------|----------|
| Custom FedAvg subclass (Recommended) | BaselineFedAvg(FedAvg) overrides aggregate_evaluate(). Mirrors SplitFedAvg pattern. | ✓ |
| Inline in server_app.py main loop | Compute ratio directly in round loop. Simpler but scatters logic. | |
| Flower built-in weighted mean | Pass FitMetricsContract keys as evaluate_metrics_aggregation_fn. Violates BSL-06 (needs sums, not averages). | |

**User's choice:** Custom FedAvg subclass
**Notes:** Captured as D-20 in CONTEXT.md.

### Q2: How should clients package the sufficient statistics on the wire?

| Option | Description | Selected |
|--------|-------------|----------|
| FitMetricsContract.to_dict() keys only (Recommended) | Strict schema, validate_fit_metrics() rejects malformed. Enforces CR-4. | ✓ |
| FitMetricsContract + free-form extras | Contract keys + arbitrary extras. More flexible; schema drift risk. | |
| Raw sufficient-stat numpy array | Pack stats in FitRes.parameters. Abuses parameters channel. | |

**User's choice:** FitMetricsContract keys only
**Notes:** Captured as D-21 in CONTEXT.md. Combined with Q3 answer, drives the CR-4 extension plan.

### Q3: Where should per-user-group (sparse/medium/dense) metrics be computed?

| Option | Description | Selected |
|--------|-------------|----------|
| Client-side, included in extras (Recommended) | Client knows its user's group from split_manifest. Mirrors adaptive module. | ✓ |
| Server-side, post-hoc | Clients report overall + user_id; server looks up groups. Extra work, needs user_id on wire. | |
| Defer to Phase 6 reporting harness | Phase 2 headline-only; per-group is post-processing. Loses live visibility. | |

**User's choice:** Client-side, included in extras
**Notes:** Captured as D-22 in CONTEXT.md. Because Q2 picked strict schema, D-22 implies extending FitMetricsContract with per-group keys (NOT free-form extras) — Phase 2 Plan 01 extends CR-4 contract.

---

## Area 3: Per-client memory model

### Q1: How should each client handle the 6040-row user embedding matrix under cross-device (1 user = 1 client)?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep full 6040×d matrix (Recommended) | Preserves 'all params GLOBAL' invariant. ~3MB/client. Simplest. | ✓ |
| Single-user row fast path | Client gets only its user's row; server gathers. Memory-optimal; violates baseline invariant. | |
| Sparse update — full on downlink, row-only on uplink | Keep full matrix client-side, return only updated row. Best comm savings without breaking contract. | |

**User's choice:** Keep full 6040×d matrix
**Notes:** Captured as D-23 in CONTEXT.md. Sparse-update option filed under Deferred Ideas.

### Q2: How should the baseline's client handle gradient masking so only the assigned user's row moves?

| Option | Description | Selected |
|--------|-------------|----------|
| Zero non-user gradients post-backward (Recommended) | After loss.backward(), zero user_embeddings.weight.grad except assigned user's row. 1 line, optimizer-agnostic. | ✓ |
| Sparse embedding + .sparse=True | Switch to nn.Embedding(sparse=True). Adam won't work; overhead for negligible speedup at N=1. | |
| Index-select + fresh parameter per round | Create fresh 1×d nn.Parameter per round. Most memory-efficient; complicates state_dict. | |

**User's choice:** Zero non-user gradients post-backward
**Notes:** Captured as D-24 in CONTEXT.md.

### Q3: Should pyproject.toml keep embedding-dim=128 as default, or sync with mode-locked value?

| Option | Description | Selected |
|--------|-------------|----------|
| Mode-locked via foundation — pyproject value fallback only (Recommended) | resolve_mode_defaults(mode) owns dim/lr/optimizer/epochs/rounds. Zero drift. | ✓ |
| pyproject.toml owns defaults; mode is advisory | pyproject is source of truth. Drift possible between modules. | |
| Both — pyproject echoes mode defaults, asserts match at startup | Documentation + safety. +3 lines of assertion per module. | |

**User's choice:** Mode-locked via foundation — pyproject fallback only
**Notes:** Captured as D-25 in CONTEXT.md.

---

## Area 4: Result artifact shape

### Q1: Where should selected client IDs per round live?

| Option | Description | Selected |
|--------|-------------|----------|
| Embedded in result JSON + W&B step log (Recommended) | JSON gets selected_clients_per_round field; W&B gets per-step log. Reproducible + queryable. | ✓ |
| Manifest-only | Lives in run_manifest.selected_clients_per_round. Lean JSON; manifest grows over training. | |
| Sidecar CSV | Separate CSV file. Easy to grep; splits audit trail. | |

**User's choice:** Embedded in result JSON + W&B step log
**Notes:** Captured as D-26 in CONTEXT.md.

### Q2: What should the default checkpoint-rule be for the baseline?

| Option | Description | Selected |
|--------|-------------|----------|
| best_round_restore by default (Recommended) | Server tracks sampled_ndcg@10, saves params at best round, restores at end. Matches PFedRec reproduction target. | ✓ |
| last by default | Report final-round metrics verbatim. Simplest; breaks cross-module comparability. | |
| Configurable — no default, set per mode | resolve_mode_defaults(mode) picks; benchmark=best_round_restore, cross_silo_legacy=last. | |

**User's choice:** best_round_restore by default
**Notes:** Captured as D-27 in CONTEXT.md. Interpretation: defensibility choice, not performance choice.

### Q3: Where should cross-device result JSONs live?

| Option | Description | Selected |
|--------|-------------|----------|
| Flat — same results/federated/, differentiated by manifest.mode (Recommended) | No directory churn. Queries filter by _manifest.mode. Matches Phase 1 D-15. | ✓ |
| Sub-folder by mode | results/federated/cross-device/ vs /cross-silo/. Cleaner visual; harder to compare. | |
| Sub-folder by module AND mode | results/federated/baseline-cf/cross-device/. Most hierarchical; complicates Phase 5 reproduction. | |

**User's choice:** Flat
**Notes:** Captured as D-28 in CONTEXT.md.

---

## Final check

**Q: Ready to write CONTEXT.md?**

| Option | Description | Selected |
|--------|-------------|----------|
| Create context (Recommended) | Write CONTEXT.md + DISCUSSION-LOG.md; exit for user to run /gsd:plan-phase 2. | ✓ |
| Explore more gray areas | Identify 2-4 additional Phase-2 gray areas. | |
| Revisit an area | Re-open one of the four areas. | |

**User's choice:** Create context

---

## Claude's Discretion (folded into CONTEXT.md `<decisions>` as Claude's Discretion subsection)

- RNG purpose names for client sampling (`client_sampling`) and batch shuffling (`batch_shuffle` or via torch.Generator)
- Test coverage strategy for Phase 2 (default: pytest-style tests inside `federated-baseline-cf/tests/` mirroring Phase 1 layout)
- Placement of `BaselineFedAvg` class (default: `federated-baseline-cf/federated_baseline_cf/strategy.py`)
- Manifest `git-commit` value semantics when working tree is dirty (default: record HEAD sha + `dirty: true` flag)
- Loud-warning wording for CLI overrides (default: `"⚠ OVERRIDE: <key>=<value> (mode default=<default>). Run is NOT comparable to benchmark thesis table."`)
- Best-round checkpoint storage location (default: in-memory only)

## Deferred Ideas (filed in CONTEXT.md `<deferred>`)

- Full `fedrec_common/` refactor (deferred to v2 per PROJECT.md)
- Per-group evaluator W&B dashboards (Phase 6 concern)
- Best-round checkpoint crash-resilience (future; default in-memory is sufficient for single-machine simulation)
- `cross_silo_legacy` regression tests (nice-to-have; can become a dedicated plan if requested)
- Sparse-update `SparseFedAvg` aggregator (rejected in favor of full-matrix; revisit post-thesis if comm cost is paper-worthy)
- Single-user fast-path model (rejected — violates baseline invariant)
- Unified test runner across all four modules (post-Phase-2 nice-to-have)
