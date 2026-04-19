---
phase: 01-foundation-contract
plan: 04
subsystem: infra
tags: [rng, sha256, reproducibility, manifest, foundation-contract, cr-3, imp-2, d-15, d-16, wave-2]

requires:
  - "01-foundation-contract-01 (fedrec-foundation package scaffold + atomic_write_json + test harness)"
provides:
  - "fedrec_foundation.rng: _derive_seed (sha256, full 256-bit digest), py_rng, np_rng, torch_gen, server_rng, derive_rng (back-compat alias), _ALLOWED_PURPOSES={train_neg, eval_neg, model_init, server_sample, dataloader}"
  - "Four-tier RNG namespace system (py / np / torch) — same (run_seed, user_idx, round_num, purpose) tuple produces independent streams across three factories"
  - "fedrec_foundation.manifest: RunManifest dataclass (23 fields), RUN_MANIFEST_SCHEMA_VERSION=1, generate_run_id (YYYYMMDD-HHMMSS-<6hex> UTC), build_run_manifest (duck-typed ModeProfile), write_manifest_sibling (atomic), embed_manifest_in_result (mutates + returns)"
  - "IMP-2 composite hash carriage: mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256 — all four travel in every manifest"
  - "D-15 double-write contract: _manifest embedded in result JSON AND <run_id>-manifest.json sibling — belt-and-suspenders for audit/reproduction"
  - "16 tests GREEN (10 test_rng + 6 test_manifest); 4 new behavioral tests added beyond the stub enumeration (dataloader order, atomic-write no-leftovers, fluent return, run_id format)"
affects: [02-foundation-contract, 03-foundation-contract, 05-foundation-contract, 06-foundation-contract]

tech-stack:
  added: []
  patterns:
    - "SHA-256 seed derivation with namespace prefix — cross-process-stable, PYTHONHASHSEED-immune (CR-3 correctness)"
    - "Duck-typed ModeProfile at the manifest/mode boundary — defers Plan 04 unblocking on Plan 05's dataclass"
    - "Double-write manifest (embedded + sibling) — D-15 guarantees at least one artifact survives partial failure"
    - "torch.Generator for EVERY DataLoader(..., generator=..., shuffle=True) — CR-3 fourth reproducibility assertion"

key-files:
  created:
    - "scripts/foundation/fedrec_foundation/rng.py"
    - "scripts/foundation/fedrec_foundation/manifest.py"
  modified:
    - "scripts/foundation/tests/test_rng.py"
    - "scripts/foundation/tests/test_manifest.py"

key-decisions:
  - "Used hashlib.sha256 with the FULL 256-bit integer (not truncated to 8 bytes) per Codex N-1 — torch.Generator.manual_seed needs the value mod'd into int64 positive range, but py_rng / np_rng accept arbitrary ints so the full digest flows through unchanged."
  - "Namespace prefix (py / np / torch) inside the sha256 payload — guarantees that py_rng(s, u, r, p) and np_rng(s, u, r, p) produce INDEPENDENT streams even with identical tuple inputs, preventing a subtle reproducibility trap where two generators unintentionally share a stream."
  - "Added 'dataloader' and 'server_sample' to _ALLOWED_PURPOSES up-front — CR-3 mandates DataLoader(generator=torch_gen(..., 'dataloader')); doing it now avoids a later ValueError and makes the contract visible in one place."
  - "Duck-typed mode_profile parameter (typed as Any) in build_run_manifest — Plan 05 ships the real ModeProfile dataclass; Plan 04 would deadlock on a circular import otherwise. The stub test class in test_manifest.py demonstrates the required attribute surface."
  - "generate_run_id uses UTC (datetime.now(timezone.utc)) not local time — run-IDs are sortable and globally comparable across collaborators in different timezones."
  - "embed_manifest_in_result mutates AND returns the dict — enables fluent use `json.dump(embed_manifest_in_result(m, result), f)` while still being safe if caller doesn't capture the return."

patterns-established:
  - "Namespaced sha256 seed derivation: `_derive_seed(namespace, run_seed, user_idx, round_num, purpose)` with payload `f'{namespace}:{run_seed}:{user_idx}:{round_num}:{purpose}'.encode('ascii')`. Reusable anywhere else in the codebase that needs cross-process-stable int seeds."
  - "Subprocess reproducibility test (PYTHONHASHSEED=0/1/random) — the canonical way to prove a hash-based seed is stable across fresh interpreters. test_rng.py::test_derive_rng_stable_across_processes is the template."
  - "_ALLOWED_PURPOSES frozenset at the top of rng.py — every downstream module that adds a new sampling use case must add its purpose string here or hit a ValueError. Keeps the set of sampling intents discoverable."
  - "Duck-typing at module boundaries where two modules would otherwise form a cycle (Plan 04 ↔ Plan 05). `mode_profile: Any` plus a docstring listing required attributes."
  - "Double-write critical artifacts (embedded + sibling) via D-15 — any downstream writer that stores run-scoped metadata should follow this pattern."

requirements-completed: [FND-06, FND-07]

duration: "4 min"
completed: "2026-04-19"
---

# Phase 01 Plan 04: FND-06 RNG Factories + FND-07 Run Manifest Summary

**Four-tier reproducibility backbone: three sha256-namespaced RNG factories (`py_rng` / `np_rng` / `torch_gen`) plus a 23-field `RunManifest` carrying all four IMP-2 foundation fingerprints with double-write (embedded + sibling) per D-15.**

## Performance

- **Duration:** ~4 min (247 seconds)
- **Started:** 2026-04-19T03:10:53Z
- **Completed:** 2026-04-19T03:15:00Z
- **Tasks:** 2 (both completed autonomously via TDD RED → GREEN)
- **Files created:** 2 (`rng.py`, `manifest.py`)
- **Files modified:** 2 (un-skipped `test_rng.py`, `test_manifest.py`)
- **Tests:** 16 GREEN (10 rng + 6 manifest) — up from 8 stubs; 0 skipped, 0 failed
- **All foundation tests:** 61 passed, 4 skipped (Plan 02+06 integration stubs), 0 failed

## Accomplishments

- **FND-06 three RNG factories land.** `py_rng`, `np_rng`, `torch_gen` all derive from the same `_derive_seed` with a per-factory namespace prefix. A subprocess test with `PYTHONHASHSEED=0,1,random` proves the seeds are byte-identical across fresh Python processes — the CR-3 anchor criterion.
- **FND-06 coverage expanded beyond the plan stubs.** Plan required 5 tests (FND-06-a..e); 10 ship. The extras codify the DataLoader iteration-order assertion, the `ValueError` on unknown purpose, the presence of `dataloader` + `server_sample` in `_ALLOWED_PURPOSES`, top-level `server_rng` reproducibility, and the `derive_rng` back-compat alias.
- **FND-07 run manifest lands with all 23 fields.** 13 mode/config (D-16) + 6 foundation fingerprints (IMP-2) + 2 metadata + 2 bookkeeping. `foundation_contract_sha256` rides alongside `mapping_sha256`, `split_hash`, `exclusion_sha256` per IMP-2 so a one-byte mutation to any foundation input is detectable at the run-manifest level.
- **D-15 double-write verified.** `embed_manifest_in_result` inserts `_manifest` into the result JSON AND `write_manifest_sibling` publishes `<run_id>-manifest.json` via `atomic_write_json` — no `.tmp-*` leftovers, no partial sibling on crash.
- **Duck-typed ModeProfile boundary.** Plan 04 does not import `fedrec_foundation.mode` (Plan 05 territory); the `mode_profile: Any` parameter in `build_run_manifest` plus a `_StubProfile` in the test carries the contract without a cyclical dependency.
- **Every acceptance criterion met.** `grep "hashlib.sha256"` / `grep "dataloader"` / `grep "def torch_gen"` in `rng.py` — all hit. `grep "foundation_contract_sha256"` / `grep "mapping_sha256"` / `grep "exclusion_sha256"` in `manifest.py` — all hit. Smoke-test `python -c "from fedrec_foundation.rng import py_rng; print(py_rng(42, 1, 0, 'train_neg').random())"` prints stable `0.06309475747172966`; `generate_run_id()` prints `20260419-031427-72bc02`.

## Task Commits

Each TDD cycle produced two commits (RED + GREEN):

**Task 1: FND-06 three RNG factories**
1. `d0eb4b1` — test(01-04): add failing tests for FND-06 RNG factories (CR-3)
2. `7003701` — feat(01-04): implement FND-06 three RNG factories (CR-3)

**Task 2: FND-07 run manifest**
3. `e8ad8a4` — test(01-04): add failing tests for FND-07 run manifest (IMP-2 + D-15)
4. `4ca6049` — feat(01-04): implement FND-07 run manifest with IMP-2 composite hashes

_Note: Plan metadata commit (SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md) is appended separately at the end of execution._

No REFACTOR commits were needed — both GREEN implementations were already minimal and well-structured thanks to the detailed `<action>` blocks in the plan.

## Files Created/Modified

### `scripts/foundation/fedrec_foundation/rng.py` (new)
- `_ALLOWED_PURPOSES = frozenset({"train_neg", "eval_neg", "model_init", "server_sample", "dataloader"})` — closed set. Unknown purpose raises `ValueError`.
- `_derive_seed(namespace, run_seed, user_idx, round_num, purpose) -> int` — sha256-based, FULL 256-bit digest (N-1), namespace-prefixed payload.
- `py_rng(run_seed, user_idx, round_num, purpose) -> random.Random` — namespace `"py"`.
- `np_rng(run_seed, user_idx, round_num, purpose) -> numpy.random.Generator` — namespace `"np"`, via `np.random.default_rng`.
- `torch_gen(run_seed, user_idx, round_num, purpose) -> torch.Generator` — namespace `"torch"`, seed mod'd into int64 positive range.
- `server_rng(run_seed) -> random.Random` — top-level server client-sampler RNG.
- `derive_rng(...)` — back-compat alias for `py_rng` matching the research file's earlier exposition.

### `scripts/foundation/fedrec_foundation/manifest.py` (new)
- `RUN_MANIFEST_SCHEMA_VERSION = 1`.
- `@dataclass class RunManifest` — 23 fields: `schema_version`, `run_id`, `mode`, `num_supernodes`, `partition_mode`, `fraction_train`, `fraction_eval`, `weight_policy`, `primary_evaluator`, `num_train_negatives`, `num_eval_negatives`, `run_seed`, `checkpoint_rule`, `mapping_sha256`, `split_hash`, `exclusion_sha256`, `foundation_contract_sha256`, `raw_data_hash`, `builder_version`, `overrides`, `module`, `flwr_version`, `torch_version`, `git_commit`.
- `generate_run_id() -> str` — `"{YYYYMMDD}-{HHMMSS}-{uuid4_hex[:6]}"` (UTC).
- `_git_commit() -> str` — best-effort `git rev-parse HEAD`; returns `"unknown"` on failure.
- `build_run_manifest(run_id, mode_profile, run_seed, mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256, raw_data_hash, builder_version, overrides, module) -> RunManifest` — reads `flwr.__version__`, `torch.__version__`, git commit internally.
- `write_manifest_sibling(manifest, result_json_path) -> Path` — D-15 sibling via `atomic_write_json`.
- `embed_manifest_in_result(manifest, result_dict) -> dict` — D-15 embedded `_manifest` key; mutates in place AND returns for fluent use.

### `scripts/foundation/tests/test_rng.py` (modified — un-skipped + expanded)
- Module-level `pytestmark = pytest.mark.skip(...)` removed.
- All 5 original stub bodies replaced with real assertions (FND-06-a..e).
- 5 new tests added: `test_dataloader_iteration_order`, `test_unknown_purpose_raises`, `test_allowed_purposes_includes_dataloader`, `test_server_rng_reproducible`, `test_derive_rng_is_py_rng_alias`.
- Subprocess test updated to pass `namespace="py"` (matches CR-3 signature change from plan 01's stub).
- **10 tests GREEN.**

### `scripts/foundation/tests/test_manifest.py` (modified — un-skipped + expanded)
- Module-level `pytestmark = pytest.mark.skip(...)` removed.
- All 3 original stub bodies replaced (FND-07-a/b/c).
- 3 new tests added: `test_run_id_format`, `test_atomic_sibling_write`, `test_embed_returns_same_dict`.
- `_StubProfile` ModeProfile stand-in to avoid circular import with Plan 05's `mode.py`.
- **6 tests GREEN.**

## Decisions Made

- **Full 256-bit sha256 digest, not truncated.** Codex N-1 flagged that truncating to 8 bytes discards entropy unnecessarily. Torch's `manual_seed` refuses values `>= 2**63`, so only `torch_gen` mods into the int64 positive range — `py_rng` and `np_rng` consume the full int directly.
- **Namespace prefix inside the sha256 payload (`"py:{run_seed}:..."`), not a separate hashing pass.** One sha256 call per seed derivation; the namespace string is part of the payload bytes. Simpler and still cryptographically independent because sha256 is collision-resistant on arbitrary prefixes.
- **`_ALLOWED_PURPOSES` as a module-level `frozenset`.** Closed set — new sampling intents must be declared here or raise `ValueError`. `dataloader` and `server_sample` are explicitly included up-front to satisfy CR-3's DataLoader-seeding requirement and the top-level server client-sampler.
- **Duck-typed `mode_profile: Any` in `build_run_manifest`.** Plan 04 cannot import `fedrec_foundation.mode` (Plan 05 territory and would create a cyclical import path). Typing as `Any` plus a docstring listing required attributes (`mode`, `num_supernodes`, …, `checkpoint_rule`) documents the contract without the import dependency. Plan 05's real `ModeProfile` will satisfy the duck-type structurally.
- **`generate_run_id` uses UTC, not local time.** `datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")` — sortable, globally comparable, timezone-unambiguous. A run ID from a German collaborator lexically sorts correctly against a run ID from a Tokyo collaborator.
- **`embed_manifest_in_result` mutates AND returns the dict.** Enables fluent use `json.dump(embed_manifest_in_result(m, r), f)` while still being correct if the caller doesn't capture the return value. Tested explicitly (`test_embed_returns_same_dict`).
- **`_git_commit` is best-effort.** Returns `"unknown"` on any exception. Bare-export / CI / container contexts that lack `.git` should not block a run from recording a manifest — the `git_commit` field still captures the intent, and upstream deploy pipelines can set it via a manifest post-processor if a stable value matters.

## Deviations from Plan

**None — plan executed exactly as written.**

The `<action>` blocks in the plan file were detailed enough that both tasks were implemented verbatim. Minor additions (more tests than the plan's minimum) fall into the "deeper coverage, not scope change" bucket:

- Plan required 5 tests for FND-06; 10 ship. The extras directly codify the CR-3 fourth reproducibility assertion (DataLoader iteration order) and the `ValueError` on unknown purpose — both behaviors are in the `<behavior>` block of the plan but weren't enumerated as separate test bodies.
- Plan required 3 tests for FND-07; 6 ship. The extras codify the `generate_run_id` format, the atomic-write no-leftovers property, and the fluent-return API — all explicitly stated in the plan's `<behavior>` block.

No auto-fixes, no blocking issues, no architectural questions.

## Issues Encountered

None. TDD cycles both went RED → GREEN on the first implementation attempt:

- Task 1 RED: expected `ModuleNotFoundError: No module named 'fedrec_foundation.rng'` — confirmed.
- Task 1 GREEN: 10/10 tests passed on first run.
- Task 2 RED: expected `ModuleNotFoundError: No module named 'fedrec_foundation.manifest'` — confirmed.
- Task 2 GREEN: 6/6 tests passed on first run.
- Cross-plan regression check: 61/61 non-integration tests pass; the 4 skipped `test_integration.py` tests are Plan 02+06 territory, not touched.

## Known Stubs

**None.** Both modules ship fully implemented with no placeholder values, no TODO markers, no mock data. Every dataclass field is populated from either a real source (mode_profile attributes, `flwr.__version__`, `torch.__version__`, git commit) or a caller-supplied sha256 digest.

The `module: str` field in `RunManifest` is typed as a free-form str with the docstring constraint `"baseline" | "personalized" | "adaptive" | "pfedrec"` — not a stub; it is a caller-provided tag. Plan 05's launcher (`scripts/run.py`) is the authoritative caller that will set this per invocation.

## User Setup Required

None — every dependency (`torch`, `numpy`, `flwr`, `pandas`) was already available in the conda env verified at plan start (torch 2.9.1, numpy 2.2.6, flwr 1.24.0).

## Next Phase Readiness

Every Wave-3 consumer can now rely on the four-tier RNG + run-manifest contract:

- **Plan 03 (evaluator / weight policy)** — no dependency on this plan's output, but evaluator negative-sampling code MUST use `np_rng(run_seed, user_idx, round_num, "eval_neg")` going forward.
- **Plan 05 (mode resolver + launcher)** — `scripts/run.py` will construct a `ModeProfile`, pass it through `build_run_manifest(...)`, and call `embed_manifest_in_result` + `write_manifest_sibling` on every run.
- **Plan 06 (validation)** — can assert every `results/federated/*.json` carries a `_manifest` key with all 23 fields populated, and that a sibling `<run_id>-manifest.json` exists next to it.

**Downstream contract for Phases 2–5:** Every DataLoader instantiation that sets `shuffle=True` MUST pass `generator=torch_gen(run_seed, user_idx, round_num, "dataloader")`. This is the CR-3 fourth reproducibility assertion — without it, DataLoader worker shuffling is non-deterministic even with all three RNG factories seeded. Grep for `DataLoader(.*shuffle=True` in every `federated-*-cf/` module during Phase 2 migration and ensure the generator is passed.

**No blockers. No open questions.** The RNG and manifest contracts are frozen — any change requires incrementing `RUN_MANIFEST_SCHEMA_VERSION` and `_ALLOWED_PURPOSES` via a new phase plan.

## Self-Check: PASSED

- **Files created:**
  - FOUND: `scripts/foundation/fedrec_foundation/rng.py` (verified via git log + pytest import success).
  - FOUND: `scripts/foundation/fedrec_foundation/manifest.py` (verified via git log + pytest import success).
- **Files modified:**
  - FOUND: `scripts/foundation/tests/test_rng.py` (10 tests passing, pytestmark removed).
  - FOUND: `scripts/foundation/tests/test_manifest.py` (6 tests passing, pytestmark removed).
- **Commits:**
  - FOUND: `d0eb4b1` (Task 1 RED), `7003701` (Task 1 GREEN), `e8ad8a4` (Task 2 RED), `4ca6049` (Task 2 GREEN) — all four confirmed on `feat/try_to_run_the_baseline` via `git log --oneline -6`.
- **Automated verify:** PASSED.
  - `pytest tests/test_rng.py -v` → 10 passed, 0 failed.
  - `pytest tests/test_manifest.py -v` → 6 passed, 0 failed.
  - `pytest tests/ -v` → 61 passed, 4 skipped (Plan 02+06 integration stubs, out of scope), 0 failed.
  - Smoke: `python -c "from fedrec_foundation.rng import py_rng; print(py_rng(42, 1, 0, 'train_neg').random())"` → `0.06309475747172966` (stable, reproducible).
  - Smoke: `python -c "from fedrec_foundation.manifest import generate_run_id; print(generate_run_id())"` → `20260419-031427-72bc02` (correct format).

---

*Phase: 01-foundation-contract*
*Plan: 04*
*Completed: 2026-04-19*
