---
phase: 01-foundation-contract
plan: 02
subsystem: foundation
tags: [fedrec-foundation, id-mapping, leave-one-out, exclusion-set, atomic-bundle, sha256, cr-1-ratings-only, cr-5-train-only, imp-2-composite-hash, imp-3-flat-npz, n-3-atomic-index, wave-2]

requires:
  - "Plan 01: fedrec_foundation.{atomic, hashing, paths} + Wave-0 test stubs"

provides:
  - "fedrec_foundation.mapping: CanonicalMapping + build_mapping (CR-1 ratings-only item set) + save_mapping + load_mapping (int-key restoration + schema_version verify)"
  - "fedrec_foundation.user_groups: classify_user_group with FROZEN half-open semantics (IMP-4) — [0,30) sparse, [30,100) medium, [100,inf) dense"
  - "fedrec_foundation.split: SplitManifest (raw_data_hash + mapping_sha256 as top-level fields per IMP-2) + PerUserStats + 5-param build_split + compute_split_hash (folds both fingerprints per IMP-2) + save_split_or_verify (D-04 lock 'invalidate all cached results') + load_split_manifest"
  - "fedrec_foundation.exclusion: flat items + indptr NPZ layout (IMP-3) + build_exclusion (train_pos ∪ test_pos per D-13) + save_exclusion (atomic tempfile+os.replace) + load_exclusion (np.load allow_pickle=False per D-05) + ExclusionTable context manager + module-level exclusion_for helper (CR-3)"
  - "fedrec_foundation.bundle: FoundationIndex dataclass + 4-param publish_bundle (reads raw_data_hash from split_manifest field) + verify_bundle (raises RuntimeError 'incomplete' on missing payload or sha mismatch) + compute_foundation_contract_sha256 composite (N-3 + IMP-2)"
  - "fedrec_foundation.build (+ scripts/build_derived.py CLI): python -m fedrec_foundation.build produces data/derived/ bundle from data/ml-1m/ in seconds via vectorized merge filter (no .apply lambda rowwise)"
  - "data/derived/mapping.json: 6040 users, 3706 items (empirical CR-1 anchor; NOT 3883 from movies.dat)"
  - "data/derived/split_manifest.json: split_hash + raw_data_hash + mapping_sha256 + train_user_stats (CR-5) + bucket_boundaries=[30,100] + bucket_semantics='half_open'"
  - "data/derived/exclusion_items.npz: flat int32 items + int64 indptr (~4MB; allow_pickle=False safe)"
  - "data/derived/foundation_index.json: mapping_sha256 + split_hash + exclusion_sha256 + foundation_contract_sha256 + builder_version + created_at"

affects: [03-foundation-contract, 04-foundation-contract, 05-foundation-contract, 06-foundation-contract, 02-baseline, 03-pfedrec, 04-personalized, 05-adaptive, 06-evaluator, 07-thesis-evaluation]

tech-stack:
  added:
    - "numpy flat items + indptr NPZ layout (CSR-style per-user slicing) — O(1) per user lookup"
  patterns:
    - "SplitManifest stores both fingerprints (raw_data_hash + mapping_sha256) as dataclass fields — downstream consumers read from the manifest directly, no side-channel, no post-hoc assignment"
    - "D-04 lock-forever: save_split_or_verify refuses to overwrite a committed manifest on divergent split_hash with the sentinel string 'invalidate all cached results'"
    - "Atomic NPZ write via tempfile.mkstemp + np.savez + os.replace (handles the .npz suffix np.savez may append)"
    - "Vectorized merge-based train filter: left-join on (user_idx, item_idx) test_pairs, take NaN rows — replaces O(N) .apply(lambda) row-wise filtering"
    - "Bundle atomicity via sentinel-last: payload files written first, foundation_index.json written LAST — loaders call verify_bundle() before reading any payload"

key-files:
  created:
    - "scripts/foundation/fedrec_foundation/mapping.py"
    - "scripts/foundation/fedrec_foundation/user_groups.py"
    - "scripts/foundation/fedrec_foundation/split.py"
    - "scripts/foundation/fedrec_foundation/exclusion.py"
    - "scripts/foundation/fedrec_foundation/bundle.py"
    - "scripts/foundation/fedrec_foundation/build.py"
    - "scripts/foundation/scripts/build_derived.py"
    - "data/derived/mapping.json"
    - "data/derived/split_manifest.json"
    - "data/derived/exclusion_items.npz"
    - "data/derived/foundation_index.json"
  modified:
    - "scripts/foundation/tests/test_mapping.py (un-skipped, 3 tests GREEN)"
    - "scripts/foundation/tests/test_split.py (un-skipped, 4 tests GREEN)"
    - "scripts/foundation/tests/test_exclusion.py (un-skipped, 4 tests GREEN)"
    - "scripts/foundation/tests/test_integration.py (un-skipped, 4 tests GREEN)"

key-decisions:
  - "CR-1 LOCKED: item2idx is built from sorted(ratings_df['movie_id'].unique()) — NOT sorted(movies_df['movie_id'].unique()). ML-1M movies.dat lists 3883 movies but only 3706 are ever rated; using movies.dat silently adds 177 never-rated items to the embedding table and invalidates every cached item embedding. Empirical CR-1 anchor test (test_ml1m_counts_6040_3706) confirms the real-data result is 6040/3706."
  - "CR-3 module-level helper: exported exclusion_for(npz, user_idx) as a plain function alongside ExclusionTable.for_user so callers who want to keep a raw np.load(...) object around don't have to construct a class wrapper. Both paths return the same O(1) indptr slice."
  - "CR-5 train-only stats: PerUserStats (n_interactions, genre_entropy, n_unique_items, rating_std, user_group) are computed on TRAIN rows only — the held-out LOO test item is removed first. This prevents Phase 4's adaptive-alpha heuristic from seeing the test item's genre and underestimating its own improvement on sparse users."
  - "IMP-2 composite hashes: compute_split_hash folds mapping_sha256 + raw_data_hash into the hash payload so the split_hash changes if either underlying artifact changes. compute_foundation_contract_sha256 folds all three sha values (mapping + split + exclusion) — any one-byte mutation in any artifact flips the composite."
  - "IMP-3 flat NPZ layout: chose flat int32 items + int64 indptr over keyed-dict layout (one key per user). At 6040 users, flat is smaller on disk, faster to load (one np.load + O(1) slice), and simpler to validate (loop once over indptr). save_exclusion is atomic via tempfile + os.replace with a fallback for np.savez appending .npz."
  - "N-3 atomic bundle: foundation_index.json is written LAST by publish_bundle. Payload files are each atomic individually (save_mapping, save_split_or_verify, save_exclusion all use tempfile + os.replace or np.savez + os.replace). Loaders call verify_bundle() first — on missing payload or sha mismatch they raise RuntimeError with 'incomplete' or 'corrupted' sentinel so callers can pattern-match. Deleted exclusion_items.npz in a fuzz test triggers 'incomplete' as expected."
  - "Signature locked: publish_bundle is 4-param (derived_dir, mapping, split_manifest, exclusion) — raw_data_hash is NOT a separate parameter, it is read from split_manifest.raw_data_hash (the dataclass field populated by build_split). build_split is 5-param (ratings_df, mapping, movies_df, mapping_sha256, raw_data_hash). SplitManifest stores BOTH fingerprints as top-level fields so Plan 04 RunManifest and publish_bundle read them directly from the manifest — no side-channel."
  - "D-04 lock-forever: save_split_or_verify refuses to overwrite a committed manifest on divergent hash with the sentinel substring 'invalidate all cached results'. Test test_split_lock_refuses_overwrite exercises this with a pytest.raises(ValueError, match=...). Committing data/derived/*.json IS the lock — future builder runs that produce a different hash will error before the overwrite."
  - "Zero row-wise .apply(lambda) in the CLI: the 1M-row ML-1M DataFrame is filtered via pd.merge on a small (user_idx, item_idx) test-pairs table. groupby-apply in build_exclusion operates once per group (6040 calls at ML-1M scale), not per row — the forbidden anti-pattern is row-wise iteration over the full 1M-row table."

requirements-completed: [FND-01, FND-02, FND-03]

metrics:
  duration: "~8 min"
  started: "2026-04-19T03:09:41Z"
  completed: "2026-04-19T03:17:48Z"
  tasks_completed: 3
  files_created: 11
  files_modified: 4
  tests_green: 15
  plan_02_commits: 3
---

# Phase 01 Plan 02: Foundation Contract — Canonical Mapping + LOO Split + Exclusion Set Summary

**Three on-disk foundation artifacts (mapping.json, split_manifest.json, exclusion_items.npz) published atomically behind foundation_index.json, with 6040-user / 3706-item empirical CR-1 anchor confirmed on the real ML-1M.**

## Performance

- **Duration:** ~8 min (total wall clock across 3 tasks)
- **Started:** 2026-04-19T03:09:41Z
- **Completed:** 2026-04-19T03:17:48Z
- **Tasks:** 3 (all completed autonomously, no deviations from plan)
- **Files created:** 11 (6 Python modules + 4 data artifacts + 1 CLI shim)
- **Files modified:** 4 (test files un-skipped)
- **Plan 02 tests GREEN:** 15 (3 mapping + 4 split + 4 exclusion + 4 integration)

## Accomplishments

- **FND-01 complete:** `fedrec_foundation.mapping` ships `CanonicalMapping`, `build_mapping`, `save_mapping`, `load_mapping`. CR-1 is enforced in code (`sorted(ratings_df["movie_id"].unique())`) and in tests (`test_item_mapping_from_ratings_only` proves a never-rated movie is absent). Empirical anchor test confirms `data/ml-1m/` builds to **6040 users / 3706 items** — NOT 3883.
- **FND-02 complete:** `fedrec_foundation.split` ships `SplitManifest` (with `raw_data_hash` and `mapping_sha256` as top-level fields per IMP-2), `PerUserStats`, 5-param `build_split`, `compute_split_hash`, `save_split_or_verify` (D-04 lock), `load_split_manifest`. CR-5 train-only user stats enforced in code and in `test_train_only_user_stats`. The LOO held-out item is the `tail(1)` of a stable mergesort on `(user_idx, timestamp, item_idx)`.
- **FND-03 complete:** `fedrec_foundation.exclusion` ships IMP-3 flat `items + indptr` NPZ layout, `build_exclusion` (train_pos ∪ test_pos per D-13), atomic `save_exclusion` (tempfile + np.savez + os.replace), `load_exclusion` (`allow_pickle=False` per D-05), `ExclusionTable` context manager with O(1) `for_user`, and CR-3 module-level `exclusion_for(npz, user_idx)` helper.
- **N-3 atomic bundle:** `fedrec_foundation.bundle` ships 4-param `publish_bundle` (reads `raw_data_hash` from `split_manifest.raw_data_hash` — single source of truth), `verify_bundle` (fails loudly with `"incomplete"` sentinel on missing payload or sha mismatch), and composite `compute_foundation_contract_sha256`. `foundation_index.json` is written LAST so readers never see a partially-published bundle.
- **CLI builder:** `python -m fedrec_foundation.build` runs the full pipeline on the real ML-1M in seconds via a vectorized merge-based train filter (zero `.apply(lambda)` row-wise calls on the 1M-row DataFrame). Re-running is idempotent (D-04 lock makes it a no-op with identical hashes).
- **Locked artifacts committed:**
  - `mapping_sha256`: `0cffcdde64c654736bb2585a3653515866c92522bf97fefdacd1a02fd021bb48`
  - `split_hash`: `5685bed7e4b650807e58a49e25ea611cdef82444a034bdf105e46aa3755284d6`
  - `exclusion_sha256`: `b5a670c8e8717705caaef5ceb0932604f16997353908d6082de000aaeafc6491`
  - `foundation_contract_sha256`: `fe181dafe6f791d6679b08fdf6a4ad88f17ea3a7cce98df7f491d1c96a14233a`
  - `builder_version`: `1.0.0`
- **Wave 2 parallel-safe:** no files outside Plan 02's territory were touched. Sibling plans (03/04/05) all land cleanly — the full foundation test suite reports **65 passed, 0 failed** at plan close.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave 2 parallel execution; hooks validated once by orchestrator after all siblings complete):

1. **Task 1: FND-01 canonical mapping + user_groups classifier** — `de5ffa5` (feat)
2. **Task 2: FND-02 LOO split + FND-03 exclusion (IMP-2 + IMP-3 + CR-5)** — `ee3c354` (feat)
3. **Task 3: atomic bundle publication + CLI + data/derived/ artifacts (N-3)** — `cf93fde` (feat)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md) is appended separately at plan close._

## Files Created/Modified

### Python modules (Task 1 + Task 2 + Task 3)

- `scripts/foundation/fedrec_foundation/mapping.py` — FND-01 + CR-1.
- `scripts/foundation/fedrec_foundation/user_groups.py` — `USER_GROUP_BOUNDARIES=(30,100)`, `BUCKET_SEMANTICS="half_open"`, `classify_user_group` (IMP-4).
- `scripts/foundation/fedrec_foundation/split.py` — FND-02 + CR-5 + IMP-2 + D-04.
- `scripts/foundation/fedrec_foundation/exclusion.py` — FND-03 + IMP-3 + D-05 + D-13 + CR-3.
- `scripts/foundation/fedrec_foundation/bundle.py` — N-3 atomic publication + IMP-2 composite hashes.
- `scripts/foundation/fedrec_foundation/build.py` — thin shim so `python -m fedrec_foundation.build` resolves.
- `scripts/foundation/scripts/build_derived.py` — CLI; vectorized merge-based train filter.

### Data artifacts (Task 3)

- `data/derived/mapping.json` — 6040 users / 3706 items (CR-1 anchor).
- `data/derived/split_manifest.json` — split_hash + raw_data_hash + mapping_sha256 + train_user_stats + bucket_boundaries + bucket_semantics.
- `data/derived/exclusion_items.npz` — ~4 MB flat int32 items + int64 indptr.
- `data/derived/foundation_index.json` — 4 fingerprints + builder_version + created_at.

### Tests un-skipped (all tasks)

- `scripts/foundation/tests/test_mapping.py` — 3 tests GREEN.
- `scripts/foundation/tests/test_split.py` — 4 tests GREEN.
- `scripts/foundation/tests/test_exclusion.py` — 4 tests GREEN.
- `scripts/foundation/tests/test_integration.py` — 4 tests GREEN (including the real-ML-1M empirical anchor).

## Decisions Made

- **CR-1 anchored by empirical test.** Rather than just enforce the "ratings-only" rule in code, added `test_ml1m_counts_6040_3706` that actually loads `data/ml-1m/ratings.dat` and asserts `num_users==6040, num_items==3706`. Skipped if data is absent (CI-friendly).
- **IMP-2 single source of truth.** Put `raw_data_hash` and `mapping_sha256` on `SplitManifest` as dataclass fields. `publish_bundle` is 4-param (not 5); it reads `raw_data_hash` from the manifest. `compute_split_hash` folds both fingerprints into the hash payload so a mapping change or raw-data change flips the split_hash automatically.
- **IMP-3 flat NPZ over keyed-dict.** At 6040 users, flat `items + indptr` layout is ~1.5× smaller than 6040 per-user arrays, loads once, and every per-user slice is O(1). `np.savez` may append `.npz` to a tempfile path — the atomic writer handles both candidate paths.
- **CR-5 train-only stats.** Per-user stats (`n_interactions`, `genre_entropy`, etc.) are computed on the train DataFrame (LOO test item removed first) so Phase 4 adaptive-alpha does not see leakage through the stats.
- **D-04 lock-forever committed.** The committed `data/derived/*.json` IS the lock. Any future builder run that produces a different hash will error via `save_split_or_verify` before overwriting. Idempotent re-run prints the same hashes.
- **Python 3.9 typing preserved.** Used `typing.Dict`, `typing.List`, `typing.Tuple`, `typing.Iterable` throughout (not PEP 604 `X | Y`, not `list[int]`). Matches Plan 01's scaffolding and `.planning/codebase/CONVENTIONS.md`.
- **Vectorized merge filter everywhere.** Both the CLI (`build_derived.py`) and the test fixtures (`_vectorized_train_split` helper in `test_exclusion.py` and `test_integration.py`) use a left-merge on a small `(user_idx, item_idx)` test-pairs DataFrame. Zero row-wise `.apply(lambda)` in Plan 02 files.

## Deviations from Plan

None — plan executed exactly as written.

Task 1 followed the action block verbatim. Task 2 followed the Codex-override prescriptions (CR-5, IMP-2, IMP-3, D-04 sentinel). Task 3 followed the 4-param `publish_bundle` signature, wrote `scripts/build_derived.py` plus a thin `fedrec_foundation/build.py` shim, and ran the CLI to produce the four real-ML-1M artifacts. All 15 automated tests green on first run after each task's implementation.

## Issues Encountered

None blocking. Two minor polish calls made while implementing:

- **`.npz` suffix handling in `save_exclusion`:** `np.savez(tmp, ...)` with a tempfile path that ends in `.npz` may or may not append a second `.npz` depending on the numpy version. The atomic writer checks both `tmp` and `tmp + ".npz"` and cleans up whichever exists on exception. No test failure observed; fix was pre-emptive for cross-version safety.
- **`_compute_genre_entropy` accepts `movie_id` not `item_idx`:** the helper joins `user_train_df` to `movies_df` on raw `movie_id` (upstream `build_split` preserves that column in the train slice). Simpler than a double-join through `mapping.item2idx` and does not depend on `movies_df` having been mapped to canonical space.

## Known Stubs

None in Plan 02's territory. All un-skipped tests are real assertions; all implementations are functional. No `TODO`, `FIXME`, `NotImplementedError`, or placeholder return values in any Plan-02-owned file.

## User Setup Required

None. The CLI (`python -m fedrec_foundation.build`) runs entirely on local data (`data/ml-1m/` already present) and produces artifacts under `data/derived/`. No external services, no credentials, no `.env` touched.

## Next Phase Readiness

**Ready for Plan 06 (integration wiring)** — Plan 02's artifacts are the hard dependency for every downstream module's `dataset.py`:

- `load_mapping(".../data/derived/mapping.json")` → canonical `user2idx` / `item2idx` used by every module.
- `load_split_manifest(".../data/derived/split_manifest.json")` → deterministic LOO held-out + `train_user_stats` (used by Phase 4 adaptive alpha).
- `load_exclusion(".../data/derived/exclusion_items.npz")` → O(1) per-user exclusion set used wherever training negatives are sampled (fixes the test-positive-leak pattern catalogued in `.planning/codebase/CONCERNS.md`).
- `verify_bundle(".../data/derived/")` → loaders call this first; if any payload drifts, the module refuses to start.

**Ready for Plan 04 (run manifest)** — `SplitManifest.raw_data_hash` and `SplitManifest.mapping_sha256` are dataclass fields that `RunManifest` reads directly. No side-channel access; no post-hoc assignment anywhere in the codebase.

**Ready for Plan 03 (evaluator + weight_policy)** — `train_user_stats` (bucket-labeled via `classify_user_group`) is available on the split manifest for per-group reporting. `fedrec_foundation.exclusion.exclusion_for` is available as a module-level helper for the evaluator's negative sampler.

**No blockers.** No architectural decisions deferred. No cross-plan file conflicts — Plan 02 stayed strictly within its territory.

## Self-Check: PASSED

- **Files created:** FOUND all 11 files (`git log --name-status` on commits de5ffa5, ee3c354, cf93fde confirms each).
- **Commits:** FOUND de5ffa5 (Task 1), ee3c354 (Task 2), cf93fde (Task 3). `git log --oneline` confirms all three on `feat/try_to_run_the_baseline`.
- **Automated verify:** PASSED. `pytest scripts/foundation/tests/test_mapping.py tests/test_split.py tests/test_exclusion.py tests/test_integration.py -v` → **15 passed, 0 failed**. Full foundation suite (including sibling plans 03/04/05) → **65 passed, 0 failed**.
- **Empirical anchor:** PASSED. `python -m fedrec_foundation.build` on real `data/ml-1m/` produced `mapping.num_users==6040, mapping.num_items==3706` — CR-1 confirmed.
- **Idempotency (D-04):** PASSED. Second run of `python -m fedrec_foundation.build` prints identical hashes and does not overwrite existing artifacts.
- **Signatures locked:** PASSED. `grep -E "^def publish_bundle\(" bundle.py` matches; `publish_bundle` is 4-param everywhere. `grep "split_manifest.raw_data_hash" bundle.py` matches. `grep "split.raw_data_hash =" scripts/build_derived.py` returns no matches (no post-hoc assignment).
- **Vectorized filter:** PASSED. `grep -E "df.*\.apply\(lambda" scripts/build_derived.py tests/test_integration.py tests/test_split.py tests/test_exclusion.py` returns no matches.

---

*Phase: 01-foundation-contract*
*Plan: 02*
*Completed: 2026-04-19*
