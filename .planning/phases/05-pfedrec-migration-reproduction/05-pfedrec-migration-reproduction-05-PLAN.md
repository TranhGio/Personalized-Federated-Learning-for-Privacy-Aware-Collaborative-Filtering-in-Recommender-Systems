---
phase: 05-pfedrec-migration-reproduction
plan: 05
type: execute
wave: 3
depends_on: ["05-pfedrec-migration-reproduction-03"]
files_modified:
  - scripts/foundation/tests/test_pfedrec_subprocess_determinism.py
autonomous: true
requirements: [PFR-06]
must_haves:
  truths:
    - "Subprocess regression guard runs `scripts/run.py pfedrec paper_compat_pfedrec` (or smaller smoke config) twice with same run-seed; asserts byte-identity on (a) selected_clients_per_round JSON field, (b) per-key torch.equal on each partition_{pid}.pt cache file (single key 'affine_output.weight'), (c) byte-identical _manifest.pfr08_verification field"
    - "Coverage guard scans at least one partition_{pid}.pt for the 'affine_output.weight' key; if checked_partitions > 0 but coverage_seen is False, pytest.fail with 'PFR-03 path not actually exercised by this run'"
    - "@pytest.mark.slow + FEDREC_SKIP_SLOW=1 escape hatch (Phase 2/3/4 precedent)"
    - "Wave-3 file-ownership disjoint with Plan 04 (server_app.py + tests/test_server_integration.py owned by Plan 04)"
  artifacts:
    - path: "scripts/foundation/tests/test_pfedrec_subprocess_determinism.py"
      provides: "@pytest.mark.slow subprocess regression guard for PFR-06 + D-16 cache disk-payload determinism + D-14 PFR-08 verification byte-identity"
  key_links:
    - from: "scripts/foundation/tests/test_pfedrec_subprocess_determinism.py"
      to: ".embedding_cache/{run_id}/partition_{pid}.pt — Plan 03 D-16 cache layout"
      via: "torch.load + torch.equal per-key comparison"
      pattern: "affine_output.weight"
    - from: "scripts/foundation/tests/test_pfedrec_subprocess_determinism.py"
      to: "results/federated/{run_id}_results.json — Plan 04 server_app output"
      via: "selected_clients_per_round + _manifest.pfr08_verification byte-identity"
      pattern: "selected_clients_per_round|pfr08_verification"
---

<objective>
Subprocess determinism regression guard for PFedRec — Wave-3 parallel with Plan 04 (disjoint file ownership: this plan touches ONLY `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py`).

Purpose:
  - PFR-06 regression-prevention axis: catch any future "deterministic RNG feeds non-deterministic domain" (G-03-01 family) regression by asserting byte-identity of `selected_clients_per_round` across two same-seed subprocess reruns.
  - D-16 cache disk-payload determinism: assert byte-identity of every overlapping `partition_{pid}.pt` cache file via per-key `torch.equal` comparison (single key `affine_output.weight` after D-01 bias move).
  - D-14 PFR-08 verification determinism: assert byte-identical `_manifest.pfr08_verification` audit dict — guarantees the auto-verify hook itself is deterministic (e.g. doesn't quietly use stdlib random for some intermediate computation).
  - Coverage guard prevents false-GREEN: scan materialized cache for at least one `partition_{pid}.pt` containing the `affine_output.weight` key; if coverage_seen is False on a non-empty selection, `pytest.fail` with explanatory message.

Mirrors Phase 2 Plan 05 + Phase 3 Plan 05 + Phase 4 Plan 06 idioms verbatim with three Phase-5-specific adaptations:
  1. Module alias = `pfedrec`; mode = `paper_compat_pfedrec`.
  2. Cache-root probe paths target `federated-pfedrec/.embedding_cache/`.
  3. Per-key comparison swapped for PFedRec's single-key payload (`affine_output.weight` only — bias is GLOBAL after D-01).
  4. Audit dict byte-identity (`_manifest.pfr08_verification`) is unique to Phase 5 (Phase 4's analog was `_manifest.best_prototype`).

Output:
  - 1 new file (`scripts/foundation/tests/test_pfedrec_subprocess_determinism.py`).
  - 1 `@pytest.mark.slow` test that COLLECTS but SKIPS under `FEDREC_SKIP_SLOW=1` (Phase 2/3/4 precedent — proves authoring correctness even when CI doesn't run the full subprocess).
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-03-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/03-personalized-migration/03-personalized-migration-05-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-06-SUMMARY.md

<interfaces>
The full reference shape lives at `scripts/foundation/tests/test_adaptive_determinism.py` (Phase 4 Plan 06). Phase-5-specific adaptations:

1. Subprocess CLI:
   ```
   scripts/run.py pfedrec paper_compat_pfedrec --run-config "run-seed=42 run-id=<rid> num-server-rounds=2 local-epochs=1 fraction-train=0.01 wandb-enabled=false reuse-cache=false"
   ```
   The `--run-config` does NOT include `enable-per-user-alpha` / `enable-item-perturbation` (Phase 4 specific). DO include `reuse-cache=false` to force per-run cache materialization.

2. Cache probe paths target `federated-pfedrec/.embedding_cache/{run_id}/`:
   ```python
   def _probe_cache_dir(run_id: str, alt_root: Optional[Path]) -> Optional[Path]:
       candidates = []
       if alt_root is not None:
           candidates.append(alt_root / run_id)
       candidates.append(_REPO_ROOT / ".embedding_cache" / run_id)
       candidates.append(_REPO_ROOT / "federated-pfedrec" / ".embedding_cache" / run_id)
       for p in candidates:
           if p.exists() and any(p.glob("partition_*.pt")):
               return p
       return None
   ```

3. Per-key comparison: single key `affine_output.weight` after D-01 bias move:
   ```python
   for key in keys_a:
       t_a, t_b = sd_a[key], sd_b[key]
       if not torch.equal(t_a, t_b):
           max_delta = float((t_a.float() - t_b.float()).abs().max().item())
           mismatches.append(
               f"partition {pid}: tensor {key!r} differs "
               f"(shape={tuple(t_a.shape)}, dtype={t_a.dtype}, max_abs_delta={max_delta})"
           )
   ```

4. Coverage guard scans for `affine_output.weight`:
   ```python
   if "affine_output.weight" in sd_a:
       coverage_seen = True
   # ...
   if checked_partitions > 0 and not coverage_seen:
       pytest.fail(
           "PFR-03 path not actually exercised by this run. "
           "No partition_{pid}.pt contains 'affine_output.weight'. "
           "Confirm Plan 03 client_app.py + Plan 01 model contract propagated correctly."
       )
   ```

5. Audit dict byte-identity (Phase-5 unique invariant):
   ```python
   audit_a = data_a.get("_manifest", {}).get("pfr08_verification")
   audit_b = data_b.get("_manifest", {}).get("pfr08_verification")
   if audit_a is None and audit_b is None:
       pytest.skip("pfr08_verification absent on both runs (smoke config too small)")
   if audit_a is None or audit_b is None:
       pytest.fail(f"D-14 VIOLATED: asymmetric pfr08_verification — run_a={audit_a is not None} run_b={audit_b is not None}")
   assert audit_a == audit_b, (
       f"D-14 VIOLATED: pfr08_verification differs across same-seed runs.\n"
       f"  run_a = {audit_a}\n  run_b = {audit_b}"
   )
   ```

6. `pytestmark` (3 skip guards + slow marker):
   ```python
   pytestmark = [
       pytest.mark.slow,
       pytest.mark.skipif(os.environ.get("FEDREC_SKIP_SLOW") == "1",
                          reason="FEDREC_SKIP_SLOW=1 — skip slow subprocess test"),
       pytest.mark.skipif(not _LAUNCHER.exists(), reason="scripts/run.py not present"),
       pytest.mark.skipif(not _BUNDLE_PATH.exists(), reason="foundation bundle not present"),
   ]
   ```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create scripts/foundation/tests/test_pfedrec_subprocess_determinism.py — single @pytest.mark.slow subprocess regression guard</name>
  <files>scripts/foundation/tests/test_pfedrec_subprocess_determinism.py</files>
  <read_first>
    - scripts/foundation/tests/test_adaptive_determinism.py — Phase 4 Plan 06 reference (clone shape with Phase-5-specific adaptations)
    - scripts/foundation/tests/test_personalized_determinism.py — Phase 3 Plan 05 reference (single-key cache idiom is closer to Phase 5 PFedRec)
    - scripts/foundation/tests/test_baseline_determinism.py — Phase 2 Plan 05 reference
    - .planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-06-SUMMARY.md — coverage guard pattern + dual-root cache probe + FEDREC_SKIP_SLOW escape hatch
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-14, D-16, D-23
    - .planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md §Don't Hand-Roll (subprocess determinism guard contract)
    - .planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-03-PLAN.md (D-16 cache layout — partition_{pid}.pt single-key payload)
    - .planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-PLAN.md (D-14 _manifest.pfr08_verification audit dict shape)
  </read_first>
  <action>
Create `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` modeled on `scripts/foundation/tests/test_adaptive_determinism.py` (Phase 4 Plan 06) with these Phase-5-specific changes:

1. **Module name + docstring**: target `paper_compat_pfedrec` (not `benchmark_cross_device`); cite PFR-06 / D-16 / D-14 not ADP-06 / ADP-02.

2. **`_run_pfedrec` helper**: invoke `scripts/run.py pfedrec paper_compat_pfedrec --run-config "..."`. The `--run-config` should NOT include `enable-per-user-alpha` / `enable-item-perturbation` (Phase 4 specific). DO include `reuse-cache=false` to force per-run cache materialization (D-22 cold-round path exercised).

3. **`_probe_cache_dir`**: probe `_REPO_ROOT / "federated-pfedrec" / ".embedding_cache" / run_id` (the module dir; per Phase 3 Plan 03's `_CACHE_BASE_DIR = _MODULE_DIR.parent / ".embedding_cache"` rule), in addition to the `FEDREC_CACHE_ROOT` env hint and the repo-root fallback.

4. **Three invariants in order**:
   - **(a)** `selected_clients_per_round` byte-identity (PFR-06 / G-03-01) — always asserted (no skip).
   - **(b)** `_manifest.pfr08_verification` byte-identity (D-14) — skip if both runs return None (degenerate smoke config); fail-asymmetric if only one is None; assert equality otherwise. NOTE: Audit dict contains floats; use Python `==` (dict equality) which is exact for the JSON-roundtrip values.
   - **(c)** Per-key `torch.equal` on every overlapping `partition_{pid}.pt` (D-16). Single key after D-01: only `affine_output.weight` is in the LOCAL payload; bias is GLOBAL and lives in the result JSON, not in cache .pt files.

5. **Coverage guard** (Phase 4 Plan 06 idiom — prevents silent-config-drift false-GREEN): scan `cache_a_dir` for at least one `partition_{pid}.pt` containing the `'affine_output.weight'` key. If `checked_partitions > 0` and `coverage_seen is False`, `pytest.fail` with the message documented in `<interfaces>`.

6. **Cold-run sanity skips** (Phase 3 Plan 05 + Phase 4 Plan 06 idiom): if `_probe_cache_dir` returns None for either run (smoke config too small, no cache .pt files materialized), `pytest.skip` cleanly — invariants (a) + (b) already asserted.

7. **Test naming**: ONE test function `test_pfedrec_determinism_subprocess_byte_identical(tmp_path)`. To satisfy VALIDATION row 5-05-02 separately, OPTIONALLY add a thin second test `test_partition_pt_byte_identical` that also runs both subprocess invocations (or shares fixtures) and asserts ONLY invariant (c). Either approach is acceptable.

8. **`@pytest.mark.slow` marker** + 3 skip guards (FEDREC_SKIP_SLOW=1, scripts/run.py missing, foundation bundle missing). The marker is intentionally NOT registered in pyproject.toml (Phase 2/3/4 precedent — harmless `PytestUnknownMarkWarning`).

9. **Module-level imports**: `pytest`, `torch`, `subprocess`, `json`, `os`, `sys`, `Path`, `Dict`, `List`, `Optional`, `Set`. Do NOT import from `federated_pfedrec.*` — the test exercises the launcher's subprocess contract, not the in-process module.

10. **CWD = `_REPO_ROOT`** for the subprocess (Phase 3 Plan 05 precedent — the launcher itself cds into the module before `flwr run`).

11. **Wave-3 file ownership disjoint with Plan 04**: `git diff --name-only` after this task should show ONLY `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py`. NO touches to:
    - `federated-pfedrec/` (everything in this dir owned by Plans 01-04)
    - `scripts/run.py` (owned by Phase 1 Plan 05)
    - `scripts/foundation/fedrec_foundation/` (owned by Plan 02 D-25 update + Phase 1 Plans)
    - Pre-existing `scripts/foundation/tests/test_*.py` files (Plan 02's `test_mode.py` extension is owned by Plan 02)

The Plan 04 sibling executor is concurrently writing `federated-pfedrec/federated_pfedrec/server_app.py` + `federated-pfedrec/tests/test_server_integration.py`. These two file sets are entirely disjoint; commits use `--no-verify` to avoid pre-commit hook contention.

After creating the file, verify it COLLECTS under `FEDREC_SKIP_SLOW=1`:
```
FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py --collect-only -q
```
should output at least `1 test collected`. Run the same with `-x`:
```
FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py -x
```
should report `1 skipped` (or `2 skipped` if the optional second test was added) — the slow tests are skipped due to FEDREC_SKIP_SLOW=1; foundation suite remains GREEN overall.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py -x</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` exists
    - `grep -c "@pytest.mark.slow\|pytest.mark.slow" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1
    - `grep -c "FEDREC_SKIP_SLOW" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1
    - `grep -c "selected_clients_per_round" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 2 (read for run_a + run_b)
    - `grep -c "pfr08_verification" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 2 (read for run_a + run_b)
    - `grep -c "affine_output.weight" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1 (coverage guard scans for this key)
    - `grep -c "torch.equal" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1 (per-key cache comparison)
    - `grep -c "paper_compat_pfedrec" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1 (subprocess CLI mode)
    - `grep -c "PFR-03 path not actually exercised" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1 (coverage-guard message)
    - `grep -c "from federated_pfedrec" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns 0 (the test must NOT import from the module — subprocess contract only)
    - `FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py -x` exits 0 with at least 1 skipped (or passed if both launcher + bundle present and the slow test runs)
    - Pre-existing foundation suite still passes: `FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/ -x` exits 0 with no regressions (the new file adds 1 or 2 SKIPPED items; existing tests unchanged)
  </acceptance_criteria>
  <done>
    - 1 new file at scripts/foundation/tests/test_pfedrec_subprocess_determinism.py
    - 1 (or 2) `@pytest.mark.slow` subprocess test(s) covering VALIDATION rows 5-05-01 and 5-05-02
    - 3 invariants asserted: selected_clients_per_round byte-identity (PFR-06), pfr08_verification byte-identity (D-14), per-key partition_{pid}.pt byte-identity via torch.equal (D-16)
    - Coverage guard scans for affine_output.weight key — prevents silent-config-drift false-GREEN
    - Foundation suite remains GREEN; new file SKIPS cleanly under FEDREC_SKIP_SLOW=1; Wave-3 file-ownership disjoint with Plan 04
  </done>
</task>

</tasks>

<verification>
- Foundation suite under FEDREC_SKIP_SLOW=1: `FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/ -x` exits 0; the new file adds 1-2 SKIPPED items
- Pre-existing test files in `scripts/foundation/tests/` are byte-identical to pre-Plan-05 state (Wave-3 disjoint with Plan 02's `test_mode.py` extension and Plan 04's federated-pfedrec/tests/test_server_integration.py)
- Foundation suite full collection: `pytest scripts/foundation/tests/ --collect-only -q` reports the same passed/skipped baseline as pre-Plan-05 plus 1-2 new tests in test_pfedrec_subprocess_determinism.py
</verification>

<success_criteria>
- 1 new file: scripts/foundation/tests/test_pfedrec_subprocess_determinism.py
- 1 (or 2) @pytest.mark.slow subprocess test(s)
- Three invariants asserted: PFR-06 selected_clients_per_round byte-identity, D-14 pfr08_verification audit byte-identity, D-16 per-key torch.equal on partition_{pid}.pt
- Coverage guard prevents silent-config-drift on the `affine_output.weight` key
- FEDREC_SKIP_SLOW=1 escape hatch verified
- Wave-3 file-ownership disjoint: zero touches to federated-pfedrec/, scripts/run.py, scripts/foundation/fedrec_foundation/, or pre-existing scripts/foundation/tests/* files
</success_criteria>

<output>
After completion, create `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-05-SUMMARY.md` covering:
- PFR-06 regression-prevention closure (selected_clients_per_round byte-identity)
- D-16 cache disk-payload determinism (per-key torch.equal on affine_output.weight)
- D-14 audit-dict byte-identity (pfr08_verification — Phase 5 unique vs Phase 4 best_prototype)
- Coverage guard pattern (Phase 4 Plan 06 idiom carried forward)
- Phase 5 closure: all 9 PFR requirements (PFR-01..09) covered across Plans 01-05
- Confirmation that Wave-3 file-ownership disjointness held: `git diff --stat HEAD~1 HEAD federated-pfedrec/` returns empty (Plan 04's commits land in parallel without overlap)
</output>
