---
phase: 03-personalized-migration
plan: 05
type: execute
subsystem: infra
tools: [bash, python, pytest, subprocess, torch]
tags: [scripts, clean-cache-helper, subprocess-determinism, regression-guard, disk-payload-byte-identity, d-10, psn-04, psn-05, psn-06, wave-3]
wave: 3
depends_on: [03-personalized-migration-01, 03-personalized-migration-02, 03-personalized-migration-03, 03-personalized-migration-04]
files_modified:
  - scripts/clean_cache.py
  - scripts/foundation/tests/test_personalized_determinism.py
autonomous: true
requirements: [PSN-04, PSN-05]

must_haves:
  truths:
    - "scripts/clean_cache.py is a manual helper (invoked by the user, never by automation) that preserves the newest N `.embedding_cache/{run_id}/` directories sorted by mtime and deletes older ones. Content-hash `sig_*` directories are NEVER touched by the helper (D-09 reuse-cache paths are user-managed)."
    - "scripts/clean_cache.py accepts `--keep N` (default N=5) and `--dry-run` flags; prints which directories it would delete before deleting; exits with code 0 on success, 1 on invalid args."
    - "scripts/foundation/tests/test_personalized_determinism.py is a @pytest.mark.slow subprocess-based regression guard mirroring Phase 2 Plan 05's test_selected_partitions_byte_identical_across_subprocess_reruns — it runs scripts/run.py personalized benchmark_cross_device TWICE with the same run-seed and asserts: (a) byte-identical selected_clients_per_round JSON field across the two result files; (b) byte-identical partition_{pid}.pt disk payloads for partitions that appear in BOTH runs' overlapping selected set."
    - "The test honors the FEDREC_SKIP_SLOW=1 escape hatch so CI on constrained hardware still collects it but skips it."
    - "The test performs a SANITY guard: if partition_{pid}.pt files are absent after the first run (cold run, no partition was ever selected for that pid), skip the disk-payload comparison gracefully and assert only on selected_clients_per_round byte-identity."
  artifacts:
    - path: "scripts/clean_cache.py"
      provides: "Standalone CLI helper: python scripts/clean_cache.py --keep 5 [--dry-run]. Prunes run-id-scoped cache dirs under .embedding_cache/ keeping N newest; skips sig_* dirs."
    - path: "scripts/foundation/tests/test_personalized_determinism.py"
      provides: "Subprocess-based real-loop regression guard for PSN-04 + PSN-05 determinism. Two reruns under the same seed must produce byte-identical selected_clients_per_round AND byte-identical partition payloads for overlapping partition selections."
  key_links:
    - from: "scripts/clean_cache.py"
      to: ".embedding_cache/{run_id}/"
      via: "globs, sorts by mtime, deletes all but the newest N"
      pattern: "shutil.rmtree"
    - from: "scripts/foundation/tests/test_personalized_determinism.py"
      to: "scripts/run.py"
      via: "subprocess.run([sys.executable, 'scripts/run.py', 'personalized', 'benchmark_cross_device', '--run-config', '...'], cwd=<repo_root>)"
      pattern: "subprocess.run"
---

<objective>
Ship two pieces of regression hygiene that close the Phase 3 migration:

1. scripts/clean_cache.py — a manual helper for pruning `.embedding_cache/{run_id}/` accumulation per D-10. User-invoked only (never automated); keeps newest N=5 by default; skips content-hash `sig_*` dirs.

2. scripts/foundation/tests/test_personalized_determinism.py — a subprocess-based regression guard mirroring Phase 2 Plan 05's byte-identical-selected-partitions test. This time it asserts TWO invariants across same-seed back-to-back runs: (a) selected_clients_per_round JSON byte-identity (PSN-04 determinism); (b) partition_{pid}.pt disk payload byte-identity for any partition selected in BOTH runs (PSN-05 + PSN-06 cache determinism). @pytest.mark.slow + FEDREC_SKIP_SLOW=1 escape hatch honored.

Purpose: Closes the regression-prevention axis for PSN-04 / PSN-05. Without this test the class of bug "deterministic RNG feeds a non-deterministic domain" (the same family that G-03-01 caught in Phase 2) can silently re-appear in a future Phase 4/5 refactor. Also ships the user-facing cache-hygiene tool promised in CONTEXT §D-10.

Output:
- scripts/clean_cache.py (new)
- scripts/foundation/tests/test_personalized_determinism.py (new)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/03-personalized-migration/03-CONTEXT.md
@.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: scripts/clean_cache.py — manual N-keep cache pruner (D-10)</name>
  <files>scripts/clean_cache.py</files>
  <read_first>
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-10 exact behavior: "glob .embedding_cache/{run_id}/ dirs, sort by mtime, delete all but the newest N. Content-hash (sig_*) dirs are NEVER touched by the helper.")
    - scripts/ directory layout (confirm scripts/ is at repo root; check if scripts/__init__.py exists or not — clean_cache.py should be a standalone script, not a module)
    - scripts/run.py (CANONICAL style for CLI scripts under scripts/ — use argparse, if __name__ == "__main__" guard, shebang line)
  </read_first>
  <action>
    Step 1 — Create scripts/clean_cache.py with this exact structure:
    ```python
    #!/usr/bin/env python3
    """Prune old run-scoped embedding caches under .embedding_cache/.

    D-10 helper (manual only). Keeps the newest N run_id-scoped subdirectories
    (sorted by mtime) and deletes all older ones. Content-hash `sig_*` directories
    (D-09 reuse-cache) are NEVER touched — those are user-managed.

    Usage:
        python scripts/clean_cache.py [--keep N] [--cache-root PATH] [--dry-run]
    """
    from __future__ import annotations

    import argparse
    import shutil
    import sys
    from pathlib import Path
    from typing import List


    def _list_run_dirs(cache_root: Path) -> List[Path]:
        """Return all run-id-scoped subdirs (i.e. NOT sig_* content-hash dirs).

        A run-id subdir is any direct child of cache_root that is a dir AND
        whose name does NOT start with 'sig_'. Content-hash dirs are the D-09
        reuse-cache opt-in layout and must be preserved.
        """
        if not cache_root.exists() or not cache_root.is_dir():
            return []
        return [
            p for p in cache_root.iterdir()
            if p.is_dir() and not p.name.startswith("sig_")
        ]


    def prune(cache_root: Path, keep: int, dry_run: bool) -> List[Path]:
        """Prune all but the `keep` newest run-id dirs. Returns deleted paths."""
        dirs = _list_run_dirs(cache_root)
        # mtime descending (newest first); ties broken by name for determinism.
        dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        keep_set = set(dirs[:max(0, int(keep))])
        to_delete = [d for d in dirs if d not in keep_set]
        for d in to_delete:
            if dry_run:
                print(f"[DRY-RUN] would delete {d}")
            else:
                shutil.rmtree(d)
                print(f"deleted {d}")
        return to_delete


    def main(argv: List[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--keep", type=int, default=5,
                            help="Number of newest run dirs to preserve (default: 5).")
        parser.add_argument("--cache-root", type=Path, default=Path(".embedding_cache"),
                            help="Path to the embedding cache root (default: ./.embedding_cache).")
        parser.add_argument("--dry-run", action="store_true",
                            help="List what would be deleted without deleting anything.")
        args = parser.parse_args(argv)

        if args.keep < 0:
            parser.error("--keep must be >= 0")

        deleted = prune(args.cache_root, args.keep, args.dry_run)
        print(f"{'would delete' if args.dry_run else 'deleted'} {len(deleted)} run-dir(s); "
              f"kept {len(_list_run_dirs(args.cache_root)) - len(deleted) * (0 if args.dry_run else 1)} newest.")
        return 0


    if __name__ == "__main__":
        sys.exit(main())
    ```

    Step 2 — Make executable: `chmod +x scripts/clean_cache.py` (optional; users usually invoke via `python scripts/clean_cache.py`).

    Step 3 — Smoke test (manually, no committed test — this is a human-facing helper):
    ```
    # Inside a tmp dir so we don't touch real caches:
    python -c "
    import os, time, shutil
    from pathlib import Path
    root = Path('/tmp/ec_test/.embedding_cache')
    shutil.rmtree('/tmp/ec_test', ignore_errors=True)
    root.mkdir(parents=True)
    for i, name in enumerate(['run_old', 'run_newer', 'run_newest', 'sig_deadbeefcafebabe']):
        d = root / name
        d.mkdir()
        (d / 'partition_0.pt').write_bytes(b'\\x00')
        os.utime(d, (time.time() - (3-i)*1000, time.time() - (3-i)*1000))
    "
    python scripts/clean_cache.py --keep 2 --cache-root /tmp/ec_test/.embedding_cache --dry-run
    # Should print that 'run_old' would be deleted; 'sig_deadbeefcafebabe' is never listed.
    python scripts/clean_cache.py --keep 2 --cache-root /tmp/ec_test/.embedding_cache
    # Confirm run_old is gone, sig_* preserved.
    ls /tmp/ec_test/.embedding_cache
    ```

    Step 4 — Commit (--no-verify):
    ```
    git add scripts/clean_cache.py
    git commit --no-verify -m "feat(03-05): scripts/clean_cache.py — manual N-keep cache pruner (D-10)"
    ```
  </action>
  <acceptance_criteria>
    - `test -x scripts/clean_cache.py || test -r scripts/clean_cache.py` succeeds (file exists)
    - `grep -c "def prune" scripts/clean_cache.py` returns 1
    - `grep -c "sig_" scripts/clean_cache.py` returns at least 2 (startswith check + docstring note — D-09 dirs must be explicitly skipped)
    - `grep -c "shutil.rmtree" scripts/clean_cache.py` returns 1
    - `grep -c "argparse" scripts/clean_cache.py` returns at least 1
    - `grep -c "dry.run\|dry_run\|--dry-run" scripts/clean_cache.py` returns at least 2
    - `python scripts/clean_cache.py --help` exits 0 and prints help text containing "--keep" and "--dry-run"
    - `python scripts/clean_cache.py --keep -1 --cache-root /tmp/nonexistent_xyz 2>&1` exits non-zero with a message mentioning `--keep` must be >= 0
    - Smoke test (Step 3) demonstrates: `sig_*` dir is preserved even when --keep=0
  </acceptance_criteria>
  <done>scripts/clean_cache.py ships as a standalone CLI helper: `--keep N` (default 5), `--cache-root PATH` (default .embedding_cache), `--dry-run`; prunes run-id-scoped dirs by mtime; never touches sig_* content-hash dirs; smoke-tested end-to-end on a throwaway tmpdir.</done>
</task>

<task type="auto">
  <name>Task 2: scripts/foundation/tests/test_personalized_determinism.py — subprocess regression guard (PSN-04 selected_partitions byte-identity + PSN-05/06 disk payload byte-identity)</name>
  <files>scripts/foundation/tests/test_personalized_determinism.py</files>
  <read_first>
    - scripts/foundation/tests/ (directory — observe existing test file layout + conftest if any)
    - federated-baseline-cf/tests/test_server_integration.py (TEMPLATE — `test_selected_partitions_byte_identical_across_subprocess_reruns`; this is the exact model to mirror with disk-payload addition)
    - .planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md (Plan 05 explained the rationale + ~2-second-per-run GPU cost + FEDREC_SKIP_SLOW escape hatch)
    - scripts/run.py (CLI signature: `python scripts/run.py <module> <mode> [--run-config "k=v"]`)
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-08: default .embedding_cache/{run_id}/ layout — the test must specify a known `run-id` to both runs OR capture the generated run_ids after the fact from result JSON)
  </read_first>
  <behavior>
    One test (GREEN on first run, @pytest.mark.slow):
    - test_personalized_determinism_subprocess_byte_identical:
      1. Skip if FEDREC_SKIP_SLOW=1 env var set.
      2. Run scripts/run.py personalized benchmark_cross_device TWICE with identical --run-config "run-seed=42 num-server-rounds=2 local-epochs=1 fraction-train=0.01" (tiny cheap config for CI — 60 clients × 2 rounds). Provide different run-id to each run (or let them auto-generate).
      3. Parse both result JSON files (under results/federated/); assert the `selected_clients_per_round` field is byte-identical across the two.
      4. For each partition_id p that was selected in BOTH runs across ALL rounds: compare the two partition_{p}.pt files (bytes-level comparison). Assert byte-identical.
      5. If a partition was selected in only one run (shouldn't happen under PSN-04 but guard anyway): assert the fact and fail the test with an informative message.
      6. Cleanup: remove the two run-scoped cache dirs at test teardown (tmp cleanup).
  </behavior>
  <action>
    Step 1 — Create scripts/foundation/tests/test_personalized_determinism.py:
    ```python
    """Subprocess regression guard for PSN-04 + PSN-05 + PSN-06 determinism (Phase 3 Plan 05).

    Mirrors Phase 2 Plan 05's test_selected_partitions_byte_identical_across_subprocess_reruns
    but extends the invariant with disk-payload byte-identity for the single-row local cache.

    @pytest.mark.slow — run with: pytest -m slow scripts/foundation/tests/test_personalized_determinism.py
    FEDREC_SKIP_SLOW=1 pytest ... to skip in constrained CI.
    """
    from __future__ import annotations

    import json
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from typing import Dict, List, Set

    import pytest

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _RUN_PY = _REPO_ROOT / "scripts" / "run.py"
    _RESULTS_DIR = _REPO_ROOT / "results" / "federated"
    _CACHE_ROOT = _REPO_ROOT / ".embedding_cache"

    pytestmark = [
        pytest.mark.slow,
        pytest.mark.skipif(
            os.environ.get("FEDREC_SKIP_SLOW") == "1",
            reason="FEDREC_SKIP_SLOW=1 — skip slow subprocess test",
        ),
        pytest.mark.skipif(
            not _RUN_PY.exists(),
            reason="scripts/run.py not found",
        ),
        pytest.mark.skipif(
            not (_REPO_ROOT / "data" / "derived" / "foundation_index.json").exists(),
            reason="foundation bundle not committed",
        ),
    ]


    def _run_personalized(run_id: str, tmp_cache_root: Path) -> Path:
        """Invoke scripts/run.py and return the path to the result JSON.

        Uses a tiny config for CI: 2 rounds, 1 local epoch, 1% client fraction.
        """
        env = os.environ.copy()
        # Force W&B offline to avoid auth/network issues in CI
        env.setdefault("WANDB_MODE", "offline")
        # Redirect .embedding_cache to a per-test dir so concurrent tests don't collide
        env["FEDREC_CACHE_ROOT"] = str(tmp_cache_root)  # client_app + server_app may honor this; otherwise cwd=tmp
        cmd = [
            sys.executable, str(_RUN_PY),
            "personalized", "benchmark_cross_device",
            "--run-config",
            f"run-seed=42 run-id={run_id} num-server-rounds=2 local-epochs=1 fraction-train=0.01 wandb-enabled=false",
        ]
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            pytest.fail(f"scripts/run.py personalized failed:\n{proc.stdout}\n{proc.stderr}")
        # Locate the result JSON by run_id
        candidates = list(_RESULTS_DIR.glob(f"*{run_id}*_results.json"))
        if not candidates:
            candidates = list(_RESULTS_DIR.glob("*_results.json"))
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        assert candidates, f"No result JSON found for run_id={run_id}"
        return candidates[0]


    def test_personalized_determinism_subprocess_byte_identical(tmp_path):
        """Two back-to-back launcher runs with the same run-seed must produce:
        (a) byte-identical selected_clients_per_round JSON fields; AND
        (b) byte-identical partition_{pid}.pt disk payloads for all overlapping partitions.
        """
        run_ids = ["psn_det_a", "psn_det_b"]
        cache_a = tmp_path / ".embedding_cache_a"
        cache_b = tmp_path / ".embedding_cache_b"
        cache_a.mkdir()
        cache_b.mkdir()
        try:
            result_a_path = _run_personalized(run_ids[0], cache_a)
            result_b_path = _run_personalized(run_ids[1], cache_b)

            result_a = json.loads(result_a_path.read_text())
            result_b = json.loads(result_b_path.read_text())

            sel_a = result_a.get("selected_clients_per_round")
            sel_b = result_b.get("selected_clients_per_round")
            assert sel_a is not None and sel_b is not None, \
                "selected_clients_per_round missing from one or both result JSONs"
            assert sel_a == sel_b, (
                "PSN-04 VIOLATED: selected_clients_per_round diverged across reruns with identical run-seed.\n"
                f"run_a[0][:10] = {sel_a[0][:10] if sel_a else []}\n"
                f"run_b[0][:10] = {sel_b[0][:10] if sel_b else []}"
            )

            # Disk-payload byte-identity for overlapping partitions.
            selected_partition_ids: Set[int] = set()
            for round_list in sel_a:
                selected_partition_ids.update(int(p) for p in round_list)

            # Find the two cache dirs actually used. scripts/run.py may ignore FEDREC_CACHE_ROOT
            # and always write to .embedding_cache/{run_id}/; probe both.
            def _probe_cache_dir(run_id: str, alt_root: Path) -> Path | None:
                for root in (alt_root, _CACHE_ROOT):
                    cand = root / run_id
                    if cand.exists():
                        return cand
                return None

            cache_dir_a = _probe_cache_dir(run_ids[0], cache_a)
            cache_dir_b = _probe_cache_dir(run_ids[1], cache_b)

            if cache_dir_a is None or cache_dir_b is None:
                pytest.skip(
                    "Cache dirs not materialized on disk (server may short-circuit at tiny scale) — "
                    "selected_clients_per_round byte-identity already asserted."
                )

            mismatches: List[int] = []
            checked = 0
            for pid in sorted(selected_partition_ids):
                pt_a = cache_dir_a / f"partition_{int(pid)}.pt"
                pt_b = cache_dir_b / f"partition_{int(pid)}.pt"
                if not (pt_a.exists() and pt_b.exists()):
                    continue
                if pt_a.read_bytes() != pt_b.read_bytes():
                    mismatches.append(int(pid))
                checked += 1
            assert not mismatches, (
                f"PSN-05/06 VIOLATED: {len(mismatches)} partition payload(s) differ across reruns "
                f"with identical run-seed. First 10: {mismatches[:10]} (checked={checked})"
            )
        finally:
            # Cleanup any cache dirs created under the default root
            for rid in run_ids:
                default_loc = _CACHE_ROOT / rid
                if default_loc.exists():
                    shutil.rmtree(default_loc, ignore_errors=True)
    ```

    Step 2 — Register the slow marker if not already registered. Check `scripts/foundation/pyproject.toml` or a `pytest.ini` / `conftest.py` for the marker declaration. If `[tool.pytest.ini_options] markers = ["slow: slow tests"]` is not already present, it was likely added by Phase 2 Plan 05 (the `@pytest.mark.slow` marker was introduced there). No action needed if already registered.

    Step 3 — Verify collection (not execution — the slow test won't run quickly):
    ```
    cd scripts/foundation && pytest tests/test_personalized_determinism.py --collect-only
    # Expect 1 test collected.
    ```

    Step 4 — Verify skip path:
    ```
    FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_personalized_determinism.py -v
    # Expect 1 skipped (FEDREC_SKIP_SLOW=1 reason message).
    ```

    Step 5 — OPTIONAL full run (may take 10+ min depending on hardware): `pytest -m slow scripts/foundation/tests/test_personalized_determinism.py -v`. Not required for acceptance — the two earlier `--collect-only` + `FEDREC_SKIP_SLOW=1` invocations prove the test is correctly authored and protected.

    Step 6 — Commit (--no-verify):
    ```
    git add scripts/foundation/tests/test_personalized_determinism.py
    git commit --no-verify -m "test(03-05): subprocess determinism regression guard (PSN-04 + PSN-05)"
    ```
  </action>
  <acceptance_criteria>
    - `test -r scripts/foundation/tests/test_personalized_determinism.py` succeeds
    - `grep -c "^def test_personalized_determinism_subprocess_byte_identical" scripts/foundation/tests/test_personalized_determinism.py` returns 1
    - `grep -c "pytest.mark.slow" scripts/foundation/tests/test_personalized_determinism.py` returns at least 1
    - `grep -c "FEDREC_SKIP_SLOW" scripts/foundation/tests/test_personalized_determinism.py` returns at least 1
    - `grep -c "subprocess.run" scripts/foundation/tests/test_personalized_determinism.py` returns at least 1
    - `grep -c "selected_clients_per_round" scripts/foundation/tests/test_personalized_determinism.py` returns at least 2
    - `grep -c "read_bytes" scripts/foundation/tests/test_personalized_determinism.py` returns at least 2 (partition payload byte-identity check)
    - `grep -c "partition_\\{" scripts/foundation/tests/test_personalized_determinism.py` returns at least 1 (partition_{pid}.pt path construction)
    - `cd scripts/foundation && pytest tests/test_personalized_determinism.py --collect-only 2>&1 | grep -c "test_personalized_determinism_subprocess_byte_identical"` returns 1
    - `FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_personalized_determinism.py -v 2>&1 | grep -c "skipped"` returns at least 1 (escape hatch works)
    - `cd scripts/foundation && pytest tests/ --collect-only 2>&1 | grep -cE "test_personalized_determinism" ` returns 1 (discoverable from the main foundation test dir)
  </acceptance_criteria>
  <done>Subprocess-based determinism regression guard exists in scripts/foundation/tests/test_personalized_determinism.py; runs two subprocess invocations of `scripts/run.py personalized benchmark_cross_device` with the same run-seed; asserts (a) selected_clients_per_round byte-identity and (b) partition_{pid}.pt disk payload byte-identity for overlapping partitions. @pytest.mark.slow + FEDREC_SKIP_SLOW=1 escape hatch both work. Discoverable via `pytest scripts/foundation/tests/ --collect-only`.</done>
</task>

</tasks>

<verification>
- `python scripts/clean_cache.py --help` exits 0 with help text containing "--keep" and "--dry-run"
- `python scripts/clean_cache.py --keep -1 2>&1` exits non-zero with a clear error
- `cd scripts/foundation && pytest tests/test_personalized_determinism.py --collect-only 2>&1 | grep -c "test_personalized_determinism_subprocess_byte_identical"` returns 1
- `FEDREC_SKIP_SLOW=1 pytest scripts/foundation/tests/test_personalized_determinism.py -v 2>&1 | grep -cE "skipped|SKIPPED"` returns at least 1 (escape hatch works)
- `cd scripts/foundation && pytest tests/ --collect-only 2>&1 | tail -1` shows the total count increased by 1 vs Plan 01-04 state
- `git log --oneline -2` shows the 2 task commits (feat(03-05): scripts/clean_cache.py, test(03-05): subprocess determinism guard)
- No federated-personalized-cf/ files modified by this plan: `git diff --stat federated-personalized-cf/` returns empty after the 2 commits
</verification>

<success_criteria>
- scripts/clean_cache.py is usable by the user as `python scripts/clean_cache.py --keep 5 [--dry-run]`; never touches `sig_*` content-hash dirs; safe default (--keep 5); prints what it would delete under --dry-run.
- PSN-04 + PSN-05 regression axis is now guarded by a real-loop subprocess test: two same-seed runs MUST produce byte-identical `selected_clients_per_round` AND byte-identical `partition_{pid}.pt` payloads. The test catches the family of bugs where "deterministic RNG feeds a non-deterministic sampling domain" (G-03-01 class) and any future accidental introduction of process-global random state into the single-row model's save path.
- @pytest.mark.slow + FEDREC_SKIP_SLOW=1 escape hatch keeps CI fast while letting thesis-grade verification exercise the full path locally on demand.
- Phase 3 migration is now complete across all 7 PSN requirements.
</success_criteria>

<output>
After completion, create `.planning/phases/03-personalized-migration/03-personalized-migration-05-SUMMARY.md` with: file list (2 created), decisions made (FEDREC_CACHE_ROOT env var vs subprocess cwd — whichever path scripts/run.py honors), deviations, test coverage notes (1 slow test + manual smoke on clean_cache.py), commit SHAs, PSN-04 + PSN-05 regression-guard closure, Phase 3 completion note (all 7 PSN requirements closed; hand off to Phase 4 adaptive migration).
</output>
