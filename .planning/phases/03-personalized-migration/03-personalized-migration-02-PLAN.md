---
phase: 03-personalized-migration
plan: 02
type: execute
subsystem: infra
tags: [pyproject-toml, dataset-adapter, cross-device, num-supernodes, partition-mode, fedrec-foundation, rip-and-replace, d-02, d-17, d-18, psn-01, wave-1]
wave: 1
depends_on: []
files_modified:
  - federated-personalized-cf/pyproject.toml
  - federated-personalized-cf/federated_personalized_cf/dataset.py
  - federated-personalized-cf/tests/test_dataset_adapter.py
autonomous: true
requirements: [PSN-01]

must_haves:
  truths:
    - "federated-personalized-cf/pyproject.toml declares num-supernodes = 6040 in BOTH [tool.flwr.federations.local-simulation] AND [tool.flwr.federations.local-sim-gpu]; partition-mode = \"natural\" under [tool.flwr.app.config]; cross-silo (num-supernodes=5) is never present as a default anywhere in the file."
    - "federated-personalized-cf/pyproject.toml declares 6 Phase-3 foundation-contract config keys: mode, run-seed=42, weight-policy=num_positives, eval-num-negatives=99, checkpoint-rule=best_round_restore, reuse-cache=false (D-09)."
    - "federated-personalized-cf/pyproject.toml declares [project.optional-dependencies] dev = [\"pytest>=7.0\"] exclusively owned by this plan/task (eliminates Wave-1 write-race with Plan 01)."
    - "federated-personalized-cf/federated_personalized_cf/dataset.py is a thin foundation adapter — mapping/split/exclusion delegated to fedrec_foundation (mirrors Phase 2 Plan 02 D-17 pattern). partition_mode='dirichlet' raises NotImplementedError with message pointing at D-02 and referencing pre-Phase-3 commit for legacy cross-silo numbers."
    - "Pre-existing uncommitted WIP in client_app.py, server_app.py, task.py, strategy.py, models/, early_stopping.py is untouched by this plan (D-18 surgical discipline — Plans 01/03/04 own those files)."
    - "3 GREEN tests in tests/test_dataset_adapter.py prove: (a) load_partition_data delegates to foundation mapping (num_users=6040, num_items=3706); (b) the held-out test_item_per_user for a known partition_id appears in the returned testloader; (c) old module-local helpers (create_global_mappings, dirichlet_partition_users, create_leave_one_out_split, create_train_test_split, compute_user_genre_distribution) are absent from the module."
  artifacts:
    - path: "federated-personalized-cf/pyproject.toml"
      provides: "Cross-device defaults (6040 supernodes in both federations) + 6 Phase-3 config keys + pytest dev dep"
      contains: "num-supernodes = 6040, partition-mode = \"natural\", reuse-cache = false"
    - path: "federated-personalized-cf/federated_personalized_cf/dataset.py"
      provides: "Thin adapter: load_partition_data + load_full_data delegate to fedrec_foundation.{mapping, split, exclusion, bundle}"
      contains: "from fedrec_foundation.mapping import, from fedrec_foundation.split import, from fedrec_foundation.exclusion import, from fedrec_foundation.bundle import verify_bundle, raise NotImplementedError"
    - path: "federated-personalized-cf/tests/test_dataset_adapter.py"
      provides: "3 GREEN pytest tests verifying D-17 rip-and-replace + D-02 NotImplementedError"
  key_links:
    - from: "federated-personalized-cf/federated_personalized_cf/dataset.py"
      to: "scripts/foundation/fedrec_foundation/bundle.py"
      via: "_load_foundation_bundle calls verify_bundle(data_derived()) on every load"
      pattern: "verify_bundle"
    - from: "federated-personalized-cf/pyproject.toml"
      to: "scripts/run.py"
      via: "Cross-device defaults in-file ensure `flwr run .` works without the launcher; launcher remains a belt-and-suspenders redundancy"
      pattern: "num-supernodes = 6040"
---

<objective>
Mirror Phase 2 Plan 02: flip federated-personalized-cf/pyproject.toml defaults to cross-device (6040 supernodes in both federation blocks, partition-mode="natural", 6 foundation-contract keys including new reuse-cache=false per D-09) and rip-and-replace dataset.py so it becomes a thin adapter that delegates mapping/split/exclusion to fedrec_foundation (D-17). Enforces D-02 NotImplementedError at the dataset layer for partition_mode='dirichlet'. Adds 3 GREEN pytest tests.

Purpose: Closes PSN-01 in-file so `flwr run .` inside federated-personalized-cf/ spawns 6040 supernodes by default. Establishes the foundation-backed dataset contract that Plan 03 (client_app.py + task.py) consumes without changing signatures.

Output:
- federated-personalized-cf/pyproject.toml (edited: 6 new keys, both federation num-supernodes flipped, pytest dev dep added)
- federated-personalized-cf/federated_personalized_cf/dataset.py (rewritten as thin foundation adapter)
- federated-personalized-cf/tests/test_dataset_adapter.py (new — 3 GREEN tests)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/03-personalized-migration/03-CONTEXT.md
@.planning/phases/02-baseline-migration/02-baseline-migration-02-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: pyproject.toml cross-device defaults + 6 Phase-3 config keys + pytest dev dep (PSN-01)</name>
  <files>federated-personalized-cf/pyproject.toml</files>
  <read_first>
    - federated-personalized-cf/pyproject.toml (ENTIRE FILE; need to see every existing key to avoid TOML duplicate-key errors + preserve existing dependencies like fedrec-foundation from Phase 1 Plan 06)
    - federated-baseline-cf/pyproject.toml (CANONICAL TEMPLATE — the block shape to clone for the Phase-3 keys + cross-device num-supernodes + [project.optional-dependencies])
    - .planning/phases/02-baseline-migration/02-baseline-migration-02-SUMMARY.md (Plan 02 Phase-2 Task 1 commit e3e4afc — exact pattern to mirror)
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-09 reuse-cache default "false"; Phase-2 carry-forward keys)
  </read_first>
  <action>
    Step 1 — Inventory with `git diff federated-personalized-cf/pyproject.toml` to capture the pre-existing uncommitted WIP state. If pre-existing WIP already sets any of the target keys, PRESERVE those values — do not downgrade them. Document each WIP hunk touched.

    Step 2 — Add `[project.optional-dependencies]` section above `[build-system]` (exact placement matches baseline):
    ```toml
    [project.optional-dependencies]
    dev = ["pytest>=7.0"]
    ```

    Step 3 — Under `[tool.flwr.app.config]`, append (or merge if already present) the following 6 Phase-3 contract keys as a new block. If any key already exists elsewhere in the file, REMOVE the duplicate and consolidate here (TOML rejects duplicate top-level keys within a table):
    ```toml
    # ==== Phase 3 cross-device / foundation-contract keys (D-19) ====
    # The canonical runtime source is fedrec_foundation.mode.resolve_mode_defaults(mode).
    # These values are FALLBACK only (consulted when the launcher did not pre-set them).
    mode = "cross_silo_legacy"  # flipped to "benchmark_cross_device" by scripts/run.py personalized benchmark_cross_device
    run-seed = 42
    weight-policy = "num_positives"
    eval-num-negatives = 99
    checkpoint-rule = "best_round_restore"
    reuse-cache = false  # D-09: set to true to opt into content-hash cache reuse across runs
    ```
    If the pre-existing `eval-num-negatives` key lives under a different section (e.g. "Evaluation protocol"), replace the old one with a pointer comment — TOML rejects duplicate keys within the same table.

    Step 4 — Under `[tool.flwr.app.config]`, ensure `partition-mode = "natural"` is set (create the key if absent; overwrite if present-with-old-value).

    Step 5 — In `[tool.flwr.federations.local-simulation]`, flip `options.num-supernodes = 5` to `options.num-supernodes = 6040` and add a cross-silo opt-in comment:
    ```toml
    [tool.flwr.federations.local-simulation]
    # ==== Cross-device default: 1 user = 1 client (N=6040). ====
    # To reproduce legacy cross-silo numbers, check out a pre-Phase-3 commit and
    # override via: flwr run . --run-config "num-supernodes=5 partition-mode=dirichlet"
    options.num-supernodes = 6040
    ```

    Step 6 — Same flip in `[tool.flwr.federations.local-sim-gpu]`. Also append the cross-silo opt-in comment. Preserve existing `options.backend.client-resources.num-gpus` / `.num-cpus` settings verbatim.

    Step 7 — Preserve verbatim: `fedrec-foundation` dep entry (Phase 1 Plan 06), `[tool.flwr.app.components]`, `[tool.flwr.federations.remote-federation]`, every other pre-existing `[tool.flwr.app.config]` key not explicitly addressed above, `[tool.hatch.build.targets.wheel] packages = ["."]`.

    Step 8 — Parse check: `python -c "import tomllib; d = tomllib.load(open('federated-personalized-cf/pyproject.toml', 'rb')); assert d['tool']['flwr']['federations']['local-simulation']['options']['num-supernodes'] == 6040; assert d['tool']['flwr']['federations']['local-sim-gpu']['options']['num-supernodes'] == 6040; assert d['tool']['flwr']['app']['config']['partition-mode'] == 'natural'; assert d['tool']['flwr']['app']['config']['run-seed'] == 42; assert d['tool']['flwr']['app']['config']['reuse-cache'] is False; assert d['project']['optional-dependencies']['dev'] == ['pytest>=7.0']; print('ok')"` prints `ok`.

    Step 9 — Commit (--no-verify per Wave-1 parallel rule):
    ```
    git add federated-personalized-cf/pyproject.toml
    git commit --no-verify -m "feat(03-02): personalized pyproject cross-device defaults + dev dep (PSN-01)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -cE "^options.num-supernodes = 6040" federated-personalized-cf/pyproject.toml` returns 2 (local-simulation + local-sim-gpu)
    - `grep -cE "^options.num-supernodes = 5" federated-personalized-cf/pyproject.toml` returns 0 (legacy value eradicated)
    - `grep -cE '^partition-mode = "natural"' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^mode = "cross_silo_legacy"' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^run-seed = 42' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^weight-policy = "num_positives"' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^eval-num-negatives = 99' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^checkpoint-rule = "best_round_restore"' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^reuse-cache = false' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '\[project.optional-dependencies\]' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -cE 'dev = \["pytest>=7.0"\]' federated-personalized-cf/pyproject.toml` returns 1
    - `grep -c "fedrec-foundation" federated-personalized-cf/pyproject.toml` returns at least 1 (Phase 1 Plan 06 dep preserved)
    - `python -c "import tomllib; d = tomllib.load(open('federated-personalized-cf/pyproject.toml', 'rb')); assert d['tool']['flwr']['federations']['local-simulation']['options']['num-supernodes'] == 6040; assert d['tool']['flwr']['app']['config']['reuse-cache'] is False"` exits 0
    - `git diff --stat federated-personalized-cf/federated_personalized_cf/` returns empty after commit (other files untouched by Task 1)
  </acceptance_criteria>
  <done>pyproject.toml flipped to cross-device defaults (6040 in both federations, partition-mode="natural", 6 Phase-3 contract keys including reuse-cache=false D-09, pytest dev dep). PSN-01 satisfied in-file. `flwr run .` inside federated-personalized-cf/ now resolves to cross-device without relying on scripts/run.py.</done>
</task>

<task type="auto">
  <name>Task 2: dataset.py rip-and-replace to foundation adapter + D-02 NotImplementedError + 3 GREEN adapter tests (D-17, D-18, PSN-01 data layer)</name>
  <files>federated-personalized-cf/federated_personalized_cf/dataset.py, federated-personalized-cf/tests/test_dataset_adapter.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/dataset.py (ENTIRE FILE; need to inventory which helpers to remove per D-17 and which pre-existing WIP functions to preserve per D-18)
    - federated-baseline-cf/federated_baseline_cf/dataset.py (CANONICAL TEMPLATE — the post-Plan-02 shape to mirror)
    - federated-baseline-cf/tests/test_dataset_adapter.py (3-test template to clone)
    - scripts/foundation/fedrec_foundation/bundle.py (verify_bundle signature)
    - scripts/foundation/fedrec_foundation/mapping.py (load_mapping → CanonicalMapping with num_users, num_items, user2idx, item2idx fields)
    - scripts/foundation/fedrec_foundation/split.py (load_split_manifest → SplitManifest with test_item_per_user dict)
    - scripts/foundation/fedrec_foundation/exclusion.py (load_exclusion → ExclusionTable with .for_user(idx))
    - scripts/foundation/fedrec_foundation/paths.py (data_derived() function)
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-02 exact NotImplementedError message + pre-Phase-3 commit reference)
  </read_first>
  <action>
    Step 1 — Inventory pre-existing dataset.py state with `git diff federated-personalized-cf/federated_personalized_cf/dataset.py`. Record which functions are D-17 rip targets (remove) vs D-18 pre-existing WIP (preserve verbatim).

    D-17 REMOVE (clone Phase-2 Plan 02's list, adjusted for this module):
    - `create_global_mappings` (if present)
    - `create_leave_one_out_split` (if present)
    - `dirichlet_partition_users` (if present — intentionally superseded by D-02 NotImplementedError)
    - `create_train_test_split` (if present)
    - `compute_user_genre_distribution` (if present)
    - `_partition_cache` module dict (if present)

    D-18 PRESERVE VERBATIM (if present):
    - `MovieLensDataset` torch.utils.data.Dataset subclass
    - `download_movielens_1m` helper
    - `load_movielens_1m` parser
    - `natural_partition_users` (per-user partitioning — essential for cross-device 1-user-per-client)

    Step 2 — Rewrite dataset.py as a thin foundation adapter. Module-level imports should include:
    ```python
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
    from fedrec_foundation.mapping import CanonicalMapping, load_mapping
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.split import SplitManifest, load_split_manifest
    ```

    Add module-level `_foundation_cache: Dict[str, Any] = {}` keyed by `foundation_contract_sha256` (replaces the old `_partition_cache`).

    Step 3 — Add `_load_foundation_bundle(data_dir: Optional[Path] = None) -> Dict[str, Any]`:
    ```python
    def _load_foundation_bundle(data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Load + cache the foundation bundle (mapping, split_manifest, exclusion)."""
        derived = data_dir if data_dir is not None else data_derived()
        index = verify_bundle(derived)  # raises RuntimeError on corruption per IMP-2/N-3
        sha = index.foundation_contract_sha256
        if sha in _foundation_cache:
            return _foundation_cache[sha]
        bundle = {
            "mapping": load_mapping(derived / "mapping.json"),
            "split_manifest": load_split_manifest(derived / "split_manifest.json"),
            "exclusion": load_exclusion(derived / "exclusion_items.npz"),
            "foundation_contract_sha256": sha,
        }
        _foundation_cache[sha] = bundle
        return bundle
    ```

    Step 4 — Rewrite `load_partition_data(partition_id, num_partitions, ..., partition_mode="natural", split_mode="leave_one_out", ...)`:
    - If `partition_mode == "dirichlet"`: raise `NotImplementedError("Personalized cross-device migration removed multi-user support per D-02. Check out a pre-Phase-3 commit (see .planning/phases/03-personalized-migration/03-CONTEXT.md §Deferred) to reproduce legacy cross-silo numbers.")`
    - If `split_mode == "random"`: raise `ValueError("split_mode='random' is no longer supported — the foundation's leave-one-out split is authoritative. Use split_mode='leave_one_out'.")`
    - Otherwise: call `_load_foundation_bundle()`, use `bundle["mapping"]` for user2idx/item2idx, use `bundle["split_manifest"].test_item_per_user[partition_id]` for the held-out test item, build the per-user partition DataLoader(s). Return signature stays `(trainloader, testloader, num_users, num_items, user2idx, item2idx)`.

    Step 5 — Rewrite `load_full_data(..., partition_mode="natural")`:
    - If `partition_mode == "dirichlet"`: raise the same `NotImplementedError` as step 4.
    - Otherwise: delegate mapping/split to foundation; build full-dataset LOO mask from `bundle["split_manifest"].test_item_per_user` across all users.

    Step 6 — Remove the 5 D-17 targets listed in Step 1 (they may not all exist — the personalized module never had genre-distribution helpers). Verify by grep after edit.

    Step 7 — Create federated-personalized-cf/tests/test_dataset_adapter.py (mirror federated-baseline-cf/tests/test_dataset_adapter.py):
    ```python
    """Adapter tests for federated_personalized_cf.dataset (Phase 3 Plan 02, D-17)."""
    from pathlib import Path
    import pytest

    _FOUNDATION_INDEX = Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json"
    pytestmark = pytest.mark.skipif(
        not _FOUNDATION_INDEX.exists(),
        reason="foundation bundle not committed (data/derived/foundation_index.json missing)",
    )

    from federated_personalized_cf.dataset import (
        _load_foundation_bundle, load_partition_data,
    )
    import federated_personalized_cf.dataset as _ds_mod

    def test_load_partition_data_uses_foundation_mapping():
        trainloader, testloader, num_users, num_items, user2idx, item2idx = load_partition_data(
            partition_id=0, num_partitions=6040, partition_mode="natural",
        )
        assert num_users == 6040
        assert num_items == 3706

    def test_load_partition_data_test_item_from_foundation_split():
        bundle = _load_foundation_bundle()
        pid = 0
        expected_test_item = bundle["split_manifest"].test_item_per_user[pid]
        trainloader, testloader, _, _, _, _ = load_partition_data(
            partition_id=pid, num_partitions=6040, partition_mode="natural",
        )
        # The testloader MUST contain that item_idx as the positive for this user.
        all_test_items = []
        for batch in testloader:
            # batch shape depends on dataset; just collect item ids
            all_test_items.extend(batch[1].tolist() if isinstance(batch, (list, tuple)) and len(batch) > 1 else [])
        assert expected_test_item in all_test_items or expected_test_item in {int(x) for x in getattr(testloader.dataset, "item_ids", [])}

    def test_removed_helpers_gone_and_d02_raises():
        # D-17: these helpers must NOT appear as module attributes
        for name in ("create_global_mappings", "create_leave_one_out_split",
                     "dirichlet_partition_users", "create_train_test_split",
                     "compute_user_genre_distribution", "_partition_cache"):
            assert not hasattr(_ds_mod, name), f"D-17 violated: {name} still present"
        # D-02: dirichlet must raise NotImplementedError
        with pytest.raises(NotImplementedError, match="cross-device"):
            load_partition_data(partition_id=0, num_partitions=5, partition_mode="dirichlet")
    ```

    Step 8 — Verify: `cd federated-personalized-cf && pytest tests/test_dataset_adapter.py -v` → 3 passed. Also run import smoke: `python -c "from federated_personalized_cf.dataset import load_partition_data, _load_foundation_bundle; b = _load_foundation_bundle(); assert b['mapping'].num_users == 6040; print('ok')"` → `ok`.

    Step 9 — D-18 scope check: `git diff --name-only HEAD~1..HEAD` should list ONLY pyproject.toml (from Task 1), dataset.py, and tests/test_dataset_adapter.py. `git diff --stat federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/early_stopping.py federated-personalized-cf/federated_personalized_cf/models/` returns empty (Plans 01/03/04 own these).

    Step 10 — Commit (--no-verify):
    ```
    git add federated-personalized-cf/federated_personalized_cf/dataset.py \
            federated-personalized-cf/tests/test_dataset_adapter.py
    git commit --no-verify -m "refactor(03-02): rip-and-replace dataset.py as foundation adapter (D-17, D-02, PSN-01)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.mapping import" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `grep -c "from fedrec_foundation.split import" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `grep -c "from fedrec_foundation.exclusion import" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `grep -c "from fedrec_foundation.bundle import verify_bundle" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `grep -c "def _load_foundation_bundle" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `grep -cE 'raise NotImplementedError' federated-personalized-cf/federated_personalized_cf/dataset.py` returns at least 1
    - `grep -c "def create_global_mappings" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 0
    - `grep -c "def create_leave_one_out_split" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 0
    - `grep -c "def dirichlet_partition_users" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 0
    - `grep -c "def create_train_test_split" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 0
    - `grep -c "_partition_cache" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 0
    - `grep -c "def natural_partition_users" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1 (D-18 preserved)
    - `grep -c "def load_partition_data" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `grep -c "def load_full_data" federated-personalized-cf/federated_personalized_cf/dataset.py` returns 1
    - `cd federated-personalized-cf && pytest tests/test_dataset_adapter.py -v` exits 0 with "3 passed"
    - `python -c "from federated_personalized_cf.dataset import load_partition_data; import pytest
try:
    load_partition_data(partition_id=0, num_partitions=5, partition_mode='dirichlet')
    raise AssertionError('expected NotImplementedError')
except NotImplementedError as e:
    assert 'cross-device' in str(e) or 'D-02' in str(e) or 'pre-Phase-3' in str(e)
print('ok')"` prints `ok`
    - `git diff --stat federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/models/` returns empty after the commit (D-18 surgical discipline)
  </acceptance_criteria>
  <done>dataset.py is a thin foundation adapter (D-17): mapping/split/exclusion delegated to fedrec_foundation; partition_mode='dirichlet' raises NotImplementedError per D-02 with explicit reference to pre-Phase-3 commit; 5 D-17 rip targets are absent from the module; 4 D-18 pre-existing functions preserved verbatim; 3 GREEN adapter tests.</done>
</task>

</tasks>

<verification>
- `python -c "import tomllib; d = tomllib.load(open('federated-personalized-cf/pyproject.toml', 'rb')); assert d['tool']['flwr']['federations']['local-simulation']['options']['num-supernodes'] == 6040; assert d['tool']['flwr']['federations']['local-sim-gpu']['options']['num-supernodes'] == 6040; assert d['tool']['flwr']['app']['config']['reuse-cache'] is False; print('ok')"` prints `ok`
- `cd federated-personalized-cf && pytest tests/test_dataset_adapter.py -v` exits 0 with "3 passed"
- `python scripts/run.py --dry-run personalized benchmark_cross_device 2>&1 | grep -c "num-supernodes=6040"` returns at least 1 (launcher consistent with in-file defaults)
- D-18 scope: `git diff --stat federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/models/` returns empty (Plan 01 owns strategy.py + models/; Plans 03/04 own the rest)
</verification>

<success_criteria>
- PSN-01 is observable in-file: `flwr run .` inside federated-personalized-cf/ spawns 6040 supernodes by default under `partition-mode="natural"`.
- Cross-silo opt-in preserved as documented fallback: `flwr run . --run-config "num-supernodes=5 partition-mode=dirichlet"` still dispatches — but `dirichlet` now raises NotImplementedError per D-02 (tests prove this).
- dataset.py contract (load_partition_data / load_full_data) signatures unchanged; Plan 03 can wire client_app.py against them without signature churn.
- Wave-1 write-race safety: this plan's 2 commits touch only pyproject.toml, dataset.py, and tests/test_dataset_adapter.py. Plan 01's 2 commits touch only strategy.py, models/bpr_mf.py, models/basic_mf.py, and tests/test_strategy.py + tests/test_single_row_model.py. Zero file overlap.
</success_criteria>

<output>
After completion, create `.planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md` with: file list, decisions made (any pre-existing WIP hunks preserved or replaced with rationale), deviations, test counts (3 GREEN adapter tests), commit SHAs, PSN-01 closure note, Plan 03 readiness confirmation.
</output>
