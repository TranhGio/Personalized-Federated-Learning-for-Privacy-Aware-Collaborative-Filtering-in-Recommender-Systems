---
phase: 04-adaptive-migration-bug-fixes
plan: 02
type: execute
subsystem: infra
tags: [pyproject-toml, dataset-adapter, cross-device, num-supernodes, partition-mode, fedrec-foundation, rip-and-replace, pytest-dev-dep, schema-version-2-keys, adp-01, d-02, d-09, d-10, d-11, d-12, d-17, d-18, wave-1]
wave: 1
depends_on: []
files_modified:
  - federated-adaptive-personalized-cf/pyproject.toml
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py
  - federated-adaptive-personalized-cf/tests/test_pyproject_shape.py
  - federated-adaptive-personalized-cf/tests/test_dataset_adapter.py
autonomous: true
requirements: [ADP-01]

must_haves:
  truths:
    - "federated-adaptive-personalized-cf/pyproject.toml declares num-supernodes = 6040 in BOTH [tool.flwr.federations.local-simulation] AND [tool.flwr.federations.local-sim-gpu]; partition-mode = \"natural\" under [tool.flwr.app.config]; the legacy cross-silo value num-supernodes = 5 is never present as a default anywhere in the file (cross-silo opt-in via --run-config override is documented in a comment)."
    - "federated-adaptive-personalized-cf/pyproject.toml declares the 6 Phase-3 carry-forward config keys: mode = \"cross_silo_legacy\" (placeholder value; the launcher scripts/run.py flips to benchmark_cross_device), run-seed = 42, weight-policy = \"num_positives\", eval-num-negatives = 99, checkpoint-rule = \"best_round_restore\", reuse-cache = false (D-09)."
    - "federated-adaptive-personalized-cf/pyproject.toml declares the 4 Phase-4 adaptive-specific defaults that go into the schema_version=2 cache signature (D-02 from Phase 4 CONTEXT): alpha-method = \"hierarchical_conditional\" (D-10), fusion-type = \"concat\" (D-11), enable-per-user-alpha = true (D-03 unconditional benchmark default), enable-item-perturbation = true (D-03 unconditional benchmark default), contrastive-lambda = 0.1 (D-12). model-type = \"dual\" already exists; confirm it is set to dual (D-09)."
    - "federated-adaptive-personalized-cf/pyproject.toml declares [project.optional-dependencies] dev = [\"pytest>=7.0\"] exclusively owned by this plan/task (eliminates Wave-1 write-race with Plan 01)."
    - "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py is a thin foundation adapter — mapping/split/exclusion delegated to fedrec_foundation (mirrors Phase 2 Plan 02 + Phase 3 Plan 02 D-17 pattern). partition_mode='dirichlet' raises NotImplementedError at BOTH load_partition_data AND load_full_data with a message pointing at D-02 and referencing pre-Phase-4 commit for legacy cross-silo numbers."
    - "Pre-existing uncommitted WIP in client_app.py, server_app.py, task.py, strategy.py, models/ is untouched by this plan (D-18 surgical discipline — Plans 01/03/05 own those files)."
    - "5 GREEN tests: 1 test_pyproject_shape.py regression (asserts pyproject has all 6 Phase-3 + 5 Phase-4 cross-device-defining keys) + 4 tests/test_dataset_adapter.py proving (a) load_partition_data delegates to foundation mapping (num_users=6040, num_items=3706); (b) the held-out test_item_per_user for a known partition_id appears in the returned testloader; (c) old module-local helpers are absent from the module; (d) partition_mode='dirichlet' raises NotImplementedError at BOTH load_partition_data AND load_full_data."
  artifacts:
    - path: "federated-adaptive-personalized-cf/pyproject.toml"
      provides: "Cross-device defaults (6040 supernodes in both federations) + 6 Phase-3 contract keys + 5 Phase-4 schema-v2 signature keys + pytest dev dep"
      contains: "num-supernodes = 6040, partition-mode = \"natural\", alpha-method = \"hierarchical_conditional\", fusion-type = \"concat\", enable-per-user-alpha = true, enable-item-perturbation = true, contrastive-lambda = 0.1, model-type = \"dual\", reuse-cache = false"
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py"
      provides: "Thin adapter: load_partition_data + load_full_data delegate to fedrec_foundation.{mapping, split, exclusion, bundle}"
      contains: "from fedrec_foundation.mapping import, from fedrec_foundation.split import, from fedrec_foundation.exclusion import, from fedrec_foundation.bundle import verify_bundle, raise NotImplementedError"
    - path: "federated-adaptive-personalized-cf/tests/test_pyproject_shape.py"
      provides: "1 GREEN grep-level regression test asserting all 11 cross-device-defining keys present in pyproject.toml"
    - path: "federated-adaptive-personalized-cf/tests/test_dataset_adapter.py"
      provides: "4 GREEN tests verifying D-17 rip-and-replace + D-02 NotImplementedError enforcement at BOTH dataset entry points"
  key_links:
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py"
      to: "scripts/foundation/fedrec_foundation/bundle.py"
      via: "_load_foundation_bundle calls verify_bundle(data_derived()) on every load"
      pattern: "verify_bundle"
    - from: "federated-adaptive-personalized-cf/pyproject.toml"
      to: "scripts/run.py"
      via: "Cross-device defaults in-file ensure `flwr run .` works without the launcher; launcher remains a belt-and-suspenders redundancy"
      pattern: "num-supernodes = 6040"
    - from: "federated-adaptive-personalized-cf/pyproject.toml"
      to: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py"
      via: "Plan 03 reads the 5 Phase-4 signature-driving keys into the schema_version=2 manifest"
      pattern: "alpha-method|fusion-type|enable-per-user-alpha|enable-item-perturbation|contrastive-lambda"
---

<objective>
Mirror Phase 2 Plan 02 + Phase 3 Plan 02: flip federated-adaptive-personalized-cf/pyproject.toml defaults to cross-device (6040 supernodes in both federation blocks, partition-mode="natural", 6 Phase-3 carry-forward contract keys + 5 Phase-4 adaptive-specific signature keys that drive schema_version=2 cache, pytest dev dep), and rip-and-replace dataset.py so it becomes a thin adapter that delegates mapping/split/exclusion to fedrec_foundation (D-17). Enforces D-02 NotImplementedError at BOTH dataset entry points for partition_mode='dirichlet'. Adds 5 GREEN tests (1 pyproject shape regression + 4 dataset-adapter).

Purpose: Closes ADP-01 in-file so `flwr run .` inside federated-adaptive-personalized-cf/ spawns 6040 supernodes by default. Establishes the foundation-backed dataset contract that Plan 03 (client_app.py + task.py) consumes without changing signatures. Fixes the 5 schema_version=2 signature keys (D-02 from CONTEXT) as pyproject defaults so Plan 03's cache manifest round-trips the thesis benchmark config.

Output:
- federated-adaptive-personalized-cf/pyproject.toml (edited: 6 Phase-3 keys + 5 Phase-4 keys + pytest dev dep + both federation num-supernodes flipped to 6040)
- federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py (rewritten as thin foundation adapter)
- federated-adaptive-personalized-cf/tests/test_pyproject_shape.py (new — 1 grep-level regression test)
- federated-adaptive-personalized-cf/tests/test_dataset_adapter.py (new — 4 GREEN tests)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md
@.planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md
@.planning/phases/02-baseline-migration/02-baseline-migration-02-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: pyproject.toml cross-device defaults + 6 Phase-3 keys + 5 Phase-4 keys + pytest dev dep (ADP-01)</name>
  <files>federated-adaptive-personalized-cf/pyproject.toml, federated-adaptive-personalized-cf/tests/test_pyproject_shape.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/pyproject.toml (ENTIRE FILE; need to see every existing key to avoid TOML duplicate-key errors + preserve existing `fedrec-foundation` dep from Phase 1 Plan 06 + all existing `[tool.flwr.app.config]` entries including model-type/alpha-method/fusion-type which already exist with different default values — e.g., current alpha-method = "multi_factor" must change to "hierarchical_conditional"; current contrastive-lambda = 0.0 must change to 0.1; current enable-per-user-alpha = false must change to true; current enable-item-perturbation = false must change to true)
    - federated-personalized-cf/pyproject.toml (CANONICAL Phase-3 TEMPLATE — the block shape to clone for the Phase-3 keys + cross-device num-supernodes + [project.optional-dependencies])
    - federated-baseline-cf/pyproject.toml (Phase-2 TEMPLATE for reference)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §D-09..D-12 (the 5 Phase-4 benchmark defaults locked by user)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §ADP-01 (exact default set)
    - .planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md (Plan 03-02 commit — exact pattern + gotcha: TOML duplicate-key autofix)
  </read_first>
  <action>
    Step 1 — Inventory with `git diff federated-adaptive-personalized-cf/pyproject.toml` to capture the pre-existing uncommitted WIP state (if any). Preserve WIP values that match or exceed the target — do not downgrade.

    Step 2 — Add `[project.optional-dependencies]` section above `[build-system]` (exact placement matches baseline):
    ```toml
    [project.optional-dependencies]
    dev = ["pytest>=7.0"]
    ```

    Step 3 — Under `[tool.flwr.app.config]`, append (or merge if any are already present) the following 6 Phase-3 carry-forward contract keys as a new block. If any key already exists elsewhere in the file, REMOVE the duplicate and consolidate here (TOML rejects duplicate top-level keys within a table):
    ```toml
    # ==== Phase 3 cross-device / foundation-contract keys (carry-forward from Phase 3 Plan 02) ====
    # The canonical runtime source is fedrec_foundation.mode.resolve_mode_defaults(mode).
    # These values are FALLBACK only (consulted when the launcher did not pre-set them).
    mode = "cross_silo_legacy"  # flipped to "benchmark_cross_device" by scripts/run.py adaptive benchmark_cross_device
    run-seed = 42
    weight-policy = "num_positives"
    eval-num-negatives = 99
    checkpoint-rule = "best_round_restore"
    reuse-cache = false  # D-09: set to true to opt into content-hash cache reuse across runs
    ```
    If the pre-existing `eval-num-negatives` key lives under a different section (e.g., "Evaluation protocol"), REMOVE the old one — TOML rejects duplicate keys.

    Step 4 — Under `[tool.flwr.app.config]`, ensure `partition-mode = "natural"` is set (create if absent; overwrite old value if present).

    Step 5 — Update the 5 Phase-4 signature-driving keys to the locked benchmark defaults (CONTEXT D-09..D-12):
    - `model-type = "dual"` — confirm present (CONTEXT D-09)
    - `alpha-method = "hierarchical_conditional"` — CHANGE FROM existing "multi_factor" (CONTEXT D-10)
    - `fusion-type = "concat"` — confirm present (CONTEXT D-11; already matches)
    - `enable-per-user-alpha = true` — CHANGE FROM existing false (CONTEXT D-03 + D-12; unconditional in benchmark mode)
    - `enable-item-perturbation = true` — CHANGE FROM existing false (CONTEXT D-03 + D-12; unconditional in benchmark mode)
    - `contrastive-lambda = 0.1` — CHANGE FROM existing 0.0 (CONTEXT D-12)

    Add a comment pointing at CONTEXT D-09..D-12 near these keys so the rationale is traceable.

    Step 6 — In `[tool.flwr.federations.local-simulation]`, flip `options.num-supernodes = 5` to `options.num-supernodes = 6040` and add a cross-silo opt-in comment:
    ```toml
    [tool.flwr.federations.local-simulation]
    # ==== Cross-device default: 1 user = 1 client (N=6040). ====
    # To reproduce legacy cross-silo numbers, check out a pre-Phase-4 commit and
    # override via: flwr run . --run-config "num-supernodes=5 partition-mode=dirichlet"
    options.num-supernodes = 6040
    ```

    Step 7 — Same flip in `[tool.flwr.federations.local-sim-gpu]`. Also append the cross-silo opt-in comment. Preserve existing `options.backend.client-resources.num-gpus = 0.2` / `options.backend.client-resources.num-cpus = 6` settings verbatim.

    Step 8 — Preserve verbatim: `fedrec-foundation` dep entry (Phase 1 Plan 06), `[tool.flwr.app.components]`, `[tool.flwr.federations.remote-federation]`, every other pre-existing `[tool.flwr.app.config]` key not explicitly addressed above (e.g., num-server-rounds, fraction-train, local-epochs, embedding-dim, dropout, lr, weight-decay, num-negatives, all mlp-hidden-dims / alpha-weight-* / alpha-max-* / alpha-hc-* / prototype-momentum / user-group-* / ranking-k-values / early-stopping-* / wandb-* keys), `[tool.hatch.build.targets.wheel] packages = ["."]`.

    Step 9 — Parse + assertion check:
    ```
    python -c "
    import tomllib
    with open('federated-adaptive-personalized-cf/pyproject.toml', 'rb') as f:
        d = tomllib.load(f)
    cfg = d['tool']['flwr']['app']['config']
    feds = d['tool']['flwr']['federations']
    assert feds['local-simulation']['options']['num-supernodes'] == 6040
    assert feds['local-sim-gpu']['options']['num-supernodes'] == 6040
    assert cfg['partition-mode'] == 'natural'
    assert cfg['run-seed'] == 42
    assert cfg['weight-policy'] == 'num_positives'
    assert cfg['eval-num-negatives'] == 99
    assert cfg['checkpoint-rule'] == 'best_round_restore'
    assert cfg['reuse-cache'] is False
    assert cfg['model-type'] == 'dual'
    assert cfg['alpha-method'] == 'hierarchical_conditional'
    assert cfg['fusion-type'] == 'concat'
    assert cfg['enable-per-user-alpha'] is True
    assert cfg['enable-item-perturbation'] is True
    assert abs(cfg['contrastive-lambda'] - 0.1) < 1e-9
    assert d['project']['optional-dependencies']['dev'] == ['pytest>=7.0']
    print('ok')
    "
    ```
    Expect `ok`.

    Step 10 — Create federated-adaptive-personalized-cf/tests/test_pyproject_shape.py (new test file):
    ```python
    """ADP-01 regression: federated-adaptive-personalized-cf/pyproject.toml cross-device defaults.

    Asserts the 11 cross-device-defining keys are present with the expected values:
    - 6 Phase-3 carry-forward (mode, run-seed, weight-policy, eval-num-negatives,
      checkpoint-rule, reuse-cache)
    - 5 Phase-4 signature-driving (alpha-method, fusion-type, enable-per-user-alpha,
      enable-item-perturbation, contrastive-lambda)
    PLUS the 2 federation num-supernodes flips AND the [dev] pytest extra.
    """
    import tomllib
    from pathlib import Path

    _PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


    def _load():
        with open(_PYPROJECT, "rb") as f:
            return tomllib.load(f)


    def test_num_supernodes_flipped_in_both_federations():
        d = _load()
        feds = d["tool"]["flwr"]["federations"]
        assert feds["local-simulation"]["options"]["num-supernodes"] == 6040
        assert feds["local-sim-gpu"]["options"]["num-supernodes"] == 6040


    def test_phase3_foundation_contract_keys_present():
        d = _load()
        cfg = d["tool"]["flwr"]["app"]["config"]
        assert cfg["partition-mode"] == "natural"
        assert cfg["mode"] == "cross_silo_legacy"  # launcher flips to benchmark_cross_device
        assert cfg["run-seed"] == 42
        assert cfg["weight-policy"] == "num_positives"
        assert cfg["eval-num-negatives"] == 99
        assert cfg["checkpoint-rule"] == "best_round_restore"
        assert cfg["reuse-cache"] is False


    def test_phase4_signature_keys_at_thesis_defaults():
        d = _load()
        cfg = d["tool"]["flwr"]["app"]["config"]
        assert cfg["model-type"] == "dual"
        assert cfg["alpha-method"] == "hierarchical_conditional"
        assert cfg["fusion-type"] == "concat"
        assert cfg["enable-per-user-alpha"] is True
        assert cfg["enable-item-perturbation"] is True
        assert abs(cfg["contrastive-lambda"] - 0.1) < 1e-9


    def test_dev_pytest_extra_declared():
        d = _load()
        assert d["project"]["optional-dependencies"]["dev"] == ["pytest>=7.0"]


    def test_fedrec_foundation_dep_preserved():
        d = _load()
        deps = d["project"]["dependencies"]
        assert any("fedrec-foundation" in dep for dep in deps), \
            "Phase 1 Plan 06 dep must be preserved"
    ```

    Step 11 — Verify tests: `cd federated-adaptive-personalized-cf && pip install --quiet "pytest>=7.0" 2>/dev/null; pytest tests/test_pyproject_shape.py -v` → 5 passed.

    Step 12 — Commit (--no-verify per Wave-1 parallel rule):
    ```
    git add federated-adaptive-personalized-cf/pyproject.toml \
            federated-adaptive-personalized-cf/tests/test_pyproject_shape.py
    git commit --no-verify -m "feat(04-02): adaptive pyproject cross-device defaults + schema-v2 keys + dev dep (ADP-01)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -cE "^options\\.num-supernodes = 6040" federated-adaptive-personalized-cf/pyproject.toml` returns 2 (local-simulation + local-sim-gpu)
    - `grep -cE "^options\\.num-supernodes = 5$" federated-adaptive-personalized-cf/pyproject.toml` returns 0 (legacy value eradicated)
    - `grep -cE '^partition-mode = "natural"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^mode = "cross_silo_legacy"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^run-seed = 42' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^weight-policy = "num_positives"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^eval-num-negatives = 99' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^checkpoint-rule = "best_round_restore"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^reuse-cache = false' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^model-type = "dual"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^alpha-method = "hierarchical_conditional"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^alpha-method = "multi_factor"' federated-adaptive-personalized-cf/pyproject.toml` returns 0 (old default eradicated)
    - `grep -cE '^fusion-type = "concat"' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^enable-per-user-alpha = true' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^enable-per-user-alpha = false' federated-adaptive-personalized-cf/pyproject.toml` returns 0
    - `grep -cE '^enable-item-perturbation = true' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^enable-item-perturbation = false' federated-adaptive-personalized-cf/pyproject.toml` returns 0
    - `grep -cE '^contrastive-lambda = 0\\.1$' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE '^contrastive-lambda = 0\\.0$' federated-adaptive-personalized-cf/pyproject.toml` returns 0 (old default eradicated)
    - `grep -cE '\[project\.optional-dependencies\]' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -cE 'dev = \["pytest>=7\\.0"\]' federated-adaptive-personalized-cf/pyproject.toml` returns 1
    - `grep -c "fedrec-foundation" federated-adaptive-personalized-cf/pyproject.toml` returns at least 1 (Phase 1 Plan 06 dep preserved)
    - `python -c "import tomllib; d = tomllib.load(open('federated-adaptive-personalized-cf/pyproject.toml', 'rb')); cfg = d['tool']['flwr']['app']['config']; assert cfg['enable-per-user-alpha'] is True and cfg['enable-item-perturbation'] is True and cfg['alpha-method'] == 'hierarchical_conditional' and cfg['fusion-type'] == 'concat' and abs(cfg['contrastive-lambda'] - 0.1) < 1e-9; print('ok')"` prints `ok`
    - `cd federated-adaptive-personalized-cf && pytest tests/test_pyproject_shape.py -v` exits 0 with "5 passed"
    - `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` returns empty after commit (other source files untouched by Task 1)
  </acceptance_criteria>
  <done>pyproject.toml flipped to cross-device defaults (6040 in both federations, partition-mode="natural", 6 Phase-3 + 5 Phase-4 contract keys including thesis defaults per CONTEXT D-09..D-12, pytest dev dep). ADP-01 satisfied in-file. `flwr run .` inside federated-adaptive-personalized-cf/ now resolves to cross-device with the full thesis benchmark config (dual + hierarchical_conditional + concat + per-user-alpha-ON + item-perturbation-ON + contrastive-lambda=0.1) without relying on scripts/run.py.</done>
</task>

<task type="auto">
  <name>Task 2: dataset.py rip-and-replace to foundation adapter + D-02 NotImplementedError at BOTH entry points + 4 GREEN adapter tests (D-17, D-18, ADP-01 data layer)</name>
  <files>federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py, federated-adaptive-personalized-cf/tests/test_dataset_adapter.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py (ENTIRE FILE; need to inventory which helpers to remove per D-17 and which pre-existing WIP functions to preserve per D-18)
    - federated-personalized-cf/federated_personalized_cf/dataset.py (CANONICAL Phase-3 TEMPLATE — the post-Plan-02 shape to mirror)
    - federated-personalized-cf/tests/test_dataset_adapter.py (3-test TEMPLATE — clone and extend with the "raises at BOTH load_partition_data AND load_full_data" sharpening)
    - scripts/foundation/fedrec_foundation/bundle.py (verify_bundle signature + FoundationIndex fields)
    - scripts/foundation/fedrec_foundation/mapping.py (load_mapping → CanonicalMapping with num_users, num_items, user2idx, item2idx fields)
    - scripts/foundation/fedrec_foundation/split.py (load_split_manifest → SplitManifest with test_item_per_user dict + train_user_stats per-user PerUserStats)
    - scripts/foundation/fedrec_foundation/exclusion.py (load_exclusion → ExclusionTable with .for_user(idx))
    - scripts/foundation/fedrec_foundation/paths.py (data_derived() function)
    - .planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md (Phase-3 Plan 02 commit — D-02 NotImplementedError at BOTH entry points, tightened from baseline Plan 02 which raised in only one)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §"Claude's Discretion > Cross-silo legacy freeze" (Phase 3 D-02 mirror)
  </read_first>
  <action>
    Step 1 — Inventory pre-existing dataset.py state with `git diff federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py`. Record which functions are D-17 rip targets (remove) vs D-18 pre-existing WIP (preserve verbatim).

    D-17 REMOVE (clone Phase-3 Plan 02's list):
    - `create_global_mappings` (if present)
    - `create_leave_one_out_split` (if present)
    - `dirichlet_partition_users` (if present — superseded by D-02 NotImplementedError)
    - `create_train_test_split` (if present)
    - `compute_user_genre_distribution` (if present — this module specifically uses it for alpha computation; check if the adaptive task.py still imports it. If yes, move the helper to a private adapter-internal function that reads the equivalent stats from `bundle["split_manifest"].train_user_stats[partition_id]` PerUserStats — Phase 1 Plan 02 CR-5 already exposes n_interactions / genre_entropy / n_unique_items / rating_std per user via PerUserStats.)
    - `_partition_cache` module dict (if present)

    D-18 PRESERVE VERBATIM (if present):
    - `MovieLensDataset` torch.utils.data.Dataset subclass (used by trainloader/testloader construction)
    - `download_movielens_1m` helper (with ML-1M URL)
    - `load_movielens_1m` parser
    - `natural_partition_users` (per-user partitioning — essential for cross-device 1-user-per-client)

    Step 2 — Rewrite dataset.py as a thin foundation adapter. Module-level imports should include:
    ```python
    import hashlib
    import json
    import os
    from pathlib import Path
    from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
    from urllib.request import urlretrieve
    import zipfile

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset

    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
    from fedrec_foundation.mapping import CanonicalMapping, load_mapping
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.split import SplitManifest, load_split_manifest
    ```

    Add module-level `_foundation_cache: Dict[str, Any] = {}` keyed by `foundation_contract_sha256` (replaces the old `_partition_cache`).

    Step 3 — Add `_load_foundation_bundle(data_dir: Optional[Path] = None) -> Dict[str, Any]` (verbatim clone of Phase-3 `federated-personalized-cf/federated_personalized_cf/dataset.py`):
    ```python
    def _load_foundation_bundle(data_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Load + cache the foundation bundle (mapping, split_manifest, exclusion)."""
        derived = data_dir if data_dir is not None else data_derived()
        index = verify_bundle(derived)
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

    Step 4 — Rewrite `load_partition_data(partition_id, num_partitions, ..., partition_mode="natural", split_mode="leave_one_out", compute_stats=False, ...)` keeping the existing signature (the adaptive client_app.py calls with compute_stats=True):
    - If `partition_mode == "dirichlet"`: raise `NotImplementedError("Adaptive cross-device migration removed multi-user-per-client support per D-02. Check out a pre-Phase-4 commit (see .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §Deferred) to reproduce legacy cross-silo numbers.")`.
    - If `split_mode == "random"`: raise `ValueError("split_mode='random' is no longer supported — the foundation's leave-one-out split is authoritative. Use split_mode='leave_one_out'.")`
    - Otherwise: call `_load_foundation_bundle()`, use `bundle["mapping"]` for user2idx/item2idx, use `bundle["split_manifest"].test_item_per_user[partition_id]` for the held-out test item, build the per-user partition DataLoader(s). When `compute_stats=True`, derive `user_stats: Dict[str, float]` from `bundle["split_manifest"].train_user_stats[partition_id]` PerUserStats (expose n_interactions / genre_entropy / n_unique_items / rating_std as the dict keys the adaptive client_app.py expects — confirm key names by reading existing task.py::compute_client_alpha).
    - Return signature MUST preserve the adaptive-module contract: `(trainloader, testloader, user_stats)` (3 items) — this is different from baseline/personalized which return 6 items. Read the existing client_app.py call at around line 323 (`trainloader, _, user_stats = load_data(...)`) to confirm the return arity and adjust accordingly.

    Step 5 — Rewrite `load_full_data(..., partition_mode="natural")`:
    - If `partition_mode == "dirichlet"`: raise the same `NotImplementedError` as step 4. **D-02 tightening (Phase-3 Plan 02 pattern): BOTH entry points raise — this is a departure from baseline Plan 02 which raised only in load_partition_data.**
    - Otherwise: delegate mapping/split to foundation; build full-dataset LOO mask from `bundle["split_manifest"].test_item_per_user` across all users.

    Step 6 — Add a `load_data(partition_id, num_partitions, alpha=0.5, compute_stats=False, split_mode="leave-one-out", partition_mode="natural")` wrapper if the existing client_app.py calls `load_data(...)` rather than `load_partition_data(...)`. The adaptive module specifically uses `load_data`. Wire it to call `load_partition_data` internally (D-18 surgical — same keyword args, routed to the new adapter).

    Step 7 — Remove the 5-6 D-17 targets listed in Step 1. Verify by grep after edit.

    Step 8 — Create federated-adaptive-personalized-cf/tests/test_dataset_adapter.py (mirror federated-personalized-cf/tests/test_dataset_adapter.py with D-02 tightening):
    ```python
    """Adapter tests for federated_adaptive_personalized_cf.dataset (Phase 4 Plan 02, D-17 + D-02 tightening)."""
    from pathlib import Path
    import pytest

    _FOUNDATION_INDEX = Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json"
    pytestmark = pytest.mark.skipif(
        not _FOUNDATION_INDEX.exists(),
        reason="foundation bundle not committed (data/derived/foundation_index.json missing)",
    )

    from federated_adaptive_personalized_cf.dataset import (
        _load_foundation_bundle, load_data, load_full_data,
    )
    import federated_adaptive_personalized_cf.dataset as _ds_mod


    def test_load_data_uses_foundation_mapping():
        trainloader, testloader, user_stats = load_data(
            partition_id=0, num_partitions=6040, alpha=0.5,
            compute_stats=True, split_mode="leave-one-out", partition_mode="natural",
        )
        bundle = _load_foundation_bundle()
        assert bundle["mapping"].num_users == 6040
        assert bundle["mapping"].num_items == 3706


    def test_load_data_test_item_from_foundation_split():
        bundle = _load_foundation_bundle()
        pid = 0
        expected_test_item = bundle["split_manifest"].test_item_per_user[pid]
        trainloader, testloader, _ = load_data(
            partition_id=pid, num_partitions=6040, alpha=0.5,
            compute_stats=False, split_mode="leave-one-out", partition_mode="natural",
        )
        all_test_items: list[int] = []
        for batch in testloader:
            items = batch.get("item") if isinstance(batch, dict) else None
            if items is not None:
                all_test_items.extend(int(x) for x in items.tolist())
        assert expected_test_item in all_test_items, \
            f"Foundation split's test_item={expected_test_item} missing from testloader contents {all_test_items[:20]}..."


    def test_removed_helpers_gone():
        # D-17: these helpers must NOT appear as module attributes
        for name in ("create_global_mappings", "create_leave_one_out_split",
                     "dirichlet_partition_users", "create_train_test_split",
                     "compute_user_genre_distribution", "_partition_cache"):
            assert not hasattr(_ds_mod, name), f"D-17 violated: {name} still present in dataset.py"


    def test_dirichlet_raises_at_both_entry_points():
        # D-02 tightening (Phase-3 Plan 02 pattern): BOTH load_data/load_partition_data AND load_full_data raise.
        with pytest.raises(NotImplementedError, match="cross-device|D-02|pre-Phase-4"):
            load_data(partition_id=0, num_partitions=5, alpha=0.5,
                      partition_mode="dirichlet")
        with pytest.raises(NotImplementedError, match="cross-device|D-02|pre-Phase-4"):
            load_full_data(partition_mode="dirichlet")
    ```

    Step 9 — Verify: `cd federated-adaptive-personalized-cf && pytest tests/test_dataset_adapter.py -v` → 4 passed. Also run import smoke: `python -c "from federated_adaptive_personalized_cf.dataset import load_data, _load_foundation_bundle; b = _load_foundation_bundle(); assert b['mapping'].num_users == 6040; print('ok')"` → `ok`.

    Step 10 — D-18 scope check: `git diff --name-only HEAD~1..HEAD` should list ONLY pyproject.toml + test_pyproject_shape.py (from Task 1), dataset.py, and tests/test_dataset_adapter.py. `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/early_stopping.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/ federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/` returns empty (Plans 01/03/05 own these).

    Step 11 — Commit (--no-verify):
    ```
    git add federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py \
            federated-adaptive-personalized-cf/tests/test_dataset_adapter.py
    git commit --no-verify -m "refactor(04-02): rip-and-replace dataset.py as foundation adapter (D-17, D-02 both entry points, ADP-01)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.mapping import" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1
    - `grep -c "from fedrec_foundation.split import" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1
    - `grep -c "from fedrec_foundation.exclusion import" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1
    - `grep -c "from fedrec_foundation.bundle import verify_bundle" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1
    - `grep -c "def _load_foundation_bundle" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1
    - `grep -cE "raise NotImplementedError" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns at least 2 (D-02 at BOTH load_data/load_partition_data AND load_full_data — Phase-3 tightening)
    - `grep -c "def create_global_mappings" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 0
    - `grep -c "def create_leave_one_out_split" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 0
    - `grep -c "def dirichlet_partition_users" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 0
    - `grep -c "def create_train_test_split" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 0
    - `grep -c "_partition_cache" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 0
    - `grep -c "def natural_partition_users" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1 (D-18 preserved)
    - `grep -cE "def (load_data|load_partition_data)" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns at least 1 (adaptive client_app.py calls load_data)
    - `grep -c "def load_full_data" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` returns 1
    - `cd federated-adaptive-personalized-cf && pytest tests/test_dataset_adapter.py -v` exits 0 with "4 passed"
    - `python -c "from federated_adaptive_personalized_cf.dataset import load_data
try:
    load_data(partition_id=0, num_partitions=5, alpha=0.5, partition_mode='dirichlet')
    raise AssertionError('expected NotImplementedError')
except NotImplementedError as e:
    assert 'cross-device' in str(e) or 'D-02' in str(e) or 'pre-Phase-4' in str(e)
print('ok')"` prints `ok`
    - `python -c "from federated_adaptive_personalized_cf.dataset import load_full_data
try:
    load_full_data(partition_mode='dirichlet')
    raise AssertionError('expected NotImplementedError')
except NotImplementedError as e:
    assert 'cross-device' in str(e) or 'D-02' in str(e) or 'pre-Phase-4' in str(e)
print('ok')"` prints `ok`
    - `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/` returns empty after the commit (D-18 surgical discipline)
  </acceptance_criteria>
  <done>dataset.py is a thin foundation adapter (D-17): mapping/split/exclusion delegated to fedrec_foundation; partition_mode='dirichlet' raises NotImplementedError per D-02 at BOTH load_data/load_partition_data AND load_full_data with explicit reference to pre-Phase-4 commits; 5 D-17 rip targets absent from module; 4 D-18 pre-existing functions preserved verbatim; 4 GREEN adapter tests.</done>
</task>

</tasks>

<verification>
- `python -c "import tomllib; d = tomllib.load(open('federated-adaptive-personalized-cf/pyproject.toml', 'rb')); cfg = d['tool']['flwr']['app']['config']; feds = d['tool']['flwr']['federations']; assert feds['local-simulation']['options']['num-supernodes'] == 6040; assert feds['local-sim-gpu']['options']['num-supernodes'] == 6040; assert cfg['reuse-cache'] is False and cfg['enable-per-user-alpha'] is True and cfg['enable-item-perturbation'] is True and cfg['alpha-method'] == 'hierarchical_conditional'; print('ok')"` prints `ok`
- `cd federated-adaptive-personalized-cf && pytest tests/test_pyproject_shape.py tests/test_dataset_adapter.py -v` exits 0 with "9 passed" (5 pyproject + 4 dataset)
- `python scripts/run.py --dry-run adaptive benchmark_cross_device 2>&1 | grep -c "num-supernodes=6040"` returns at least 1 (launcher consistent with in-file defaults)
- D-18 scope: `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/` returns empty (Plan 01 owns strategy.py; Plans 03/05 own client/server/task)
</verification>

<success_criteria>
- ADP-01 is observable in-file: `flwr run .` inside federated-adaptive-personalized-cf/ spawns 6040 supernodes by default under `partition-mode="natural"` with the full thesis benchmark config (dual + hierarchical_conditional + concat + per-user-alpha ON + item-perturbation ON + contrastive-lambda 0.1).
- Cross-silo opt-in preserved as documented fallback: `flwr run . --run-config "num-supernodes=5 partition-mode=dirichlet"` still dispatches — but `dirichlet` now raises NotImplementedError per D-02 at BOTH load_data AND load_full_data (tests prove this).
- dataset.py contract (load_data / load_full_data) signatures unchanged; Plan 03 can wire client_app.py against them without signature churn; user_stats dict continues to be populated from `bundle["split_manifest"].train_user_stats[partition_id]` PerUserStats, so compute_client_alpha / compute_per_user_alpha still work unchanged.
- Wave-1 write-race safety: this plan's 2 commits touch only pyproject.toml, dataset.py, test_pyproject_shape.py, and test_dataset_adapter.py. Plan 01's 2 commits touch only strategy.py + 4 test files. Zero file overlap.
</success_criteria>

<output>
After completion, create `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-02-SUMMARY.md` with: file list (4 files: pyproject.toml, dataset.py, test_pyproject_shape.py, test_dataset_adapter.py), decisions made (any pre-existing WIP hunks preserved or replaced with rationale; how user_stats dict keys were sourced from PerUserStats; any load_data vs load_partition_data signature harmonization), deviations, test counts (5 pyproject + 4 dataset = 9 GREEN), commit SHAs, ADP-01 closure note, Plan 03 readiness confirmation.
</output>
</content>
</invoke>