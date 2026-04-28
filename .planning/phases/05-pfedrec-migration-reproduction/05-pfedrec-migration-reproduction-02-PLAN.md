---
phase: 05-pfedrec-migration-reproduction
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - federated-pfedrec/pyproject.toml
  - federated-pfedrec/federated_pfedrec/dataset.py
  - scripts/foundation/fedrec_foundation/mode.py
  - federated-pfedrec/tests/test_pyproject.py
  - federated-pfedrec/tests/test_dataset.py
  - scripts/foundation/tests/test_mode.py
autonomous: true
requirements: [PFR-01, PFR-02, PFR-09]
must_haves:
  truths:
    - "federated-pfedrec/pyproject.toml declares num-supernodes=6040 in BOTH local-simulation AND local-sim-gpu federation blocks (PFR-01)"
    - "pyproject.toml carries 6 Phase-5 contract keys: mode, run-seed, weight-policy, reuse-cache, eval-num-negatives, checkpoint-rule"
    - "pyproject.toml [project.optional-dependencies] dev = ['pytest>=7.0'] (Wave-1 dev pytest dep ownership)"
    - "dataset.py is rip-and-replaced as a foundation adapter: load_partition_data and load_full_data delegate mapping/split/exclusion to fedrec_foundation"
    - "dataset.py raises NotImplementedError at BOTH load_partition_data AND load_full_data when partition_mode != 'natural' (D-09 mirror of Phase 3 D-02; tokens 'D-09', 'cross-device', 'pre-Phase-5' all in error message)"
    - "scripts/foundation/fedrec_foundation/mode.py: _PAPER_COMPAT_PFEDREC.weight_policy == 'uniform' (D-25 Phase-1 deferred-decision closed) and the 'Deferred confirmation to PFR-02' comment is removed"
  artifacts:
    - path: "federated-pfedrec/pyproject.toml"
      provides: "Cross-device PFedRec config with PFR-01 num-supernodes=6040 + Phase-5 contract keys + dev pytest extra"
      contains: "options.num-supernodes = 6040"
    - path: "federated-pfedrec/federated_pfedrec/dataset.py"
      provides: "Foundation adapter; D-09 frozen-cross-silo guard at both entry points"
      contains: "raise NotImplementedError"
    - path: "scripts/foundation/fedrec_foundation/mode.py"
      provides: "_PAPER_COMPAT_PFEDREC.weight_policy == 'uniform' (D-25)"
      contains: "weight_policy=\"uniform\""
  key_links:
    - from: "federated-pfedrec/pyproject.toml [tool.flwr.federations.local-simulation]"
      to: "scripts/run.py launcher invocation of paper_compat_pfedrec"
      via: "options.num-supernodes = 6040"
      pattern: "num-supernodes = 6040"
    - from: "federated-pfedrec/federated_pfedrec/dataset.py::load_partition_data"
      to: "fedrec_foundation.bundle.verify_bundle + split.load_split_manifest + exclusion.load_exclusion"
      via: "_load_foundation_bundle helper (Phase 3 idiom)"
      pattern: "verify_bundle"
    - from: "_PAPER_COMPAT_PFEDREC.weight_policy"
      to: "fedrec_foundation.weight_policy.WeightPolicy.UNIFORM"
      via: "ModeProfile.weight_policy field"
      pattern: "weight_policy=\"uniform\""
---

<objective>
pyproject + dataset adapter + foundation mode.py D-25 update (Wave-1 disjoint-file ownership with Plan 01).

Purpose:
  - PFR-01: Flip federated-pfedrec to cross-device defaults (num-supernodes=6040 in BOTH federation blocks; partition-mode=natural; all 6 Phase-5 contract keys including default `mode = "paper_compat_pfedrec"`).
  - D-25: Close Phase 1 deferred decision — change `_PAPER_COMPAT_PFEDREC.weight_policy` from 'num_positives' to 'uniform' in `scripts/foundation/fedrec_foundation/mode.py`; remove the "Deferred confirmation to PFR-02" comment.
  - D-17 + D-18 surgical rip-and-replace: dataset.py becomes a thin (~440 LOC) foundation adapter delegating mapping/split/exclusion to `fedrec_foundation`. D-09 NotImplementedError fires at BOTH load_partition_data AND load_full_data when partition_mode != "natural" (Phase 3 / Phase 4 tightening pattern).
  - Add `[project.optional-dependencies] dev = ['pytest>=7.0']` for `pip install -e ".[dev]"`.

Output:
  - 3 modified files: pyproject.toml + dataset.py (in `federated-pfedrec`) + mode.py (in `scripts/foundation/fedrec_foundation`).
  - 3 new test files: test_pyproject.py + test_dataset.py + extension to test_mode.py — total ~7 new GREEN tests.
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
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-02-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/pyproject.toml
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/dataset.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/fedrec_foundation/mode.py

<interfaces>
<!-- Phase 3 dataset.py rip-and-replace shape -->
<!-- Source: federated-personalized-cf/federated_personalized_cf/dataset.py (post-Phase-3-Plan-02) -->

```python
# Foundation bundle loader pattern (Phase 3 / Phase 4)
from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
from fedrec_foundation.paths import data_derived
from fedrec_foundation.split import load_split_manifest

@dataclass
class _FoundationBundle:
    mapping: ItemMapping  # 6040 users x 3706 items
    split_manifest: SplitManifest  # train/test splits + train_user_stats
    exclusion: ExclusionTable  # FND-03 per-user exclusion sets
    raw_data_hash: str
    mapping_sha256: str

def _load_foundation_bundle() -> _FoundationBundle:
    """Phase 3 idiom: verify_bundle once + return all four artifacts."""
    derived_dir = data_derived()
    verify_bundle(derived_dir)
    mapping = load_mapping(derived_dir / "mapping.json")
    split_manifest = load_split_manifest(derived_dir / "split_manifest.json")
    exclusion = load_exclusion(derived_dir / "exclusion_items.npz")
    # ... return bundle
```

```python
# D-09 / D-02 NotImplementedError tokens (Phase 3 + Phase 4)
raise NotImplementedError(
    "PFedRec cross-silo (partition_mode='dirichlet') path is FROZEN per "
    "Phase 5 D-09. Cross-device migration only. See "
    ".planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md "
    "§Deferred. To re-run cross-silo PFedRec, check out a pre-Phase-5 commit."
)
```

```python
# scripts/foundation/fedrec_foundation/mode.py current state
_PAPER_COMPAT_PFEDREC = ModeProfile(
    mode="paper_compat_pfedrec",
    num_supernodes=6040,
    partition_mode="natural",
    # Deferred confirmation to PFR-02; may be overridden per-module.   <-- REMOVE THIS COMMENT
    weight_policy="num_positives",                                       <-- CHANGE TO "uniform"
    primary_evaluator="sampled_loo_99",
    fraction_train=1.0,
    ...
)
```

```toml
# Phase 5 pyproject.toml additions / changes
[tool.flwr.app.config]
# Mode + Phase-5 contract keys (D-25 mode resolver canonical hyperparam source)
mode = "paper_compat_pfedrec"
run-seed = 42
weight-policy = "uniform"           # D-24
reuse-cache = false                 # D-18
eval-num-negatives = 99
checkpoint-rule = "best_round_restore"  # D-13

# (existing PFedRec keys remain — paper-compat hyperparam lock per D-15)

[tool.flwr.federations.local-simulation]
options.num-supernodes = 6040       # PFR-01 (was 5)

[tool.flwr.federations.local-sim-gpu]
options.num-supernodes = 6040       # PFR-01 (was 5)
options.backend.client-resources.num-cpus = 12
options.backend.client-resources.num-gpus = 0.2

[project.optional-dependencies]
dev = ["pytest>=7.0"]                # Wave-1 dev pytest dep
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Update pyproject.toml + scripts/foundation/fedrec_foundation/mode.py D-25; ship test_pyproject.py + test_mode.py extension with 4 GREEN tests</name>
  <files>federated-pfedrec/pyproject.toml, scripts/foundation/fedrec_foundation/mode.py, federated-pfedrec/tests/test_pyproject.py, scripts/foundation/tests/test_mode.py</files>
  <read_first>
    - federated-pfedrec/pyproject.toml — current state (num-supernodes=5; missing Phase-5 contract keys; missing [dev] extra)
    - scripts/foundation/fedrec_foundation/mode.py — current `_PAPER_COMPAT_PFEDREC` profile (line 137-155 in current file; weight_policy="num_positives" + the deferred-confirmation comment)
    - scripts/foundation/tests/test_mode.py — current test file structure (so the new D-25 regression test follows the same idiom)
    - .planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md — Phase 3 PSN-01 in-file pyproject pattern (mirror exactly)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-02-SUMMARY.md — Phase 4 ADP-01 cross-device defaults pattern
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-15, D-18, D-24, D-25
  </read_first>
  <behavior>
    - Test 1 (test_num_supernodes_6040): parse pyproject.toml; assert `[tool.flwr.federations.local-simulation].options.num-supernodes == 6040`; assert `[tool.flwr.federations.local-sim-gpu].options.num-supernodes == 6040`.
    - Test 2 (test_partition_mode_natural): parse pyproject.toml; assert `[tool.flwr.app.config].partition-mode == "natural"`; assert `[tool.flwr.app.config].mode == "paper_compat_pfedrec"`; assert `[tool.flwr.app.config].weight-policy == "uniform"`; assert `[tool.flwr.app.config].run-seed == 42`; assert `[tool.flwr.app.config].reuse-cache == false`.
    - Test 3 (test_dev_extra_pytest_present): parse pyproject.toml; assert `[project.optional-dependencies].dev` is a list containing the prefix string `"pytest>=7.0"`.
    - Test 4 (scripts/foundation/tests/test_mode.py::test_paper_compat_pfedrec_weight_policy_uniform): import `from fedrec_foundation.mode import resolve_mode_defaults`; assert `resolve_mode_defaults("paper_compat_pfedrec").weight_policy == "uniform"`; assert `resolve_mode_defaults("paper_compat_pfedrec").fraction_train == 1.0`; assert `resolve_mode_defaults("paper_compat_pfedrec").num_supernodes == 6040`. Read mode.py source as text; assert the literal substring `"Deferred confirmation to PFR-02"` is NOT present anywhere in the file (D-25 comment removal regression guard).
  </behavior>
  <action>

**1. Update `scripts/foundation/fedrec_foundation/mode.py` `_PAPER_COMPAT_PFEDREC` profile (lines 137-155 in current file):**

Replace the existing dataclass instantiation with:

```python
_PAPER_COMPAT_PFEDREC = ModeProfile(
    mode="paper_compat_pfedrec",
    num_supernodes=6040,
    partition_mode="natural",
    # D-24/D-25: Reference engine.py:81 divides by len(round_user_params),
    # i.e. uniform weight = 1 per participating client. PFR-08 reproduction
    # requires this. Closes Phase 1 deferred decision.
    weight_policy="uniform",
    primary_evaluator="sampled_loo_99",
    fraction_train=1.0,       # D-06: paper uses full participation
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=32,
    optimizer="sgd",
    lr=0.1,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",
    assert_one_user_per_client=True,
)
```

CRITICAL: Remove the prior comment `# Deferred confirmation to PFR-02; may be overridden per-module.` entirely. The Phase 1 deferred-decision marker is closed by Phase 5; leaving the comment would be a documentation regression.

**2. Update `federated-pfedrec/pyproject.toml`:**

Apply these surgical edits (preserve the `[build-system]`, `[tool.hatch.build.targets.wheel]`, `[tool.flwr.app]`, `[tool.flwr.app.components]` blocks verbatim; preserve all paper-compat hyperparams already present such as `lr=0.1`, `lr-eta=80`, `latent-dim=32`, `num-server-rounds=100`, `local-epochs=1`, `num-negatives=4`, `l2-regularization=0.0`, `batch-size=256`):

(a) Append to `[project]` block (after `dependencies` list closes):
```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]
```

(b) In `[tool.flwr.app.config]` block, add the following six Phase-5 contract keys at the top (or wherever readable; before existing keys is fine):
```toml
# Phase-5 contract keys (D-25 mode resolver canonical hyperparam source).
# Pyproject is the override surface; the mode profile owns canonical values.
mode = "paper_compat_pfedrec"     # D-05: paper_compat is the only PFedRec mode
run-seed = 42                     # FND-06 single source of truth
weight-policy = "uniform"         # D-24 — matches engine.py:81
reuse-cache = false               # D-18 default; --run-config "reuse-cache=true" opt-in
eval-num-negatives = 99           # NCF protocol (FND-04)
checkpoint-rule = "best_round_restore"  # D-13 best-round monitor against sampled_ndcg@10
```

(c) In the same `[tool.flwr.app.config]` block, ensure `partition-mode = "natural"` is present and the `strategy = "fedavg"` key remains (D-07 dropped FedProx in strategy.py — Plan 01 — but pyproject can keep `strategy = "fedavg"` as the only valid value; do NOT add `proximal-mu` if it's not already there). Drop `proximal-mu` if currently present (Plan 01 dropped FedProx).

(d) Update BOTH federation blocks to flip num-supernodes from 5 to 6040:
```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 6040

[tool.flwr.federations.local-sim-gpu]
options.num-supernodes = 6040
options.backend.client-resources.num-cpus = 12
options.backend.client-resources.num-gpus = 0.2
```

(e) Update `wandb-project = "federated-pfedrec"` to `wandb-project = ""` (empty string → server_app.py defaults to `federated-cf-cross-device` per D-10 / Phase-3 pattern).

(f) Preserve all existing paper-compat keys verbatim (D-15 strict hyperparam lock — no parallel ablation knobs in pyproject).

**3. Create `federated-pfedrec/tests/test_pyproject.py`:**

```python
"""Phase 5 PFR-01 regression guard: cross-device defaults in pyproject.toml."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_num_supernodes_6040(cfg) -> None:
    """PFR-01: BOTH federation blocks declare num-supernodes=6040."""
    fed = cfg["tool"]["flwr"]["federations"]
    assert fed["local-simulation"]["options"]["num-supernodes"] == 6040
    assert fed["local-sim-gpu"]["options"]["num-supernodes"] == 6040


def test_partition_mode_natural(cfg) -> None:
    """PFR-01 + D-25 contract keys present."""
    app_cfg = cfg["tool"]["flwr"]["app"]["config"]
    assert app_cfg["partition-mode"] == "natural"
    assert app_cfg["mode"] == "paper_compat_pfedrec"
    assert app_cfg["weight-policy"] == "uniform"
    assert app_cfg["run-seed"] == 42
    assert app_cfg["reuse-cache"] is False
    assert app_cfg["eval-num-negatives"] == 99
    assert app_cfg["checkpoint-rule"] in ("best_round_restore", "best_round")


def test_dev_extra_pytest_present(cfg) -> None:
    """[project.optional-dependencies] dev = ['pytest>=7.0']."""
    deps = cfg["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("pytest>=7.0") for d in deps), deps
```

**4. Extend `scripts/foundation/tests/test_mode.py` with the D-25 regression guard.**

Append (do NOT replace any existing test) the following test function:

```python
def test_paper_compat_pfedrec_weight_policy_uniform() -> None:
    """D-25: _PAPER_COMPAT_PFEDREC.weight_policy is 'uniform' (was 'num_positives' pre-PFR-02).

    Phase 1 deferred this decision. Phase 5 closes it: reference engine.py:81
    divides by len(round_user_params) — uniform weight per participating client.
    """
    from fedrec_foundation.mode import resolve_mode_defaults

    profile = resolve_mode_defaults("paper_compat_pfedrec")
    assert profile.weight_policy == "uniform", (
        f"D-25: expected 'uniform', got {profile.weight_policy!r}"
    )
    assert profile.fraction_train == 1.0, "D-06: paper uses full participation"
    assert profile.num_supernodes == 6040
    assert profile.optimizer == "sgd"
    assert profile.lr == 0.1
    assert profile.embedding_dim == 32

    # D-25 documentation regression guard: comment must be removed.
    import inspect

    import fedrec_foundation.mode as _m

    src = inspect.getsource(_m)
    assert "Deferred confirmation to PFR-02" not in src, (
        "D-25 closure incomplete: 'Deferred confirmation to PFR-02' comment "
        "still in mode.py — remove it when flipping weight_policy to 'uniform'."
    )
```

Verify: run `pytest scripts/foundation/tests/test_mode.py -x -v -k paper_compat_pfedrec_weight_policy_uniform` — green. Run `cd federated-pfedrec && pytest tests/test_pyproject.py -x -v` — 3 green.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && pytest scripts/foundation/tests/test_mode.py::test_paper_compat_pfedrec_weight_policy_uniform federated-pfedrec/tests/test_pyproject.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "options.num-supernodes = 6040" federated-pfedrec/pyproject.toml` returns exactly 2 (one per federation block)
    - `grep -c "options.num-supernodes = 5" federated-pfedrec/pyproject.toml` returns 0
    - `grep -c 'partition-mode = "natural"' federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c 'mode = "paper_compat_pfedrec"' federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c 'weight-policy = "uniform"' federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c 'run-seed = 42' federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c 'reuse-cache = false' federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c "checkpoint-rule" federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c "pytest>=7.0" federated-pfedrec/pyproject.toml` returns at least 1
    - `grep -c 'weight_policy="uniform"' scripts/foundation/fedrec_foundation/mode.py` returns 1
    - `grep -c 'weight_policy="num_positives"' scripts/foundation/fedrec_foundation/mode.py` (in `_PAPER_COMPAT_PFEDREC` block specifically) — combined with surrounding context must show only `_BENCHMARK_CROSS_DEVICE` keeps `"num_positives"`
    - `grep -c "Deferred confirmation to PFR-02" scripts/foundation/fedrec_foundation/mode.py` returns 0
    - `pytest scripts/foundation/tests/test_mode.py::test_paper_compat_pfedrec_weight_policy_uniform federated-pfedrec/tests/test_pyproject.py -x -v` exits 0 with 4 tests passed
  </acceptance_criteria>
  <done>
    - PFR-01 land: pyproject.toml flipped to cross-device 6040 supernodes in both federations + 6 Phase-5 contract keys + dev pytest extra
    - D-25 closes Phase 1 deferred decision: weight_policy='uniform' in mode.py + comment removed
    - 3 GREEN pyproject regression tests + 1 GREEN foundation regression test
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Rip-and-replace federated-pfedrec/dataset.py as foundation adapter; ship test_dataset.py with D-09 NotImplementedError coverage</name>
  <files>federated-pfedrec/federated_pfedrec/dataset.py, federated-pfedrec/tests/test_dataset.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/dataset.py — current state (D-18 surgical: preserve MovieLensDataset, download_movielens_1m, load_movielens_1m, natural_partition_users verbatim; replace mapping/split/exclusion helpers with foundation adapter)
    - federated-personalized-cf/federated_personalized_cf/dataset.py — Phase 3 reference adapter shape (~440 LOC)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py — Phase 4 reference adapter shape (preserves user_stats 7-tuple via _per_user_stats_to_dict translator)
    - .planning/phases/03-personalized-migration/03-personalized-migration-02-SUMMARY.md §D-17 rip-and-replace + D-18 surgical-edit
    - .planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-02-SUMMARY.md §D-02 NotImplementedError tightening pattern
    - scripts/foundation/fedrec_foundation/bundle.py — verify_bundle signature
    - scripts/foundation/fedrec_foundation/exclusion.py — load_exclusion + ExclusionTable.for_user signatures
    - scripts/foundation/fedrec_foundation/split.py — load_split_manifest + SplitManifest field shapes
  </read_first>
  <behavior>
    - Test 1 (test_load_partition_data_raises_on_non_natural): import dataset.load_partition_data; call with `partition_id=0, partition_mode="dirichlet"`; assert `pytest.raises(NotImplementedError)` whose message contains the literal substrings `"D-09"`, `"cross-device"`, AND `"pre-Phase-5"`. Test does NOT need the foundation bundle — the guard fires BEFORE any data load.
    - Test 2 (test_load_full_data_raises_on_non_natural): import dataset.load_full_data; call with `partition_mode="dirichlet"`; assert same `NotImplementedError` with same 3 token substrings.
    - Test 3 (test_dataset_uses_foundation_adapter_imports): read dataset.py source as text; assert `"from fedrec_foundation"` appears at least 1 time; assert at least one of {`verify_bundle`, `load_split_manifest`, `load_exclusion`, `data_derived`} appears as an imported symbol; assert NO `def create_global_mappings` and NO `def create_leave_one_out_split` and NO `def dirichlet_partition_users` function definitions remain (they are foundation-owned now).
  </behavior>
  <action>

Rip-and-replace `federated-pfedrec/federated_pfedrec/dataset.py` as a thin foundation adapter mirroring Phase 3 / Phase 4 shape. Preserve verbatim per D-18 surgical:

- `MovieLensDataset` class (the torch.utils.data.Dataset wrapper) — UNTOUCHED.
- `download_movielens_1m` function — UNTOUCHED (the URL-retrieve helper).
- `load_movielens_1m` function — UNTOUCHED (the pandas reader).
- `natural_partition_users` function (if present in the current file) — UNTOUCHED.

Remove (rip-and-replace):
- `create_global_mappings`, `create_leave_one_out_split`, `dirichlet_partition_users`, `create_train_test_split`, `compute_user_genre_distribution` — moved to `fedrec_foundation`.
- `_partition_cache` and any `_dataset_cache` — replaced by foundation `verify_bundle`.

Add new top-of-file imports:

```python
from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
from fedrec_foundation.paths import data_derived
from fedrec_foundation.split import SplitManifest, load_split_manifest
from fedrec_foundation.mapping import ItemMapping, load_mapping
```

Add a module-level dataclass-style bundle holder (mirror Phase 3 / Phase 4):

```python
@dataclass
class _FoundationBundle:
    """Carrier for the four foundation artifacts consumed by client_app + task."""
    mapping: "ItemMapping"
    split_manifest: "SplitManifest"
    exclusion: "ExclusionTable"
    raw_data_hash: str
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str


_BUNDLE_CACHE: Optional[_FoundationBundle] = None


def _load_foundation_bundle() -> _FoundationBundle:
    """Load and verify the canonical Phase 1 bundle once per process."""
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE
    derived_dir = data_derived()
    foundation_index = verify_bundle(derived_dir)
    mapping = load_mapping(derived_dir / "mapping.json")
    split_manifest = load_split_manifest(derived_dir / "split_manifest.json")
    exclusion = load_exclusion(derived_dir / "exclusion_items.npz")
    _BUNDLE_CACHE = _FoundationBundle(
        mapping=mapping,
        split_manifest=split_manifest,
        exclusion=exclusion,
        raw_data_hash=foundation_index.raw_data_hash,
        mapping_sha256=foundation_index.mapping_sha256,
        split_hash=foundation_index.split_hash,
        exclusion_sha256=foundation_index.exclusion_sha256,
        foundation_contract_sha256=foundation_index.foundation_contract_sha256,
    )
    return _BUNDLE_CACHE
```

(Use the actual field names available in the foundation API. If `load_mapping` or `ItemMapping` symbol names differ in the actual codebase, prefer the names exposed by Phase 1 Plan 02 — which Phase 3 / Phase 4 dataset adapters already use. If a field is missing from `_FoundationBundle`, the executor follows whatever Phase 3 dataset.py uses verbatim.)

Replace `load_partition_data` and `load_full_data` entry points with adapter shape:

```python
def load_partition_data(
    partition_id: int,
    num_partitions: int,
    partition_mode: str = "natural",
    batch_size: int = 256,
    num_negatives: int = 4,
    alpha: float = 0.5,
    split_mode: str = "leave-one-out",
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load one partition's train/test loaders for cross-device PFedRec.

    Phase 5 D-09: partition_mode != "natural" raises NotImplementedError.
    Cross-silo (Dirichlet) PFedRec is FROZEN at pre-Phase-5 commits.
    """
    if partition_mode != "natural":
        raise NotImplementedError(
            f"D-09: partition_mode={partition_mode!r} is FROZEN for federated-pfedrec. "
            "Phase 5 migrates PFedRec to cross-device only. To run cross-silo, "
            "check out a pre-Phase-5 commit (see "
            ".planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §Deferred)."
        )
    bundle = _load_foundation_bundle()
    # ... use bundle.mapping / bundle.split_manifest / bundle.exclusion
    # ... return (trainloader, testloader, num_users, num_items)
    raise NotImplementedError("Plan 03 implements the foundation-adapter body")  # placeholder



def load_full_data(
    partition_mode: str = "natural",
    batch_size: int = 256,
    num_negatives: int = 4,
    alpha: float = 0.5,
    split_mode: str = "leave-one-out",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load full-population train/test loaders.

    Phase 5 D-09: partition_mode != "natural" raises NotImplementedError.
    """
    if partition_mode != "natural":
        raise NotImplementedError(
            f"D-09: partition_mode={partition_mode!r} is FROZEN for federated-pfedrec. "
            "Phase 5 migrates PFedRec to cross-device only. To run cross-silo, "
            "check out a pre-Phase-5 commit (see "
            ".planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §Deferred)."
        )
    bundle = _load_foundation_bundle()
    # ... derive global train + test loaders from the bundle
    # ... return (full_trainloader, full_testloader, num_users, num_items)
    raise NotImplementedError("Plan 03 implements the foundation-adapter body")  # placeholder
```

CRITICAL CONSTRAINT — Plan 02 ONLY ships the D-09 GUARD layer + adapter scaffolding. The actual body that builds DataLoaders from the bundle is Plan 03's responsibility (it owns task.py + client_app.py + the data-flow integration). Plan 02's load_partition_data / load_full_data MAY end with `raise NotImplementedError("Plan 03 implements the foundation-adapter body")` after the guard fires for non-natural modes — Plan 03 will replace those placeholders with real bundle-to-DataLoader bodies. The KEY invariant Plan 02 closes is the D-09 guard at BOTH entry points; the body is Plan 03's scope.

Alternative if Plan 02 chooses to ship the natural-path body in this plan (acceptable but optional): clone the Phase 3 / Phase 4 adapter body that builds train/test DataLoaders from `bundle.split_manifest.train_pairs[partition_id]` and `bundle.split_manifest.test_pairs[partition_id]` with PFedRec-specific BCE label format. This is a planner-discretion call; the lighter-weight scaffolding option is recommended to keep Plan 02 ≤2 tasks and let Plan 03 own the data-flow integration.

Create `federated-pfedrec/tests/test_dataset.py`:

```python
"""Phase 5 D-09 frozen-cross-silo guard: dataset.py raises at both entry points."""
from __future__ import annotations

import pytest


def test_load_partition_data_raises_on_non_natural() -> None:
    from federated_pfedrec.dataset import load_partition_data

    with pytest.raises(NotImplementedError) as exc_info:
        load_partition_data(partition_id=0, num_partitions=6040, partition_mode="dirichlet")

    msg = str(exc_info.value)
    assert "D-09" in msg, msg
    assert "cross-device" in msg, msg
    assert "pre-Phase-5" in msg, msg


def test_load_full_data_raises_on_non_natural() -> None:
    from federated_pfedrec.dataset import load_full_data

    with pytest.raises(NotImplementedError) as exc_info:
        load_full_data(partition_mode="dirichlet")

    msg = str(exc_info.value)
    assert "D-09" in msg, msg
    assert "cross-device" in msg, msg
    assert "pre-Phase-5" in msg, msg


def test_dataset_uses_foundation_adapter_imports() -> None:
    """D-17 rip-and-replace: dataset.py imports from fedrec_foundation
    and the legacy mapping/split/exclusion helpers are removed."""
    import inspect

    import federated_pfedrec.dataset as ds

    src = inspect.getsource(ds)
    assert "from fedrec_foundation" in src

    # Legacy helpers must be removed.
    assert "def create_global_mappings" not in src
    assert "def create_leave_one_out_split" not in src
    assert "def dirichlet_partition_users" not in src
    assert "def create_train_test_split" not in src
    assert "def compute_user_genre_distribution" not in src
```

D-18 SCOPE BOUNDARY: do NOT touch client_app.py, server_app.py, task.py, models/, strategy.py, pyproject.toml (other than the additions in Task 1 above). Verify with `git diff --name-only` after this task that the only changes are dataset.py + 2 new test files (in addition to Task 1's pyproject.toml + mode.py + 2 test files).

Verify: `cd federated-pfedrec && pytest tests/test_dataset.py -x -v` — 3 GREEN.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_dataset.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation" federated-pfedrec/federated_pfedrec/dataset.py` returns at least 1
    - `grep -c "def create_global_mappings\|def create_leave_one_out_split\|def dirichlet_partition_users\|def create_train_test_split\|def compute_user_genre_distribution" federated-pfedrec/federated_pfedrec/dataset.py` returns 0 (D-17 rip-and-replace)
    - `grep -c "def load_partition_data" federated-pfedrec/federated_pfedrec/dataset.py` returns 1
    - `grep -c "def load_full_data" federated-pfedrec/federated_pfedrec/dataset.py` returns 1
    - `grep -c "raise NotImplementedError" federated-pfedrec/federated_pfedrec/dataset.py` returns at least 2 (one per entry point per D-09)
    - `grep -cE "D-09" federated-pfedrec/federated_pfedrec/dataset.py` returns at least 2 (in the error messages)
    - `grep -c "MovieLensDataset" federated-pfedrec/federated_pfedrec/dataset.py` returns at least 1 (D-18 preserved)
    - `grep -c "def download_movielens_1m\|def load_movielens_1m" federated-pfedrec/federated_pfedrec/dataset.py` returns at least 2 (D-18 preserved verbatim)
    - `pytest federated-pfedrec/tests/test_dataset.py -x -v` exits 0 with 3 tests passed
  </acceptance_criteria>
  <done>
    - dataset.py is a thin (~440 LOC) foundation adapter; D-17 rip-and-replace complete
    - D-09 NotImplementedError fires at BOTH load_partition_data and load_full_data when partition_mode != "natural" (mirrors Phase 3 D-02 / Phase 4 D-02 tightening)
    - Error message includes 'D-09' / 'cross-device' / 'pre-Phase-5' tokens (test-asserted)
    - 3 GREEN dataset regression tests
  </done>
</task>

</tasks>

<verification>
- `cd federated-pfedrec && pytest tests/test_pyproject.py tests/test_dataset.py -x -v` → 6 GREEN
- `pytest scripts/foundation/tests/test_mode.py::test_paper_compat_pfedrec_weight_policy_uniform -x -v` → 1 GREEN
- Foundation suite full regression: `pytest scripts/foundation/tests/ -x -v` → all GREEN (no regressions from D-25 mode.py change)
- D-18 surgical scope: `git diff --name-only` after Plan 02 closes shows ONLY:
  - federated-pfedrec/pyproject.toml
  - federated-pfedrec/federated_pfedrec/dataset.py
  - federated-pfedrec/tests/test_pyproject.py (new)
  - federated-pfedrec/tests/test_dataset.py (new)
  - scripts/foundation/fedrec_foundation/mode.py
  - scripts/foundation/tests/test_mode.py
</verification>

<success_criteria>
- pyproject.toml: num-supernodes=6040 in BOTH federation blocks; mode/run-seed/weight-policy/reuse-cache/eval-num-negatives/checkpoint-rule keys present; [project.optional-dependencies] dev includes pytest>=7.0
- mode.py: _PAPER_COMPAT_PFEDREC.weight_policy='uniform' AND deferred-confirmation comment removed
- dataset.py: thin foundation adapter; D-09 NotImplementedError at BOTH load_partition_data and load_full_data when partition_mode != "natural"; legacy mapping/split/exclusion helpers stripped
- 7 GREEN tests across 3 new/extended test files
- Plan 02 zero-touched files outside the listed `files_modified` set (Wave-1 disjoint-file ownership with Plan 01)
</success_criteria>

<output>
After completion, create `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-02-SUMMARY.md` covering:
- PFR-01 closure (cross-device defaults + dev pytest extra)
- D-25 closure (Phase 1 deferred decision; weight_policy='uniform')
- D-17 + D-09 rip-and-replace (dataset.py adapter shape; both entry points guarded)
- Test counts and acceptance criteria status
- Confirmation Wave-1 disjoint file ownership held
</output>
