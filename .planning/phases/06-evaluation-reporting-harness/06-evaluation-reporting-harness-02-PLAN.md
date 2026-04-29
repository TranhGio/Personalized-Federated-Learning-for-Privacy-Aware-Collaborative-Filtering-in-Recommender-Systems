---
phase: 06-evaluation-reporting-harness
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/foundation/fedrec_foundation/manifest.py
  - scripts/foundation/tests/test_manifest.py
autonomous: true
requirements: [EVL-01, EVL-06]
must_haves:
  truths:
    - "RUN_MANIFEST_SCHEMA_VERSION == 2 (bumped from 1; one-line constant change)"
    - "RunManifest carries final_eval_round_index: int = 0 (sentinel: 0 = no extra eval ran; >=1 = post-restore broadcast index)"
    - "RunManifest carries metrics: Dict[str, Dict[str, float]] = field(default_factory=dict) (mirrors final_metrics block, top-level keys 'best' and 'last')"
    - "Backward-compat: existing v1 RunManifest test fixtures construct without passing the two new fields (Pitfall 3 — defaults must be safe)"
    - "write_manifest_sibling accepts optional sibling_name kwarg overriding default <run_id>-manifest.json (D-04 clean filename support)"
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/manifest.py"
      provides: "Schema v2 RunManifest + write_manifest_sibling sibling_name kwarg"
      contains: "RUN_MANIFEST_SCHEMA_VERSION: int = 2"
    - path: "scripts/foundation/tests/test_manifest.py"
      provides: "3 NEW tests + ensure existing tests stay GREEN under schema v2"
      contains: "def test_run_manifest_schema_version_2"
  key_links:
    - from: "scripts/foundation/fedrec_foundation/manifest.py::RunManifest.metrics"
      to: "Wave-2/3 server_app.py final_metrics dict (best/last/best_round/last_round/final_eval_round_index)"
      via: "manifest = replace(manifest, final_eval_round_index=N, metrics=results_data['final_metrics']) post-build mutation"
      pattern: "metrics: Dict\\[str, Any\\] = field\\(default_factory=dict\\)"
    - from: "scripts/foundation/fedrec_foundation/manifest.py::write_manifest_sibling"
      to: "Wave-2/3 server_app.py D-04 clean filename"
      via: "write_manifest_sibling(manifest, results_filename, sibling_name='manifest.json')"
      pattern: "sibling_name: Optional\\[str\\] = None"
---

<objective>
Bump `RunManifest` schema from version 1 to version 2, adding two safe-default fields (`final_eval_round_index`, `metrics`) and one optional kwarg on `write_manifest_sibling` (`sibling_name`). This is the Wave-1 manifest primitive every Wave-2/3 server_app plan depends on.

Purpose:
  - Close EVL-01: the manifest now carries `final_eval_round_index` so a reader can prove the canonical `best_*` block came from a post-restore broadcast (sentinel: 0 = no extra eval; >=1 = round index of the broadcast).
  - Close EVL-06: the manifest now carries the full `metrics: {best: {...}, last: {...}}` block as a typed field, so readers do not have to parse `results_data["final_metrics"]` separately.
  - Pitfall 3 closure: BOTH new fields are added with safe defaults (`0`, `field(default_factory=dict)`) so existing v1 test fixtures construct without modification.
  - D-04 closure: `write_manifest_sibling` gains an optional `sibling_name` kwarg (defaults to existing behavior); Wave-2/3 plans pass `sibling_name="manifest.json"` to land the clean per-run-dir filename. Cross-silo callers omit the kwarg and keep `<run_id>-manifest.json`.

Output:
  - `scripts/foundation/fedrec_foundation/manifest.py` modified: `RUN_MANIFEST_SCHEMA_VERSION` bumped 1 → 2; `RunManifest` dataclass extended with two defaulted fields; `write_manifest_sibling` extended with one optional kwarg; `from dataclasses import field` added if not already imported.
  - `scripts/foundation/tests/test_manifest.py` extended: 3 NEW tests (schema_version=2, backward-compat v1 fixtures, sibling_name kwarg).
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/fedrec_foundation/manifest.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/tests/test_manifest.py

<interfaces>
<!-- Existing manifest.py public surface -->
```python
# scripts/foundation/fedrec_foundation/manifest.py — current state (lines 26-84)

RUN_MANIFEST_SCHEMA_VERSION: int = 1   # CHANGE TO 2

@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    # Mode + locked config (13 fields).
    mode: str
    num_supernodes: int
    partition_mode: str
    fraction_train: float
    fraction_eval: float
    weight_policy: str
    primary_evaluator: str
    num_train_negatives: int
    num_eval_negatives: int
    run_seed: int
    checkpoint_rule: str
    # Foundation fingerprints (IMP-2).
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str
    raw_data_hash: str
    builder_version: str
    # Overrides + module metadata.
    overrides: Dict[str, Any]
    module: str
    # Environment.
    flwr_version: str
    torch_version: str
    git_commit: str
    # NEW (this plan, both with safe defaults so v1 fixtures still construct):
    # final_eval_round_index: int = 0
    # metrics: Dict[str, Any] = field(default_factory=dict)
```

```python
# write_manifest_sibling current signature (manifest.py:203-226):
def write_manifest_sibling(manifest: RunManifest, result_json_path: Path) -> Path:
    """Write <result_json_path.parent>/<run_id>-manifest.json sibling. Returns sibling Path."""
# CHANGE: add optional sibling_name kwarg:
def write_manifest_sibling(
    manifest: RunManifest,
    result_json_path: Path,
    sibling_name: Optional[str] = None,
) -> Path:
    # Default behavior preserved when sibling_name=None: write to <run_id>-manifest.json.
    # Wave-2/3 callers pass sibling_name="manifest.json" for D-04 clean filenames.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Bump RUN_MANIFEST_SCHEMA_VERSION to 2, add final_eval_round_index + metrics fields with safe defaults to RunManifest, and extend write_manifest_sibling with sibling_name kwarg; ship 3 NEW tests covering schema_version, v1 backward-compat, and sibling_name override</name>
  <files>scripts/foundation/fedrec_foundation/manifest.py, scripts/foundation/tests/test_manifest.py</files>
  <read_first>
    - scripts/foundation/fedrec_foundation/manifest.py — current state (full file; pay attention to lines 21 for `from dataclasses import asdict, dataclass`, 28-29 for RUN_MANIFEST_SCHEMA_VERSION, 32-84 for RunManifest, 203-226 for write_manifest_sibling)
    - scripts/foundation/tests/test_manifest.py — current shape; existing tests must keep passing under schema v2
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-04, D-06, D-07
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Pattern 4 (manifest schema bump 1->2) + §Common Pitfalls Pitfall 3 (default-arg compatibility) + §Code Examples Example 2
  </read_first>
  <behavior>
    - Test 1 (test_run_manifest_schema_version_2): Import `RUN_MANIFEST_SCHEMA_VERSION` from `fedrec_foundation.manifest`; assert it equals `2` (bumped from 1). Build a `RunManifest` via `build_run_manifest(...)` from a `_StubProfile` (mirror existing test_manifest.py fixture); assert `manifest.schema_version == 2`. Embed via `embed_manifest_in_result(manifest, results_data)` and assert `results_data["_manifest"]["schema_version"] == 2`.
    - Test 2 (test_run_manifest_backward_compat_v1): Construct a `RunManifest` directly with the original 23 v1 fields (NO `final_eval_round_index`, NO `metrics`); assert no `TypeError` is raised; assert `manifest.final_eval_round_index == 0` (sentinel default — Pitfall 3); assert `manifest.metrics == {}` (empty dict default — Pitfall 3). The point: existing v1 test fixtures continue to work unchanged.
    - Test 3 (test_run_manifest_carries_final_eval_round_index): Build a `RunManifest` then use `dataclasses.replace(manifest, final_eval_round_index=87, metrics={"best": {"sampled_ndcg@10": 0.4413}, "last": {"sampled_ndcg@10": 0.4321}, "best_round": 87, "last_round": 100, "final_eval_round_index": 87})`; assert `embed_manifest_in_result(replaced, results_data)` writes those values into `results_data["_manifest"]["final_eval_round_index"] == 87` AND `results_data["_manifest"]["metrics"]["best"]["sampled_ndcg@10"] == 0.4413`.
    - Test 4 (test_write_manifest_sibling_default_filename): Write a manifest via `write_manifest_sibling(manifest, tmp_path / "results.json")` (NO sibling_name); assert the sibling path equals `tmp_path / f"{manifest.run_id}-manifest.json"` (legacy default preserved).
    - Test 5 (test_write_manifest_sibling_custom_name): Write a manifest via `write_manifest_sibling(manifest, tmp_path / "results.json", sibling_name="manifest.json")`; assert the sibling path equals `tmp_path / "manifest.json"` (D-04 clean filename); assert the file exists and `json.loads(sibling.read_text())["schema_version"] == 2`.
  </behavior>
  <action>
Edit `scripts/foundation/fedrec_foundation/manifest.py` with surgical edits:

1. **Add `field` to dataclass imports.** Locate `from dataclasses import asdict, dataclass` (line 21). Change to:
```python
from dataclasses import asdict, dataclass, field
```

2. **Add `Optional` to typing imports.** Locate `from typing import Any, Dict` (line 24). Change to:
```python
from typing import Any, Dict, Optional
```

3. **Bump schema version.** Locate `RUN_MANIFEST_SCHEMA_VERSION: int = 1` (line 29). Change to:
```python
RUN_MANIFEST_SCHEMA_VERSION: int = 2  # Phase 6: adds final_eval_round_index + metrics fields
```

4. **Append two defaulted fields to RunManifest dataclass.** AFTER the existing `git_commit: str` line (line 84), append:
```python
    # Phase 6 additions (both with safe defaults so v1 fixtures still construct
    # without TypeError — Pitfall 3 from RESEARCH.md):
    final_eval_round_index: int = 0
    """Index of the post-restore extra-eval-round broadcast (D-06).

    Sentinel ``0`` = no extra eval ran (mode is ``last_round``, or ``best_round``
    with no best-round recorded). Values ``>= 1`` mean a fresh evaluation ran
    on the restored best-round state and produced ``metrics["best"]``.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    """Mirror of ``results_data["final_metrics"]`` block (D-07).

    Top-level keys: ``best``, ``last``, ``best_round``, ``last_round``,
    ``final_eval_round_index``. The ``best`` and ``last`` sub-dicts carry
    ``sampled_hr@10``, ``sampled_ndcg@10``, ``evaluated_users``, plus per-group
    variants (``sampled_hr@10/sparse``, ``sampled_ndcg@10/sparse``,
    ``evaluated_users_sparse``, ...). Defaults to ``{}`` on a fresh manifest;
    server_app overwrites via ``dataclasses.replace`` post-build mutation.
    """
```

5. **Extend write_manifest_sibling signature.** Locate `def write_manifest_sibling(manifest: RunManifest, result_json_path: Path) -> Path:` (around line 203). Change to:
```python
def write_manifest_sibling(
    manifest: RunManifest,
    result_json_path: Path,
    sibling_name: Optional[str] = None,
) -> Path:
```

Then update the body to honor `sibling_name`. Locate the body line that derives the sibling filename (currently `f"{manifest.run_id}-manifest.json"`); change the filename derivation to:
```python
    sibling_filename = sibling_name if sibling_name is not None else f"{manifest.run_id}-manifest.json"
    sibling_path = Path(result_json_path).parent / sibling_filename
```
Update the docstring to document the new kwarg:
```python
    """Write the manifest as a sibling JSON next to the result file.

    Parameters
    ----------
    manifest : RunManifest
    result_json_path : Path
        Path to the main result JSON. The sibling is written into the same
        parent directory.
    sibling_name : Optional[str]
        Override the sibling filename. Defaults to ``None``, which preserves
        the legacy ``<run_id>-manifest.json`` naming. Phase-6 callers pass
        ``"manifest.json"`` for D-04 clean per-run-dir filenames; cross-silo
        legacy callers omit this kwarg.

    Returns
    -------
    pathlib.Path
        Absolute path to the sibling manifest JSON.
    """
```

6. **Append 5 new tests to test_manifest.py.** Locate the existing test functions (build_run_manifest fixtures, _StubProfile, etc. — read the file first to identify the conftest fixtures available). Append at the END of `scripts/foundation/tests/test_manifest.py`:

```python
# ============================================================================
# Phase 6 — schema v2 tests (RUN_MANIFEST_SCHEMA_VERSION = 2)
# ============================================================================

import json
from dataclasses import replace as dataclass_replace

from fedrec_foundation.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    RunManifest,
    build_run_manifest,
    embed_manifest_in_result,
    write_manifest_sibling,
)


def _build_v2_manifest_via_helper(stub_profile_factory):
    """Build a RunManifest via build_run_manifest using whatever _StubProfile
    fixture pattern this test_manifest.py file already uses. Implementer:
    inline-replace ``stub_profile_factory()`` with the existing fixture / stub
    construction call (e.g., ``_StubProfile()`` or ``make_stub_profile()``).
    """
    return build_run_manifest(
        run_id="20260429-104530-phase6t",
        mode_profile=stub_profile_factory(),
        run_seed=42,
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
    )


def test_run_manifest_schema_version_2():
    """Phase 6: schema_version constant bumped from 1 to 2."""
    assert RUN_MANIFEST_SCHEMA_VERSION == 2, (
        f"Expected RUN_MANIFEST_SCHEMA_VERSION=2, got {RUN_MANIFEST_SCHEMA_VERSION}"
    )


def test_run_manifest_backward_compat_v1(tmp_path):
    """Pitfall 3: existing v1 test fixtures must construct without TypeError.

    The two NEW fields (final_eval_round_index, metrics) carry safe defaults
    so legacy callers never see a missing-kwarg TypeError.
    """
    # Construct directly using ONLY the v1 field set (no final_eval_round_index,
    # no metrics). The point is: this MUST NOT raise TypeError under v2.
    manifest = RunManifest(
        schema_version=2,
        run_id="20260429-104530-v1back",
        mode="benchmark_cross_device",
        num_supernodes=6040,
        partition_mode="natural",
        fraction_train=0.05,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
    )
    # Defaults must be the documented sentinels.
    assert manifest.final_eval_round_index == 0, (
        "Expected sentinel default 0 (no extra eval ran)"
    )
    assert manifest.metrics == {}, (
        "Expected default_factory=dict for metrics field"
    )


def test_run_manifest_carries_final_eval_round_index():
    """EVL-01 + EVL-06: post-build mutation populates the new fields."""
    manifest = RunManifest(
        schema_version=2,
        run_id="20260429-104530-evl",
        mode="benchmark_cross_device",
        num_supernodes=6040,
        partition_mode="natural",
        fraction_train=0.05,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
    )
    nested_metrics = {
        "best": {"sampled_ndcg@10": 0.4413, "sampled_hr@10": 0.7287},
        "last": {"sampled_ndcg@10": 0.4321, "sampled_hr@10": 0.7102},
        "best_round": 87,
        "last_round": 100,
        "final_eval_round_index": 101,
    }
    replaced = dataclass_replace(manifest, final_eval_round_index=101, metrics=nested_metrics)

    results_data: Dict[str, Any] = {}
    embed_manifest_in_result(replaced, results_data)
    embedded = results_data["_manifest"]
    assert embedded["schema_version"] == 2
    assert embedded["final_eval_round_index"] == 101
    assert embedded["metrics"]["best"]["sampled_ndcg@10"] == 0.4413
    assert embedded["metrics"]["last"]["sampled_ndcg@10"] == 0.4321
    assert embedded["metrics"]["best_round"] == 87
    assert embedded["metrics"]["last_round"] == 100


def test_write_manifest_sibling_default_filename(tmp_path):
    """Default behavior preserved: <run_id>-manifest.json (cross-silo legacy)."""
    manifest = RunManifest(
        schema_version=2, run_id="20260429-104530-defflt",
        mode="cross_silo_legacy", num_supernodes=5, partition_mode="dirichlet",
        fraction_train=1.0, fraction_eval=1.0, weight_policy="num_positives",
        primary_evaluator="sampled_loo_99", num_train_negatives=4,
        num_eval_negatives=99, run_seed=42, checkpoint_rule="last_round",
        mapping_sha256="m" * 64, split_hash="s" * 12,
        exclusion_sha256="e" * 64, foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64, builder_version="1.0.0", overrides={},
        module="baseline", flwr_version="1.22.0", torch_version="2.7.1",
        git_commit="abc1234",
    )
    result_json = tmp_path / "results.json"
    result_json.write_text("{}")
    sibling = write_manifest_sibling(manifest, result_json)
    assert sibling.name == "20260429-104530-defflt-manifest.json"
    assert sibling.exists()


def test_write_manifest_sibling_custom_name(tmp_path):
    """D-04: sibling_name='manifest.json' lands the clean per-run-dir filename."""
    manifest = RunManifest(
        schema_version=2, run_id="20260429-104530-clean",
        mode="benchmark_cross_device", num_supernodes=6040, partition_mode="natural",
        fraction_train=0.05, fraction_eval=1.0, weight_policy="num_positives",
        primary_evaluator="sampled_loo_99", num_train_negatives=4,
        num_eval_negatives=99, run_seed=42, checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64, split_hash="s" * 12,
        exclusion_sha256="e" * 64, foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64, builder_version="1.0.0", overrides={},
        module="baseline", flwr_version="1.22.0", torch_version="2.7.1",
        git_commit="abc1234",
    )
    result_json = tmp_path / "results.json"
    result_json.write_text("{}")
    sibling = write_manifest_sibling(manifest, result_json, sibling_name="manifest.json")
    assert sibling.name == "manifest.json"
    assert sibling.exists()
    payload = json.loads(sibling.read_text())
    assert payload["schema_version"] == 2
```

NOTE: If the existing test_manifest.py file ALREADY has fixtures named `_StubProfile` or `make_stub_profile` for `build_run_manifest` testing, the executor MUST reuse those fixtures verbatim instead of duplicating them. Read the file first; do NOT introduce a second stub.

Verify: `cd scripts/foundation && pytest tests/test_manifest.py -x -v` — all existing tests + 5 new tests pass.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation && pytest tests/test_manifest.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "RUN_MANIFEST_SCHEMA_VERSION: int = 2" scripts/foundation/fedrec_foundation/manifest.py` returns 1
    - `grep -c "RUN_MANIFEST_SCHEMA_VERSION: int = 1" scripts/foundation/fedrec_foundation/manifest.py` returns 0 (old constant removed)
    - `grep -c "from dataclasses import asdict, dataclass, field" scripts/foundation/fedrec_foundation/manifest.py` returns 1
    - `grep -c "final_eval_round_index: int = 0" scripts/foundation/fedrec_foundation/manifest.py` returns 1
    - `grep -c "metrics: Dict\\[str, Any\\] = field(default_factory=dict)" scripts/foundation/fedrec_foundation/manifest.py` returns 1
    - `grep -c "sibling_name: Optional\\[str\\] = None" scripts/foundation/fedrec_foundation/manifest.py` returns 1
    - `python -c "from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION; assert RUN_MANIFEST_SCHEMA_VERSION == 2; print('ok')"` prints "ok"
    - `python -c "from fedrec_foundation.manifest import RunManifest; import dataclasses; fnames = {f.name for f in dataclasses.fields(RunManifest)}; assert 'final_eval_round_index' in fnames; assert 'metrics' in fnames; print('ok')"` prints "ok"
    - `cd scripts/foundation && pytest tests/test_manifest.py -x -v` exits 0 with all existing tests + 5 new tests passing
    - `cd scripts/foundation && pytest tests/ -q -m "not slow"` exits 0 (full foundation suite green; existing non-manifest tests still pass)
  </acceptance_criteria>
  <done>
    - RUN_MANIFEST_SCHEMA_VERSION bumped to 2
    - Two new RunManifest fields (final_eval_round_index, metrics) added with safe defaults — Pitfall 3 closure
    - write_manifest_sibling extended with optional sibling_name kwarg; default behavior preserved
    - 5 NEW tests in test_manifest.py: schema_version=2, v1 backward-compat (Pitfall 3), post-build mutation, default filename preserved, sibling_name override (D-04)
    - Foundation suite full green
  </done>
</task>

</tasks>

<verification>
- Schema constant: `python -c "from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION; print(RUN_MANIFEST_SCHEMA_VERSION)"` prints `2`
- v1 backward-compat: existing test_manifest.py fixtures construct RunManifest without passing the two new fields — no TypeError
- D-04 sibling override: `write_manifest_sibling(m, p, sibling_name="manifest.json")` writes to `<parent>/manifest.json`
- Full foundation suite green: `cd scripts/foundation && pytest tests/ -q -m "not slow"` exits 0
- D-18 surgical scope: `git diff --stat` shows ONLY changes to scripts/foundation/fedrec_foundation/manifest.py + scripts/foundation/tests/test_manifest.py; nothing else
- No existing manifest.py callers (build_run_manifest, embed_manifest_in_result, write_manifest_sibling — all 4 modules' server_app.py) need code changes today; their existing call sites continue to work because both new fields default and the new kwarg is optional. Wave 2/3 plans extend those call sites.
</verification>

<success_criteria>
- `RUN_MANIFEST_SCHEMA_VERSION` bumped from 1 to 2 with comment citing Phase 6
- `RunManifest` extends with `final_eval_round_index: int = 0` and `metrics: Dict[str, Any] = field(default_factory=dict)` — both with safe defaults so existing v1 callers/fixtures construct unchanged (Pitfall 3)
- `write_manifest_sibling` extends with optional `sibling_name: Optional[str] = None` kwarg; legacy default `<run_id>-manifest.json` preserved when omitted
- 5 NEW tests pin schema_version=2 (Test 1), v1 backward-compat (Test 2), post-build mutation field embedding (Test 3), default sibling filename (Test 4), D-04 clean filename via sibling_name (Test 5)
- All existing tests in test_manifest.py continue to pass (zero regressions)
- Full foundation suite: `pytest scripts/foundation/tests/ -q -m "not slow"` green
- No touches to mode.py / atomic.py / paths.py (Plan 01 owns paths.py)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-02-SUMMARY.md` covering:
- Schema bump 1 -> 2 + two new fields with safe defaults (Pitfall 3 closure)
- write_manifest_sibling sibling_name kwarg signature change (D-04 enabler for Wave 2/3)
- Test counts and which decisions each pins (schema_version, v1 compat, sibling_name override)
- Cross-phase contract: Wave 2/3 plans extend call sites via `dataclasses.replace(manifest, ...)` + `write_manifest_sibling(..., sibling_name="manifest.json")`
</output>
</content>
</invoke>