---
phase: 07-thesis-evaluation-run
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/foundation/fedrec_foundation/atomic.py
  - scripts/foundation/fedrec_foundation/mode.py
  - scripts/foundation/fedrec_foundation/manifest.py
  - scripts/run.py
  - scripts/foundation/tests/test_atomic.py
  - scripts/foundation/tests/test_mode.py
  - scripts/foundation/tests/test_manifest.py
  - scripts/foundation/tests/test_launcher.py
autonomous: true
requirements:
  - THS-01
user_setup: []

must_haves:
  truths:
    - "resolve_mode_defaults('thesis_crossdevice_main') returns a ModeProfile with embedding_dim=64, optimizer='adam', lr=0.001, num_server_rounds=100, weight_policy='num_positives', fraction_train=0.1"
    - "MODE_NAMES contains exactly 4 entries: ('benchmark_cross_device', 'thesis_crossdevice_main', 'paper_compat_pfedrec', 'cross_silo_legacy')"
    - "scripts/run.py adaptive thesis_crossdevice_main --dry-run exits 0 and prints `mode=\"thesis_crossdevice_main\"`"
    - "RUN_MANIFEST_SCHEMA_VERSION == 3"
    - "RunManifest dataclass carries 3 new fields: thesis_run_label (default ''), ablation_dimension (default 'none'), ablation_value (default '')"
    - "v1/v2 RunManifest construction without thesis kwargs still succeeds (backward compat)"
    - "atomic_write_text() writes UTF-8 text via tempfile + os.replace, leaves no .tmp-* leftovers"
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/atomic.py"
      provides: "atomic_write_text() function for markdown/CSV writes"
      contains: "def atomic_write_text"
    - path: "scripts/foundation/fedrec_foundation/mode.py"
      provides: "_THESIS_CROSSDEVICE_MAIN ModeProfile + registry entry"
      contains: "_THESIS_CROSSDEVICE_MAIN = ModeProfile("
    - path: "scripts/foundation/fedrec_foundation/manifest.py"
      provides: "RunManifest schema v3 with thesis fields"
      contains: "RUN_MANIFEST_SCHEMA_VERSION: int = 3"
    - path: "scripts/run.py"
      provides: "Updated MODE_NUM_SUPERNODES dict with thesis_crossdevice_main"
      contains: '"thesis_crossdevice_main": 6040'
  key_links:
    - from: "scripts/foundation/fedrec_foundation/mode.py"
      to: "_REGISTRY dict"
      via: "_THESIS_CROSSDEVICE_MAIN registered under key 'thesis_crossdevice_main'"
      pattern: '"thesis_crossdevice_main":\s*_THESIS_CROSSDEVICE_MAIN'
    - from: "scripts/run.py"
      to: "argparse choices"
      via: "MODE_NUM_SUPERNODES dict (sorted keys feed argparse)"
      pattern: '"thesis_crossdevice_main":\s*6040'
---

<objective>
Add the foundation primitives that all downstream Phase 7 work consumes:
1. A new `_THESIS_CROSSDEVICE_MAIN` ModeProfile (cloned verbatim from `_BENCHMARK_CROSS_DEVICE` per D-04) registered in `_REGISTRY`.
2. RunManifest schema bump v2→v3 with three new thesis-tagging fields (`thesis_run_label`, `ablation_dimension`, `ablation_value`) per D-22, all with safe defaults so v1/v2 fixtures continue to construct without TypeError.
3. `atomic_write_text` companion to `atomic_write_json` for markdown/CSV writes by the aggregator.
4. `scripts/run.py` MODE_NUM_SUPERNODES dict updated with `thesis_crossdevice_main: 6040` so the launcher's `argparse choices=` accepts the new mode.
5. Tests pinning all four invariants (4 unit tests in `test_mode.py` extension + 3 unit tests in `test_manifest.py` extension + 1 unit test in `test_atomic.py` + 1 launcher dry-run test in `test_launcher.py`).

Purpose: Plan 01 is the gate for the entire phase. Without these foundation extensions, Plan 02 (server_app patches), Plan 03 (orchestrator), Plan 04 (aggregator), and Plan 05 (manual runbook) all fail loudly. Specifically:
- Without the mode profile + registry entry: `resolve_mode_defaults("thesis_crossdevice_main")` raises `ValueError("Unknown mode")` (Pitfall 4 from RESEARCH.md).
- Without the MODE_NUM_SUPERNODES update: `python scripts/run.py adaptive thesis_crossdevice_main` is rejected by `argparse` with `invalid choice` (Pitfall 5).
- Without the manifest schema v3: thesis runs cannot record `thesis_run_label`, so the aggregator filter excludes every run (Pitfall 2 + Pitfall 7).
- Without `atomic_write_text`: the aggregator's markdown writes have to hand-roll the tempfile pattern (anti-pattern per "Don't Hand-Roll").

Output: 4 source files extended, 4 test files extended, 9 new pytest functions all GREEN.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/07-thesis-evaluation-run/07-CONTEXT.md
@.planning/phases/07-thesis-evaluation-run/07-RESEARCH.md
@.planning/phases/07-thesis-evaluation-run/07-VALIDATION.md

<interfaces>
<!-- Key types and signatures the executor needs. Do not re-explore the codebase. -->

From scripts/foundation/fedrec_foundation/mode.py (existing):
```python
@dataclass(frozen=True)
class ModeProfile:
    mode: str
    num_supernodes: int
    partition_mode: str
    weight_policy: str
    primary_evaluator: str
    fraction_train: float
    fraction_eval: float
    num_train_negatives: int
    num_eval_negatives: int
    embedding_dim: int
    optimizer: str
    lr: float
    local_epochs: int
    num_server_rounds: int
    checkpoint_rule: str
    assert_one_user_per_client: bool

_BENCHMARK_CROSS_DEVICE = ModeProfile(
    mode="benchmark_cross_device",
    num_supernodes=6040,
    partition_mode="natural",
    weight_policy="num_positives",
    primary_evaluator="sampled_loo_99",
    fraction_train=0.1,
    fraction_eval=1.0,
    num_train_negatives=4,
    num_eval_negatives=99,
    embedding_dim=64,
    optimizer="adam",
    lr=0.001,
    local_epochs=1,
    num_server_rounds=100,
    checkpoint_rule="best_round",
    assert_one_user_per_client=True,
)

_REGISTRY: Dict[str, ModeProfile] = {
    "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
    "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
    "cross_silo_legacy": _CROSS_SILO_LEGACY,
}
MODE_NAMES = tuple(_REGISTRY.keys())
```

From scripts/foundation/fedrec_foundation/manifest.py (existing):
```python
RUN_MANIFEST_SCHEMA_VERSION: int = 2  # Phase 6: adds final_eval_round_index + metrics fields

@dataclass
class RunManifest:
    schema_version: int
    run_id: str
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
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str
    raw_data_hash: str
    builder_version: str
    overrides: Dict[str, Any]
    module: str
    flwr_version: str
    torch_version: str
    git_commit: str
    final_eval_round_index: int = 0  # Phase 6
    metrics: Dict[str, Any] = field(default_factory=dict)  # Phase 6
```

From scripts/foundation/fedrec_foundation/atomic.py (existing):
```python
def atomic_write_json(path: str, data: object) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
```

From scripts/run.py (existing, line 68):
```python
MODE_NUM_SUPERNODES = {
    "benchmark_cross_device": 6040,
    "paper_compat_pfedrec": 6040,
    "cross_silo_legacy": 5,
}
# argparse choices=sorted(MODE_NUM_SUPERNODES.keys()) — adding to dict updates choices automatically.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Foundation source edits — mode.py + manifest.py + atomic.py + scripts/run.py</name>
  <read_first>
    - scripts/foundation/fedrec_foundation/mode.py (full file — 338 lines; read so the executor sees existing _BENCHMARK_CROSS_DEVICE / _PAPER_COMPAT_PFEDREC structure and the _REGISTRY dict at line 179)
    - scripts/foundation/fedrec_foundation/manifest.py (full file — 280 lines; read so the executor sees RUN_MANIFEST_SCHEMA_VERSION at line 29 and the RunManifest dataclass field block ending at line 103 with `metrics: Dict[str, Any] = field(default_factory=dict)`)
    - scripts/foundation/fedrec_foundation/atomic.py (full file — 64 lines; read so the executor sees the atomic_write_json + _json_default pattern)
    - scripts/run.py (full file — 207 lines; read so the executor sees MODE_NUM_SUPERNODES at lines 68-72 and the argparse hookup at line 167)
    - .planning/phases/07-thesis-evaluation-run/07-CONTEXT.md §D-04 + §D-22 (the exact decisions implemented here)
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pattern 1: Mode Profile Cloning", "Pattern 2: Manifest Schema Bump v2 → v3", "Pattern 6: Atomic Markdown Write", and "Pitfall 4 + Pitfall 5"
  </read_first>
  <behavior>
    - mode.py: A new module-level constant `_THESIS_CROSSDEVICE_MAIN = ModeProfile(...)` exists immediately after `_BENCHMARK_CROSS_DEVICE` (line 135). Its 16 field values match `_BENCHMARK_CROSS_DEVICE` byte-for-byte EXCEPT `mode="thesis_crossdevice_main"`. The `_REGISTRY` dict gains a new key `"thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN` placed between `"benchmark_cross_device"` and `"paper_compat_pfedrec"`. `MODE_NAMES` automatically picks up the new entry (no edit needed).
    - manifest.py: `RUN_MANIFEST_SCHEMA_VERSION` bumped from `2` to `3`. Three new fields appended to `RunManifest` AFTER the existing `metrics: Dict[str, Any] = field(default_factory=dict)`: `thesis_run_label: str = ""`, `ablation_dimension: str = "none"`, `ablation_value: str = ""`. All three carry NumPy-style docstrings. `build_run_manifest` is NOT touched — defaults flow through implicitly so callers don't have to pass these kwargs (Pitfall 7 backward-compat invariant).
    - atomic.py: New `atomic_write_text(path: str, content: str) -> None` function added AFTER `atomic_write_json` (after line 48). Mirrors the `atomic_write_json` body but: (a) no JSON serialization, (b) `tempfile.mkstemp(..., suffix=".txt")`, (c) `os.fdopen(fd, "w", encoding="utf-8")`. Same exception-cleanup semantics.
    - scripts/run.py: `MODE_NUM_SUPERNODES` dict at lines 68-72 gains a new key `"thesis_crossdevice_main": 6040` placed between `"benchmark_cross_device"` and `"paper_compat_pfedrec"`. argparse `choices=sorted(MODE_NUM_SUPERNODES.keys())` (line 167) inherits the new value automatically — no edit needed.
  </behavior>
  <action>
1. Edit `scripts/foundation/fedrec_foundation/mode.py`:
   - After the `_BENCHMARK_CROSS_DEVICE = ModeProfile(...)` block (existing lines 118-135), insert exactly:
     ```python


     _THESIS_CROSSDEVICE_MAIN = ModeProfile(
         mode="thesis_crossdevice_main",
         num_supernodes=6040,
         partition_mode="natural",
         weight_policy="num_positives",
         primary_evaluator="sampled_loo_99",
         fraction_train=0.1,       # sweep-tunable default
         fraction_eval=1.0,
         num_train_negatives=4,
         num_eval_negatives=99,
         embedding_dim=64,
         optimizer="adam",
         lr=0.001,
         local_epochs=1,
         num_server_rounds=100,
         checkpoint_rule="best_round",
         assert_one_user_per_client=True,
     )
     ```
     (D-04: clones `_BENCHMARK_CROSS_DEVICE` byte-for-byte except the `mode` string. The provenance tag IS the mode name.)
   - Replace the existing `_REGISTRY` dict literal (current at line 179):
     ```python
     _REGISTRY: Dict[str, ModeProfile] = {
         "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
         "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
         "cross_silo_legacy": _CROSS_SILO_LEGACY,
     }
     ```
     with:
     ```python
     _REGISTRY: Dict[str, ModeProfile] = {
         "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
         "thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN,  # Phase 7 D-04
         "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
         "cross_silo_legacy": _CROSS_SILO_LEGACY,
     }
     ```

2. Edit `scripts/foundation/fedrec_foundation/manifest.py`:
   - Replace line 29 `RUN_MANIFEST_SCHEMA_VERSION: int = 2  # Phase 6: adds final_eval_round_index + metrics fields` with:
     ```python
     RUN_MANIFEST_SCHEMA_VERSION: int = 3  # Phase 7 D-22: adds thesis_run_label + ablation_dimension + ablation_value
     ```
   - After the existing `metrics: Dict[str, Any] = field(default_factory=dict)` block (and its closing docstring at ~line 103), add three new fields BEFORE the closing of the `RunManifest` class:
     ```python
         # Phase 7 additions (D-22) — all with safe defaults so v1/v2 fixtures
         # construct without TypeError (Pitfall 7 backward-compat invariant):
         thesis_run_label: str = ""
         """Phase 7 D-22: thesis run provenance tag.

         Sentinel ``""`` (empty string) = non-thesis run (Phase 1-6 backward compat).
         ``"main"`` = main-comparison run.
         ``"ablation_<knob>=<value>"`` = ablation run (e.g., ``"ablation_fusion_type=add"``).
         """
         ablation_dimension: str = "none"
         """Phase 7 D-22: which knob is being ablated.

         One of ``{"none", "alpha_method", "per_user_alpha", "item_perturbation",
         "contrastive_lambda", "fusion_type"}``. ``"none"`` for main runs.
         """
         ablation_value: str = ""
         """Phase 7 D-22: specific value of the ablated knob.

         Empty for main runs. Examples: ``"add"`` when ``ablation_dimension="fusion_type"``;
         ``"true"`` when ``ablation_dimension="per_user_alpha"``.
         """
     ```
   - Do NOT modify `build_run_manifest` — the new fields default to safe values and are populated server-side via `dataclass_replace` in Plan 02 (Pitfall 2 mitigation pattern from Phase 6 D-07).

3. Edit `scripts/foundation/fedrec_foundation/atomic.py`:
   - After the `atomic_write_json` function body (after line 48 closing `raise`), insert before `def _json_default(...)`:
     ```python


     def atomic_write_text(path: str, content: str) -> None:
         """Write a UTF-8 text string atomically via tempfile + ``os.replace``.

         Companion to :func:`atomic_write_json` for plain-text payloads
         (markdown, CSV, etc.). Phase 7 aggregator uses this for
         ``main_comparison.md`` / ``main_comparison.csv`` writes.

         Parameters
         ----------
         path : str
             Destination path. Parent directories are created if absent.
         content : str
             UTF-8 text payload.

         Returns
         -------
         None
         """
         parent = Path(path).parent
         parent.mkdir(parents=True, exist_ok=True)
         fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".txt")
         try:
             with os.fdopen(fd, "w", encoding="utf-8") as f:
                 f.write(content)
             os.replace(tmp, path)
         except Exception:
             try:
                 os.unlink(tmp)
             except FileNotFoundError:
                 pass
             raise
     ```

4. Edit `scripts/run.py`:
   - Replace lines 68-72 (the `MODE_NUM_SUPERNODES = {...}` dict) with:
     ```python
     MODE_NUM_SUPERNODES = {
         "benchmark_cross_device": 6040,
         "thesis_crossdevice_main": 6040,  # Phase 7 D-04
         "paper_compat_pfedrec": 6040,
         "cross_silo_legacy": 5,
     }
     ```
     (Pitfall 5: argparse `choices=sorted(MODE_NUM_SUPERNODES.keys())` automatically picks up the new mode; no other edits needed in scripts/run.py.)

5. Run `pytest scripts/foundation/tests/test_mode.py scripts/foundation/tests/test_manifest.py scripts/foundation/tests/test_launcher.py -x -v` to confirm existing tests do NOT regress. Expected: 1 failure (`test_all_three_modes_registered` — now stale; Task 2 updates it). Other tests must still pass.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && python -c "from fedrec_foundation.mode import resolve_mode_defaults, MODE_NAMES; p = resolve_mode_defaults('thesis_crossdevice_main'); assert p.embedding_dim == 64 and p.optimizer == 'adam' and p.lr == 0.001 and p.num_server_rounds == 100 and p.weight_policy == 'num_positives' and p.fraction_train == 0.1; assert 'thesis_crossdevice_main' in MODE_NAMES; print('mode OK')" && python -c "from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION, RunManifest; assert RUN_MANIFEST_SCHEMA_VERSION == 3; import inspect; src = inspect.getsource(RunManifest); assert 'thesis_run_label' in src and 'ablation_dimension' in src and 'ablation_value' in src; print('manifest OK')" && python -c "from fedrec_foundation.atomic import atomic_write_text; import tempfile, os, pathlib; td = tempfile.mkdtemp(); p = pathlib.Path(td) / 'out.txt'; atomic_write_text(str(p), 'hello'); assert p.read_text() == 'hello' and not list(pathlib.Path(td).glob('.tmp-*')); print('atomic OK')" && python scripts/run.py adaptive thesis_crossdevice_main --dry-run 2>&1 | grep -q 'mode="thesis_crossdevice_main"' && echo "launcher OK"</automated>
  </verify>
  <done>
    - `scripts/foundation/fedrec_foundation/mode.py` has `_THESIS_CROSSDEVICE_MAIN = ModeProfile(` and `"thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN` in `_REGISTRY`.
    - `scripts/foundation/fedrec_foundation/manifest.py` has `RUN_MANIFEST_SCHEMA_VERSION: int = 3` and three new fields in `RunManifest`.
    - `scripts/foundation/fedrec_foundation/atomic.py` has `def atomic_write_text(path: str, content: str) -> None:`.
    - `scripts/run.py` has `"thesis_crossdevice_main": 6040,` in `MODE_NUM_SUPERNODES`.
    - All four existing-pattern smoke checks (mode resolve, schema version, atomic_write_text round-trip, launcher dry-run) pass.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Foundation tests — extend test_mode.py + test_manifest.py + test_atomic.py + test_launcher.py</name>
  <read_first>
    - scripts/foundation/tests/test_mode.py (full file — 80 lines; read so the executor sees `test_all_three_modes_registered` at line 16 which MUST be renamed to `test_all_four_modes_registered`)
    - scripts/foundation/tests/test_manifest.py (lines 1-260; read so the executor sees `_StubProfile` at line 19, `_build()` at line 38, `test_run_manifest_schema_version_2` at line 167, `test_run_manifest_backward_compat_v1` at line 181, and the existing import block)
    - scripts/foundation/tests/test_launcher.py (full file — 79 lines; read so the executor sees the `_run` helper and the existing `test_launcher_paper_compat_pfedrec` pattern)
    - scripts/foundation/tests/conftest.py (so the executor sees if there's any pytest fixture infrastructure to reuse)
    - .planning/phases/07-thesis-evaluation-run/07-VALIDATION.md "Per-Task Verification Map" rows 7-01-01 through 7-01-05 (the exact 5 unit tests this task creates)
  </read_first>
  <behavior>
    - test_mode.py: `test_all_three_modes_registered` is RENAMED to `test_all_four_modes_registered` and the asserted set gains `"thesis_crossdevice_main"`. A new `test_thesis_crossdevice_main_profile` test asserts every field of the resolved ModeProfile matches the `_BENCHMARK_CROSS_DEVICE` clone exactly (mode name being the only difference).
    - test_manifest.py: `test_run_manifest_schema_version_2` updated to assert v3. A new `test_run_manifest_backward_compat_v2` test constructs a manifest WITHOUT thesis kwargs and asserts the three new fields default to `""` / `"none"` / `""`. A new `test_run_manifest_carries_thesis_fields` test uses `dataclass_replace` to populate the three thesis fields and asserts they roundtrip through `embed_manifest_in_result`.
    - test_atomic.py: NEW file with `test_atomic_write_text` covering: (a) content correctness, (b) UTF-8 handling, (c) no `.tmp-*` leftovers in the destination directory after success, (d) parent directory auto-creation.
    - test_launcher.py: New `test_thesis_mode_dry_run` test asserts `python scripts/run.py adaptive thesis_crossdevice_main --dry-run` exits 0 and stdout contains `mode="thesis_crossdevice_main"`.
  </behavior>
  <action>
1. Edit `scripts/foundation/tests/test_mode.py`:
   - Replace the existing `test_all_three_modes_registered` function (lines 16-19) with:
     ```python
     def test_all_four_modes_registered() -> None:
         """Phase 7 D-04: thesis_crossdevice_main joins the registry alongside the existing 3 modes."""
         assert set(MODE_NAMES) == {
             "benchmark_cross_device",
             "thesis_crossdevice_main",
             "paper_compat_pfedrec",
             "cross_silo_legacy",
         }
     ```
   - Add a NEW test function immediately after `test_paper_compat_profile` (around line 44):
     ```python
     def test_thesis_crossdevice_main_profile() -> None:
         """Phase 7 D-04: thesis_crossdevice_main clones benchmark_cross_device byte-for-byte except mode name."""
         p = resolve_mode_defaults("thesis_crossdevice_main")
         # Mode name is the provenance tag — the ONLY difference from benchmark_cross_device.
         assert p.mode == "thesis_crossdevice_main"
         # Every other field matches benchmark_cross_device verbatim (D-01).
         assert p.num_supernodes == 6040
         assert p.partition_mode == "natural"
         assert p.weight_policy == "num_positives"
         assert p.primary_evaluator == "sampled_loo_99"
         assert p.fraction_train == 0.1
         assert p.fraction_eval == 1.0
         assert p.num_train_negatives == 4
         assert p.num_eval_negatives == 99
         assert p.embedding_dim == 64
         assert p.optimizer == "adam"
         assert p.lr == 0.001
         assert p.local_epochs == 1
         assert p.num_server_rounds == 100
         assert p.checkpoint_rule == "best_round"
         assert p.assert_one_user_per_client is True
         # Sanity: byte-for-byte clone except mode name.
         from fedrec_foundation.mode import _BENCHMARK_CROSS_DEVICE
         from dataclasses import replace as _replace
         assert p == _replace(_BENCHMARK_CROSS_DEVICE, mode="thesis_crossdevice_main")
     ```

2. Edit `scripts/foundation/tests/test_manifest.py`:
   - Replace the existing `test_run_manifest_schema_version_2` function (lines 167-178) with:
     ```python
     def test_run_manifest_schema_version_3() -> None:
         """Phase 7 D-22: schema_version constant bumped from 2 to 3."""
         assert RUN_MANIFEST_SCHEMA_VERSION == 3, (
             f"Expected RUN_MANIFEST_SCHEMA_VERSION=3, got {RUN_MANIFEST_SCHEMA_VERSION}"
         )
         m = _build()
         assert m.schema_version == 3
         result_dict: Dict[str, Any] = {}
         embed_manifest_in_result(m, result_dict)
         assert result_dict["_manifest"]["schema_version"] == 3
     ```
   - Add TWO new tests immediately after the existing `test_run_manifest_backward_compat_v1` (line 181):
     ```python
     def test_run_manifest_backward_compat_v2() -> None:
         """Phase 7 D-22 + Pitfall 7: pre-v3 callers (no thesis kwargs) must construct without TypeError.

         All three new fields carry safe defaults: thesis_run_label="" (non-thesis sentinel),
         ablation_dimension="none", ablation_value="".
         """
         # Construct with the v2 field set ONLY (no thesis_run_label / ablation_dimension / ablation_value).
         manifest = RunManifest(
             schema_version=3,
             run_id="20260429-104530-v2back",
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
             # Phase 6 v2 fields populated explicitly (no thesis fields):
             final_eval_round_index=0,
             metrics={},
         )
         # Defaults must be the documented sentinels per D-22.
         assert manifest.thesis_run_label == "", "Default thesis_run_label is empty string"
         assert manifest.ablation_dimension == "none", "Default ablation_dimension is 'none'"
         assert manifest.ablation_value == "", "Default ablation_value is empty string"


     def test_run_manifest_carries_thesis_fields() -> None:
         """Phase 7 D-22: post-build mutation via dataclasses.replace populates the 3 thesis fields.

         Mirrors the Phase 6 D-07 pattern: server_app builds a manifest, then replaces it
         with the thesis-tagging fields read from context.run_config BEFORE embed_manifest_in_result.
         """
         m = _build()
         # Roundtrip empty defaults first.
         assert m.thesis_run_label == ""
         assert m.ablation_dimension == "none"
         assert m.ablation_value == ""
         # Now mutate via dataclass_replace (the canonical pattern Plan 02 server_apps will use).
         m2 = dataclass_replace(
             m,
             thesis_run_label="ablation_fusion_type=add",
             ablation_dimension="fusion_type",
             ablation_value="add",
         )
         assert m2.thesis_run_label == "ablation_fusion_type=add"
         assert m2.ablation_dimension == "fusion_type"
         assert m2.ablation_value == "add"
         # Embedded surface must also carry the mutated values.
         result_dict: Dict[str, Any] = {}
         embed_manifest_in_result(m2, result_dict)
         assert result_dict["_manifest"]["thesis_run_label"] == "ablation_fusion_type=add"
         assert result_dict["_manifest"]["ablation_dimension"] == "fusion_type"
         assert result_dict["_manifest"]["ablation_value"] == "add"
     ```

3. CREATE the new file `scripts/foundation/tests/test_atomic.py` with content:
   ```python
   """Tests for fedrec_foundation.atomic (Phase 7 D-17 / Pattern 6)."""
   from __future__ import annotations

   from pathlib import Path

   from fedrec_foundation.atomic import atomic_write_text


   def test_atomic_write_text(tmp_path: Path) -> None:
       """Phase 7: atomic_write_text writes UTF-8 content via tempfile+os.replace; no .tmp-* leftovers."""
       target = tmp_path / "out.md"
       payload = "# Header\n\n| col1 | col2 |\n|---|---|\n| 0.4123 ± 0.0089 | 0.7290 ± 0.0123 |\n"
       atomic_write_text(str(target), payload)
       # File exists + content is byte-identical.
       assert target.exists()
       assert target.read_text(encoding="utf-8") == payload
       # No .tmp-* leftovers in the parent dir (atomicity contract).
       leftovers = list(tmp_path.glob(".tmp-*"))
       assert leftovers == [], f"Expected no .tmp-* leftovers; found {leftovers}"


   def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
       """Phase 7: parent directories auto-created if absent (matches atomic_write_json semantics)."""
       target = tmp_path / "deeply" / "nested" / "dir" / "out.csv"
       atomic_write_text(str(target), "module,ndcg10_mean\nbaseline,0.4123\n")
       assert target.exists()
       assert "0.4123" in target.read_text(encoding="utf-8")


   def test_atomic_write_text_overwrites_existing(tmp_path: Path) -> None:
       """Phase 7: atomic write replaces existing file content (idempotent re-aggregation)."""
       target = tmp_path / "out.md"
       atomic_write_text(str(target), "first")
       atomic_write_text(str(target), "second")
       assert target.read_text(encoding="utf-8") == "second"
   ```

4. Edit `scripts/foundation/tests/test_launcher.py`:
   - Append a NEW test function at the end of the file (after `test_launcher_malformed_run_config_rejected`):
     ```python


     def test_thesis_mode_dry_run() -> None:
         """Phase 7 D-04: scripts/run.py accepts the new mode and emits TOML-quoted mode value."""
         r = _run("--dry-run", "adaptive", "thesis_crossdevice_main")
         assert r.returncode == 0, r.stderr
         assert 'mode="thesis_crossdevice_main"' in r.stdout
         assert "federated-adaptive-personalized-cf" in r.stdout
         # Regression: num-supernodes is federation-level, must NOT appear in --run-config.
         assert "num-supernodes" not in r.stdout
     ```

5. Run `pytest scripts/foundation/tests/test_mode.py scripts/foundation/tests/test_manifest.py scripts/foundation/tests/test_atomic.py scripts/foundation/tests/test_launcher.py -x -v` and confirm ALL tests GREEN.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && pytest scripts/foundation/tests/test_mode.py::test_all_four_modes_registered scripts/foundation/tests/test_mode.py::test_thesis_crossdevice_main_profile scripts/foundation/tests/test_manifest.py::test_run_manifest_schema_version_3 scripts/foundation/tests/test_manifest.py::test_run_manifest_backward_compat_v2 scripts/foundation/tests/test_manifest.py::test_run_manifest_carries_thesis_fields scripts/foundation/tests/test_atomic.py::test_atomic_write_text scripts/foundation/tests/test_atomic.py::test_atomic_write_text_creates_parent_dirs scripts/foundation/tests/test_atomic.py::test_atomic_write_text_overwrites_existing scripts/foundation/tests/test_launcher.py::test_thesis_mode_dry_run -x -v</automated>
  </verify>
  <done>
    - `pytest scripts/foundation/tests/test_mode.py -x -v` reports all tests passing including `test_thesis_crossdevice_main_profile` and `test_all_four_modes_registered`.
    - `pytest scripts/foundation/tests/test_manifest.py -x -v` reports all tests passing including `test_run_manifest_schema_version_3`, `test_run_manifest_backward_compat_v2`, `test_run_manifest_carries_thesis_fields`.
    - `pytest scripts/foundation/tests/test_atomic.py -x -v` reports 3 tests passing.
    - `pytest scripts/foundation/tests/test_launcher.py::test_thesis_mode_dry_run -x -v` passes.
    - The previous (renamed) `test_all_three_modes_registered` no longer exists — `grep -c "test_all_three_modes_registered" scripts/foundation/tests/test_mode.py` returns 0.
    - Full foundation suite via `cd scripts/foundation && pytest -ra` reports no regressions (existing 100+ tests remain green; 9 new tests added).
  </done>
</task>

</tasks>

<verification>
- All foundation primitives Plan 02..05 depend on are in place.
- 9 new pytest functions GREEN (1 per validation map row 7-01-01 through 7-01-05, plus 4 supplementary atomic/manifest tests).
- No regressions in existing foundation tests.
- `python scripts/run.py adaptive thesis_crossdevice_main --dry-run` exits 0.
- `from fedrec_foundation.atomic import atomic_write_text` succeeds.
- `from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION` returns `3`.
</verification>

<success_criteria>
- [ ] `scripts/foundation/fedrec_foundation/mode.py` contains `_THESIS_CROSSDEVICE_MAIN = ModeProfile(` and the registry has 4 entries (`grep -c "thesis_crossdevice_main" scripts/foundation/fedrec_foundation/mode.py >= 2`).
- [ ] `scripts/foundation/fedrec_foundation/manifest.py` contains `RUN_MANIFEST_SCHEMA_VERSION: int = 3` and the three new fields (`grep -c "thesis_run_label\|ablation_dimension\|ablation_value" scripts/foundation/fedrec_foundation/manifest.py >= 3`).
- [ ] `scripts/foundation/fedrec_foundation/atomic.py` contains `def atomic_write_text(path: str, content: str) -> None:`.
- [ ] `scripts/run.py` MODE_NUM_SUPERNODES dict contains `"thesis_crossdevice_main": 6040`.
- [ ] All 9 new pytest functions GREEN.
- [ ] Full foundation suite (`cd scripts/foundation && pytest -ra`) reports no regressions.
</success_criteria>

<output>
After completion, create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-01-SUMMARY.md` documenting:
- Final field values for `_THESIS_CROSSDEVICE_MAIN` (16 fields).
- Schema bump v2→v3 with the three new field names + defaults.
- `atomic_write_text` signature.
- Total test count (foundation suite before/after).
- Any deviations from the action text.
</output>
</content>
</invoke>