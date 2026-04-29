---
phase: 07-thesis-evaluation-run
plan: 02
type: execute
wave: 2
depends_on:
  - 07-thesis-evaluation-run-01-PLAN.md
files_modified:
  - federated-baseline-cf/federated_baseline_cf/server_app.py
  - federated-personalized-cf/federated_personalized_cf/server_app.py
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py
  - federated-pfedrec/federated_pfedrec/server_app.py
  - federated-baseline-cf/pyproject.toml
  - federated-personalized-cf/pyproject.toml
  - federated-adaptive-personalized-cf/pyproject.toml
  - federated-pfedrec/pyproject.toml
  - federated-baseline-cf/tests/test_server_integration.py
  - federated-personalized-cf/tests/test_server_integration.py
  - federated-adaptive-personalized-cf/tests/test_server_integration.py
  - federated-pfedrec/tests/test_server_integration.py
autonomous: true
requirements:
  - THS-01
  - THS-02
user_setup: []

must_haves:
  truths:
    - "All 4 server_app.py files accept thesis_crossdevice_main mode and route to federated-cf-cross-device W&B project + module_run_results_dir results path"
    - "All 4 server_app.py files read thesis-run-label, ablation-dimension, ablation-value from context.run_config and mutate the manifest via dataclass_replace BEFORE embed_manifest_in_result"
    - "All 4 pyproject.toml files declare default values for thesis-run-label, ablation-dimension, ablation-value (so flwr's fuse_dicts validation accepts the orchestrator's --run-config overrides)"
    - "All 4 modules' test_server_integration.py have a new test_thesis_label_in_manifest test that asserts the run-config -> manifest field flow"
  artifacts:
    - path: "federated-baseline-cf/federated_baseline_cf/server_app.py"
      provides: "thesis_crossdevice_main mode routing + thesis manifest mutation"
      contains: '"thesis_crossdevice_main"'
    - path: "federated-personalized-cf/federated_personalized_cf/server_app.py"
      provides: "thesis_crossdevice_main mode routing + thesis manifest mutation"
      contains: '"thesis_crossdevice_main"'
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      provides: "thesis_crossdevice_main mode routing + thesis manifest mutation"
      contains: '"thesis_crossdevice_main"'
    - path: "federated-pfedrec/federated_pfedrec/server_app.py"
      provides: "thesis_crossdevice_main mode routing + thesis manifest mutation"
      contains: '"thesis_crossdevice_main"'
  key_links:
    - from: "context.run_config['thesis-run-label']"
      to: "manifest.thesis_run_label"
      via: "dataclass_replace(manifest, thesis_run_label=str(context.run_config.get('thesis-run-label', '')), ...)"
      pattern: 'thesis_run_label=str\(context\.run_config\.get\("thesis-run-label"'
    - from: "manifest.thesis_run_label"
      to: "results.json _manifest.thesis_run_label"
      via: "embed_manifest_in_result(manifest, results_data) called AFTER dataclass_replace"
      pattern: "embed_manifest_in_result"
---

<objective>
Wire the Phase 1 foundation (mode + schema v3) into all 4 module server_apps so thesis runs:
1. Route to the correct W&B project (`federated-cf-cross-device`) and per-run results directory (`results/federated/<module>/<run_id>/`) when `mode=="thesis_crossdevice_main"`. Currently each `server_app.py` has TWO `mode in ("benchmark_cross_device", "paper_compat_pfedrec")` gates per Phase 6 D-02; both need `"thesis_crossdevice_main"` added (Pitfall 3).
2. Read three new run-config keys (`thesis-run-label`, `ablation-dimension`, `ablation-value`) and populate the manifest's three new fields via `dataclass_replace` BEFORE `embed_manifest_in_result` (Pitfall 2 — without this, manifests come out empty and the aggregator filter excludes every run).
3. Each module's `pyproject.toml` declares safe defaults for the three new keys so Flower's `fuse_dicts` validation accepts orchestrator overrides without "Key not present" errors.
4. Each module gains one `test_thesis_label_in_manifest` integration test that exercises the run-config -> manifest flow with a synthetic Context shim.

Purpose: Without this plan, the orchestrator (Plan 03) fires runs that produce manifests with `thesis_run_label=""`, the aggregator (Plan 04) silently filters them out, and `D-20 hard-fail-on-missing-cells` reports "Missing 33 cells" even though all 33 runs succeeded.

PFedRec runs at `paper_compat_pfedrec` mode (D-06), but the orchestrator passes `thesis-run-label=main` regardless of mode — so PFedRec's server_app needs the SAME manifest-mutation patch even though it never runs at `thesis_crossdevice_main`. Adding `"thesis_crossdevice_main"` to PFedRec's `mode in (...)` tuples is defensive (a future thesis-mode run from pfedrec would still route correctly) but the load-bearing change for PFedRec is the manifest mutation.

Output: 4 server_apps patched (8 line edits for mode tuples + 4 manifest-mutation patches), 4 pyproject.toml updates (3 new keys per file), 4 new integration tests.
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
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-01-PLAN.md

<interfaces>
Site map for the 4 server_app.py files (verified 2026-04-29):

federated-baseline-cf/federated_baseline_cf/server_app.py:
- line 48: `from dataclasses import replace as dataclass_replace` (already imported)
- line 294: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec")` <- W&B project gate (Pitfall 3 site #1)
- line 889: `manifest = dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` <- Phase 6 mutation site; thesis fields go HERE
- line 896: `embed_manifest_in_result(manifest, results_data)` <- MUST execute AFTER thesis mutation
- line 903: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):` <- results-path gate (Pitfall 3 site #2)

federated-personalized-cf/federated_personalized_cf/server_app.py:
- line 381: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec")` <- W&B project gate
- line 988: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):` <- results-path gate

federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:
- line 489: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec")` <- W&B project gate
- line 1292: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):` <- results-path gate

federated-pfedrec/federated_pfedrec/server_app.py:
- line 513: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec")` <- W&B project gate
- line 1156: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):` <- results-path gate

Existing pyproject.toml [tool.flwr.app.config] block already declares (verified):
- `wandb-run-name = ""` (line ~90 in baseline / ~221 in adaptive)
- `mode = "cross_silo_legacy"` (Phase 2 contract key)
- `run-seed = 42`
- `weight-policy = "num_positives"`
- `eval-num-negatives = 99`
- `checkpoint-rule = "best_round_restore"`
THREE NEW KEYS go in this block.

Existing source-string assertions in test_server_integration.py that pin the OLD 2-element tuple:
- federated-pfedrec/tests/test_server_integration.py line 321
- federated-personalized-cf/tests/test_server_integration.py line 324
These MUST be updated to the 3-element tuple form.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: server_app.py edits across all 4 modules + pyproject.toml run-config keys</name>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py lines 280-310 (W&B project gate site #1)
    - federated-baseline-cf/federated_baseline_cf/server_app.py lines 880-915 (manifest mutation site + results-path gate site #2)
    - federated-personalized-cf/federated_personalized_cf/server_app.py lines 365-395 (W&B project gate)
    - federated-personalized-cf/federated_personalized_cf/server_app.py lines 980-1000 (results-path gate)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py lines 478-500 (W&B project gate)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py lines 1280-1310 (results-path gate)
    - federated-pfedrec/federated_pfedrec/server_app.py lines 505-525 (W&B project gate)
    - federated-pfedrec/federated_pfedrec/server_app.py lines 1149-1170 (results-path gate)
    - federated-baseline-cf/pyproject.toml lines 44-100 ([tool.flwr.app.config] block)
    - federated-personalized-cf/pyproject.toml lines 44-100 ([tool.flwr.app.config] block)
    - federated-adaptive-personalized-cf/pyproject.toml lines 44-225 ([tool.flwr.app.config] block)
    - federated-pfedrec/pyproject.toml lines 45-100 ([tool.flwr.app.config] block)
    - .planning/phases/07-thesis-evaluation-run/07-CONTEXT.md sections D-22 and D-04
    - .planning/phases/07-thesis-evaluation-run/07-RESEARCH.md "Pitfall 2" + "Pitfall 3"
    - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-01-PLAN.md (Wave 1 dependency)
  </read_first>
  <behavior>
    - All 4 server_app.py files: BOTH `mode in ("benchmark_cross_device", "paper_compat_pfedrec")` tuples are extended to `mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")`. (8 line edits total — 2 per module x 4 modules.)
    - All 4 server_app.py files: 3-line patch at the existing `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` site adding `thesis_run_label`, `ablation_dimension`, `ablation_value` kwargs read from `context.run_config`. (4 patch sites.)
    - All 4 pyproject.toml files: 3 new keys added in `[tool.flwr.app.config]` block: `thesis-run-label = ""`, `ablation-dimension = "none"`, `ablation-value = ""`.
    - Existing source-string assertions in test_server_integration.py pinned at the OLD 2-element tuple are updated to the NEW 3-element tuple form so they don't break.
  </behavior>
  <action>
**Step 1 — Patch all 4 server_app.py files (8 mode-tuple edits + 4 manifest-mutation patches).**

For each of the 4 modules, perform these 3 edits:

(a) **W&B project gate (site #1) — Pitfall 3**: change the tuple in the W&B project default expression.
  - baseline server_app.py line 294: change `if mode in ("benchmark_cross_device", "paper_compat_pfedrec")` to `if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")`.
  - personalized server_app.py line 381: same string change.
  - adaptive server_app.py line 489: same string change.
  - pfedrec server_app.py line 513: same string change.

(b) **Manifest mutation patch — Pitfall 2**: locate the existing `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` call. The call is BEFORE `embed_manifest_in_result(manifest, results_data)`. Replace the existing kwargs-only call with one that ALSO passes the three thesis fields read from `context.run_config`.

EXACT replacement pattern (executor adapts indentation):

BEFORE (in baseline server_app.py around line 889):
```python
manifest = dataclass_replace(
    manifest,
    final_eval_round_index=final_eval_round_index,
    metrics=results_data["final_metrics"],
)
```

AFTER:
```python
# Phase 7 D-22: thesis-tagging fields read from run_config; sentinels for non-thesis runs.
manifest = dataclass_replace(
    manifest,
    final_eval_round_index=final_eval_round_index,
    metrics=results_data["final_metrics"],
    thesis_run_label=str(context.run_config.get("thesis-run-label", "")),
    ablation_dimension=str(context.run_config.get("ablation-dimension", "none")),
    ablation_value=str(context.run_config.get("ablation-value", "")),
)
```

Apply this same patch in all 4 server_app.py files. Variable `context` is in scope at the mutation site in every module (Flower entry-point parameter; verify via grep). MATCH existing indentation — do NOT reformat surrounding code.

(c) **Results-path gate (site #2) — Pitfall 3**: change the second `mode in (...)` tuple gating per-run-dir vs legacy results path.
  - baseline server_app.py line 903: change to `if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec"):`.
  - personalized server_app.py line 988: same string change.
  - adaptive server_app.py line 1292: same string change.
  - pfedrec server_app.py line 1156: same string change.

**Step 2 — Patch all 4 pyproject.toml files.**

For each of the 4 module pyproject.toml files, locate the `[tool.flwr.app.config]` block and add the following three keys at the END of the block (e.g., after the existing `checkpoint-rule = "best_round_restore"` line):

```toml

# Phase 7 D-22 thesis-tagging keys. Default sentinels = non-thesis run.
# Orchestrator scripts/thesis/run_thesis_sweep.py overrides these via --run-config.
# fuse_dicts requires the keys to exist with default values — declaring here is the contract.
thesis-run-label = ""        # "" = non-thesis | "main" | "ablation_<knob>=<value>"
ablation-dimension = "none"  # "none" | "alpha_method" | "per_user_alpha" | "item_perturbation" | "contrastive_lambda" | "fusion_type"
ablation-value = ""           # specific value of the ablated knob; empty for main/non-thesis
```

Apply identically in:
- federated-baseline-cf/pyproject.toml
- federated-personalized-cf/pyproject.toml
- federated-adaptive-personalized-cf/pyproject.toml
- federated-pfedrec/pyproject.toml

**Step 3 — Update existing source-string assertions in test_server_integration.py.**

First, run the grep safety net to find ALL occurrences across all 4 module test directories — do NOT trust the per-site list at face value:
```bash
grep -rn '"benchmark_cross_device", "paper_compat_pfedrec"' federated-baseline-cf/tests/ federated-personalized-cf/tests/ federated-adaptive-personalized-cf/tests/ federated-pfedrec/tests/
```

Update every match to the 3-element tuple form. Survey at planning time confirms 2 sites contain the literal:
- federated-pfedrec/tests/test_server_integration.py line ~321
- federated-personalized-cf/tests/test_server_integration.py line ~324

Baseline and adaptive-personalized `test_server_integration.py` files do NOT currently contain this 2-element literal (their `test_client_assertion.py` files already use the 3-element tuple form via `("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy")` which is a different code path and stays untouched). The grep step is the safety net to catch any drift between planning and execution.

Locate any assertion of the form:
```python
assert 'if mode in ("benchmark_cross_device", "paper_compat_pfedrec")' in src, (...)
```
and replace the literal-tuple substring with the 3-element tuple.

EXACT replacement string:
```python
assert 'if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")' in src, (
    "Phase 7 D-04: thesis_crossdevice_main mode joins the per-run-dir gate"
)
```

After the edit, re-run the grep with a stricter pattern to verify ZERO 2-tuple literals remain:
```bash
grep -rn '"benchmark_cross_device", "paper_compat_pfedrec"[^,]' federated-*-cf/tests/test_server_integration.py
```
This MUST return zero matches — proves all old 2-tuple literals are upgraded to 3-tuples (the trailing `[^,]` excludes the 3-tuple form which has a comma after the second element).
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && for module in federated-baseline-cf/federated_baseline_cf federated-personalized-cf/federated_personalized_cf federated-adaptive-personalized-cf/federated_adaptive_personalized_cf federated-pfedrec/federated_pfedrec; do count=$(grep -c '"thesis_crossdevice_main"' "$module/server_app.py"); test "$count" -ge 2 || (echo "FAIL: $module/server_app.py has only $count thesis_crossdevice_main occurrences (need >=2)" && exit 1); done && for module in federated-baseline-cf/federated_baseline_cf federated-personalized-cf/federated_personalized_cf federated-adaptive-personalized-cf/federated_adaptive_personalized_cf federated-pfedrec/federated_pfedrec; do count=$(grep -c 'thesis_run_label=str(context.run_config.get' "$module/server_app.py"); test "$count" -eq 1 || (echo "FAIL: $module/server_app.py manifest mutation patch missing or duplicated (count=$count)" && exit 1); done && for tomlfile in federated-baseline-cf/pyproject.toml federated-personalized-cf/pyproject.toml federated-adaptive-personalized-cf/pyproject.toml federated-pfedrec/pyproject.toml; do grep -q '^thesis-run-label = ""' "$tomlfile" || (echo "FAIL: $tomlfile missing thesis-run-label" && exit 1); grep -q '^ablation-dimension = "none"' "$tomlfile" || (echo "FAIL: $tomlfile missing ablation-dimension" && exit 1); grep -q '^ablation-value = ""' "$tomlfile" || (echo "FAIL: $tomlfile missing ablation-value" && exit 1); done && stale_2tuple=$(grep -rln '"benchmark_cross_device", "paper_compat_pfedrec"[^,]' federated-baseline-cf/tests/test_server_integration.py federated-personalized-cf/tests/test_server_integration.py federated-adaptive-personalized-cf/tests/test_server_integration.py federated-pfedrec/tests/test_server_integration.py 2>/dev/null | wc -l) && test "$stale_2tuple" -eq 0 || (echo "FAIL: $stale_2tuple test_server_integration.py file(s) still contain the OLD 2-tuple literal" && exit 1) && echo "All server_app + pyproject + 3-tuple-assertion patches verified"</automated>
  </verify>
  <done>
    - All 4 server_app.py files contain `"thesis_crossdevice_main"` at least 2 times (mode-tuple gates).
    - All 4 server_app.py files contain `thesis_run_label=str(context.run_config.get("thesis-run-label"` exactly once (manifest mutation patch).
    - All 4 pyproject.toml files declare `thesis-run-label = ""`, `ablation-dimension = "none"`, `ablation-value = ""`.
    - Existing source-string assertions in test_server_integration.py updated to expect the 3-element tuple.
    - **Warning 3 closure:** `grep -rn '"benchmark_cross_device", "paper_compat_pfedrec"[^,]' federated-*-cf/tests/test_server_integration.py` returns ZERO matches across ALL 4 module test directories after the edit (proves all old 2-tuple literals are upgraded to 3-tuples; survey at planning time confirmed only pfedrec + personalized contain the 2-tuple, but the grep across all 4 dirs is the safety net against drift between planning and execution).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Per-module integration tests — test_thesis_label_in_manifest in all 4 modules</name>
  <read_first>
    - federated-baseline-cf/tests/test_server_integration.py (full file — read so the executor sees existing test patterns, conftest fixtures, and import style)
    - federated-personalized-cf/tests/test_server_integration.py (full file)
    - federated-adaptive-personalized-cf/tests/test_server_integration.py (full file)
    - federated-pfedrec/tests/test_server_integration.py (full file)
    - .planning/phases/07-thesis-evaluation-run/07-VALIDATION.md "Per-Task Verification Map" rows 7-02-01 through 7-02-04
  </read_first>
  <behavior>
    - Each module's test_server_integration.py gains ONE new test function `test_thesis_label_in_manifest` that asserts source-level wiring: the `dataclass_replace(manifest, ...)` call in the module's server_app.py contains all 5 expected kwargs (final_eval_round_index, metrics, thesis_run_label, ablation_dimension, ablation_value). It also verifies the call ordering (dataclass_replace must run BEFORE embed_manifest_in_result) and that BOTH `mode in (...)` tuples carry "thesis_crossdevice_main".
    - The test does NOT spawn a real Flower run (those take 30+ minutes); instead it reads the server_app.py source via Path read_text and asserts substring presence. Same pattern as the existing pfedrec test_server_integration.py source-string assertions.
    - All 4 tests use the SAME logic, parameterized only by the module's package directory name. Code duplication across 4 files is expected and matches the existing test_server_integration.py copy-paste pattern.
  </behavior>
  <action>
For EACH of the 4 modules `(baseline, personalized, adaptive, pfedrec)`, append the following NEW test function at the END of `federated-<module>-cf/tests/test_server_integration.py`. The package directory name varies per module — substitute the right one:
- baseline: `federated_baseline_cf`
- personalized: `federated_personalized_cf`
- adaptive: `federated_adaptive_personalized_cf`
- pfedrec: `federated_pfedrec`

Before adding the test, check whether a `_server_app_src()` helper or equivalent (e.g., `_src` or inline `Path(...).read_text()`) already exists in the file. If yes, reuse it; if no, add this helper at the top (after imports):

```python
def _server_app_src() -> str:
    """Read this module's server_app.py source for static wiring assertions."""
    src_path = Path(__file__).resolve().parents[1] / "<package_dir>" / "server_app.py"
    return src_path.read_text(encoding="utf-8")
```
(Replace `<package_dir>` with the module's package directory name.)

Then append this test at the END of the file:

```python


def test_thesis_label_in_manifest() -> None:
    """Phase 7 D-22 + Pitfall 2: server_app reads thesis-run-label / ablation-dimension /
    ablation-value from context.run_config and mutates the manifest via dataclass_replace
    BEFORE embed_manifest_in_result so results.json's _manifest carries the thesis-tagging fields.

    This is a STATIC source-level wiring test — it does NOT spawn a Flower run.
    The integration loop is exercised by Plan 05's smoke-run gate.
    """
    src = _server_app_src()
    # The 3 thesis kwargs MUST appear inside the dataclass_replace(manifest, ...) call.
    assert 'thesis_run_label=str(context.run_config.get("thesis-run-label"' in src, (
        "Phase 7 D-22: server_app must read thesis-run-label from run_config and pass to dataclass_replace"
    )
    assert 'ablation_dimension=str(context.run_config.get("ablation-dimension"' in src, (
        "Phase 7 D-22: server_app must read ablation-dimension from run_config and pass to dataclass_replace"
    )
    assert 'ablation_value=str(context.run_config.get("ablation-value"' in src, (
        "Phase 7 D-22: server_app must read ablation-value from run_config and pass to dataclass_replace"
    )
    # Phase-6 final_eval_round_index + metrics kwargs must coexist (regression guard for Phase 6).
    assert "final_eval_round_index=final_eval_round_index" in src, (
        "Phase 6 D-07: final_eval_round_index kwarg MUST coexist with Phase 7 thesis kwargs"
    )
    assert 'metrics=results_data["final_metrics"]' in src, (
        "Phase 6 D-07: metrics kwarg MUST coexist with Phase 7 thesis kwargs"
    )
    # Mutation MUST execute BEFORE embed_manifest_in_result.
    idx_thesis_kwarg = src.find('thesis_run_label=str(context.run_config.get')
    idx_embed = src.find("embed_manifest_in_result(manifest, results_data)")
    assert idx_thesis_kwarg != -1, "Could not locate the thesis_run_label kwarg in server_app.py"
    assert idx_embed != -1, "Could not locate embed_manifest_in_result(manifest, results_data) call site"
    assert idx_thesis_kwarg < idx_embed, (
        "Phase 7 D-22 + Pitfall 2 invariant: dataclass_replace(manifest, ...thesis fields...) "
        "MUST execute BEFORE embed_manifest_in_result so the embedded _manifest dict "
        "carries the thesis-tagging fields. Found embed_manifest_in_result first — order is wrong."
    )
    # Pitfall 3 site #1 + #2: BOTH mode tuples include thesis_crossdevice_main.
    assert src.count('"thesis_crossdevice_main"') >= 2, (
        "Phase 7 Pitfall 3: BOTH `mode in (...)` tuples in this server_app must include "
        "'thesis_crossdevice_main' (W&B project gate + results-path gate)"
    )
```

After adding all 4 tests, run the verification command in the `<verify>` block.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && pytest federated-baseline-cf/tests/test_server_integration.py::test_thesis_label_in_manifest federated-personalized-cf/tests/test_server_integration.py::test_thesis_label_in_manifest federated-adaptive-personalized-cf/tests/test_server_integration.py::test_thesis_label_in_manifest federated-pfedrec/tests/test_server_integration.py::test_thesis_label_in_manifest -x -v</automated>
  </verify>
  <done>
    - 4 PASSED: one `test_thesis_label_in_manifest` per module, each green.
    - `grep -l "test_thesis_label_in_manifest" federated-*-cf/tests/test_server_integration.py` returns 4 files.
    - Full per-module test suites do NOT regress: `cd federated-baseline-cf && pytest tests/ -ra` (and same for personalized / adaptive / pfedrec) report no new failures.
  </done>
</task>

</tasks>

<verification>
- All 4 server_app.py files have `thesis_crossdevice_main` in BOTH mode tuples + the 3-thesis-kwarg manifest mutation patch.
- All 4 pyproject.toml files declare the 3 new run-config keys.
- All 4 modules have a green `test_thesis_label_in_manifest` source-level wiring test.
- A fresh smoke run via `python scripts/run.py adaptive thesis_crossdevice_main --run-config "thesis-run-label=main run-seed=42 num-server-rounds=2 fraction-train=0.001 wandb-enabled=false"` (Plan 05's job, NOT this plan) will produce a manifest carrying `thesis_run_label="main"`. Plan 02's static source-level test is the proxy for that smoke run.
</verification>

<success_criteria>
- [ ] All 4 server_app.py files have count `"thesis_crossdevice_main"` >= 2 (BOTH mode tuples).
- [ ] All 4 server_app.py files have count `thesis_run_label=str(context.run_config.get("thesis-run-label"` == 1 (manifest mutation).
- [ ] All 4 pyproject.toml files have `thesis-run-label = ""`, `ablation-dimension = "none"`, `ablation-value = ""` lines.
- [ ] 4 PASSED: `test_thesis_label_in_manifest` in all 4 modules.
- [ ] Existing source-string assertions updated (no `assert 'if mode in ("benchmark_cross_device", "paper_compat_pfedrec")'` substring remains in any test file).
</success_criteria>

<output>
After completion, create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-02-SUMMARY.md` documenting:
- The 8 mode-tuple line numbers (post-edit) per module.
- The 4 manifest-mutation patch sites + post-edit line numbers.
- The 4 pyproject.toml line numbers where the 3 new keys land.
- Any deviations from the action text (e.g., did all 4 modules already have a `_server_app_src` helper, or did some need the helper added).
- The "before" and "after" tuple-literal strings updated in test_server_integration.py.
</output>
</content>
</invoke>