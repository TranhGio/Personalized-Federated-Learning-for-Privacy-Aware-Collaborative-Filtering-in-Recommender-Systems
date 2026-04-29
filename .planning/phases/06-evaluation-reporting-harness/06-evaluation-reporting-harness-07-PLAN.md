---
phase: 06-evaluation-reporting-harness
plan: 07
type: execute
wave: 3
depends_on:
  - 06-evaluation-reporting-harness-03
  - 06-evaluation-reporting-harness-04
  - 06-evaluation-reporting-harness-05
  - 06-evaluation-reporting-harness-06
files_modified:
  - federated-adaptive-personalized-cf/sweep.yaml
  - federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py
  - federated-baseline-cf/tests/test_server_integration.py
  - federated-personalized-cf/tests/test_server_integration.py
  - federated-adaptive-personalized-cf/tests/test_server_integration.py
  - federated-pfedrec/tests/test_server_integration.py
autonomous: true
requirements: [EVL-03, EVL-05, EVL-06]
must_haves:
  truths:
    - "federated-adaptive-personalized-cf/sweep.yaml metric.name migrated from `final/sampled_ndcg@10` to `best/sampled_ndcg@10` (Pitfall 7 closure)"
    - "NEW test_wandb_summary_keys.py asserts wandb.run.summary uses best/* and last/* namespaces (final/* removed for thesis metrics)"
    - "Per-round exposure history regression guard added to ALL FOUR test_server_integration.py files (D-09 — evaluated_users_{sparse,medium,dense} present per round)"
    - "All four modules' subprocess determinism guards still pass with the new layout (handled by Plans 03-06 already; this plan only adds cross-cutting unit-level assertions)"
    - "EVL-05 verification: wandb-project default 'federated-cf-cross-device' is preserved verbatim — D-05 zero-churn"
  artifacts:
    - path: "federated-adaptive-personalized-cf/sweep.yaml"
      provides: "metric.name migrated to best/sampled_ndcg@10 (line 18)"
      contains: "name: best/sampled_ndcg@10"
    - path: "federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py"
      provides: "NEW unit test for W&B summary key migration + sweep.yaml metric.name grep guard"
      contains: "def test_summary_keys_use_best_last_namespace"
    - path: "federated-baseline-cf/tests/test_server_integration.py"
      provides: "EXTENDED with D-09 per-round exposure history assertion"
      contains: "test_round_metrics_history_carries_per_group_exposure"
    - path: "federated-personalized-cf/tests/test_server_integration.py"
      provides: "EXTENDED with D-09 per-round exposure history assertion"
      contains: "test_round_metrics_history_carries_per_group_exposure"
    - path: "federated-adaptive-personalized-cf/tests/test_server_integration.py"
      provides: "EXTENDED with D-09 per-round exposure history assertion"
      contains: "test_round_metrics_history_carries_per_group_exposure"
    - path: "federated-pfedrec/tests/test_server_integration.py"
      provides: "EXTENDED with D-09 per-round exposure history assertion"
      contains: "test_round_metrics_history_carries_per_group_exposure"
  key_links:
    - from: "federated-adaptive-personalized-cf/sweep.yaml::metric.name"
      to: "All four server_app.py wandb.run.summary[f'best/sampled_ndcg@10']"
      via: "Bayesian sweep optimizer reads the metric key from W&B summary; key namespace migrated final/* -> best/* per Pitfall 7"
      pattern: "name: best/sampled_ndcg@10"
    - from: "federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py"
      to: "Plans 03-06 server_app.py W&B summary write blocks"
      via: "Test reads server_app.py source AND sweep.yaml; asserts no final/* reference for thesis metrics + sweep metric.name uses best/*"
      pattern: "wandb\\.run\\.summary\\[f.best/|wandb\\.run\\.summary\\[f.last/"
---

<objective>
Cross-cutting Wave-3 work that closes EVL-05 (W&B project routing — already wired by Plans 03-06; this plan only verifies), EVL-06 (canonical reporting uses best_*; sweep optimizer must follow the namespace migration), and EVL-03 (per-round exposure history surfaces in all four modules' result JSON).

Purpose:
  - **Pitfall 7 closure**: Bump `federated-adaptive-personalized-cf/sweep.yaml:18` `metric.name` from `final/sampled_ndcg@10` to `best/sampled_ndcg@10`. Plans 03-06 migrate the wandb summary write keys; this plan migrates the SWEEP CONFIG that reads them. Without this, the next `wandb agent` run would silently report NaN for the metric and stop converging.
  - Add a NEW unit test `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` that grep-guards both surfaces: (a) the four `server_app.py` files contain `wandb.run.summary[f"best/...` and `wandb.run.summary[f"last/...` and DO NOT contain `wandb.run.summary[f"final/...` for thesis metrics, and (b) `sweep.yaml` line 18 reads `name: best/sampled_ndcg@10`.
  - Add a per-round exposure history regression guard (`test_round_metrics_history_carries_per_group_exposure`) to all FOUR `test_server_integration.py` files. Each test asserts that the per-round `eval_metrics_history` entries serialized into `results.json` carry `evaluated_users_sparse`, `evaluated_users_medium`, `evaluated_users_dense` keys (D-09 — the support counts that let readers interpret per-group metrics with the right variance lens).

Output:
  - `federated-adaptive-personalized-cf/sweep.yaml` line 18 mutated (single-line edit).
  - `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` (NEW): 2 tests covering namespace migration and sweep.yaml grep.
  - 4 `test_server_integration.py` files extended with one new test each: `test_round_metrics_history_carries_per_group_exposure` (the test name is identical across all four modules to maximize convention reuse).
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
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf/sweep.yaml
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-personalized-cf/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/tests/test_server_integration.py

<interfaces>
<!-- Pre-Phase-6 sweep.yaml line 18 (Pitfall 7 anchor) -->
```yaml
# federated-adaptive-personalized-cf/sweep.yaml:17-19
metric:
  name: final/sampled_ndcg@10   # <-- MUST migrate to best/sampled_ndcg@10
  goal: maximize
```

<!-- Plans 03-06 wandb.run.summary write surface (already migrated by upstream plans) -->
```python
# Every server_app.py post-Phase-6 contains:
for key, value in final_metrics["best"].items():
    if isinstance(value, (int, float)):
        wandb.run.summary[f"best/{key}"] = value   # <-- this is what the sweep optimizer needs to read
for key, value in final_metrics["last"].items():
    if isinstance(value, (int, float)):
        wandb.run.summary[f"last/{key}"] = value
```

<!-- Per-round eval_metrics_history shape (D-09) -->
```python
# Already populated by Plans 03-06 via strategy.aggregate_evaluate output:
eval_metrics_history[round_num] = {
    "sampled_ndcg@10": ..., "sampled_hr@10": ...,
    "sampled_ndcg@10/sparse": ..., "sampled_ndcg@10/medium": ..., "sampled_ndcg@10/dense": ...,
    "sampled_hr@10/sparse": ..., "sampled_hr@10/medium": ..., "sampled_hr@10/dense": ...,
    "evaluated_users": ...,
    "evaluated_users_sparse": ...,    # <-- D-09 per-round support count
    "evaluated_users_medium": ...,    # <-- D-09 per-round support count
    "evaluated_users_dense": ...,     # <-- D-09 per-round support count
    ...
}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Bump federated-adaptive-personalized-cf/sweep.yaml metric.name from final/sampled_ndcg@10 to best/sampled_ndcg@10 (Pitfall 7 closure); ship NEW test_wandb_summary_keys.py with 2 unit tests grepping both server_app.py files and sweep.yaml</name>
  <files>federated-adaptive-personalized-cf/sweep.yaml, federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/sweep.yaml — line 18 (current `name: final/sampled_ndcg@10`); confirm by `grep -n "final/sampled_ndcg" federated-adaptive-personalized-cf/sweep.yaml` returns line 18 only
    - federated-baseline-cf/federated_baseline_cf/server_app.py — confirm `wandb.run.summary[f"best/{key}"]` is present (Plan 03 output)
    - federated-personalized-cf/federated_personalized_cf/server_app.py — same (Plan 04 output)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py — same (Plan 05 output)
    - federated-pfedrec/federated_pfedrec/server_app.py — same (Plan 06 output); note: pfedrec also has standalone wandb.run.summary["pfr08"] etc. — those are NOT thesis-metric keys, do not flag them
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Common Pitfalls Pitfall 7 (sweep.yaml metric.name)
    - .planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md §Per-Task Verification Map row 6-07-01
    - **MAJOR fix note (plan-checker iteration 1):** before authoring the sweep test, executor MUST read `federated-adaptive-personalized-cf/sweep.yaml` end-to-end so the YAML schema (top-level `metric` block, `name` / `goal` keys) is concretely understood. The test then uses `yaml.safe_load` to navigate that schema, NOT a substring grep. PyYAML is already a wandb transitive dep so no new install is required; document this in the SUMMARY.
  </read_first>
  <action>
**Edit 1: Single-line sweep.yaml mutation.**

Open `federated-adaptive-personalized-cf/sweep.yaml`, locate line 18 (currently `  name: final/sampled_ndcg@10`), and change it to:

```yaml
  name: best/sampled_ndcg@10
```

That is the entire mutation — one literal string change. The `goal: maximize` line below remains unchanged.

**Edit 2: Create NEW test file.** Create `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py`:

```python
"""Phase 6: W&B summary key namespace migration regression guard.

Closes Pitfall 7 (sweep.yaml metric.name) and EVL-06 canonical reporting
contract. The four module server_app.py files migrated from
``wandb.run.summary[f"final/{key}"] = value`` to ``wandb.run.summary[f"best/...
+ wandb.run.summary[f"last/..."]`` per D-07. This file pins both surfaces
mechanically:

1. Sweep optimizer reads `best/sampled_ndcg@10` (NOT final/sampled_ndcg@10).
2. No server_app.py file emits a `final/{thesis_metric}` summary key.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Resolve the repo root from the test's location (this test file lives at
# federated-adaptive-personalized-cf/tests/, so repo root is parent[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]

_SWEEP_YAML = _REPO_ROOT / "federated-adaptive-personalized-cf" / "sweep.yaml"

_SERVER_APPS = [
    _REPO_ROOT / "federated-baseline-cf" / "federated_baseline_cf" / "server_app.py",
    _REPO_ROOT / "federated-personalized-cf" / "federated_personalized_cf" / "server_app.py",
    _REPO_ROOT / "federated-adaptive-personalized-cf" / "federated_adaptive_personalized_cf" / "server_app.py",
    _REPO_ROOT / "federated-pfedrec" / "federated_pfedrec" / "server_app.py",
]


def test_sweep_yaml_metric_name_uses_best_namespace():
    """Pitfall 7: sweep.yaml metric.name MUST be best/sampled_ndcg@10.

    The Bayesian sweep optimizer reads this name from W&B summary keys.
    After Phase 6 the canonical thesis metric lives at
    ``wandb.run.summary["best/sampled_ndcg@10"]``; the legacy
    ``final/sampled_ndcg@10`` would silently report NaN, breaking convergence.

    MAJOR fix (plan-checker iteration 1): we YAML-parse the file and assert
    against the structured ``loaded["metric"]["name"]`` field, NOT a substring
    grep. A future comment like ``# was final/sampled_ndcg@10`` would have
    spuriously satisfied the substring check; structured parse cannot be
    fooled. PyYAML is already a wandb transitive dependency — no new install.
    """
    import yaml  # PyYAML — wandb transitive dep, no new install needed

    loaded = yaml.safe_load(_SWEEP_YAML.read_text())
    assert isinstance(loaded, dict), (
        f"sweep.yaml did not parse as a dict: {type(loaded).__name__}"
    )
    assert "metric" in loaded, (
        f"sweep.yaml is missing top-level 'metric' block; cannot verify name."
    )
    metric = loaded["metric"]
    assert isinstance(metric, dict), (
        f"sweep.yaml metric block is not a dict: {type(metric).__name__}"
    )
    assert metric.get("name") == "best/sampled_ndcg@10", (
        f"sweep.yaml metric.name not migrated to best/* namespace. "
        f"Expected 'best/sampled_ndcg@10', got {metric.get('name')!r}."
    )
    assert metric.get("name") != "final/sampled_ndcg@10", (
        f"sweep.yaml still references legacy final/* namespace. "
        f"Active wandb agents would silently report NaN."
    )


@pytest.mark.parametrize("server_app_path", _SERVER_APPS, ids=lambda p: p.parts[-3])
def test_summary_keys_use_best_last_namespace(server_app_path):
    """EVL-06 + D-07: every module's server_app.py uses best/* + last/* for
    thesis metrics; no module still emits final/{thesis_metric}.

    The pfedrec module also writes standalone summary keys
    ``wandb.run.summary["pfr08"]`` and ``wandb.run.summary["pfr08_delta_*"]``
    — those are PFR-08 audit surface keys, NOT thesis metrics, and are
    intentionally namespaced at the top level (no best/last prefix). The
    grep below specifically targets ``f"final/`` (the f-string-prefixed key
    syntax used for thesis-metric loops); the standalone audit keys remain
    matchable but never appear under the f-string prefix.
    """
    src = server_app_path.read_text()

    # Positive surfaces: every module must emit at least one best/* and last/*
    # f-string summary key (the thesis-metric write loop).
    assert 'wandb.run.summary[f"best/' in src, (
        f"{server_app_path} missing best/* summary write loop"
    )
    assert 'wandb.run.summary[f"last/' in src, (
        f"{server_app_path} missing last/* summary write loop"
    )

    # Negative surface: no module may still emit a final/{thesis_metric} loop.
    assert 'wandb.run.summary[f"final/' not in src, (
        f"{server_app_path} still references legacy final/* namespace "
        f"for thesis metrics (D-07 / Pitfall 7 regression)."
    )

    # MINOR fix (plan-checker iteration 1): also catch the NON-f-string variant
    # ``wandb.run.summary["final/..."]`` (raw string literal). Plan 06 removed
    # both forms (`final/pfr08*` migrated to top-level `pfr08*`, thesis metrics
    # migrated to best/last namespaces). This guards against either future
    # regression. The pfedrec PFR-08 audit surface uses
    # ``wandb.run.summary["pfr08"]`` etc. which does NOT match the
    # ``"final/`` prefix — so this assertion does not flag pfedrec audit keys.
    assert 'wandb.run.summary["final/' not in src, (
        f"{server_app_path} still references legacy final/* namespace via "
        f"a raw (non-f-string) literal — likely a pre-Phase-6 holdover that "
        f"the migration missed (Pitfall 7 regression, raw-string variant)."
    )
```

Verify by running:

```bash
cd federated-adaptive-personalized-cf && pytest tests/test_wandb_summary_keys.py -x -v
```

The 2 test functions × 4 parametrize expansion = 5 test items MUST pass.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf && pytest tests/test_wandb_summary_keys.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "name: best/sampled_ndcg@10" federated-adaptive-personalized-cf/sweep.yaml` returns 1
    - `grep -c "name: final/sampled_ndcg@10" federated-adaptive-personalized-cf/sweep.yaml` returns 0 (legacy line removed)
    - `test -f federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` exits 0
    - `cd federated-adaptive-personalized-cf && pytest tests/test_wandb_summary_keys.py -x -v` exits 0 with all 5 test items (1 sweep test + 4 server_app parametrized) passing
    - File otherwise unchanged: `git diff federated-adaptive-personalized-cf/sweep.yaml` shows ONLY the line-18 mutation (single line)
    - **MAJOR fix (sweep YAML structured parse, plan-checker iteration 1):** Sweep test uses `yaml.safe_load` + `loaded["metric"]["name"] == "best/sampled_ndcg@10"`, NOT a substring `in text` check. A comment like `# was final/sampled_ndcg@10` cannot satisfy the structured parse. Acceptance: `grep -c "yaml.safe_load(_SWEEP_YAML.read_text())" federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` returns 1; `grep -c 'loaded\["metric"\]\["name"\]' federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` returns at least 1; `grep -c '"name: best/sampled_ndcg@10" in text' federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` returns 0 (the legacy substring assertion is removed). PyYAML is a wandb transitive dependency — no new install needed.
    - **MINOR fix (raw-string final/* assertion, plan-checker iteration 1):** Test asserts BOTH `wandb.run.summary[f"final/` (f-string variant) AND `wandb.run.summary["final/` (raw-string variant) are absent from every module's server_app.py. Acceptance: `grep -c 'wandb.run.summary\["final/' federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` returns at least 1 (the new third assertion is present in the test body)
  </acceptance_criteria>
  <done>
    - sweep.yaml metric.name migrated to best/sampled_ndcg@10 (Pitfall 7 closure)
    - test_wandb_summary_keys.py NEW file with 2 tests pinning sweep + all 4 server_app namespace migrations
    - **MAJOR closure (sweep YAML structured parse, plan-checker iteration 1):** sweep test uses `yaml.safe_load` + `loaded["metric"]["name"]` structured navigation; substring grep variant explicitly removed
    - **MINOR closure (raw-string final/* assertion, plan-checker iteration 1):** test asserts BOTH f-string and raw-string forms of `wandb.run.summary["final/...]` are absent
    - All 5 test items pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Add cross-cutting D-09 per-round exposure history assertion to all four test_server_integration.py files (one new test per module: test_round_metrics_history_carries_per_group_exposure)</name>
  <files>federated-baseline-cf/tests/test_server_integration.py, federated-personalized-cf/tests/test_server_integration.py, federated-adaptive-personalized-cf/tests/test_server_integration.py, federated-pfedrec/tests/test_server_integration.py</files>
  <read_first>
    - federated-baseline-cf/tests/test_server_integration.py — current state AFTER Plan 03 (already extended with 4 NEW tests for path/extra_eval/best_last_blocks/per_group_exposure). NOTE: Plan 03's Test 4 (test_round_metrics_history_carries_per_group_exposure) is the same surface as this Task. The executor MUST verify whether Plan 03 already added this exact test — if so, skip the baseline edit and only update the other 3 modules.
    - federated-personalized-cf/tests/test_server_integration.py — same check post-Plan-04
    - federated-adaptive-personalized-cf/tests/test_server_integration.py — same check post-Plan-05
    - federated-pfedrec/tests/test_server_integration.py — same check post-Plan-06
    - .planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md §Per-Task Verification Map row 6-07-02
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-09
  </read_first>
  <action>
**Important coordination note**: Plans 03-06 each include a `test_round_metrics_history_carries_per_group_exposure` test in their respective `test_server_integration.py` files (Tests 4 in Plans 03-06's Task 1 behavior block). This Plan 07 Task 2 acts as the cross-cutting safety net: if any Wave-2/3 plan executor accidentally drops or weakens that test, this task re-asserts the convention.

**Step 1: Inspect each test_server_integration.py for the existing Plan 03/04/05/06 test.**

For each of the four test files, run:

```bash
grep -c "test_round_metrics_history_carries_per_group_exposure" federated-baseline-cf/tests/test_server_integration.py
grep -c "test_round_metrics_history_carries_per_group_exposure" federated-personalized-cf/tests/test_server_integration.py
grep -c "test_round_metrics_history_carries_per_group_exposure" federated-adaptive-personalized-cf/tests/test_server_integration.py
grep -c "test_round_metrics_history_carries_per_group_exposure" federated-pfedrec/tests/test_server_integration.py
```

**Step 2: For each module that returns 0** (i.e., the upstream plan executor did NOT add the test), append the following test function. Use the same fixture/mocking style as the other tests in that file (read the existing test functions to identify the conftest fixtures and the result-loading pattern). Below is the canonical assertion body — adapt the fixture wiring per module:

```python
def test_round_metrics_history_carries_per_group_exposure(MODULE_FIXTURE):
    """D-09: every per-round eval_metrics_history entry carries
    evaluated_users_{sparse,medium,dense} so per-group metrics can be read
    with the right variance lens.

    Phase 6's strategy.aggregate_evaluate already emits these keys (sufficient-
    stat sum + ratio at the end). This test pins the surface so a future
    silent removal cannot land without tripping the regression guard.
    """
    results = MODULE_FIXTURE.run_and_load_results()  # adapt to module fixture
    eval_history = results["eval_metrics_history"]
    assert len(eval_history) > 0, "Expected at least one round in eval_metrics_history"

    required_keys = {
        "evaluated_users",
        "evaluated_users_sparse",
        "evaluated_users_medium",
        "evaluated_users_dense",
    }
    rounds_with_full_exposure = [
        r for r, m in eval_history.items()
        if required_keys.issubset(set(m.keys()))
    ]
    assert rounds_with_full_exposure, (
        f"D-09 regression: no round in eval_metrics_history carries all of "
        f"{required_keys}. Sample round keys: "
        f"{sorted(next(iter(eval_history.values())).keys())[:20]}"
    )
```

**Step 3: For each module that returns >= 1** (upstream plan already added the test), VERIFY the test still passes:

```bash
cd federated-{module} && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure
```

If a module's upstream plan added a weaker version of the test (e.g., only checks one of the 4 keys), STRENGTHEN it to the canonical assertion body above. Document the strengthening in this plan's SUMMARY.

**Step 4: Run all four module suites:**

```bash
cd federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure
cd ../federated-personalized-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure
cd ../federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure
cd ../federated-pfedrec && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure
```

All 4 module test invocations MUST exit 0.

**Step 5: Aggregate cross-cutting unit-test verification across all four module suites:**

```bash
for mod in federated-baseline-cf federated-personalized-cf federated-adaptive-personalized-cf federated-pfedrec; do
    echo "=== $mod ==="
    cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/$mod
    pytest tests/ -q -m "not slow"
done
```

Every module's full unit-test suite (excluding @pytest.mark.slow) MUST exit 0.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure && cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-personalized-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure && cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure && cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure</automated>
  </verify>
  <acceptance_criteria>
    - **MAJOR fix (per-module verification, plan-checker iteration 1):** Each module's regression guard MUST be verifiable in isolation so a failure points to a SPECIFIC module rather than reporting "one of four failed". The four explicit per-module greps + four explicit per-module pytest invocations below collectively give that diagnostic granularity.
    - `grep -c "test_round_metrics_history_carries_per_group_exposure" federated-baseline-cf/tests/test_server_integration.py` returns 1 (exactly one definition — duplicates would indicate Plan 03 + Plan 07 both added it)
    - `grep -c "test_round_metrics_history_carries_per_group_exposure" federated-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_round_metrics_history_carries_per_group_exposure" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_round_metrics_history_carries_per_group_exposure" federated-pfedrec/tests/test_server_integration.py` returns 1
    - `cd federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure` exits 0 (baseline-only invocation — failure isolates to baseline)
    - `cd federated-personalized-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure` exits 0 (personalized-only invocation — failure isolates to personalized)
    - `cd federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure` exits 0 (adaptive-only invocation — failure isolates to adaptive)
    - `cd federated-pfedrec && pytest tests/test_server_integration.py -x -v -k test_round_metrics_history_carries_per_group_exposure` exits 0 (pfedrec-only invocation — failure isolates to pfedrec)
    - All four `pytest tests/ -q -m "not slow"` invocations exit 0 (no regressions in any module)
    - The test body in each file checks for the FULL set `{evaluated_users, evaluated_users_sparse, evaluated_users_medium, evaluated_users_dense}` (not a subset)
  </acceptance_criteria>
  <done>
    - All four test_server_integration.py files contain (or had already contained) the cross-cutting D-09 per-round exposure regression guard
    - All test invocations green
    - Documentation of any test strengthening from upstream Plan 03-06 weaker variants captured in this plan's SUMMARY
    - **MAJOR closure (per-module test verification, plan-checker iteration 1):** Acceptance criteria + verify both run pytest against each module separately (4 commands, not one chained union), so a failure points to the SPECIFIC failing module rather than reporting "one of four failed"
  </done>
</task>

</tasks>

<verification>
- sweep.yaml: `grep -c "name: best/sampled_ndcg@10" federated-adaptive-personalized-cf/sweep.yaml` returns 1; `grep -c "name: final/sampled_ndcg@10" federated-adaptive-personalized-cf/sweep.yaml` returns 0
- New test file: `test -f federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` exits 0
- W&B summary key migration parametrized test green for all 4 modules
- Per-round exposure history regression guard present in all 4 test_server_integration.py files
- Cross-module unit-test sweep across all four modules exits 0 (no regressions)
- D-18 surgical scope: this plan touches sweep.yaml + 5 test files; NO server_app.py / strategy.py / dataset.py / task.py / client_app.py / models/ files (Plans 03-06 own those)
- Wave-3 file-disjointness held: this plan modifies a SINGLE config file (sweep.yaml) + 4 test files (one per module). It does NOT race with Plan 06 because Plan 06 modifies federated-pfedrec/federated_pfedrec/server_app.py + federated-pfedrec/tests/test_server_integration.py + scripts/foundation/tests/test_pfedrec_subprocess_determinism.py — and this plan ALSO modifies federated-pfedrec/tests/test_server_integration.py. **POTENTIAL WAVE-3 WRITE RACE on test_server_integration.py files**. Mitigation: Plan 06 (and 03/04/05) add `test_round_metrics_history_carries_per_group_exposure` AS PART OF their integration-test extensions. This Plan 07 Task 2 only adds the test if upstream plans omitted it (Step 1 inspection check), and only strengthens it if a weaker variant exists. The execute-phase orchestrator runs plans sequentially within a wave when file conflicts exist; if write-race surfaces empirically, escalate to merging Plan 07 into a post-Wave-3 cleanup plan.
</verification>

<success_criteria>
- federated-adaptive-personalized-cf/sweep.yaml line 18 mutated: `name: final/sampled_ndcg@10` -> `name: best/sampled_ndcg@10` (Pitfall 7 closure)
- federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py NEW: 2 test functions × 5 test items pin sweep.yaml metric.name + all 4 server_app.py W&B summary namespaces
- All 4 test_server_integration.py files contain `test_round_metrics_history_carries_per_group_exposure` asserting D-09 evaluated_users_{,sparse,medium,dense} keys per round
- Cross-module unit-test sweep green
- No server_app.py / strategy.py / data-pipeline files modified (Plans 03-06 own those — Wave-3 file-disjointness held under the documented mitigation)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-07-SUMMARY.md` covering:
- sweep.yaml metric.name migration (Pitfall 7 closure)
- NEW test_wandb_summary_keys.py with 2 tests + 4 parametrized server_app namespace asserts
- D-09 per-round exposure history regression guards in all 4 test_server_integration.py files
- Notes on whether the cross-module test was already added by Plan 03-06 vs added by this plan; documentation of any strengthening of weaker upstream variants
- Final cross-cutting verification: pytest sweep across all 4 modules + foundation tests green (excluding @pytest.mark.slow gates which run separately)
</output>
</content>
</invoke>