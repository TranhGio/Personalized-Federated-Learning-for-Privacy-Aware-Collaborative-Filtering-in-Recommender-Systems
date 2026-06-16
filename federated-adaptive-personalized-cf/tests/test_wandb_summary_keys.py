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
