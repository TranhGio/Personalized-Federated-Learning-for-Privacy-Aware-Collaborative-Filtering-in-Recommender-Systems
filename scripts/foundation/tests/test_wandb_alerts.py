"""Tests for fedrec_foundation.wandb_alerts (end-of-run W&B alert)."""
from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest

from fedrec_foundation.wandb_alerts import build_run_finished_alert, send_run_finished_alert


def _final_metrics(**best_overrides: Any) -> Dict[str, Any]:
    best: Dict[str, Any] = {
        "sampled_ndcg@10": 0.3447117925,
        "sampled_ndcg@10/sparse": 0.4269672556,
        "sampled_ndcg@10/medium": 0.3801613590,
        "sampled_ndcg@10/dense": 0.2935400623,
        "sampled_hr@10": 0.6110927152,
        "evaluated_users": 6040.0,
        "d06_cache_misses": 0,
    }
    best.update(best_overrides)
    return {"best": best, "best_round": 83, "last_round": 100}


def test_healthy_run_is_info_with_headline_numbers() -> None:
    title, text, level = build_run_finished_alert(
        "pfedrec", "20260726-083121-cf72e0", _final_metrics(),
        expected_users=6040, thesis_run_label="main", results_path="/x/results.json",
    )
    assert level == "INFO"
    assert title.startswith("[OK] pfedrec 20260726-083121-cf72e0")
    assert "0.3447" in title
    assert "sparse 0.4270" in text
    assert "d06_cache_misses: 0" in text
    assert "label: main" in text
    assert "ISSUES" not in text


def test_cache_misses_trigger_warn() -> None:
    title, text, level = build_run_finished_alert(
        "adaptive", "r", _final_metrics(d06_cache_misses=5400), expected_users=6040,
    )
    assert level == "WARN"
    assert title.startswith("[CHECK]")
    assert "COLD-EVAL SUSPECT" in text


def test_population_mismatch_triggers_warn() -> None:
    _title, text, level = build_run_finished_alert(
        "personalized", "r", _final_metrics(evaluated_users=604.0), expected_users=6040,
    )
    assert level == "WARN"
    assert "evaluated_users=604 != expected 6040" in text


def test_unstamped_and_missing_metrics_do_not_crash() -> None:
    fm = {"best": {"sampled_ndcg@10": 0.2}, "best_round": 25, "last_round": 100}
    _title, text, level = build_run_finished_alert("baseline", "r", fm, expected_users=6040)
    assert level == "INFO"
    assert "not stamped" in text
    assert "sparse n/a" in text


def test_send_is_noop_without_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    fake = types.SimpleNamespace(
        run=None,
        alert=lambda **kw: calls.append(kw),
        AlertLevel=types.SimpleNamespace(INFO="INFO", WARN="WARN"),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    send_run_finished_alert("pfedrec", "r", _final_metrics(), expected_users=6040)
    assert calls == []


def test_send_calls_wandb_alert_with_level(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []
    fake = types.SimpleNamespace(
        run=object(),
        alert=lambda **kw: calls.append(kw),
        AlertLevel=types.SimpleNamespace(INFO="INFO", WARN="WARN"),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    send_run_finished_alert("pfedrec", "r", _final_metrics(d06_cache_misses=3), expected_users=6040)
    assert len(calls) == 1
    assert calls[0]["level"] == "WARN"


def test_send_swallows_wandb_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kw: Any) -> None:
        raise RuntimeError("network down")

    fake = types.SimpleNamespace(
        run=object(), alert=_boom,
        AlertLevel=types.SimpleNamespace(INFO="INFO", WARN="WARN"),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    send_run_finished_alert("pfedrec", "r", _final_metrics(), expected_users=6040)  # must not raise
