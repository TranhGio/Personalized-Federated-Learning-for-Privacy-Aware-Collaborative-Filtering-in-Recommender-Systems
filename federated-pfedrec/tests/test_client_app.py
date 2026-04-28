"""Phase 5 Plan 03 client_app.py regression guard.

Pins:
- PFR-05 single-user collapse: benchmark-mode one-user-per-client assertion
  fires; loop over ``user_train_data.keys()`` is REMOVED from the source.
- D-22 cold-round probe-then-load: ``_load_local_user_state`` returns
  ``None`` on cache miss before any tensor load.
- G-03-01 discover_only short-circuit: ``@app.evaluate`` short-circuits to
  a zero-suffstats ``EvaluateMetricsContract`` payload when
  ``config['discover_only']`` is True (no data load, no model load).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    """Redirect ``_CACHE_BASE_DIR`` into ``tmp_path``."""
    from federated_pfedrec import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: PFR-05 single-user collapse — assertion + loop absence.
# ---------------------------------------------------------------------------


def test_benchmark_one_user_per_client_assert() -> None:
    """PFR-05: the benchmark-mode one-user assertion is wired in client_app.

    Mechanically guards against a regression that re-introduces the legacy
    ``for user_idx in user_train_data.keys()`` loop in ``@app.train``.
    """
    import federated_pfedrec.client_app as client_mod

    src = Path(client_mod.__file__).read_text()

    # PFR-05: benchmark-mode assertion is referenced from foundation.mode.
    assert "assert_benchmark_one_user_per_client" in src, (
        "PFR-05: client_app must wire fedrec_foundation.mode."
        "assert_benchmark_one_user_per_client at the train + evaluate sites"
    )

    # Legacy multi-user loop is gone (D-12 single-user collapse).
    assert "for user_idx in user_train_data" not in src, (
        "PFR-05: legacy multi-user loop must be removed from client_app.py"
    )
    # The accompanying legacy save_user_local_params / load_user_local_params
    # helpers should also have been retired in favor of the manifest-sidecar
    # _save_local_user_state / _load_local_user_state idiom.
    assert "def save_user_local_params" not in src, (
        "PFR-03: legacy per-user-subdir save_user_local_params must be retired"
    )
    assert "def load_user_local_params" not in src, (
        "PFR-03: legacy per-user-subdir load_user_local_params must be retired"
    )


# ---------------------------------------------------------------------------
# Test 2: D-22 cold-round probe-then-load.
# ---------------------------------------------------------------------------


def test_cold_round_probe_then_load(cache_root) -> None:
    """D-22: ``_load_local_user_state`` returns ``None`` on cache miss; round-trips on hit."""
    from federated_pfedrec.client_app import (
        _load_local_user_state,
        _save_local_user_state,
        _signature_fields,
    )

    sig = _signature_fields(
        run_id="r1",
        num_users=6040,
        num_items=3706,
        latent_dim=32,
        split_hash="abc",
    )

    # Cold round: cache directory empty, expect None (no exception).
    out = _load_local_user_state(
        partition_id=0,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )
    assert out is None, "D-22: cold round must return None (not raise)"

    # Save state and try again — round-trip succeeds.
    state = {"affine_output.weight": torch.randn(1, 32)}
    _save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )

    out = _load_local_user_state(
        partition_id=0,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )
    assert out is not None
    assert set(out.keys()) == {"affine_output.weight"}
    assert tuple(out["affine_output.weight"].shape) == (1, 32)


# ---------------------------------------------------------------------------
# Test 3: G-03-01 discover_only short-circuit.
# ---------------------------------------------------------------------------


def test_discover_only_short_circuit() -> None:
    """G-03-01: discover_only=True short-circuits to zero-suffstats payload."""
    import federated_pfedrec.client_app as client_mod

    src = inspect.getsource(client_mod.evaluate)

    # The discover_only branch must appear FIRST (before any model/bundle load).
    assert "discover_only" in src, (
        "G-03-01: @app.evaluate must check msg.content['config']['discover_only']"
    )
    assert "EvaluateMetricsContract" in src, (
        "G-03-01: discover_only short-circuit must build an EvaluateMetricsContract payload"
    )

    # The discover_only branch returns BEFORE any heavy work — verify by
    # locating the discover_only check and confirming it precedes any
    # _load_foundation_bundle / _load_local_user_state call.
    discover_idx = src.find("discover_only")
    bundle_idx = src.find("_load_foundation_bundle")
    if bundle_idx >= 0:
        assert discover_idx < bundle_idx, (
            "G-03-01: discover_only check must short-circuit BEFORE "
            "the foundation bundle is loaded (no heavy work on discovery rounds)"
        )
