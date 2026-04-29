"""Tests for fedrec_foundation.mode (D-06..D-11 + CR-2 + Pitfalls 6, 8)."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import pytest

from fedrec_foundation.mode import (
    MODE_NAMES, ModeProfile,
    resolve_mode_defaults, log_mode_and_overrides,
    assert_benchmark_one_user_per_client,
)


def test_all_four_modes_registered() -> None:
    """Phase 7 D-04: thesis_crossdevice_main joins the registry alongside the existing 3 modes."""
    assert set(MODE_NAMES) == {
        "benchmark_cross_device",
        "thesis_crossdevice_main",
        "paper_compat_pfedrec",
        "cross_silo_legacy",
    }


def test_benchmark_profile() -> None:
    p = resolve_mode_defaults("benchmark_cross_device")
    assert p.num_supernodes == 6040
    assert p.partition_mode == "natural"
    assert p.weight_policy == "num_positives"
    assert p.primary_evaluator == "sampled_loo_99"
    assert p.assert_one_user_per_client is True


def test_cross_silo_legacy_profile() -> None:
    p = resolve_mode_defaults("cross_silo_legacy")
    assert p.num_supernodes == 5
    assert p.partition_mode == "dirichlet"
    assert p.assert_one_user_per_client is False


def test_paper_compat_profile() -> None:
    p = resolve_mode_defaults("paper_compat_pfedrec")
    assert p.num_supernodes == 6040
    assert p.embedding_dim == 32
    assert p.optimizer == "sgd"
    assert p.lr == 0.1


def test_thesis_crossdevice_main_profile() -> None:
    """Phase 7 D-04: thesis_crossdevice_main clones benchmark_cross_device byte-for-byte except mode name."""
    p = resolve_mode_defaults("thesis_crossdevice_main")
    # Mode name is the provenance tag — the ONLY difference from benchmark_cross_device.
    assert p.mode == "thesis_crossdevice_main"
    # Every other field matches benchmark_cross_device verbatim (D-04).
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


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        resolve_mode_defaults("made_up_mode")


def test_module_override() -> None:
    p = resolve_mode_defaults(
        "paper_compat_pfedrec", module_overrides={"weight_policy": "uniform"}
    )
    assert p.weight_policy == "uniform"
    # Other fields unchanged.
    assert p.embedding_dim == 32


def test_override_logging() -> None:
    """Pitfall 6: kebab keys (from run_config) convert to snake before comparison."""
    p = resolve_mode_defaults("benchmark_cross_device")
    run_config = {
        "weight-policy": "uniform",          # kebab override of snake field
        "num-server-rounds": 50,             # different value -> override
        "embedding-dim": 64,                 # same value as default -> NOT an override
    }
    buf = StringIO()
    with redirect_stdout(buf):
        overrides = log_mode_and_overrides("benchmark_cross_device", p, run_config)
    assert "weight_policy" in overrides
    assert overrides["weight_policy"] == "uniform"
    assert "num_server_rounds" in overrides
    assert overrides["num_server_rounds"] == 50
    assert "embedding_dim" not in overrides  # same as default
    # Stdout carries the loud warning prefix.
    out = buf.getvalue()
    assert "[MODE OVERRIDE]" in out


def test_assertion_flags_benchmark() -> None:
    p = resolve_mode_defaults("benchmark_cross_device")
    # One user: passes.
    assert_benchmark_one_user_per_client(p, num_users_in_client=1, overrides={})
    # More than one: raises.
    with pytest.raises(AssertionError, match="exactly one user"):
        assert_benchmark_one_user_per_client(p, num_users_in_client=5, overrides={})


def test_assertion_flags_cross_silo_legacy_skipped() -> None:
    """Pitfall 8: cross_silo_legacy must NOT trigger the one-user assertion."""
    p = resolve_mode_defaults("cross_silo_legacy")
    # 1200 users in client -> no error because assert_one_user_per_client=False.
    assert_benchmark_one_user_per_client(p, num_users_in_client=1200, overrides={})


def test_assertion_skipped_on_override() -> None:
    """D-10: explicit override bypasses the lock (and emits a visible skip log)."""
    p = resolve_mode_defaults("benchmark_cross_device")
    # Simulate the override dict returned by log_mode_and_overrides.
    overrides = {"num_supernodes": 10}
    assert_benchmark_one_user_per_client(p, num_users_in_client=604, overrides=overrides)


def test_paper_compat_pfedrec_weight_policy_uniform() -> None:
    """D-25: _PAPER_COMPAT_PFEDREC.weight_policy is 'uniform' (was 'num_positives' pre-PFR-02).

    Phase 1 deferred this decision. Phase 5 closes it: reference engine.py:81
    divides by len(round_user_params) — uniform weight per participating client.
    """
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
