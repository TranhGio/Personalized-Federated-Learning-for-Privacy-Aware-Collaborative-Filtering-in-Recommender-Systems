"""Mode resolver (D-06..D-11) and CR-2 in-app assertion helpers.

D-06/D-07: A **mode** is the single source of truth for "what experiment is
this?". One mode picks ``num_supernodes``, partition mode, weight policy,
primary evaluator, training hyperparameters, and a policy flag
(``assert_one_user_per_client``).

D-08: Per-module overrides are allowed where paper-compat setting legitimately
differs. Use ``resolve_mode_defaults(mode, module_overrides={...})``.

D-10: Run-config CLI overrides are visible (each override emits a loud
``[MODE OVERRIDE]`` log) and captured in the run manifest. The override dict
returned by ``log_mode_and_overrides`` is exactly what goes into
``manifest.overrides``.

D-11 + CR-2: ``num-supernodes`` cannot be set from inside a Flower app — it
is federation-level. ``assert_benchmark_one_user_per_client`` is the in-app
assertion layer that verifies the launcher (``scripts/run.py``) agreed with
the client-side ``mode`` declaration.

Pitfall 6: run_config keys are kebab-case (``weight-policy``); dataclass
fields are snake_case (``weight_policy``). Conversion happens via
``key.replace('-', '_')`` before comparison.

Pitfall 8: the single-user assertion must NOT trigger in
``cross_silo_legacy``. Each profile carries an explicit
``assert_one_user_per_client`` flag; only the benchmark / paper-compat modes
set it to True.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional


# ============================================================================
# Canonical mode profiles (D-06..D-11)
# ============================================================================


@dataclass(frozen=True)
class ModeProfile:
    """Complete experiment profile for one mode. D-07: a mode IS an experiment.

    Attributes
    ----------
    mode : str
        Mode name (one of :data:`MODE_NAMES`).
    num_supernodes : int
        Number of Flower supernodes at the federation level. 6040 = cross-
        device (1 user = 1 client), 5 = legacy cross-silo. CANNOT be set from
        inside a Flower app (Codex CR-2); must be set by ``scripts/run.py``.
    partition_mode : str
        ``"natural"`` for cross-device (1 user = 1 partition), ``"dirichlet"``
        for cross-silo.
    weight_policy : str
        Aggregation weight policy (see ``fedrec_foundation.weight_policy``).
        One of ``"uniform"``, ``"num_positives"``, ``"num_training_examples"``.
    primary_evaluator : str
        Primary evaluator string (see ``fedrec_foundation.evaluator``).
        Always ``"sampled_loo_99"`` for this phase (D-12).
    fraction_train : float
        Flower strategy ``fraction-train`` (C — per-round client sampling).
    fraction_eval : float
        Flower strategy ``fraction-eval``.
    num_train_negatives : int
        Negative samples per positive during training.
    num_eval_negatives : int
        Negative samples per positive during evaluation (99 = NCF protocol).
    embedding_dim : int
        Latent factor dimensionality.
    optimizer : str
        ``"adam"`` or ``"sgd"``.
    lr : float
        Learning rate for the client-side optimizer.
    local_epochs : int
        Local training steps per round (``K`` in the paper notation).
    num_server_rounds : int
        Total federated rounds (``R`` in the paper notation).
    checkpoint_rule : str
        ``"best_round"`` (restore best metric) or ``"last_round"`` (report last
        round's metric).
    assert_one_user_per_client : bool
        D-11: whether the in-app assertion should fire when the local client
        partition contains more than one distinct user. True for
        benchmark/paper-compat; False for cross_silo_legacy.
    """

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


# ============================================================================
# Registered profiles
# ----------------------------------------------------------------------------
# Values for `weight_policy` and `primary_evaluator` are STRING LITERALS that
# match the `.value` of `fedrec_foundation.weight_policy.WeightPolicy` and
# `fedrec_foundation.evaluator.EvalProtocol` respectively. Using literals here
# avoids an import-time coupling to a sibling module; downstream callers can
# still compare against `WeightPolicy.NUM_POSITIVES.value` if they prefer.
# ============================================================================


_BENCHMARK_CROSS_DEVICE = ModeProfile(
    mode="benchmark_cross_device",
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


# Phase 7 D-04: thesis_crossdevice_main is a byte-for-byte clone of
# _BENCHMARK_CROSS_DEVICE with only the ``mode`` string differing. The mode
# name itself IS the provenance tag — it is the discriminant the manifest
# carries forward and the aggregator filters on (THS-01).
#
# Warning 4 closure: ``checkpoint_rule="best_round"`` is inherited verbatim;
# Phase 2 Plan 04 established that all 4 server_app.py files accept BOTH
# ``"best_round"`` (ModeProfile) and ``"best_round_restore"`` (pyproject)
# spellings in the same checkpoint-rule branch — see STATE.md Phase 2:
# "checkpoint_rule branch accepts both 'best_round_restore' (pyproject) and
# 'best_round' (ModeProfile) spellings to avoid bikeshed."
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

_CROSS_SILO_LEGACY = ModeProfile(
    mode="cross_silo_legacy",
    num_supernodes=5,
    partition_mode="dirichlet",
    weight_policy="num_training_examples",
    primary_evaluator="sampled_loo_99",
    fraction_train=1.0,
    fraction_eval=1.0,
    num_train_negatives=1,
    num_eval_negatives=99,
    embedding_dim=128,
    optimizer="adam",
    lr=0.001,
    local_epochs=5,
    num_server_rounds=10,
    checkpoint_rule="last_round",
    assert_one_user_per_client=False,  # Pitfall 8: legacy DISABLES the lock
)


_REGISTRY: Dict[str, ModeProfile] = {
    "benchmark_cross_device": _BENCHMARK_CROSS_DEVICE,
    "thesis_crossdevice_main": _THESIS_CROSSDEVICE_MAIN,  # Phase 7 D-04
    "paper_compat_pfedrec": _PAPER_COMPAT_PFEDREC,
    "cross_silo_legacy": _CROSS_SILO_LEGACY,
}


MODE_NAMES = tuple(_REGISTRY.keys())


# ============================================================================
# Public API
# ============================================================================


def resolve_mode_defaults(
    mode: str,
    module_overrides: Optional[Dict[str, object]] = None,
) -> ModeProfile:
    """Return the ``ModeProfile`` for a mode name, with optional per-module overrides.

    Parameters
    ----------
    mode : str
        Mode identifier (see :data:`MODE_NAMES`).
    module_overrides : Optional[Dict[str, object]]
        D-08: per-module overrides allowed where paper-compat setting differs.
        Keys must be snake_case ``ModeProfile`` field names. Example:
        ``{"weight_policy": "uniform"}`` for PFedRec's paper mode.

    Returns
    -------
    ModeProfile
        The resolved profile (a fresh frozen instance if overrides applied).

    Raises
    ------
    ValueError
        If ``mode`` is not a registered mode name.
    """
    if mode not in _REGISTRY:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected one of {sorted(_REGISTRY)}."
        )
    profile = _REGISTRY[mode]
    if not module_overrides:
        return profile
    # Use dataclass replace so we return a fresh (still frozen) instance.
    return replace(profile, **module_overrides)


def log_mode_and_overrides(
    mode: str,
    profile: ModeProfile,
    run_config: Dict[str, object],
) -> Dict[str, object]:
    """Print loud warnings for any run_config key that overrides a mode field.

    D-10: overrides must be VISIBLE. Each override prints a ``[MODE OVERRIDE]``
    line. The returned dict is exactly what goes into ``manifest.overrides``.

    Pitfall 6: run_config keys are kebab-case (``weight-policy``); dataclass
    fields are snake_case (``weight_policy``). Conversion happens via
    ``key.replace('-', '_')`` before the ``hasattr`` check.

    Parameters
    ----------
    mode : str
        Mode identifier (for the log lines).
    profile : ModeProfile
        Resolved profile to compare against.
    run_config : Dict[str, object]
        Flower ``context.run_config`` dict (kebab-case keys).

    Returns
    -------
    Dict[str, object]
        Subset of ``run_config`` (re-keyed to snake_case) that actually
        overrides a profile field — value differs from the mode default.
    """
    overrides: Dict[str, object] = {}
    for key, val in run_config.items():
        # kebab-case run_config key -> snake_case dataclass field
        snake = key.replace("-", "_")
        if hasattr(profile, snake):
            profile_val = getattr(profile, snake)
            if profile_val != val:
                overrides[snake] = val
                print(
                    f"[MODE OVERRIDE] {key}: mode={mode} default={profile_val!r} "
                    f"user-override={val!r}"
                )
    if overrides:
        print(
            f"[MODE OVERRIDE] {len(overrides)} override(s) active; "
            f"captured in manifest.overrides"
        )
    return overrides


def assert_benchmark_one_user_per_client(
    profile: ModeProfile,
    num_users_in_client: int,
    overrides: Dict[str, object],
) -> None:
    """Phase 2-5 client_app.py assertion entry point (D-11 + CR-2).

    Raises ``AssertionError`` if the profile requires single-user clients AND
    no ``num-supernodes`` override is in play AND ``num_users_in_client != 1``.

    Pitfall 8: cross_silo_legacy has ``assert_one_user_per_client=False``, so
    this helper returns immediately for that profile (no false positive on
    multi-user legacy clients).

    D-10: a visible ``num_supernodes`` override bypasses the lock (with a log
    line) — overrides are legitimate escape hatches, not silent weakenings.

    Parameters
    ----------
    profile : ModeProfile
        Profile returned by :func:`resolve_mode_defaults`.
    num_users_in_client : int
        The count of distinct users in this client's partition, typically
        ``len(client_partition_df["user_idx"].unique())``.
    overrides : Dict[str, object]
        Return value of :func:`log_mode_and_overrides` — if ``num_supernodes``
        was overridden, the assertion is skipped (visible override bypasses
        the lock per D-10).

    Raises
    ------
    AssertionError
        If the profile locks single-user clients, no override bypasses it,
        and the partition has more than one user.
    """
    if not profile.assert_one_user_per_client:
        return
    if "num_supernodes" in overrides or "num-supernodes" in overrides:
        print(
            f"[MODE] single-user-per-client assertion SKIPPED because "
            f"num-supernodes was overridden (value={num_users_in_client})"
        )
        return
    if num_users_in_client != 1:
        raise AssertionError(
            f"Benchmark mode {profile.mode!r} requires exactly one user per "
            f"client; got {num_users_in_client}. Either run via `scripts/run.py "
            f"<module> {profile.mode}` (which sets num-supernodes=6040) or "
            f"override num-supernodes explicitly."
        )


__all__ = [
    "ModeProfile",
    "MODE_NAMES",
    "resolve_mode_defaults",
    "log_mode_and_overrides",
    "assert_benchmark_one_user_per_client",
]
