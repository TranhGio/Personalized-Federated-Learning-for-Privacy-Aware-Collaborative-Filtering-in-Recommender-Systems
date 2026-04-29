"""Run manifest / protocol fingerprint (FND-07 + IMP-2).

Written twice per run (D-15, belt-and-suspenders):
  1. Embedded under the ``_manifest`` key inside the main result JSON.
  2. Sibling file ``<run_id>-manifest.json`` next to the result file.

Carries all four foundation fingerprints (IMP-2):
  - ``mapping_sha256``        — canonical user/item ID mapping hash
  - ``split_hash``            — LOO split manifest hash
  - ``exclusion_sha256``      — per-user exclusion-set CSR hash
  - ``foundation_contract_sha256`` — composite of the three above +
    raw_data_hash + builder_version

Any attempt to reproduce a run starts by comparing these four hashes against
the bundle index; a single-byte mismatch invalidates the reproduction claim.
"""
from __future__ import annotations

import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fedrec_foundation.atomic import atomic_write_json

# Bump when the manifest schema gains/loses a field or changes semantics.
RUN_MANIFEST_SCHEMA_VERSION: int = 3  # Phase 7 D-22: adds thesis_run_label + ablation_dimension + ablation_value


@dataclass
class RunManifest:
    """Run fingerprint.

    Field groupings
    ---------------
    - Bookkeeping: ``schema_version``, ``run_id``.
    - Mode + locked config (13 fields from the ModeProfile, per D-16):
      ``mode``, ``num_supernodes``, ``partition_mode``, ``fraction_train``,
      ``fraction_eval``, ``weight_policy``, ``primary_evaluator``,
      ``num_train_negatives``, ``num_eval_negatives``, ``run_seed``,
      ``checkpoint_rule``.
    - Foundation fingerprints (IMP-2 + D-16): ``mapping_sha256``,
      ``split_hash``, ``exclusion_sha256``, ``foundation_contract_sha256``,
      ``raw_data_hash``, ``builder_version``.
    - Overrides + module metadata: ``overrides``, ``module``.
    - Environment: ``flwr_version``, ``torch_version``, ``git_commit``.

    Downstream writers
    ------------------
    Callers assemble a ``RunManifest`` via :func:`build_run_manifest`, then
    call both :func:`embed_manifest_in_result` and
    :func:`write_manifest_sibling` to satisfy D-15.
    """

    schema_version: int
    run_id: str
    # Mode + locked config (from ModeProfile).
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
    module: str  # one of: "baseline" | "personalized" | "adaptive" | "pfedrec"
    # Environment.
    flwr_version: str
    torch_version: str
    git_commit: str
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
    # Phase 7 additions (D-22) — all with safe defaults so v1/v2 fixtures
    # construct without TypeError (Pitfall 7 backward-compat invariant).
    # ``build_run_manifest`` is NOT touched: server_app populates these via
    # ``dataclasses.replace`` post-build mutation (mirrors Phase 6 D-07).
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


def generate_run_id() -> str:
    """Return a run id of the form ``YYYYMMDD-HHMMSS-<6hex>`` (UTC).

    Example
    -------
    ``"20260419-142301-a1b2c3"``

    The ``datetime`` portion is sortable; the short uuid tail disambiguates
    simultaneous launches.

    Returns
    -------
    str
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _git_commit() -> str:
    """Best-effort ``git rev-parse HEAD``; returns ``"unknown"`` on failure.

    Never raises — a missing ``.git`` directory (CI, container, bare export)
    should not block a run from recording a manifest.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    run_id: str,
    mode_profile: Any,  # duck-typed ModeProfile (avoid circular import with mode.py)
    run_seed: int,
    mapping_sha256: str,
    split_hash: str,
    exclusion_sha256: str,
    foundation_contract_sha256: str,
    raw_data_hash: str,
    builder_version: str,
    overrides: Dict[str, Any],
    module: str,
) -> RunManifest:
    """Assemble a :class:`RunManifest` from a ModeProfile + foundation hashes.

    Reads ``flwr.__version__``, ``torch.__version__``, and ``git rev-parse HEAD``
    internally; callers do not need to pass environment info.

    Parameters
    ----------
    run_id : str
        Typically from :func:`generate_run_id`.
    mode_profile : Any
        Any object exposing attributes ``mode``, ``num_supernodes``,
        ``partition_mode``, ``fraction_train``, ``fraction_eval``,
        ``weight_policy``, ``primary_evaluator``, ``num_train_negatives``,
        ``num_eval_negatives``, ``checkpoint_rule``. The real
        ``ModeProfile`` dataclass is defined in Plan 05's
        ``fedrec_foundation.mode`` module; duck-typing avoids a circular
        import.
    run_seed : int
    mapping_sha256 : str
    split_hash : str
    exclusion_sha256 : str
    foundation_contract_sha256 : str
    raw_data_hash : str
    builder_version : str
    overrides : Dict[str, Any]
        CLI / config overrides that diverge from ``mode_profile``'s defaults.
        Stored verbatim for audit — copied to avoid aliasing.
    module : str
        One of ``"baseline"``, ``"personalized"``, ``"adaptive"``, ``"pfedrec"``.

    Returns
    -------
    RunManifest
    """
    # Local imports: these are deferred so that importing the manifest module
    # does not pull the full torch/flwr stack at package-init time.
    import flwr
    import torch

    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        mode=mode_profile.mode,
        num_supernodes=mode_profile.num_supernodes,
        partition_mode=mode_profile.partition_mode,
        fraction_train=mode_profile.fraction_train,
        fraction_eval=mode_profile.fraction_eval,
        weight_policy=mode_profile.weight_policy,
        primary_evaluator=mode_profile.primary_evaluator,
        num_train_negatives=mode_profile.num_train_negatives,
        num_eval_negatives=mode_profile.num_eval_negatives,
        run_seed=run_seed,
        checkpoint_rule=mode_profile.checkpoint_rule,
        mapping_sha256=mapping_sha256,
        split_hash=split_hash,
        exclusion_sha256=exclusion_sha256,
        foundation_contract_sha256=foundation_contract_sha256,
        raw_data_hash=raw_data_hash,
        builder_version=builder_version,
        overrides=dict(overrides),
        module=module,
        flwr_version=getattr(flwr, "__version__", "unknown"),
        torch_version=torch.__version__,
        git_commit=_git_commit(),
    )


def write_manifest_sibling(
    manifest: RunManifest,
    result_json_path: Path,
    sibling_name: Optional[str] = None,
) -> Path:
    """D-15 sibling file: write the manifest as a sibling JSON next to the result.

    The sibling is written via :func:`atomic_write_json` — partial writes on
    crash are impossible and no ``.tmp-*`` leftovers are left behind.

    Parameters
    ----------
    manifest : RunManifest
    result_json_path : Path
        The main result JSON. Its parent directory is used as the sibling's
        parent.
    sibling_name : Optional[str]
        Override the sibling filename. Defaults to ``None``, which preserves
        the legacy ``<run_id>-manifest.json`` naming used by cross-silo callers.
        Phase-6 callers (D-04 clean per-run-dir filenames) pass
        ``"manifest.json"`` to land the sibling at
        ``<parent>/manifest.json`` instead.

    Returns
    -------
    Path
        Absolute path of the newly written sibling file.
    """
    sibling_filename = (
        sibling_name if sibling_name is not None else f"{manifest.run_id}-manifest.json"
    )
    sibling = Path(result_json_path).parent / sibling_filename
    atomic_write_json(str(sibling), asdict(manifest))
    return sibling


def embed_manifest_in_result(
    manifest: RunManifest,
    result_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """D-15 embedded: inject ``_manifest`` key into an existing result dict.

    Mutates ``result_dict`` in place AND returns it, enabling fluent
    use: ``json.dump(embed_manifest_in_result(m, result), f)``.

    Parameters
    ----------
    manifest : RunManifest
    result_dict : Dict[str, Any]
        The main result payload. Top-level key ``_manifest`` is overwritten.

    Returns
    -------
    Dict[str, Any]
        The same dict (not a copy).
    """
    result_dict["_manifest"] = asdict(manifest)
    return result_dict
