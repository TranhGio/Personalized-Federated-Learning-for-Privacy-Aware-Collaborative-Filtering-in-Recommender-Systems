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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fedrec_foundation.atomic import atomic_write_json

# Bump when the manifest schema gains/loses a field or changes semantics.
RUN_MANIFEST_SCHEMA_VERSION: int = 1


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
) -> Path:
    """D-15 sibling file: write ``<run_id>-manifest.json`` next to the result.

    The sibling is written via :func:`atomic_write_json` — partial writes on
    crash are impossible and no ``.tmp-*`` leftovers are left behind.

    Parameters
    ----------
    manifest : RunManifest
    result_json_path : Path
        The main result JSON. Its parent directory is used as the sibling's
        parent — ``<parent>/<run_id>-manifest.json``.

    Returns
    -------
    Path
        Absolute path of the newly written sibling file.
    """
    sibling = Path(result_json_path).parent / f"{manifest.run_id}-manifest.json"
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
