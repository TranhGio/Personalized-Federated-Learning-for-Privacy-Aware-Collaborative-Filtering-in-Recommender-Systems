"""Atomic bundle publication for foundation artifacts (N-3 + IMP-2).

Publishes the four-file ``data/derived/`` bundle in a fixed order:

1. ``mapping.json`` (atomic via ``save_mapping``)
2. ``split_manifest.json`` (atomic via ``save_split_or_verify``; D-04 lock)
3. ``exclusion_items.npz`` (atomic via ``save_exclusion``)
4. ``foundation_index.json`` (atomic via ``atomic_write_json``) -- written LAST

``foundation_index.json`` carries three SHA-256 fingerprints
(``mapping_sha256``, ``split_hash``, ``exclusion_sha256``) plus a
composite ``foundation_contract_sha256`` that changes if ANY of the
three underlying artifacts changes (IMP-2). Loaders call
``verify_bundle(derived_dir)`` before reading any payload; if the
index is missing or any fingerprint mismatches, they refuse to load.

``publish_bundle`` takes 4 parameters: ``derived_dir``, ``mapping``,
``split_manifest``, ``exclusion``. The ``raw_data_hash`` is NOT a
separate parameter -- it is read from ``split_manifest.raw_data_hash``
(a dataclass field populated by ``build_split``). Single source of
truth; no side-channel, no post-hoc assignment.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.exclusion import save_exclusion
from fedrec_foundation.hashing import sha256_file
from fedrec_foundation.mapping import CanonicalMapping, save_mapping
from fedrec_foundation.split import SplitManifest, save_split_or_verify

BUNDLE_SCHEMA_VERSION: int = 1


@dataclass
class FoundationIndex:
    """Sentinel published LAST by ``publish_bundle``.

    All four fingerprints are required fields. ``verify_bundle`` uses
    them to reject an incomplete or tampered bundle.
    """

    schema_version: int
    builder_version: str
    created_at: str
    mapping_sha256: str
    split_hash: str
    exclusion_sha256: str
    foundation_contract_sha256: str


def compute_foundation_contract_sha256(
    mapping_sha256: str,
    split_hash: str,
    exclusion_sha256: str,
) -> str:
    """Composite SHA-256 that changes if ANY of the three inputs changes.

    Parameters
    ----------
    mapping_sha256 : str
        SHA-256 of ``mapping.json``.
    split_hash : str
        ``split_manifest.split_hash`` (already composite over
        mapping_sha256 + raw_data_hash + train/test keys per IMP-2).
    exclusion_sha256 : str
        SHA-256 of ``exclusion_items.npz``.

    Returns
    -------
    str
        Lowercase 64-character hex digest.
    """
    h = hashlib.sha256()
    h.update(b"mapping:" + mapping_sha256.encode("ascii") + b";")
    h.update(b"split:" + split_hash.encode("ascii") + b";")
    h.update(b"exclusion:" + exclusion_sha256.encode("ascii"))
    return h.hexdigest()


def publish_bundle(
    derived_dir,
    mapping: CanonicalMapping,
    split_manifest: SplitManifest,
    exclusion: Dict[int, np.ndarray],
) -> FoundationIndex:
    """Atomically publish the 4-file bundle. Index file is written LAST.

    Reads ``raw_data_hash`` from ``split_manifest.raw_data_hash`` -- no
    extra parameter needed (the manifest owns that fingerprint after
    ``build_split`` populates it). The 4-param signature is the LOCKED
    contract (IMP-2).

    Parameters
    ----------
    derived_dir : pathlib.Path or str
        Output directory (created if missing).
    mapping : CanonicalMapping
        Canonical ID mapping.
    split_manifest : SplitManifest
        LOO split manifest carrying ``raw_data_hash`` + ``mapping_sha256``.
    exclusion : Dict[int, numpy.ndarray]
        Per-user exclusion arrays.

    Returns
    -------
    FoundationIndex
        Published index sentinel (also written to
        ``foundation_index.json``).
    """
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = derived_dir / "mapping.json"
    split_path = derived_dir / "split_manifest.json"
    excl_path = derived_dir / "exclusion_items.npz"
    index_path = derived_dir / "foundation_index.json"

    # Sanity-touch the raw_data_hash field (single source of truth).
    _ = split_manifest.raw_data_hash

    # Step 1-3: payload files (each atomic individually).
    save_mapping(mapping, str(mapping_path))
    save_split_or_verify(split_manifest, split_path)
    save_exclusion(exclusion, excl_path)

    # Step 4: compute fingerprints and write the index LAST.
    mapping_sha = sha256_file(mapping_path)
    excl_sha = sha256_file(excl_path)
    contract = compute_foundation_contract_sha256(
        mapping_sha, split_manifest.split_hash, excl_sha
    )
    idx = FoundationIndex(
        schema_version=BUNDLE_SCHEMA_VERSION,
        builder_version=split_manifest.builder_version,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mapping_sha256=mapping_sha,
        split_hash=split_manifest.split_hash,
        exclusion_sha256=excl_sha,
        foundation_contract_sha256=contract,
    )
    atomic_write_json(str(index_path), asdict(idx))
    return idx


def verify_bundle(derived_dir) -> FoundationIndex:
    """Load ``foundation_index.json`` and verify every fingerprint matches.

    Called by loaders before reading any payload file. If the index is
    missing, if any of the three payload files is missing, or if any
    computed fingerprint differs from the index's declared value, a
    ``RuntimeError`` is raised (error message contains ``"incomplete"``
    for the missing-file case).

    Parameters
    ----------
    derived_dir : pathlib.Path or str
        Directory containing the bundle.

    Returns
    -------
    FoundationIndex
        The loaded, verified index.

    Raises
    ------
    RuntimeError
        If the bundle is incomplete, missing, or any fingerprint mismatches.
    """
    derived_dir = Path(derived_dir)
    index_path = derived_dir / "foundation_index.json"
    if not index_path.exists():
        raise RuntimeError(
            f"Bundle incomplete: {index_path} missing. "
            f"Run `python -m fedrec_foundation.build`."
        )
    with open(index_path) as f:
        data = json.load(f)
    idx = FoundationIndex(**data)

    for name in ("mapping.json", "split_manifest.json", "exclusion_items.npz"):
        if not (derived_dir / name).exists():
            raise RuntimeError(
                f"Bundle incomplete: {name} missing but index present."
            )

    mapping_sha = sha256_file(derived_dir / "mapping.json")
    excl_sha = sha256_file(derived_dir / "exclusion_items.npz")
    contract = compute_foundation_contract_sha256(
        mapping_sha, idx.split_hash, excl_sha
    )
    if mapping_sha != idx.mapping_sha256:
        raise RuntimeError(
            f"Bundle corrupted: mapping.json sha mismatch "
            f"(index={idx.mapping_sha256} actual={mapping_sha})"
        )
    if excl_sha != idx.exclusion_sha256:
        raise RuntimeError(
            f"Bundle corrupted: exclusion_items.npz sha mismatch "
            f"(index={idx.exclusion_sha256} actual={excl_sha})"
        )
    if contract != idx.foundation_contract_sha256:
        raise RuntimeError(
            "Bundle corrupted: foundation_contract_sha256 mismatch "
            "(index may have been hand-edited)"
        )
    return idx
