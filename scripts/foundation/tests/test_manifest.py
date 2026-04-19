"""Tests for fedrec_foundation.manifest (FND-07 + IMP-2 + D-15)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fedrec_foundation.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    RunManifest,
    build_run_manifest,
    embed_manifest_in_result,
    generate_run_id,
    write_manifest_sibling,
)


class _StubProfile:
    """Duck-typed ModeProfile stand-in (Plan 05 ships the real dataclass).

    We only need attribute access compatible with build_run_manifest(...).
    """

    def __init__(self) -> None:
        self.mode = "benchmark_cross_device"
        self.num_supernodes = 6040
        self.partition_mode = "natural"
        self.fraction_train = 0.1
        self.fraction_eval = 1.0
        self.weight_policy = "num_positives"
        self.primary_evaluator = "sampled_loo_99"
        self.num_train_negatives = 4
        self.num_eval_negatives = 99
        self.checkpoint_rule = "best_round"


def _build(run_seed: int = 42) -> RunManifest:
    return build_run_manifest(
        run_id=generate_run_id(),
        mode_profile=_StubProfile(),
        run_seed=run_seed,
        mapping_sha256="m" * 64,
        split_hash="s" * 64,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={"num-supernodes": 100},
        module="baseline",
    )


def test_all_fields_populated() -> None:
    """FND-07-a: every manifest field (D-16 + IMP-2 + bookkeeping) is present."""
    m = _build()
    d = m.__dict__
    for key in (
        "schema_version",
        "run_id",
        "mode",
        "num_supernodes",
        "partition_mode",
        "fraction_train",
        "fraction_eval",
        "weight_policy",
        "primary_evaluator",
        "num_train_negatives",
        "num_eval_negatives",
        "run_seed",
        "checkpoint_rule",
        "mapping_sha256",
        "split_hash",
        "exclusion_sha256",
        "foundation_contract_sha256",
        "raw_data_hash",
        "builder_version",
        "overrides",
        "module",
        "flwr_version",
        "torch_version",
        "git_commit",
    ):
        assert key in d, f"missing {key}"
    assert m.schema_version == RUN_MANIFEST_SCHEMA_VERSION
    assert len(m.mapping_sha256) == 64
    assert m.overrides == {"num-supernodes": 100}


def test_both_writes(tmp_path: Path) -> None:
    """FND-07-b + D-15: manifest embedded in result JSON AND written as sibling file."""
    m = _build()
    result_path = tmp_path / f"{m.run_id}-results.json"
    result = {"final_metrics": {"ndcg@10": 0.42}}
    embed_manifest_in_result(m, result)
    with open(result_path, "w") as f:
        json.dump(result, f)
    sibling = write_manifest_sibling(m, result_path)

    # Embedded key exists in the result JSON.
    loaded = json.loads(result_path.read_text())
    assert "_manifest" in loaded
    assert loaded["_manifest"]["run_id"] == m.run_id

    # Sibling file exists and matches.
    assert sibling.exists()
    sibling_data = json.loads(sibling.read_text())
    assert sibling_data["run_id"] == m.run_id
    assert sibling_data["foundation_contract_sha256"] == "c" * 64


def test_composite_foundation_hash() -> None:
    """FND-07-c + IMP-2: manifest carries the composite sha alongside three inputs."""
    m = _build()
    assert m.mapping_sha256 == "m" * 64
    assert m.split_hash == "s" * 64
    assert m.exclusion_sha256 == "e" * 64
    assert m.foundation_contract_sha256 == "c" * 64
    # A manifest carrying only split_hash (pre-IMP-2) would fail this test.


def test_run_id_format() -> None:
    """generate_run_id format: YYYYMMDD-HHMMSS-<6hex> (UTC)."""
    rid = generate_run_id()
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8  # date
    assert len(parts[1]) == 6  # time
    assert len(parts[2]) == 6  # short uuid hex
    # All hex after hyphen removal.
    assert all(c in "0123456789abcdef" for c in parts[2])


def test_atomic_sibling_write(tmp_path: Path) -> None:
    """write_manifest_sibling uses atomic_write_json — no partial .tmp-* leftovers."""
    m = _build()
    result_path = tmp_path / "res.json"
    result_path.write_text("{}")
    sibling = write_manifest_sibling(m, result_path)
    assert sibling.exists()
    # Atomic write leaves no temp leftovers.
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")] == []


def test_embed_returns_same_dict() -> None:
    """embed_manifest_in_result mutates AND returns the dict (fluent API)."""
    m = _build()
    result = {"metric": 0.5}
    returned = embed_manifest_in_result(m, result)
    assert returned is result
    assert "_manifest" in result
    assert result["_manifest"]["run_id"] == m.run_id
