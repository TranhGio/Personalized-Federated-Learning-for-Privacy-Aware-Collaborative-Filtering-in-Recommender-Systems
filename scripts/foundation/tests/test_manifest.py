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


# ============================================================================
# Phase 6 — schema v2 tests (RUN_MANIFEST_SCHEMA_VERSION = 2)
#
# These tests pin (1) the constant bump 1->2, (2) v1 backward-compat for legacy
# fixtures (Pitfall 3), (3) post-build mutation embedding the new fields,
# (4) preserved default sibling filename, (5) D-04 sibling_name override.
# ============================================================================

from dataclasses import replace as dataclass_replace
from typing import Any, Dict


def test_run_manifest_schema_version_3() -> None:
    """Phase 7 D-22: schema_version constant bumped from 2 to 3."""
    assert RUN_MANIFEST_SCHEMA_VERSION == 3, (
        f"Expected RUN_MANIFEST_SCHEMA_VERSION=3, got {RUN_MANIFEST_SCHEMA_VERSION}"
    )
    # Builder must propagate the bumped constant into the dataclass instance.
    m = _build()
    assert m.schema_version == 3
    # Embedded dict surface must agree.
    result_dict: Dict[str, Any] = {}
    embed_manifest_in_result(m, result_dict)
    assert result_dict["_manifest"]["schema_version"] == 3


def test_run_manifest_backward_compat_v1() -> None:
    """Pitfall 3: existing v1 test fixtures must construct without TypeError.

    The two NEW fields (final_eval_round_index, metrics) carry safe defaults
    so legacy callers never see a missing-kwarg TypeError.
    """
    # Construct directly using ONLY the v1 field set (no final_eval_round_index,
    # no metrics). The point is: this MUST NOT raise TypeError under v3.
    manifest = RunManifest(
        schema_version=3,
        run_id="20260429-104530-v1back",
        mode="benchmark_cross_device",
        num_supernodes=6040,
        partition_mode="natural",
        fraction_train=0.05,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
    )
    # Defaults must be the documented sentinels.
    assert manifest.final_eval_round_index == 0, (
        "Expected sentinel default 0 (no extra eval ran)"
    )
    assert manifest.metrics == {}, (
        "Expected default_factory=dict for metrics field"
    )


def test_run_manifest_backward_compat_v2() -> None:
    """Phase 7 D-22 + Pitfall 7: pre-v3 callers (no thesis kwargs) must construct without TypeError.

    All three new fields carry safe defaults: thesis_run_label="" (non-thesis sentinel),
    ablation_dimension="none", ablation_value="".
    """
    # Construct with the v2 field set ONLY (no thesis_run_label / ablation_dimension / ablation_value).
    manifest = RunManifest(
        schema_version=3,
        run_id="20260429-104530-v2back",
        mode="benchmark_cross_device",
        num_supernodes=6040,
        partition_mode="natural",
        fraction_train=0.05,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
        # Phase 6 v2 fields populated explicitly (no thesis fields):
        final_eval_round_index=0,
        metrics={},
    )
    # Defaults must be the documented sentinels per D-22.
    assert manifest.thesis_run_label == "", "Default thesis_run_label is empty string"
    assert manifest.ablation_dimension == "none", "Default ablation_dimension is 'none'"
    assert manifest.ablation_value == "", "Default ablation_value is empty string"


def test_run_manifest_carries_thesis_fields() -> None:
    """Phase 7 D-22: post-build mutation via dataclasses.replace populates the 3 thesis fields.

    Mirrors the Phase 6 D-07 pattern: server_app builds a manifest, then replaces it
    with the thesis-tagging fields read from context.run_config BEFORE embed_manifest_in_result.
    """
    m = _build()
    # Roundtrip empty defaults first.
    assert m.thesis_run_label == ""
    assert m.ablation_dimension == "none"
    assert m.ablation_value == ""
    # Now mutate via dataclass_replace (the canonical pattern Plan 02 server_apps will use).
    m2 = dataclass_replace(
        m,
        thesis_run_label="ablation_fusion_type=add",
        ablation_dimension="fusion_type",
        ablation_value="add",
    )
    assert m2.thesis_run_label == "ablation_fusion_type=add"
    assert m2.ablation_dimension == "fusion_type"
    assert m2.ablation_value == "add"
    # Embedded surface must also carry the mutated values.
    result_dict: Dict[str, Any] = {}
    embed_manifest_in_result(m2, result_dict)
    assert result_dict["_manifest"]["thesis_run_label"] == "ablation_fusion_type=add"
    assert result_dict["_manifest"]["ablation_dimension"] == "fusion_type"
    assert result_dict["_manifest"]["ablation_value"] == "add"


def test_run_manifest_carries_final_eval_round_index() -> None:
    """EVL-01 + EVL-06: post-build mutation populates the new fields."""
    manifest = RunManifest(
        schema_version=2,
        run_id="20260429-104530-evl",
        mode="benchmark_cross_device",
        num_supernodes=6040,
        partition_mode="natural",
        fraction_train=0.05,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
    )
    nested_metrics = {
        "best": {"sampled_ndcg@10": 0.4413, "sampled_hr@10": 0.7287},
        "last": {"sampled_ndcg@10": 0.4321, "sampled_hr@10": 0.7102},
        "best_round": 87,
        "last_round": 100,
        "final_eval_round_index": 101,
    }
    replaced = dataclass_replace(
        manifest, final_eval_round_index=101, metrics=nested_metrics
    )

    results_data: Dict[str, Any] = {}
    embed_manifest_in_result(replaced, results_data)
    embedded = results_data["_manifest"]
    assert embedded["schema_version"] == 2
    assert embedded["final_eval_round_index"] == 101
    assert embedded["metrics"]["best"]["sampled_ndcg@10"] == 0.4413
    assert embedded["metrics"]["last"]["sampled_ndcg@10"] == 0.4321
    assert embedded["metrics"]["best_round"] == 87
    assert embedded["metrics"]["last_round"] == 100


def test_write_manifest_sibling_default_filename(tmp_path: Path) -> None:
    """Default behavior preserved: <run_id>-manifest.json (cross-silo legacy)."""
    manifest = RunManifest(
        schema_version=2,
        run_id="20260429-104530-defflt",
        mode="cross_silo_legacy",
        num_supernodes=5,
        partition_mode="dirichlet",
        fraction_train=1.0,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="last_round",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
    )
    result_json = tmp_path / "results.json"
    result_json.write_text("{}")
    sibling = write_manifest_sibling(manifest, result_json)
    assert sibling.name == "20260429-104530-defflt-manifest.json"
    assert sibling.exists()


def test_write_manifest_sibling_custom_name(tmp_path: Path) -> None:
    """D-04: sibling_name='manifest.json' lands the clean per-run-dir filename."""
    manifest = RunManifest(
        schema_version=2,
        run_id="20260429-104530-clean",
        mode="benchmark_cross_device",
        num_supernodes=6040,
        partition_mode="natural",
        fraction_train=0.05,
        fraction_eval=1.0,
        weight_policy="num_positives",
        primary_evaluator="sampled_loo_99",
        num_train_negatives=4,
        num_eval_negatives=99,
        run_seed=42,
        checkpoint_rule="best_round_restore",
        mapping_sha256="m" * 64,
        split_hash="s" * 12,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides={},
        module="baseline",
        flwr_version="1.22.0",
        torch_version="2.7.1",
        git_commit="abc1234",
    )
    result_json = tmp_path / "results.json"
    result_json.write_text("{}")
    sibling = write_manifest_sibling(
        manifest, result_json, sibling_name="manifest.json"
    )
    assert sibling.name == "manifest.json"
    assert sibling.exists()
    payload = json.loads(sibling.read_text())
    assert payload["schema_version"] == 2


def test_effective_values_overrides_first() -> None:
    """Run-id audit 2026-06-10: top-level protocol fields record EFFECTIVE values.

    - snake_case override keys matching a manifest field take precedence over
      the ModeProfile default;
    - kebab-case keys do NOT match (recorded in ``overrides`` verbatim only);
    - override keys that are not manifest fields (e.g. ``lr``) never create or
      change a top-level field.
    """
    overrides = {
        "checkpoint_rule": "last_round",     # snake_case -> effective
        "fraction_eval": -1.0,                # sentinel recorded as configured
        "num-supernodes": 100,                # kebab-case -> ignored by _eff
        "lr": 0.005,                          # non-manifest key -> overrides only
    }
    m = build_run_manifest(
        run_id=generate_run_id(),
        mode_profile=_StubProfile(),
        run_seed=42,
        mapping_sha256="m" * 64,
        split_hash="s" * 64,
        exclusion_sha256="e" * 64,
        foundation_contract_sha256="c" * 64,
        raw_data_hash="r" * 64,
        builder_version="1.0.0",
        overrides=overrides,
        module="baseline",
    )
    assert m.checkpoint_rule == "last_round"          # effective, not "best_round"
    assert m.fraction_eval == -1.0                    # configured sentinel
    assert m.num_supernodes == 6040                   # kebab key ignored
    assert not hasattr(m, "lr")                       # no field invented
    assert m.overrides == overrides                   # verbatim audit copy
