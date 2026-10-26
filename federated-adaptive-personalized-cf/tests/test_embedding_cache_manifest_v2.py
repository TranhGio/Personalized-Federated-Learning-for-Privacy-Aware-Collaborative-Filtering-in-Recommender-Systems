"""Tests for the D-01..D-04 schema_version=2 manifest-sidecar embedding cache (Phase 4 Plan 03).

Covers:
  - D-01: atomic single-file cache per partition — ``.embedding_cache/{run_id}/partition_{pid}.pt``
    + sibling ``.embedding_cache/{run_id}/manifest.json``.
  - D-02: schema_version=2 fingerprint — 12 signature fields (6 Phase-3
    + 6 Phase-4: alpha_method, fusion_type, mlp_hidden_dims,
    per_user_alpha_enabled, item_perturbation_enabled, contrastive_lambda).
  - D-04: loud-mismatch ``RuntimeError`` with per-field delta + literal
    ``rm -rf .embedding_cache/{run_id}/`` hint. Schema_version 1→2
    mismatch also raises loudly.
  - D-09: opt-in reuse — ``reuse_cache=True`` switches path to
    ``.embedding_cache/sig_{sha256(fields)[:16]}/``; two runs with the
    same Phase-4 signature collide on the same sig-dir.
  - Extended LOCAL key payload shape: base + MLP + fusion + logit_alpha
    + item_perturbation all round-trip byte-identical.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


def _build_sig(**overrides):
    """Build a Phase-4 schema_version=2 signature via the client_app helper."""
    from federated_adaptive_personalized_cf.client_app import _signature_fields_v2

    base = dict(
        run_id="r1",
        method="dual",
        num_users=6040,
        num_items=3706,
        dim=64,
        split_hash="abc123",
        alpha_method="hierarchical_conditional",
        fusion_type="concat",
        mlp_hidden_dims="512,256,128",
        per_user_alpha_enabled=True,
        item_perturbation_enabled=True,
        contrastive_lambda=0.1,
    )
    base.update(overrides)
    return _signature_fields_v2(**base)


def _build_full_local_state(num_users: int = 6040, num_items: int = 3706, dim: int = 64):
    """Build a state dict with every LOCAL key that the Phase-4 cache persists."""
    return OrderedDict(
        [
            ("user_embeddings.weight", torch.zeros(num_users, dim)),
            ("user_bias.weight", torch.zeros(num_users, 1)),
            ("_logit_alpha.weight", torch.zeros(num_users, 1)),
            ("_item_perturbation.weight", torch.zeros(num_items, dim)),
        ]
    )


def test_manifest_v2_sidecar_written_and_loaded(tmp_path, monkeypatch) -> None:
    """D-01 + D-02: sidecar layout + schema_version=2 + 12 signature fields."""
    from federated_adaptive_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig = _build_sig()
    state = _build_full_local_state(num_users=6040, num_items=3706, dim=64)

    client_app._save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )

    cache_dir = tmp_path / "r1"
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "partition_0.pt").exists()

    with open(cache_dir / "manifest.json", "r") as f:
        cached = json.load(f)
    assert cached.get("schema_version") == 2, "D-02: schema_version=2 required"
    # All 12 signature fields present.
    for field in (
        "run_id",
        "method",
        "num_users",
        "num_items",
        "dim",
        "split_hash",
        "alpha_method",
        "fusion_type",
        "mlp_hidden_dims",
        "per_user_alpha_enabled",
        "item_perturbation_enabled",
        "contrastive_lambda",
    ):
        assert field in cached, f"D-02: manifest missing signature field {field}"

    loaded = client_app._load_local_user_state(
        partition_id=0,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )
    assert loaded is not None
    assert set(loaded.keys()) == set(state.keys())


def test_manifest_v2_mismatch_raises_runtime_error(tmp_path, monkeypatch) -> None:
    """D-04: loud mismatch on alpha_method change + rm -rf hint."""
    from federated_adaptive_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig_a = _build_sig(alpha_method="hierarchical_conditional")
    sig_b = _build_sig(alpha_method="multi_factor")
    state = _build_full_local_state()

    client_app._save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig_a,
    )
    with pytest.raises(RuntimeError) as excinfo:
        client_app._load_local_user_state(
            partition_id=0,
            run_id="r1",
            reuse_cache=False,
            signature=sig_b,
        )
    msg = str(excinfo.value)
    assert "alpha_method" in msg, f"D-04 error should reference diverging field; got: {msg}"
    assert "rm -rf" in msg, f"D-04 error must include rm -rf hint; got: {msg}"
    assert "r1" in msg, f"D-04 error must include the run_id path; got: {msg}"


def test_reuse_cache_sig_path_v2(tmp_path, monkeypatch) -> None:
    """D-09: reuse_cache=True routes to .embedding_cache/sig_{hash}/; two runs collide."""
    from federated_adaptive_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig_a = _build_sig(run_id="run_alpha")
    sig_b = _build_sig(run_id="run_beta")

    path_a = client_app._cache_dir_for_run(
        run_id="run_alpha", reuse_cache=True, signature=sig_a
    )
    path_b = client_app._cache_dir_for_run(
        run_id="run_beta", reuse_cache=True, signature=sig_b
    )
    assert path_a == path_b, (
        f"D-09: reuse_cache=True should collide on signature; got {path_a} != {path_b}"
    )
    name = Path(path_a).name
    assert name.startswith("sig_"), f"D-09: reuse_cache path should start with 'sig_'; got {name}"
    assert len(name) == len("sig_") + 16


def test_extended_local_key_payload_shape(tmp_path, monkeypatch) -> None:
    """D-01 extended payload: base + MLP + fusion + logit_alpha + item_perturbation round-trip."""
    from federated_adaptive_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig = _build_sig()
    state = OrderedDict(
        [
            ("user_embeddings.weight", torch.randn(6040, 64)),
            ("user_bias.weight", torch.randn(6040, 1)),
            ("personal_mlp.0.weight", torch.randn(512, 64)),
            ("personal_mlp.0.bias", torch.randn(512)),
            ("fusion_layer.weight", torch.randn(1, 2)),
            ("fusion_layer.bias", torch.randn(1)),
            ("_logit_alpha.weight", torch.randn(6040, 1)),
            ("_item_perturbation.weight", torch.randn(3706, 64)),
        ]
    )
    client_app._save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )
    loaded = client_app._load_local_user_state(
        partition_id=0,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )
    assert loaded is not None
    for k in state:
        assert torch.equal(state[k], loaded[k]), f"round-trip failure on {k}"


def test_schema_v1_manifest_raises_when_loading_under_v2(tmp_path, monkeypatch) -> None:
    """D-04: Phase-3 schema_version=1 cache triggers loud fail under v2 code."""
    from federated_adaptive_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    cache_dir = tmp_path / "r1"
    cache_dir.mkdir(parents=True)
    # Seed a Phase-3 v1 manifest (6 fields, schema_version=1).
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "r1",
                "method": "dual",
                "num_users": 6040,
                "num_items": 3706,
                "dim": 64,
                "split_hash": "abc",
            }
        )
    )
    (cache_dir / "partition_0.pt").write_bytes(b"\x00")

    sig = _build_sig()
    with pytest.raises(RuntimeError) as excinfo:
        client_app._load_local_user_state(
            partition_id=0,
            run_id="r1",
            reuse_cache=False,
            signature=sig,
        )
    msg = str(excinfo.value)
    assert "schema_version" in msg, f"D-04 schema mismatch must reference 'schema_version'; got: {msg}"
    assert "rm -rf" in msg
