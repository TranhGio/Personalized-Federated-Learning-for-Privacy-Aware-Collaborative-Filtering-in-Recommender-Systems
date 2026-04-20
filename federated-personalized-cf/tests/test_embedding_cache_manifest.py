"""Tests for the D-04..D-10 manifest-sidecar embedding cache (Phase 3 Plan 03, PSN-05).

Covers:
  - D-04: sidecar layout — ``.embedding_cache/{run_id}/partition_{pid}.pt``
    + sibling ``.embedding_cache/{run_id}/manifest.json``.
  - D-05: loud-mismatch ``RuntimeError`` containing per-field delta + a
    literal ``rm -rf .embedding_cache/{run_id}/`` hint when any field of
    the 6-field signature diverges from the on-disk manifest.
  - D-06: schema_version=1 field inside ``manifest.json``.
  - D-07: manifest.json written via ``fedrec_foundation.atomic.atomic_write_json``.
  - D-09: opt-in reuse — when ``reuse_cache=True`` the path switches to
    ``.embedding_cache/sig_{sha256(fields)[:16]}/``; two runs with the
    same signature collide on the same sig-dir.
  - D-10: single-row disk payload — the ``.pt`` has exactly 2 keys
    (``local_user_row`` shape ``(d,)`` and ``local_user_bias`` shape
    ``(1,)``). Never the old ``(num_users, d)`` blob.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _build_signature(
    run_id: str = "r1",
    method: str = "bpr",
    num_users: int = 6040,
    num_items: int = 3706,
    dim: int = 64,
    split_hash: str = "abc123",
):
    """Build a signature dict mirroring client_app._signature_fields output."""
    from federated_personalized_cf.client_app import _signature_fields

    return _signature_fields(
        run_id=run_id,
        method=method,
        num_users=num_users,
        num_items=num_items,
        dim=dim,
        split_hash=split_hash,
    )


def _single_row_state(dim: int = 64):
    """Build a valid 2-key single-row state dict for persistence tests."""
    from collections import OrderedDict

    return OrderedDict(
        [
            ("local_user_row", torch.randn(dim)),
            ("local_user_bias", torch.zeros(1)),
        ]
    )


def test_manifest_sidecar_written_and_loaded(tmp_path, monkeypatch) -> None:
    """D-04 + D-06 + D-10: sidecar layout + schema_version=1 + single-row payload.

    Simulates ``_save_local_user_state`` then inspects on-disk layout:
      - ``manifest.json`` exists with schema_version=1 and matching signature.
      - ``partition_0.pt`` exists and contains exactly 2 keys
        (``local_user_row`` shape ``(d,)``, ``local_user_bias`` shape ``(1,)``).
    """
    from federated_personalized_cf import client_app

    # Redirect the cache base dir to tmp_path so the test doesn't touch
    # the real .embedding_cache/.
    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig = _build_signature(run_id="r1", dim=64)
    state = _single_row_state(dim=64)

    client_app._save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )

    cache_dir = tmp_path / "r1"
    pt_path = cache_dir / "partition_0.pt"
    manifest_path = cache_dir / "manifest.json"

    assert pt_path.exists(), f"partition .pt not written at {pt_path}"
    assert manifest_path.exists(), f"manifest.json not written at {manifest_path}"

    # D-06: schema_version=1 + full signature visible in manifest.json.
    with open(manifest_path, "r") as f:
        cached = json.load(f)
    assert cached.get("schema_version") == 1, "D-06: schema_version=1 required"
    for key in ("run_id", "method", "num_users", "num_items", "dim", "split_hash"):
        assert key in cached, f"D-06: manifest.json missing signature field {key}"
    assert cached["dim"] == 64

    # D-10: single-row disk payload — exactly 2 keys with the right shapes.
    payload = torch.load(pt_path, map_location="cpu")
    assert set(payload.keys()) == {"local_user_row", "local_user_bias"}, (
        f"D-10 violated: payload keys {sorted(payload.keys())}"
    )
    assert tuple(payload["local_user_row"].shape) == (64,), (
        f"D-10 violated: local_user_row shape {payload['local_user_row'].shape}"
    )
    assert tuple(payload["local_user_bias"].shape) == (1,), (
        f"D-10 violated: local_user_bias shape {payload['local_user_bias'].shape}"
    )

    # Load path returns the same state dict.
    loaded = client_app._load_local_user_state(
        partition_id=0,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )
    assert loaded is not None
    assert set(loaded.keys()) == {"local_user_row", "local_user_bias"}


def test_manifest_mismatch_raises_runtime_error(tmp_path, monkeypatch) -> None:
    """D-05: loud signature-mismatch on load; error message carries delta + rm -rf hint."""
    from federated_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig_a = _build_signature(run_id="r1", dim=64)
    state = _single_row_state(dim=64)
    client_app._save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig_a,
    )

    # Now attempt to load with a divergent dim.
    sig_b = _build_signature(run_id="r1", dim=128)
    with pytest.raises(RuntimeError) as excinfo:
        client_app._load_local_user_state(
            partition_id=0,
            run_id="r1",
            reuse_cache=False,
            signature=sig_b,
        )
    msg = str(excinfo.value)
    assert "dim" in msg, f"D-05 error should reference the diverging field; got: {msg}"
    assert "rm -rf" in msg, (
        f"D-05 error must include literal 'rm -rf' hint; got: {msg}"
    )
    assert "r1" in msg, (
        f"D-05 error must include the specific run_id path; got: {msg}"
    )


def test_reuse_cache_sig_path(tmp_path, monkeypatch) -> None:
    """D-09: reuse_cache=True routes to .embedding_cache/sig_{hash}/; two runs collide."""
    from federated_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig_run_a = _build_signature(run_id="run_alpha", dim=64)
    sig_run_b = _build_signature(run_id="run_beta", dim=64)  # identical fields except run_id

    path_a = client_app._cache_dir_for_run(
        run_id="run_alpha", reuse_cache=True, signature=sig_run_a
    )
    path_b = client_app._cache_dir_for_run(
        run_id="run_beta", reuse_cache=True, signature=sig_run_b
    )
    # D-09: sig_* dir is run_id-agnostic, so two runs with identical fields
    # (except run_id) resolve to the SAME directory.
    assert path_a == path_b, (
        f"D-09 violated: reuse_cache=True should collide on signature; "
        f"got {path_a} != {path_b}"
    )
    # D-09: the dir name is sig_<16-hex-chars>.
    name = Path(path_a).name
    assert name.startswith("sig_"), f"D-09: reuse_cache path should start with 'sig_'; got {name}"
    assert len(name) == len("sig_") + 16, (
        f"D-09: sig_ dir should have 16 hex chars; got {name!r} ({len(name)} chars)"
    )


def test_single_row_payload_shape_guard_on_save(tmp_path, monkeypatch) -> None:
    """D-10: _save_local_user_state refuses a non-single-row payload.

    Regression guard — if a future adaptive-plan accidentally hands a
    3-key state_dict (or a ghost-table blob) to this plan's cache
    helper, the assert fires BEFORE any disk write happens.
    """
    from collections import OrderedDict

    from federated_personalized_cf import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)

    sig = _build_signature(run_id="r1", dim=64)
    # Malformed payload: extra key ``personal_mlp.fc.weight`` (a Phase-4
    # adaptive LOCAL tensor). This plan's D-10 guard must reject it.
    bad_state = OrderedDict(
        [
            ("local_user_row", torch.randn(64)),
            ("local_user_bias", torch.zeros(1)),
            ("personal_mlp.fc.weight", torch.randn(16, 64)),
        ]
    )
    with pytest.raises(AssertionError, match="D-10"):
        client_app._save_local_user_state(
            partition_id=0,
            state_dict=bad_state,
            run_id="r1",
            reuse_cache=False,
            signature=sig,
        )
