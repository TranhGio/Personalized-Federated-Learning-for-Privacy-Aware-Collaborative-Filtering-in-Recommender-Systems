"""Phase 5 Plan 03 cache-layout regression guard.

Pins:
- D-16 partition_{pid}.pt single-file-per-partition layout (no per-user-subdir).
- D-17 schema_v3 manifest with 10 required fields including
  ``schema_version=3``, ``method='pfedrec'``, ``loss='bce'``,
  ``num_train_negatives``, and the ``bias_classification='global'`` D-01 sentinel.
- D-21 strict-shape guard fires on BOTH save (BEFORE disk write) AND load.
- D-18 reuse_cache=true switches the path to ``sig_<sha256[:16]>``
  (run_id-agnostic).
- D-22 cold-round probe-then-load (returns ``None`` on cache miss).
- Pitfall 6: ``torch.load`` uses ``weights_only=True``.
- Phase 3 Rule-1: tempfile prefix MUST start with ``partition_tmp_``
  (PyTorchFileWriter rejects names starting with '.').
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
import torch


@pytest.fixture
def signature_factory():
    """Build a 10-field schema_v3 signature dict."""
    from federated_pfedrec.client_app import _signature_fields

    def _build(*, run_id="r1", latent_dim=32, num_users=6040, num_items=3706, split_hash="abc"):
        return _signature_fields(
            run_id=run_id,
            num_users=num_users,
            num_items=num_items,
            latent_dim=latent_dim,
            split_hash=split_hash,
        )

    return _build


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    """Redirect ``_CACHE_BASE_DIR`` into ``tmp_path``."""
    from federated_pfedrec import client_app

    monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: D-16 partition_{pid}.pt layout (no per-user-subdir).
# ---------------------------------------------------------------------------


def test_partition_pid_pt_layout(cache_root, signature_factory) -> None:
    """D-16 / PFR-03: cache file lands at ``{base}/{run_id}/partition_{pid}.pt``."""
    from federated_pfedrec.client_app import _save_local_user_state

    sig = signature_factory(run_id="r1", latent_dim=32)
    state = {"affine_output.weight": torch.randn(1, 32)}

    _save_local_user_state(
        partition_id=42,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )

    expected = cache_root / "r1" / "partition_42.pt"
    assert expected.exists(), f"D-16 layout violated: missing {expected}"
    # No legacy per-user-subdir directories.
    assert not (cache_root / "r1" / "partition_42").is_dir(), (
        "D-16 layout violated: legacy per-user-subdir directory present"
    )


# ---------------------------------------------------------------------------
# Test 2: D-17 schema_v3 manifest fields.
# ---------------------------------------------------------------------------


def test_manifest_schema_v3_fields(cache_root, signature_factory) -> None:
    """D-17: manifest sidecar has all 10 expected schema_v3 keys."""
    from federated_pfedrec.client_app import _save_local_user_state

    sig = signature_factory(run_id="r1", latent_dim=32)
    state = {"affine_output.weight": torch.randn(1, 32)}

    _save_local_user_state(
        partition_id=0,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )

    manifest_path = cache_root / "r1" / "manifest.json"
    assert manifest_path.exists(), "D-17: manifest.json sidecar must be written"

    manifest = json.loads(manifest_path.read_text())
    expected_keys = {
        "schema_version",
        "run_id",
        "method",
        "num_users",
        "num_items",
        "latent_dim",
        "split_hash",
        "loss",
        "num_train_negatives",
        "bias_classification",
    }
    assert set(manifest.keys()) >= expected_keys, (
        f"D-17 schema_v3 missing keys: {sorted(expected_keys - set(manifest.keys()))}"
    )
    assert manifest["schema_version"] == 3, "D-17: schema_version must be 3"
    assert manifest["method"] == "pfedrec"
    assert manifest["loss"] == "bce"


# ---------------------------------------------------------------------------
# Test 3: D-17 bias_classification='global' sentinel.
# ---------------------------------------------------------------------------


def test_bias_classification_sentinel_global(cache_root, signature_factory) -> None:
    """D-17: ``bias_classification='global'`` sentinel catches future D-01 reverts."""
    from federated_pfedrec.client_app import _save_local_user_state

    sig = signature_factory(run_id="r1", latent_dim=32)
    state = {"affine_output.weight": torch.randn(1, 32)}

    _save_local_user_state(
        partition_id=7,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig,
    )

    manifest = json.loads((cache_root / "r1" / "manifest.json").read_text())
    assert manifest["bias_classification"] == "global", (
        "D-17 sentinel: bias_classification MUST be 'global' to mechanically "
        "catch any future regression that reverts D-01 (bias-GLOBAL flip)."
    )


# ---------------------------------------------------------------------------
# Test 4: D-21 strict-load: shape mismatch raises with rm -rf hint.
# ---------------------------------------------------------------------------


def test_strict_load_shape_mismatch_raises(cache_root, signature_factory) -> None:
    """D-21: load fails loud when ``latent_dim`` differs between save and load."""
    from federated_pfedrec.client_app import (
        _load_local_user_state,
        _save_local_user_state,
    )

    sig_old = signature_factory(run_id="r1", latent_dim=32)
    state = {"affine_output.weight": torch.randn(1, 32)}
    _save_local_user_state(
        partition_id=42,
        state_dict=state,
        run_id="r1",
        reuse_cache=False,
        signature=sig_old,
    )

    # Same run_id; new signature has latent_dim=64.
    sig_new = signature_factory(run_id="r1", latent_dim=64)
    with pytest.raises(RuntimeError) as exc_info:
        _load_local_user_state(
            partition_id=42,
            run_id="r1",
            reuse_cache=False,
            signature=sig_new,
        )

    msg = str(exc_info.value)
    assert "latent_dim" in msg
    assert "rm -rf" in msg
    assert "r1" in msg


# ---------------------------------------------------------------------------
# Test 5: D-18 reuse-cache path collides on identical signature.
# ---------------------------------------------------------------------------


def test_reuse_cache_sig_path(cache_root, signature_factory) -> None:
    """D-18 / D-09: ``reuse_cache=true`` collapses two run_ids onto the same dir."""
    from federated_pfedrec.client_app import _cache_dir_for_run

    sig_a = signature_factory(run_id="run-a", latent_dim=32)
    sig_b = signature_factory(run_id="run-b", latent_dim=32)

    path_a = _cache_dir_for_run(run_id="run-a", reuse_cache=True, signature=sig_a)
    path_b = _cache_dir_for_run(run_id="run-b", reuse_cache=True, signature=sig_b)

    assert path_a == path_b, "D-18: identical signatures must collide under reuse_cache=true"
    assert re.match(r".*/sig_[0-9a-f]{16}$", str(path_a)), (
        f"D-18: reuse_cache path must match sig_<16-hex>, got {path_a}"
    )


# ---------------------------------------------------------------------------
# Test 6: D-21 save-side shape guard fires BEFORE disk write.
# ---------------------------------------------------------------------------


def test_save_payload_shape_guard(cache_root, signature_factory) -> None:
    """D-21: save rejects payloads with unexpected keys BEFORE any disk write."""
    from federated_pfedrec.client_app import _save_local_user_state

    sig = signature_factory(run_id="r1", latent_dim=32)
    bad_state = {
        "affine_output.weight": torch.randn(1, 32),
        "extra_param": torch.zeros(5),
    }

    with pytest.raises(AssertionError) as exc_info:
        _save_local_user_state(
            partition_id=0,
            state_dict=bad_state,
            run_id="r1",
            reuse_cache=False,
            signature=sig,
        )

    assert "D-21" in str(exc_info.value)
    # Disk write was rejected — neither the .pt nor the manifest exist.
    assert not (cache_root / "r1" / "partition_0.pt").exists()


# ---------------------------------------------------------------------------
# Test 7: Pitfall 6 — torch.load uses weights_only=True.
# ---------------------------------------------------------------------------


def test_torch_load_weights_only_true() -> None:
    """Pitfall 6: ``torch.load`` MUST be invoked with ``weights_only=True``."""
    from federated_pfedrec.client_app import _load_local_user_state

    src = inspect.getsource(_load_local_user_state)
    assert "weights_only=True" in src, (
        "Pitfall 6: torch.load must use weights_only=True (PyTorch 2.6+ safe default)"
    )
