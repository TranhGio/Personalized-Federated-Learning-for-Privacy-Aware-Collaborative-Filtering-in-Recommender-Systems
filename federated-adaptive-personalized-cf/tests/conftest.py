"""Shared fixtures for federated-adaptive-personalized-cf tests (Phase 4)."""
from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

import pytest

from flwr.common import EvaluateRes, Status, Code


def _make_eval_res(num_examples: int, metrics: Dict[str, float]) -> EvaluateRes:
    """Construct a Flower EvaluateRes with given num_examples + metrics."""
    return EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=float(metrics.get("eval_loss", 0.0)),
        num_examples=int(num_examples),
        metrics=dict(metrics),
    )


@pytest.fixture
def fake_evaluate_res():
    """Factory fixture returning EvaluateRes builders for strategy tests."""
    return _make_eval_res


@pytest.fixture
def fake_client_proxy():
    """Minimal MagicMock ClientProxy so strategy.aggregate_evaluate can index into results."""
    proxy = MagicMock()
    proxy.cid = "test_client"
    return proxy
