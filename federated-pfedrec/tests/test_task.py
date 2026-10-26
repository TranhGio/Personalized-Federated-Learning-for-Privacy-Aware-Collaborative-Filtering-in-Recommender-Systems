"""Phase 5 PFR-04 + PFR-06 + PFR-07 + D-04 client task regression guard.

Pins:
- PFR-04 / FND-03: training negatives exclude held-out test positive (exclusion threading).
- PFR-07 / D-02: per-round RNG produces different negatives across rounds; FND-06 determinism.
- PFR-06 / FND-06: eval function uses ``np_rng`` (no stdlib ``random.seed``/``random.sample``);
  BSL-05-style cross-file regression: zero ``random.seed(``, zero ``random.sample(``,
  zero module-level ``import random`` in BOTH ``task.py`` and ``client_app.py``.
- D-04: eval-time BCE loss is computed over (positive + 99 negatives) — matches
  ``IJCAI-23-PFedRec/engine.py:195-196`` (``torch.cat((test_score, negative_score))``).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_train_negs_exclude_held_out_test_positive() -> None:
    """PFR-04 / FND-03: held-out test positive is NEVER drawn as a training negative."""
    from federated_pfedrec.task import _sample_train_negatives_seeded

    user_rated = {1, 2, 3}
    exclude = {5, 6}  # represents held-out test positive(s)
    no_go = user_rated | exclude

    rng = np.random.default_rng(42)
    negs = _sample_train_negatives_seeded(
        user_rated_items=no_go,
        num_items=100,
        num_negatives=20,
        rng=rng,
    )
    neg_set = set(int(x) for x in negs.tolist())
    assert neg_set.isdisjoint(no_go), (
        f"PFR-04 violated: train negs intersect held-out: {neg_set & no_go}"
    )


def test_train_negs_resampled_every_round() -> None:
    """PFR-07 / D-02: per-round RNG differs across rounds; same key produces same output (FND-06)."""
    from fedrec_foundation.rng import np_rng

    rng_r1_a = np_rng(run_seed=42, user_idx=0, round_num=1, purpose="train_neg")
    rng_r1_b = np_rng(run_seed=42, user_idx=0, round_num=1, purpose="train_neg")
    rng_r2 = np_rng(run_seed=42, user_idx=0, round_num=2, purpose="train_neg")

    out_r1_a = rng_r1_a.integers(0, 1000, size=20).tolist()
    out_r1_b = rng_r1_b.integers(0, 1000, size=20).tolist()
    out_r2 = rng_r2.integers(0, 1000, size=20).tolist()

    assert out_r1_a == out_r1_b, "FND-06: same key must give same output"
    assert out_r1_a != out_r2, "PFR-07: round 1 vs round 2 must differ"


def test_eval_neg_rng_factory_used() -> None:
    """PFR-06 / FND-06: eval function uses ``np_rng`` (not stdlib random); BSL-05 cross-file regression."""
    import federated_pfedrec.task as task_mod

    eval_fn = None
    for name in (
        "evaluate_pfedrec_sampled",
        "evaluate_ranking_sampled",
        "evaluate_pfedrec",
    ):
        if hasattr(task_mod, name):
            eval_fn = getattr(task_mod, name)
            break
    assert eval_fn is not None, "task.py must export a sampled eval function"

    sig = inspect.signature(eval_fn)
    assert "run_seed" in sig.parameters
    assert "user_idx" in sig.parameters
    assert "round_num" in sig.parameters
    assert "exclude_items" in sig.parameters

    src_eval = inspect.getsource(eval_fn)
    assert "np_rng" in src_eval, "PFR-06: eval must use FND-06 np_rng factory"
    assert "random.seed(" not in src_eval, "BSL-05: stdlib random.seed must be eradicated"
    assert "random.sample(" not in src_eval, "BSL-05: stdlib random.sample must be eradicated"

    # Cross-file regression: scan task.py and client_app.py
    task_path = Path(task_mod.__file__)
    client_path = task_path.parent / "client_app.py"
    for path in (task_path, client_path):
        text = path.read_text()
        assert "\nimport random\n" not in text and not text.startswith("import random\n"), (
            f"BSL-05: module-level `import random` must be removed from {path.name}"
        )
        assert "random.seed(" not in text, f"BSL-05: random.seed( must be removed from {path.name}"
        assert "random.sample(" not in text, f"BSL-05: random.sample( must be removed from {path.name}"


def test_eval_bce_over_positives_plus_99_negs() -> None:
    """D-04: eval BCE loss computed over (positive + 99 negatives) — matches engine.py:195-196."""
    import federated_pfedrec.task as task_mod

    eval_fn = None
    for name in (
        "evaluate_pfedrec_sampled",
        "evaluate_ranking_sampled",
        "evaluate_pfedrec",
    ):
        if hasattr(task_mod, name):
            eval_fn = getattr(task_mod, name)
            break
    assert eval_fn is not None

    src = inspect.getsource(eval_fn)
    # Reference engine.py:195-196 idiom — torch.cat over (test_score, negative_score)
    assert (
        "torch.cat((test_score, negative_score))" in src
        or "torch.cat([test_score, negative_score])" in src
        or ("ratings_pred" in src and "torch.cat" in src and "negative_score" in src)
    ), "D-04: eval-time BCE must concatenate positive + 99 negs (engine.py:195-196)"
