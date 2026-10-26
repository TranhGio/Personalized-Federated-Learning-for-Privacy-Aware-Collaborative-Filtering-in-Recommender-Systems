"""End-of-run W&B alert (email / Slack) for federated runs.

Every server_app calls :func:`send_run_finished_alert` once, right before
``wandb.finish()``, so you learn a run finished — and whether its number is
trustworthy — without ssh-ing into the training machine.

The alert carries the headline numbers plus the cheap trust signals that
the eval-validity tooling checks later:

- ``evaluated_users`` vs the expected population (``num_supernodes``);
- ``d06_cache_misses`` (0 = warm full-pop eval proven; >0 = cold-eval
  suspect; -1/absent = not stamped).

Level is ``WARN`` when a trust signal fails, ``INFO`` otherwise.

Delivery requires "Scriptable run alerts" to be enabled once in the W&B
user settings (Settings > Alerts); without it the call is a silent no-op
server-side. The helper is NON-FATAL by design: any failure prints a
warning and returns — an alert must never crash a finished run before
``wandb.finish()`` uploads its summary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a metric value for the alert body (``n/a`` when missing)."""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def build_run_finished_alert(
    module: str,
    run_id: str,
    final_metrics: Dict[str, Any],
    *,
    expected_users: Optional[int] = None,
    thesis_run_label: str = "",
    results_path: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Build ``(title, text, level)`` for the end-of-run alert.

    Pure function (no W&B import) so it is unit-testable.

    Parameters
    ----------
    module : str
        Module short name, e.g. ``"pfedrec"``.
    run_id : str
        The run's id (``YYYYMMDD-HHMMSS-xxxxxx``).
    final_metrics : Dict[str, Any]
        The server's ``final_metrics`` dict (``best`` / ``best_round`` /
        ``last_round`` keys).
    expected_users : Optional[int]
        Population size the full-pop eval should cover (``num_supernodes``).
    thesis_run_label : str
        Thesis tag ("main", "ablation_...", or "" for non-thesis runs).
    results_path : Optional[str]
        Where ``results.json`` was written.

    Returns
    -------
    Tuple[str, str, str]
        ``(title, text, level)`` with level ``"INFO"`` or ``"WARN"``.
    """
    best: Dict[str, Any] = final_metrics.get("best") or {}
    issues: List[str] = []

    evaluated = best.get("evaluated_users")
    if expected_users and isinstance(evaluated, (int, float)) and int(evaluated) != int(expected_users):
        issues.append(f"evaluated_users={int(evaluated)} != expected {int(expected_users)}")

    misses = best.get("d06_cache_misses")
    if isinstance(misses, (int, float)) and int(misses) > 0:
        issues.append(f"d06_cache_misses={int(misses)} (COLD-EVAL SUSPECT)")
    if isinstance(misses, (int, float)):
        misses_txt = str(int(misses))
    else:
        misses_txt = "not stamped"

    level = "WARN" if issues else "INFO"
    label = thesis_run_label or "non-thesis"
    status = "CHECK" if issues else "OK"
    title = f"[{status}] {module} {run_id} finished — NDCG@10 {_fmt(best.get('sampled_ndcg@10'))}"

    lines = [
        f"module: {module} | run_id: {run_id} | label: {label}",
        (
            f"best_round: {final_metrics.get('best_round')} / "
            f"last_round: {final_metrics.get('last_round')}"
        ),
        (
            f"NDCG@10: {_fmt(best.get('sampled_ndcg@10'))} "
            f"(sparse {_fmt(best.get('sampled_ndcg@10/sparse'))} | "
            f"medium {_fmt(best.get('sampled_ndcg@10/medium'))} | "
            f"dense {_fmt(best.get('sampled_ndcg@10/dense'))})"
        ),
        f"HR@10: {_fmt(best.get('sampled_hr@10'))}",
        (
            f"evaluated_users: {int(evaluated) if isinstance(evaluated, (int, float)) else 'n/a'}"
            f" | d06_cache_misses: {misses_txt}"
        ),
    ]
    if issues:
        lines.append("ISSUES: " + "; ".join(issues))
    if results_path:
        lines.append(f"results: {results_path}")
    lines.append("Next: run scripts/thesis/eval_validity.py before citing this number.")
    return title, "\n".join(lines), level


def send_run_finished_alert(
    module: str,
    run_id: str,
    final_metrics: Dict[str, Any],
    *,
    expected_users: Optional[int] = None,
    thesis_run_label: str = "",
    results_path: Optional[str] = None,
) -> None:
    """Send the end-of-run alert via ``wandb.alert`` (non-fatal no-op if no run).

    Parameters are as in :func:`build_run_finished_alert`.
    """
    try:
        import wandb

        if wandb.run is None:
            return
        title, text, level = build_run_finished_alert(
            module,
            run_id,
            final_metrics,
            expected_users=expected_users,
            thesis_run_label=thesis_run_label,
            results_path=results_path,
        )
        wandb.alert(
            title=title,
            text=text,
            level=getattr(wandb.AlertLevel, level),
        )
        print(f"  W&B alert sent: {title}")
    except Exception as exc:  # noqa: BLE001 — alerts must never fail a finished run
        print(f"[WARN] W&B alert failed (non-fatal): {exc}")
