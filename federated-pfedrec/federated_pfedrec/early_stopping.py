"""Early stopping utility for federated learning.

Monitors a metric across rounds and signals when to stop training
if no improvement is observed for a specified number of rounds (patience).
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EarlyStoppingState:
    """Tracks the state of early stopping."""

    best_metric: float = float('-inf')  # For maximization (e.g., NDCG)
    best_round: int = 0
    rounds_without_improvement: int = 0
    should_stop: bool = False

    # Store best model parameters (optional)
    best_parameters: Optional[Dict[str, Any]] = field(default=None, repr=False)


class EarlyStopping:
    """
    Early stopping callback for federated learning.

    Monitors a specified metric and stops training when no improvement
    is seen for a specified number of rounds.

    Args:
        patience: Number of rounds with no improvement to wait before stopping.
        metric_name: Name of the metric to monitor (e.g., 'ndcg@10', 'sampled_ndcg@10').
        mode: 'max' for metrics where higher is better, 'min' for lower is better.
        min_delta: Minimum change to qualify as an improvement.
        restore_best: Whether to track best parameters for restoration.
        verbose: Whether to print early stopping messages.

    Example:
        >>> early_stopping = EarlyStopping(patience=5, metric_name='ndcg@10', mode='max')
        >>> for round_num in range(num_rounds):
        ...     metrics = train_and_evaluate(round_num)
        ...     if early_stopping.step(round_num, metrics):
        ...         print(f"Early stopping triggered at round {round_num}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        metric_name: str = "ndcg@10",
        mode: str = "max",
        min_delta: float = 0.0,
        restore_best: bool = False,
        verbose: bool = True,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.verbose = verbose

        # Initialize state
        self._state = EarlyStoppingState(
            best_metric=float('-inf') if mode == "max" else float('inf')
        )

    @property
    def state(self) -> EarlyStoppingState:
        """Get current early stopping state."""
        return self._state

    @property
    def best_metric(self) -> float:
        """Get the best metric value observed."""
        return self._state.best_metric

    @property
    def best_round(self) -> int:
        """Get the round number with the best metric."""
        return self._state.best_round

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current metric is an improvement over best."""
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

    def step(
        self,
        round_num: int,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if training should stop based on current metrics.

        Args:
            round_num: Current round number.
            metrics: Dictionary of metrics from evaluation.
            parameters: Optional model parameters to store if this is the best round.

        Returns:
            True if training should stop, False otherwise.
        """
        # Get the monitored metric
        current_metric = metrics.get(self.metric_name)

        if current_metric is None:
            if self.verbose:
                print(f"  Warning: Metric '{self.metric_name}' not found in metrics")
            return False

        # Check for improvement
        if self._is_improvement(current_metric, self._state.best_metric):
            # Improvement found
            if self.verbose:
                improvement = current_metric - self._state.best_metric
                direction = "+" if self.mode == "max" else ""
                print(f"  Early stopping: {self.metric_name} improved from "
                      f"{self._state.best_metric:.4f} to {current_metric:.4f} "
                      f"({direction}{improvement:.4f})")

            self._state.best_metric = current_metric
            self._state.best_round = round_num
            self._state.rounds_without_improvement = 0

            # Store best parameters if requested
            if self.restore_best and parameters is not None:
                self._state.best_parameters = parameters.copy()
        else:
            # No improvement
            self._state.rounds_without_improvement += 1

            if self.verbose:
                print(f"  Early stopping: No improvement in {self.metric_name}. "
                      f"Best: {self._state.best_metric:.4f} at round {self._state.best_round}. "
                      f"Patience: {self._state.rounds_without_improvement}/{self.patience}")

        # Check if we should stop
        if self._state.rounds_without_improvement >= self.patience:
            self._state.should_stop = True
            if self.verbose:
                print(f"\n  ⏹ Early stopping triggered! No improvement for {self.patience} rounds.")
                print(f"    Best {self.metric_name}: {self._state.best_metric:.4f} at round {self._state.best_round}")
            return True

        return False

    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the parameters from the best round (if restore_best was enabled)."""
        return self._state.best_parameters

    def reset(self):
        """Reset the early stopping state."""
        self._state = EarlyStoppingState(
            best_metric=float('-inf') if self.mode == "max" else float('inf')
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of early stopping state for logging."""
        return {
            "best_metric": self._state.best_metric,
            "best_round": self._state.best_round,
            "rounds_without_improvement": self._state.rounds_without_improvement,
            "stopped_early": self._state.should_stop,
            "monitored_metric": self.metric_name,
            "patience": self.patience,
        }
