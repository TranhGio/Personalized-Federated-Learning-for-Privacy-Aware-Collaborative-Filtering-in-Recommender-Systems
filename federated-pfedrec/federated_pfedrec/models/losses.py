"""Loss functions for PFedRec."""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Binary Cross-Entropy loss for implicit feedback prediction.

    Used by PFedRec for binarized ratings (0/1).
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted scores in [0, 1], shape (batch_size,) or (batch_size, 1).
        targets : torch.Tensor
            Binary targets (0 or 1), shape (batch_size,).

        Returns
        -------
        torch.Tensor
            BCE loss value.
        """
        return self.bce(predictions.view(-1), targets)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss (optional, for future use)."""

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """Compute BPR loss."""
        if neg_scores.dim() == 1:
            diff = pos_scores - neg_scores - self.margin
        else:
            pos_scores_expanded = pos_scores.unsqueeze(1)
            diff = pos_scores_expanded - neg_scores - self.margin
        return -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
