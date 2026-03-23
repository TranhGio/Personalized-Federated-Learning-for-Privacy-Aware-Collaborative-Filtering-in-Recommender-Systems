"""Loss functions for collaborative filtering."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error loss for rating prediction."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Compute MSE loss.

        Args:
            predictions: Predicted ratings, shape (batch_size,)
            targets: True ratings, shape (batch_size,)

        Returns:
            MSE loss value
        """
        return self.mse(predictions, targets)


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.

    Based on: "BPR: Bayesian Personalized Ranking from Implicit Feedback"
    Rendle et al., UAI 2009

    Optimizes pairwise ranking: users prefer observed items over unobserved items.

    Key insight from RecSys 2024 research:
    - Proper implementation critical (50% performance variance)
    - Negative sampling strategy matters
    - Regularization essential
    """

    def __init__(self, margin=0.0):
        """
        Initialize BPR loss.

        Args:
            margin: Margin for the ranking loss (default: 0.0)
                   Higher margin = stricter ranking constraint
        """
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        """
        Compute BPR loss for a batch.

        Args:
            pos_scores: Scores for positive (observed) items, shape (batch_size,)
            neg_scores: Scores for negative (unobserved) items, shape (batch_size, num_negatives)
                       or (batch_size,) if num_negatives=1

        Returns:
            BPR loss value

        BPR Assumption:
            User prefers positive item over negative item
            Maximize: score(positive) - score(negative)
            Minimize: -log(sigmoid(score(pos) - score(neg)))
        """
        # Handle both single and multiple negatives
        if neg_scores.dim() == 1:
            # Single negative per positive: shape (batch_size,)
            diff = pos_scores - neg_scores - self.margin
        else:
            # Multiple negatives: shape (batch_size, num_negatives)
            # Expand pos_scores to match
            pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch_size, 1)
            diff = pos_scores_expanded - neg_scores - self.margin  # (batch_size, num_negatives)

        # BPR loss: -mean(log(sigmoid(diff)))
        # Equivalent to: log(1 + exp(-diff))
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))

        return loss


class BPRLossWithRegularization(nn.Module):
    """
    BPR Loss with L2 regularization on model parameters.

    Combines BPR ranking loss with weight decay regularization.
    Following best practices from RecSys 2024 research.
    """

    def __init__(self, margin=0.0, weight_decay=1e-5):
        """
        Initialize BPR loss with regularization.

        Args:
            margin: Margin for ranking loss
            weight_decay: L2 regularization strength (λ)
                         Typical values: 1e-5 to 1e-3
        """
        super().__init__()
        self.bpr_loss = BPRLoss(margin=margin)
        self.weight_decay = weight_decay

    def forward(self, pos_scores, neg_scores, model_parameters=None):
        """
        Compute BPR loss with regularization.

        Args:
            pos_scores: Positive item scores
            neg_scores: Negative item scores
            model_parameters: List of parameters to regularize (embeddings)

        Returns:
            Total loss (BPR + L2 regularization)
        """
        # BPR ranking loss
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # L2 regularization (if parameters provided)
        reg_loss = 0.0
        if model_parameters is not None and self.weight_decay > 0:
            for param in model_parameters:
                reg_loss += torch.sum(param ** 2)
            reg_loss = self.weight_decay * reg_loss

        return bpr_loss + reg_loss


class InfoNCEContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for local-global user embedding alignment.

    Positive pair: (p_local[u], p_effective[u]) for same user
    Negative pairs: (p_local[u], p_effective[v]) for different users in batch

    Encourages:
    1. Local embeddings to remain aligned with blended embeddings (not collapse)
    2. Different users to maintain distinct representations (discrimination)
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Scaling temperature for similarity scores.
                        Lower = sharper distribution (harder negatives).
                        Typical values: 0.05 to 0.5
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        local_embeddings: torch.Tensor,
        effective_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            local_embeddings: Local user embeddings, shape (batch_size, embed_dim)
            effective_embeddings: Alpha-blended user embeddings, shape (batch_size, embed_dim)

        Returns:
            Scalar contrastive loss
        """
        batch_size = local_embeddings.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=local_embeddings.device)

        # L2 normalize for cosine similarity
        local_norm = F.normalize(local_embeddings, dim=1)
        effective_norm = F.normalize(effective_embeddings, dim=1)

        # Similarity matrix: (B, B) where [i,j] = sim(local[i], effective[j])
        sim_matrix = torch.mm(local_norm, effective_norm.t()) / self.temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=local_embeddings.device)

        # Cross-entropy loss with diagonal as positive
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
