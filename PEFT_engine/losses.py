"""Loss functions for seizure binary classification."""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: positive class weight (0.25 means positive loss × 0.25,
               since positives are already boosted by sampling).
        gamma: focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class WeightedBCE(nn.Module):
    """BCEWithLogitsLoss with automatic positive class weighting."""

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        else:
            pw = None
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pw
        )
        return loss


def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory: build loss function by type string.

    Args:
        loss_type: 'focal' | 'bce' | 'bce_weighted'
        **kwargs: focal_alpha, focal_gamma, pos_weight, etc.
    """
    if loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get("focal_alpha", 0.25),
            gamma=kwargs.get("focal_gamma", 2.0),
        )
    elif loss_type == "bce":
        return WeightedBCE(pos_weight=None)
    elif loss_type == "bce_weighted":
        return WeightedBCE(pos_weight=kwargs.get("pos_weight", 1.0))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
