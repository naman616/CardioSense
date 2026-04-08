"""
Module: src/training/focal_loss.py

Responsibility:
    Focal Loss implementation for handling the severe class imbalance in
    MIT-BIH (83% Normal vs 0.7% Fusion beats) as specified in §8.5.

Formula:
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where:
        p_t    = model's predicted probability for the correct class
        γ = 2  = focusing parameter (down-weights well-classified easy examples)
        α_t    = 1 / class_frequency_t  (per-class inverse frequency weight)

The (1 - p_t)^2 factor:
    - When p_t ≈ 1 (easy, well-classified sample): (1 - p_t)^2 ≈ 0 → loss ≈ 0
    - When p_t ≈ 0 (hard, misclassified sample): (1 - p_t)^2 ≈ 1 → full loss
    - Concentrates gradient updates on hard, rare arrhythmia classes (S, V, F, Q)
      which have the greatest clinical importance.

Why not standard CrossEntropy:
    - Standard CE causes the model to overfit the dominant Normal class (83%).
    - The model would achieve ~83% accuracy by predicting Normal for everything.
    - Focal Loss forces the model to learn minority class distinctions.

Design Notes:
    - α_t weights are computed from training set class frequencies.
    - γ=2 is the paper-recommended default (Lin et al., IEEE ICCV 2017).
    - Compatible with PyTorch's reduction='mean' convention.
    - Class weights are registered as a buffer (moved to device automatically).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with per-class inverse frequency weighting."""

    def __init__(self, gamma: float = 2.0, class_frequencies: np.ndarray | None = None):
        """
        Args:
            gamma: Focusing parameter. Higher γ = more focus on hard examples.
            class_frequencies: Array of per-class proportions from training set.
                               Used to compute α_t = 1 / class_frequency.
                               If None, uniform weights (standard focal loss).
        """
        super().__init__()
        self.gamma = gamma

        if class_frequencies is not None:
            # α_t = 1 / freq, then normalize so weights sum to num_classes
            alpha = 1.0 / np.clip(class_frequencies, 1e-6, None)
            alpha = (alpha / alpha.sum() * len(alpha)).astype(np.float32)
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw class scores from model (before softmax), shape (batch, num_classes)
            targets: Integer class labels, shape (batch,)
        Returns:
            Scalar focal loss (mean over batch)
        """
        # Numerically stable log-probabilities via log_softmax (avoids softmax→log chain)
        log_probs = F.log_softmax(logits, dim=1)  # (batch, num_classes)

        # Apply per-class alpha weights if available
        if self.alpha is not None:
            log_probs_weighted = log_probs * self.alpha.unsqueeze(0)
        else:
            log_probs_weighted = log_probs

        # Per-sample CE: -log(p_t) weighted
        batch_idx = torch.arange(len(targets), device=targets.device)
        ce = -log_probs_weighted[batch_idx, targets]

        # p_t = predicted probability of the correct class
        # Not detached — gradient flows through both the CE term and the focal modulation term.
        pt = torch.exp(log_probs[batch_idx, targets])
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


def compute_class_frequencies(y: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Compute per-class proportion from training labels.

    Returns:
        Array of shape (num_classes,) with proportions summing to 1.0.
    """
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(float)
    return counts / counts.sum()
