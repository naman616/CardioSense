"""
Module: src/models/baseline_cnn.py

Responsibility:
    Lightweight 3-layer 1D CNN used as a performance baseline before
    introducing the full ResNet architecture (§8.3).

Architecture:
    Input (187, 1)
    --> Conv1D(32, kernel=7) --> BatchNorm --> ReLU --> MaxPool1D(2)
    --> Conv1D(64, kernel=5) --> BatchNorm --> ReLU --> MaxPool1D(2)
    --> Conv1D(128, kernel=3) --> BatchNorm --> ReLU --> GlobalAvgPool
    --> Dense(64) --> ReLU --> Dropout(0.3) --> Dense(5) --> Softmax

Purpose:
    - Establishes first optimizer comparison data point:
        Baseline CNN + SGD  vs  Baseline CNN + Adam
    - Expected performance: 88-91% accuracy with SGD, 93-95% with Adam.
    - Deliberately simpler than ResNet to isolate optimizer impact.

Design Notes:
    - No skip connections (unlike ResNet) — this tests Adam's benefit
      on a vanilla convolutional architecture.
    - Trained under identical conditions as ResNet for fair comparison.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """3-layer 1D CNN baseline for ECG arrhythmia classification."""

    NUM_CLASSES = 5

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 187, 1)
        Returns:
            logits: Tensor of shape (batch, 5)
        """
        raise NotImplementedError
