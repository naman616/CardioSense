"""
Module: src/models/resnet1d.py

Responsibility:
    Full 1D ResNet architecture for 5-class ECG arrhythmia classification (§8.4).

Architecture (exact spec from proposal):
    Input (187, 1)
    --> Conv1D(32 filters, kernel=7, padding=same) --> BatchNorm --> ReLU
    --> ResidualBlock(32 filters) x 2
    --> ResidualBlock(64 filters, stride=2) x 2   [downsampling, temporal compression]
    --> ResidualBlock(128 filters, stride=2) x 2  [downsampling, high-level features]
    --> Global Average Pooling                     [temporal invariance]
    --> Dense(128) --> ReLU --> Dropout(p=0.4)
    --> Dense(5) --> Softmax

Total parameters: ~790K
Training time: ~10 minutes (RTX 4070 Laptop GPU)

Feature Hierarchy:
    - 32-filter blocks: low-level morphology (P-wave, QRS onset, T-wave shape)
    - 64-filter blocks: mid-level patterns (beat intervals, rhythm features)
    - 128-filter blocks: high-level arrhythmia patterns (class-discriminative)
    - Global Average Pooling: aggregates temporal features, provides spatial
      invariance to beat alignment variations across patients.

Design Notes:
    - Trained with Adam (α=0.001, β1=0.9, β2=0.999, ε=1e-8) per §8.6.
    - Dropout(0.4) + weight decay (λ=1e-4) for regularization.
    - The final convolutional layer (128-filter blocks) is used as the
      target layer for Grad-CAM explainability in §8.9.
"""

import torch
import torch.nn as nn
from .residual_block import ResidualBlock


class ResNet1D(nn.Module):
    """1D Residual Neural Network for 5-class ECG arrhythmia classification."""

    NUM_CLASSES = 5

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Residual groups
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=7, stride=1),
            ResidualBlock(32, 32, kernel_size=7, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=7, stride=2),
            ResidualBlock(64, 64, kernel_size=7, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, kernel_size=7, stride=2),
            ResidualBlock(128, 128, kernel_size=7, stride=1),
            ResidualBlock(128, 128, kernel_size=7, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 187)
        Returns:
            logits: Raw class scores, shape (batch, num_classes).
                    Apply softmax externally when probabilities are needed.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        return self.head(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the final conv layer for Grad-CAM gradient hooks."""
        return self.layer3[-1].conv2
