"""
Module: src/models/residual_block.py

Responsibility:
    A single 1D Residual Block as specified in §8.4 of the proposal.

Architecture per block:
    Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm ──(+skip)──► ReLU

Skip Connection:
    - Identity skip if input and output channels match AND stride == 1.
    - 1x1 Conv1D projection skip if channels differ OR stride > 1
      (used for downsampling blocks at stride=2).

Design Notes:
    - BatchNorm after each Conv1D stabilizes activations and smooths
      the loss landscape, complementing Adam's adaptive updates.
    - Skip connections prevent vanishing gradients in deep architectures
      (He et al., CVPR 2016).
    - kernel_size=7, padding=same to preserve temporal receptive field
      for ECG morphology feature extraction (P-wave, QRS, T-wave).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """1D Residual Block with optional downsampling via stride."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
