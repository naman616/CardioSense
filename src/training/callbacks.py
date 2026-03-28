"""
Module: src/training/callbacks.py

Responsibility:
    Training callbacks for early stopping and model checkpointing.

EarlyStopping:
    - Monitors validation macro F1-score (not loss — per §8.6).
    - Patience: 15 epochs without improvement triggers stop.
    - Restores model weights to the best checkpoint on stopping.
    - Macro F1 is monitored (not accuracy) because class imbalance makes
      accuracy misleading — a model predicting only Normal achieves 83%.

ModelCheckpoint:
    - Saves model state_dict to results/checkpoints/ whenever val macro F1 improves.
    - Naming: {model_name}_{optimizer_name}_epoch{epoch:03d}_f1{f1:.4f}.pth
    - Keeps only the best checkpoint (overwrites on improvement).

Design Notes:
    - Both callbacks follow the sklearn-style interface: call(epoch, metrics_dict).
    - EarlyStopping raises StopIteration when patience is exhausted.
    - Callbacks are passed as a list to Trainer and called after each epoch.
"""

import torch
import torch.nn as nn
from pathlib import Path


class EarlyStopping:
    """Stop training when validation macro F1 stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        raise NotImplementedError

    def __call__(self, val_f1: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        raise NotImplementedError

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore model to best weights observed during training."""
        raise NotImplementedError


class ModelCheckpoint:
    """Save model checkpoint whenever validation macro F1 improves."""

    def __init__(
        self,
        save_dir: str = "results/checkpoints",
        model_name: str = "resnet1d",
        optimizer_name: str = "adam",
    ):
        raise NotImplementedError

    def __call__(self, epoch: int, val_f1: float, model: nn.Module) -> None:
        """Save checkpoint if val_f1 improved."""
        raise NotImplementedError
