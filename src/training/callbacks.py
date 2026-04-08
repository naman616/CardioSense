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

import copy
import torch
import torch.nn as nn
from pathlib import Path


class EarlyStopping:
    """Stop training when validation macro F1 stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = float("-inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_f1: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_f1 >= self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore model to best weights observed during training."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelCheckpoint:
    """Save model checkpoint whenever validation macro F1 improves."""

    def __init__(
        self,
        save_dir: str = "results/checkpoints",
        model_name: str = "resnet1d",
        optimizer_name: str = "adam",
        min_delta: float = 1e-4,
    ):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.min_delta = min_delta
        self.best_f1 = float("-inf")
        self.best_path: Path | None = None

    def __call__(self, epoch: int, val_f1: float, model: nn.Module) -> None:
        """Save checkpoint if val_f1 improved by at least min_delta."""
        if val_f1 >= self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            # Remove previous best checkpoint to save disk space
            if self.best_path is not None and self.best_path.exists():
                self.best_path.unlink()
            fname = f"{self.model_name}_{self.optimizer_name}_epoch{epoch:03d}_f1{val_f1:.4f}.pth"
            self.best_path = self.save_dir / fname
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.best_path)
            # Also save a stable alias so the app can load without knowing epoch/f1 in filename
            best_alias = self.save_dir / f"{self.model_name}_{self.optimizer_name}_best.pth"
            torch.save(model.state_dict(), best_alias)
