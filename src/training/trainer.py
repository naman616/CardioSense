"""
Module: src/training/trainer.py

Responsibility:
    Core training loop for both BaselineCNN and ResNet1D models (§8.3, §8.6).

Training Configuration (§8.6):
    Optimizer : Adam (α=0.001, β1=0.9, β2=0.999, ε=1e-8)
    Epochs    : 100 with early stopping (patience=15 on val macro F1)
    Batch size: 128
    LR schedule: CosineAnnealingLR (per 1/√t decay motivation in Adam §4)
    Regularization: Dropout(0.4) + weight decay (λ=1e-4) via Adam's weight_decay

Per-Epoch Loop:
    1. Forward pass → logits
    2. FocalLoss(logits, targets) → scalar loss
    3. loss.backward() → gradients
    4. optimizer.step() → parameter update (Adam adjusts per-parameter lr)
    5. lr_scheduler.step() → cosine annealing update
    6. Compute val metrics: loss, accuracy, macro F1
    7. EarlyStopping check → stop if patience exhausted
    8. ModelCheckpoint → save if val macro F1 improved

Tracked Metrics (per epoch, per optimizer — for comparison study):
    - train_loss, train_acc, train_f1
    - val_loss, val_acc, val_f1
    - learning_rate (for LR schedule visualization)
    - epoch wall-clock time

Design Notes:
    - Trainer is optimizer-agnostic: accepts any torch.optim.Optimizer instance.
    - This allows running the identical training loop with 5 different optimizers
      for the comparison study (§8.7) without code duplication.
    - History dict returned at end for plotting convergence curves.
    - tqdm progress bar per epoch shows live loss and F1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .focal_loss import FocalLoss
from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """Optimizer-agnostic training loop for ECG classification models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: FocalLoss,
        device: torch.device,
        callbacks: list | None = None,
        use_lr_scheduler: bool = True,
        max_epochs: int = 100,
    ):
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """Run training loop.

        Returns:
            history: dict with keys train_loss, val_loss, train_f1, val_f1,
                     train_acc, val_acc, lr — each a list of per-epoch values.
        """
        raise NotImplementedError

    def _train_epoch(self, loader: DataLoader) -> dict:
        """One training epoch. Returns dict of train metrics."""
        raise NotImplementedError

    def _val_epoch(self, loader: DataLoader) -> dict:
        """One validation epoch. Returns dict of val metrics."""
        raise NotImplementedError

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Final evaluation on test set. Returns dict of test metrics."""
        raise NotImplementedError
