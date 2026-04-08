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

import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

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
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.callbacks = callbacks or []
        self.max_epochs = max_epochs
        self.scheduler = (
            CosineAnnealingLR(optimizer, T_max=max_epochs) if use_lr_scheduler else None
        )

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
        history = defaultdict(list)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch(train_loader)
            val_m = self._val_epoch(val_loader)
            elapsed = time.time() - t0

            # Log the LR used this epoch BEFORE the scheduler updates it
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            for k, v in train_m.items():
                history[f"train_{k}"].append(v)
            for k, v in val_m.items():
                history[f"val_{k}"].append(v)
            history["epoch_time"].append(elapsed)

            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"loss {train_m['loss']:.4f}/{val_m['loss']:.4f} | "
                f"f1 {train_m['f1']:.4f}/{val_m['f1']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Run callbacks
            val_f1 = val_m["f1"]
            should_stop = False
            for cb in self.callbacks:
                if isinstance(cb, EarlyStopping):
                    if cb(val_f1, self.model):
                        should_stop = True
                elif isinstance(cb, ModelCheckpoint):
                    cb(epoch, val_f1, self.model)

            if should_stop:
                print(f"Early stopping at epoch {epoch}.")
                for cb in self.callbacks:
                    if isinstance(cb, EarlyStopping):
                        cb.restore_best_weights(self.model)
                break

        return dict(history)

    def _train_epoch(self, loader: DataLoader) -> dict:
        """One training epoch. Returns dict of train metrics."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for X, y in tqdm(loader, desc="  train", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        return {"loss": total_loss / len(loader), "f1": f1, "acc": acc}

    def _val_epoch(self, loader: DataLoader) -> dict:
        """One validation epoch. Returns dict of val metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        return {"loss": total_loss / len(loader), "f1": f1, "acc": acc}

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Final evaluation on test set. Returns dict of test metrics."""
        return self._val_epoch(test_loader)
