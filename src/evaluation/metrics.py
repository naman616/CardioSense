"""
Module: src/evaluation/metrics.py

Responsibility:
    Compute all evaluation metrics for the final model assessment (§8.8).

Metrics Computed:
    - Overall classification accuracy
    - Per-class precision, recall, F1-score (5 classes)
    - Macro-averaged F1-score (equal weight per class — favors minority classes)
    - Weighted F1-score (weighted by support — standard benchmark metric)
    - One-vs-rest ROC-AUC per class (5 curves)
    - Mean ROC-AUC across all classes

Expected Final Results (1D ResNet + Adam, per Table 3):
    Class 0 (Normal)          : F1 ~99%
    Class 1 (Supraventricular): F1 ~88%
    Class 2 (Ventricular)     : F1 ~95%
    Class 3 (Fusion)          : F1 ~82%  <- hardest: 0.7% of data
    Class 4 (Unknown)         : F1 ~91%
    Overall accuracy          : ≥98%
    Macro F1                  : ≥93%

Design Notes:
    - Uses scikit-learn for all metrics for consistency and correctness.
    - All metrics computed on mitbih_test.csv (untouched hold-out set).
    - ROC-AUC requires softmax probabilities (not argmax predictions).
    - Returns a MetricsResult dataclass for clean downstream usage.
"""

import numpy as np
import torch
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize


@dataclass
class MetricsResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: np.ndarray  # shape (5,)
    per_class_recall: np.ndarray     # shape (5,)
    per_class_f1: np.ndarray         # shape (5,)
    per_class_roc_auc: np.ndarray    # shape (5,)
    mean_roc_auc: float


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
) -> MetricsResult:
    """Compute full evaluation metrics.

    Args:
        y_true: Ground truth labels, shape (N,)
        y_pred: Predicted class indices, shape (N,)
        y_prob: Softmax probabilities, shape (N, 5) — needed for ROC-AUC
        num_classes: Number of arrhythmia classes

    Returns:
        MetricsResult with all computed metrics.
    """
    labels = list(range(num_classes))

    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    # One-vs-rest ROC-AUC per class
    y_bin = label_binarize(y_true, classes=labels)
    if y_bin.shape[1] == 1:  # binary edge case guard
        y_bin = np.hstack([1 - y_bin, y_bin])
    per_auc = np.array([
        roc_auc_score(y_bin[:, i], y_prob[:, i]) for i in range(num_classes)
    ])

    return MetricsResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class_precision=per_pre.astype(float),
        per_class_recall=per_rec.astype(float),
        per_class_f1=per_f1.astype(float),
        per_class_roc_auc=per_auc.astype(float),
        mean_roc_auc=float(per_auc.mean()),
    )


def get_predictions(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return y_true, y_pred, y_prob."""
    model.eval()
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y.numpy())

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.vstack(all_probs),
    )
