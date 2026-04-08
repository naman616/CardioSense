"""
Module: src/evaluation/roc_curves.py

Responsibility:
    Generate one-vs-rest ROC curves for each of the 5 arrhythmia classes (§8.8).

Output:
    - Single figure with 5 ROC curves (one per class).
    - Each curve labeled with class name and AUC score.
    - Random classifier diagonal reference line.
    - AUC values annotated on the plot.

Design Notes:
    - One-vs-rest: for each class k, binary classification of (k vs not-k).
    - Requires softmax probability outputs (y_prob), not hard predictions.
    - Classes with low support (Fusion: 0.7%) will have noisier ROC curves.
    - Expected AUC values: all ≥ 0.97 for the Adam-trained ResNet.
    - Saved to results/plots/roc_curves/roc_curves_all_classes.png.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | None = None,
) -> dict:
    """Plot one-vs-rest ROC curves for all 5 classes.

    Args:
        y_true: Ground truth labels, shape (N,)
        y_prob: Softmax probabilities, shape (N, 5)

    Returns:
        auc_scores: dict mapping class name -> AUC value
    """
    num_classes = 5
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random Classifier (AUC=0.50)")

    auc_scores = {}
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_title("One-vs-Rest ROC Curves — 1D ResNet + Adam", fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return auc_scores
