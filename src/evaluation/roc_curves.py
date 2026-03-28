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
    raise NotImplementedError
