"""
Module: src/evaluation/confusion_matrix.py

Responsibility:
    Generate and visualize the 5x5 normalized confusion matrix (§8.8).

Output:
    - 5x5 heatmap with normalized cell values (row-normalized: recall per class).
    - Rows = true class, Columns = predicted class.
    - Color scale: 0.0 (white) to 1.0 (dark blue/green).
    - Cell text shows both normalized value and raw count.
    - Axis labels use full class names.

Expected Pattern:
    - Strong diagonal (high recall per class).
    - Main confusion: Class 3 (Fusion) may be confused with Class 0 (Normal)
      or Class 2 (Ventricular) due to morphological overlap.
    - Class 4 (Unknown) may scatter across all classes by definition.

Design Notes:
    - Normalization is by row (true class) to show recall per class.
    - Saved to results/plots/confusion_matrices/confusion_matrix.png.
    - Also returns the raw confusion matrix array for further analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["Normal\n(N)", "Supraventricular\n(S)", "Ventricular\n(V)", "Fusion\n(F)", "Unknown\n(Q)"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    save_path: str | None = None,
) -> np.ndarray:
    """Plot 5x5 confusion matrix heatmap.

    Returns:
        cm: The (optionally normalized) confusion matrix array, shape (5, 5).
    """
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm = cm_raw.astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
        vmin=0, vmax=1 if normalize else None,
    )
    ax.set_title(
        "Confusion Matrix (row-normalized, shows per-class recall)",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return cm
