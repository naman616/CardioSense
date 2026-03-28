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
    raise NotImplementedError
