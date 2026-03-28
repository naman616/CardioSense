"""
Module: src/eda/class_distribution.py

Responsibility:
    Visualize and quantify the severe class imbalance in the MIT-BIH dataset.

Key Output:
    - Bar chart of per-class sample counts (absolute and %).
    - Expected distribution:
        Class 0 (Normal Beat)             : ~83.0%
        Class 1 (Supraventricular Ectopic): ~3.0%
        Class 2 (Ventricular Ectopic)     : ~6.0%
        Class 3 (Fusion Beat)             : ~0.7%   <- rarest, sparse gradients
        Class 4 (Unknown/Unclassifiable)  : ~7.3%

Design Notes:
    - This imbalance directly motivates Adam's use (sparse gradient handling)
      and Focal Loss (down-weighting easy majority-class examples).
    - Saves plot to results/plots/class_distribution.png.
    - Also prints a tabular summary to stdout for notebook use.
"""

import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)",
]


def plot_class_distribution(y: np.ndarray, save_path: str | None = None) -> None:
    """Plot and optionally save the class distribution bar chart."""
    raise NotImplementedError


def print_class_statistics(y: np.ndarray) -> None:
    """Print per-class count and percentage to stdout."""
    raise NotImplementedError
