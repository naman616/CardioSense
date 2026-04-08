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
from pathlib import Path

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)",
]


def plot_class_distribution(y: np.ndarray, save_path: str | None = None) -> None:
    """Plot and optionally save the class distribution bar chart."""
    num_classes = 5
    counts = [int(np.sum(y == i)) for i in range(num_classes)]
    total = len(y)
    percentages = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASS_NAMES, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])

    for bar, pct, cnt in zip(bars, percentages, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title("MIT-BIH Arrhythmia Dataset — Class Distribution", fontsize=13)
    ax.set_xlabel("Arrhythmia Class")
    ax.set_ylabel("Sample Count")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def print_class_statistics(y: np.ndarray) -> None:
    """Print per-class count and percentage to stdout."""
    total = len(y)
    print(f"\n{'Class':<28} {'Count':>8} {'Percentage':>12}")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        cnt = int(np.sum(y == i))
        pct = cnt / total * 100
        print(f"{name:<28} {cnt:>8,} {pct:>11.1f}%")
    print("-" * 50)
    print(f"{'Total':<28} {total:>8,} {'100.0%':>12}\n")
