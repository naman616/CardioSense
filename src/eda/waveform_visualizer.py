"""
Module: src/eda/waveform_visualizer.py

Responsibility:
    Visualize representative ECG waveforms for each of the 5 arrhythmia classes
    to document morphological differences that the model must learn.

Key Outputs:
    - A 5-panel figure with one representative waveform per class.
    - Annotated P-wave, QRS complex, T-wave regions for Normal beats.
    - Visible morphological distinctions between classes (e.g., wide QRS in
      Ventricular Ectopic vs. narrow QRS in Supraventricular).

Design Notes:
    - Selects the median-amplitude sample per class as the representative.
    - X-axis: sample index 0-186 (187 points at 360 Hz ≈ 0.52 seconds).
    - Plots are saved to results/plots/class_waveforms.png.
    - These waveforms are also used in the Streamlit reference gallery (§8.10).
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

CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def get_representative_sample(X: np.ndarray, y: np.ndarray, class_id: int) -> np.ndarray:
    """Return the sample closest to the class mean (in L2 distance)."""
    cls_X = X[y == class_id]
    mean = cls_X.mean(axis=0)
    dists = np.linalg.norm(cls_X - mean, axis=1)
    return cls_X[np.argmin(dists)]


def plot_class_waveforms(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot one representative waveform per class in a single figure."""
    num_classes = 5
    t = np.arange(187)

    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Representative ECG Waveforms per Arrhythmia Class", fontsize=14, y=1.01)

    for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, CLASS_COLORS)):
        sample = get_representative_sample(X, y, i)
        ax.plot(t, sample, color=color, linewidth=1.2)
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(sample.min() - 0.1, sample.max() + 0.1)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate P-QRS-T regions on Normal beat only
        if i == 0:
            ax.axvspan(10, 30, alpha=0.12, color="green", label="P-wave")
            ax.axvspan(50, 80, alpha=0.12, color="red",   label="QRS")
            ax.axvspan(90, 130, alpha=0.12, color="blue", label="T-wave")
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Sample Index (0–186, ~360 Hz)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)
