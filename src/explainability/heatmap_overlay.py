"""
Module: src/explainability/heatmap_overlay.py

Responsibility:
    Overlay the 1D Grad-CAM heatmap on the raw ECG waveform for visualization.

Output:
    - ECG waveform plotted as a line graph.
    - Heatmap overlaid as a colored fill_between (red = high attention,
      blue = low attention), semi-transparent under the signal line.
    - Title shows: predicted class, true class, confidence score.
    - Annotations mark approximate P-wave, QRS, T-wave regions.

Design Notes (§8.9):
    - One correctly classified AND one misclassified example per class is shown.
    - Total: 10 example plots (5 classes × 2 examples).
    - Saved to results/plots/gradcam/gradcam_{class}_{correct|wrong}.png.
    - The Streamlit app uses this module live for uploaded ECG signals.
    - Color map: 'RdYlBu_r' — red for high attention, blue for low.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from pathlib import Path


CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]


def overlay_heatmap(
    ecg_signal: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: int,
    true_class: int,
    confidence: float,
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay Grad-CAM heatmap on ECG waveform.

    Args:
        ecg_signal: Raw ECG values, shape (187,).
        heatmap: Grad-CAM heatmap, shape (187,), values in [0, 1].
        predicted_class: Model's predicted class index (0-4).
        true_class: Ground truth class index (0-4).
        confidence: Softmax probability for the predicted class.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure for display in Streamlit or notebooks.
    """
    t = np.arange(len(ecg_signal))
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(13, 4))

    # Colored vertical bands representing heatmap intensity
    for i in range(len(t) - 1):
        ax.axvspan(t[i], t[i + 1], alpha=0.35, color=cmap(norm(heatmap[i])), linewidth=0)

    # ECG waveform on top
    ax.plot(t, ecg_signal, "k-", linewidth=1.2, zorder=3)

    # ECG region annotations
    ax.axvspan(10, 30,  alpha=0.08, color="green", label="P-wave")
    ax.axvspan(50, 80,  alpha=0.08, color="red",   label="QRS")
    ax.axvspan(90, 130, alpha=0.08, color="blue",  label="T-wave")

    correct = predicted_class == true_class
    status = "CORRECT" if correct else "WRONG"
    pred_name = CLASS_NAMES[predicted_class] if 0 <= predicted_class < 5 else str(predicted_class)
    true_name = CLASS_NAMES[true_class] if 0 <= true_class < 5 else str(true_class)
    ax.set_title(
        f"Grad-CAM — Predicted: {pred_name} ({confidence*100:.1f}%) | "
        f"True: {true_name} [{status}]",
        fontsize=11,
    )
    ax.set_xlabel("Sample Index (0–186)")
    ax.set_ylabel("Amplitude (Z-score)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, shrink=0.9)
    cbar.set_label("Attention", fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
