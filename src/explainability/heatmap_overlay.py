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
    raise NotImplementedError
