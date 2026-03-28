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

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)",
]


def plot_class_waveforms(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot one representative waveform per class in a single figure."""
    raise NotImplementedError


def get_representative_sample(X: np.ndarray, y: np.ndarray, class_id: int) -> np.ndarray:
    """Return the sample closest to the class mean (in L2 distance)."""
    raise NotImplementedError
