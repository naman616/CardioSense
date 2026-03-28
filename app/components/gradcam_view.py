"""
Module: app/components/gradcam_view.py

Responsibility:
    Streamlit panel for Grad-CAM heatmap overlay on the uploaded ECG signal.

UI Elements:
    - Section header: "Model Explainability — Grad-CAM"
    - Matplotlib figure: ECG waveform with heatmap color overlay.
    - Colorbar legend: "Low attention" → "High attention".
    - Explanatory text mapping high-attention regions to cardiac anatomy:
        * P-wave: atrial depolarization (samples ~10-30)
        * QRS complex: ventricular depolarization (samples ~50-80)
        * T-wave: ventricular repolarization (samples ~100-140)
    - Note on clinical relevance: expected high-attention regions per class.

Design Notes:
    - Grad-CAM computed live using GradCAM1D from src/explainability/gradcam.py.
    - Calls overlay_heatmap() from src/explainability/heatmap_overlay.py.
    - Displayed via st.pyplot(fig).
    - Toggle button to show/hide Grad-CAM (since it requires extra computation).
"""

import numpy as np
import torch
import torch.nn as nn
import streamlit as st


def gradcam_panel(
    model: nn.Module,
    ecg_signal: np.ndarray,
    predictions: dict,
) -> None:
    """Render Grad-CAM overlay panel.

    Args:
        model: Trained ResNet1D in eval mode.
        ecg_signal: Raw ECG values, shape (187,).
        predictions: Output from run_inference() with 'class_idx' key.
    """
    raise NotImplementedError
