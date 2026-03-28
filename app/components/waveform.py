"""
Module: app/components/waveform.py

Responsibility:
    Streamlit waveform viewer — plots the uploaded ECG signal for visual inspection.

UI Elements:
    - st.plotly_chart with an interactive line plot of the ECG signal.
    - X-axis: sample index 0-186 (optionally converted to milliseconds at 360 Hz).
    - Y-axis: normalized amplitude.
    - Hover tooltip shows sample index and amplitude.
    - Vertical reference lines for approximate P-wave, QRS, T-wave regions
      (fixed anatomical windows based on average beat morphology).

Design Notes:
    - Uses Plotly for interactivity (zoom, pan, hover) in Streamlit.
    - Input signal is already raw (pre-normalization) — plotted as-is for
      clinical familiarity.
    - Displayed before prediction to let clinicians inspect the signal first.
"""

import numpy as np
import streamlit as st


def waveform_viewer(ecg_signal: np.ndarray) -> None:
    """Render interactive ECG waveform plot.

    Args:
        ecg_signal: Raw ECG values, shape (187,).
    """
    raise NotImplementedError
