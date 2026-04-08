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
import plotly.graph_objects as go


def waveform_viewer(ecg_signal: np.ndarray) -> None:
    """Render interactive ECG waveform plot.

    Args:
        ecg_signal: Raw ECG values, shape (187,).
    """
    st.subheader("ECG Waveform")

    t = list(range(187))

    fig = go.Figure()

    # Shaded regions for P-wave, QRS, T-wave
    fig.add_vrect(x0=10, x1=30, fillcolor="rgba(0,200,0,0.08)", line_width=0,
                  annotation_text="P-wave", annotation_position="top left")
    fig.add_vrect(x0=50, x1=80, fillcolor="rgba(255,0,0,0.08)", line_width=0,
                  annotation_text="QRS", annotation_position="top left")
    fig.add_vrect(x0=90, x1=130, fillcolor="rgba(0,0,255,0.08)", line_width=0,
                  annotation_text="T-wave", annotation_position="top left")

    # ECG line
    fig.add_trace(go.Scatter(
        x=t, y=ecg_signal.tolist(),
        mode="lines",
        line=dict(color="#1f3d7a", width=1.5),
        name="ECG Signal",
        hovertemplate="Sample: %{x}<br>Amplitude: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Uploaded ECG Beat (187 samples, ~360 Hz)",
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        hovermode="x unified",
        height=300,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
