"""
Module: app/main.py

Responsibility:
    Streamlit web application entry point for CardioSense (§8.10).
    Interactive real-time ECG arrhythmia classification with Grad-CAM explainability.

Run:
    streamlit run app/main.py

Application Layout (5 sections):
    ┌────────────────────────────────────────────┐
    │  CardioSense — ECG Arrhythmia Classifier   │
    ├────────────────────────────────────────────┤
    │  [1] Upload Panel                          │
    │      Upload CSV row of 187 ECG values      │
    ├────────────────────────────────────────────┤
    │  [2] Waveform Viewer                       │
    │      Interactive plot of uploaded signal   │
    ├────────────────────────────────────────────┤
    │  [3] Prediction Panel                      │
    │      Predicted class + confidence bar chart│
    ├────────────────────────────────────────────┤
    │  [4] Grad-CAM Overlay                      │
    │      Heatmap on ECG waveform               │
    ├────────────────────────────────────────────┤
    │  [5] Reference Gallery                     │
    │      Canonical waveform per class          │
    └────────────────────────────────────────────┘

Design Notes:
    - Model loaded once via st.cache_resource for performance.
    - Accepts CSV files with exactly 187 numeric columns (no label column).
    - Shows per-class probabilities as a horizontal bar chart.
    - Grad-CAM computed live on uploaded signal.
    - Reference gallery uses pre-computed canonical waveforms from EDA.
"""

import streamlit as st
import torch
from app.utils.inference import load_model, run_inference
from app.components.upload import upload_panel
from app.components.waveform import waveform_viewer
from app.components.prediction import prediction_panel
from app.components.gradcam_view import gradcam_panel
from app.components.reference_gallery import reference_gallery


@st.cache_resource
def get_model():
    """Load and cache the trained ResNet1D model."""
    return load_model()


def main():
    st.set_page_config(
        page_title="CardioSense — ECG Arrhythmia Classifier",
        page_icon="🫀",
        layout="wide",
    )
    st.title("CardioSense")
    st.caption("Deep Learning-Based ECG Arrhythmia Classification · Adam Optimizer · 1D ResNet")

    model = get_model()
    ecg_signal = upload_panel()

    if ecg_signal is not None:
        waveform_viewer(ecg_signal)
        predictions = run_inference(model, ecg_signal)
        prediction_panel(predictions)
        gradcam_panel(model, ecg_signal, predictions)

    reference_gallery()


if __name__ == "__main__":
    main()
