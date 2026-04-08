"""
Module: app/components/prediction.py

Responsibility:
    Streamlit prediction panel — displays the arrhythmia classification result.

UI Elements:
    - Large st.metric showing predicted class name and confidence %.
    - Color-coded badge: green for Normal, red for arrhythmia classes.
    - Horizontal bar chart (st.bar_chart or Plotly) of all 5 class probabilities.
    - Clinical description of the predicted arrhythmia type.
    - Disclaimer: "For research/demonstration purposes only."

Class Descriptions:
    0 Normal Beat          : Regular sinus rhythm, normal P-QRS-T morphology.
    1 Supraventricular (S) : Originates above the ventricles; narrow QRS complex.
    2 Ventricular Ectopic  : Originates in ventricles; wide, abnormal QRS complex.
    3 Fusion Beat          : Simultaneous normal + ectopic beat; hybrid morphology.
    4 Unknown/Unclass.     : Does not fit standard morphological categories.

Design Notes:
    - Input: predictions dict with keys 'class_idx', 'class_name', 'probabilities'.
    - Probabilities displayed as percentage values (0-100%).
    - Adam paper connection: confidence in minority classes (S, V, F) demonstrates
      Adam's effective sparse gradient handling for rare beat types.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go


CLASS_DESCRIPTIONS = {
    0: "Regular sinus rhythm with normal P-wave, QRS complex, and T-wave morphology.",
    1: "Ectopic beat originating above the Bundle of His; characterized by narrow QRS complex.",
    2: "Ectopic beat originating in the ventricles; characterized by wide, abnormal QRS complex.",
    3: "Simultaneous activation from normal and ectopic pathways; hybrid beat morphology.",
    4: "Beat morphology does not match standard AAMI EC57 arrhythmia category definitions.",
}

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
CLASS_COLORS = ["#2ca02c", "#d62728", "#d62728", "#d62728", "#ff7f0e"]


def prediction_panel(predictions: dict) -> None:
    """Render prediction results panel.

    Args:
        predictions: dict with keys:
            'class_idx'    : int (0-4)
            'class_name'   : str
            'confidence'   : float (max probability)
            'probabilities': np.ndarray shape (5,)
    """
    st.subheader("Classification Result")

    cls_idx = predictions["class_idx"]
    cls_name = predictions["class_name"]
    confidence = predictions["confidence"]
    probs = predictions["probabilities"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(label="Predicted Class", value=cls_name)
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
        color = "#2ca02c" if cls_idx == 0 else "#d62728"
        st.markdown(
            f"<div style='background:{color};color:white;padding:6px 12px;"
            f"border-radius:4px;display:inline-block;font-weight:bold'>"
            f"{'NORMAL' if cls_idx == 0 else 'ARRHYTHMIA DETECTED'}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        # Horizontal probability bar chart
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in probs],
            y=CLASS_NAMES,
            orientation="h",
            marker_color=[CLASS_COLORS[i] if i == cls_idx else "#aec7e8" for i in range(5)],
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            title="Class Probabilities",
            xaxis_title="Probability (%)",
            xaxis=dict(range=[0, 115]),
            height=220,
            margin=dict(l=10, r=60, t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Clinical Note:** {CLASS_DESCRIPTIONS.get(cls_idx, '')}")
    st.caption(
        "⚠️ **Disclaimer:** For research and demonstration purposes only. "
        "Not intended for clinical diagnosis or treatment decisions."
    )
