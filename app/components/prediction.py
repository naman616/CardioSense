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


CLASS_DESCRIPTIONS = {
    0: "Regular sinus rhythm with normal P-wave, QRS complex, and T-wave morphology.",
    1: "Ectopic beat originating above the Bundle of His; characterized by narrow QRS complex.",
    2: "Ectopic beat originating in the ventricles; characterized by wide, abnormal QRS complex.",
    3: "Simultaneous activation from normal and ectopic pathways; hybrid beat morphology.",
    4: "Beat morphology does not match standard AAMI EC57 arrhythmia category definitions.",
}


def prediction_panel(predictions: dict) -> None:
    """Render prediction results panel.

    Args:
        predictions: dict with keys:
            'class_idx'    : int (0-4)
            'class_name'   : str
            'confidence'   : float (max probability)
            'probabilities': np.ndarray shape (5,)
    """
    raise NotImplementedError
