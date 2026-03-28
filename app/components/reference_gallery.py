"""
Module: app/components/reference_gallery.py

Responsibility:
    Streamlit reference gallery showing canonical ECG waveforms for each
    of the 5 arrhythmia classes for clinical comparison (§8.10).

UI Elements:
    - Section header: "Arrhythmia Reference Gallery"
    - 5-column layout (one column per class).
    - Each column shows:
        * Class name and clinical abbreviation
        * Representative ECG waveform plot
        * Brief morphological description
        * Expected F1-score from the trained model

Waveform Source:
    - Pre-computed representative samples from EDA
      (get_representative_sample() from src/eda/waveform_visualizer.py).
    - Saved as numpy arrays in data/processed/reference_waveforms.npz.
    - Loaded at app startup (no live computation needed).

Design Notes:
    - Reference waveforms must be generated once during EDA and saved.
    - Displayed even when no ECG is uploaded (static reference content).
    - Helps clinicians understand what each arrhythmia class looks like
      before interpreting the uploaded signal's classification.
"""

import numpy as np
import streamlit as st


CLASS_INFO = {
    0: {"name": "Normal Beat", "abbr": "N", "f1": "~99%",
        "desc": "Regular P-QRS-T complex, normal intervals."},
    1: {"name": "Supraventricular Ectopic", "abbr": "S", "f1": "~88%",
        "desc": "Narrow QRS, abnormal P-wave morphology."},
    2: {"name": "Ventricular Ectopic", "abbr": "V", "f1": "~95%",
        "desc": "Wide, bizarre QRS, no preceding P-wave."},
    3: {"name": "Fusion Beat", "abbr": "F", "f1": "~82%",
        "desc": "Hybrid morphology between N and V."},
    4: {"name": "Unknown / Unclassifiable", "abbr": "Q", "f1": "~91%",
        "desc": "Paced beat or unidentifiable morphology."},
}


def reference_gallery() -> None:
    """Render the 5-class arrhythmia reference gallery."""
    raise NotImplementedError
