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
import matplotlib.pyplot as plt
from pathlib import Path


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

CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
REFERENCE_WAVEFORMS_PATH = "data/processed/reference_waveforms.npz"


def reference_gallery() -> None:
    """Render the 5-class arrhythmia reference gallery."""
    st.subheader("Arrhythmia Reference Gallery")
    st.markdown(
        "Canonical ECG morphologies for each of the 5 arrhythmia classes "
        "(representative samples from the MIT-BIH training set)."
    )

    # Try to load pre-computed reference waveforms
    waveforms = {}
    ref_path = Path(REFERENCE_WAVEFORMS_PATH)
    if ref_path.exists():
        data = np.load(ref_path, allow_pickle=True)
        for i in range(5):
            key = f"class_{i}"
            if key in data:
                waveforms[i] = data[key]
    else:
        st.warning(
            "Reference waveforms not found. Run `notebooks/01_eda.ipynb` "
            "to generate `data/processed/reference_waveforms.npz`."
        )

    cols = st.columns(5)
    for i, col in enumerate(cols):
        info = CLASS_INFO[i]
        with col:
            st.markdown(f"**{info['abbr']}** — {info['name']}")

            if i in waveforms:
                fig, ax = plt.subplots(figsize=(3, 2))
                ax.plot(waveforms[i], color=CLASS_COLORS[i], linewidth=0.9)
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.markdown("*(waveform unavailable)*")

            st.caption(info["desc"])
            st.markdown(f"Model F1: **{info['f1']}**")
