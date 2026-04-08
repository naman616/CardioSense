"""
Module: app/components/upload.py

Responsibility:
    Streamlit upload panel — accepts a CSV row of 187 ECG signal values.

UI Elements:
    - st.file_uploader: accepts .csv files.
    - Validates that the uploaded file has exactly 187 numeric columns.
    - Shows error messages for:
        * Wrong number of columns
        * Non-numeric values
        * Empty file
    - Optionally accepts a comma-separated text input as an alternative
      to file upload (for quick demos).

Design Notes:
    - Returns a numpy array of shape (187,) or None if nothing uploaded.
    - Does NOT apply preprocessing here — raw values are returned.
    - Preprocessing (Z-score normalization) happens in inference pipeline.
    - Matches the MIT-BIH format: 187 signal values, no header, no label.
"""

import numpy as np
import pandas as pd
import streamlit as st


def upload_panel() -> np.ndarray | None:
    """Render upload panel and return raw ECG array if uploaded.

    Returns:
        ecg_signal: numpy array of shape (187,) or None.
    """
    st.subheader("Upload ECG Signal")
    st.markdown(
        "Upload a CSV file containing **187 numeric ECG signal values** "
        "(one row, no header, no label column — MIT-BIH format)."
    )

    st.caption("Tip: If the file picker is slow or restricted to one folder, use the paste option below instead.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.empty:
                st.error("Uploaded file is empty.")
                return None

            # Use first row; drop label column if 188 cols present
            row = df.iloc[0]
            if len(row) >= 188:
                row = row.iloc[:187]
            elif len(row) < 187:
                st.error(f"Expected 187 signal columns, got {len(row)}. Please check the file format.")
                return None

            try:
                signal = row.values.astype(np.float32)
            except (ValueError, TypeError):
                st.error("Non-numeric values found in the CSV. All 187 columns must be numbers.")
                return None

            st.success(f"ECG signal loaded: 187 samples | "
                       f"range [{signal.min():.3f}, {signal.max():.3f}]")
            return signal

        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None

    # Alternative: text input for quick demos
    with st.expander("Or paste comma-separated values (demo)", expanded=True):
        text_input = st.text_area("Paste 187 comma-separated values:", height=80)
        if st.button("Parse Input") and text_input.strip():
            try:
                values = [float(v.strip()) for v in text_input.split(",") if v.strip()]
                if len(values) != 187:
                    st.error(f"Expected 187 values, got {len(values)}.")
                    return None
                return np.array(values, dtype=np.float32)
            except ValueError:
                st.error("Could not parse values. Ensure all values are numeric.")

    return None
