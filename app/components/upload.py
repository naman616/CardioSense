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
import streamlit as st


def upload_panel() -> np.ndarray | None:
    """Render upload panel and return raw ECG array if uploaded.

    Returns:
        ecg_signal: numpy array of shape (187,) or None.
    """
    raise NotImplementedError
