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


# Expected high-attention region per class
CLASS_ATTENTION_NOTES = {
    0: "Normal beats: Grad-CAM typically attends to P-wave and QRS transition regions.",
    1: "Supraventricular (S): High attention expected at P-wave (~samples 10-30) — abnormal P morphology.",
    2: "Ventricular (V): High attention expected at wide QRS complex (~samples 50-80).",
    3: "Fusion (F): Mixed attention across P-wave and QRS — hybrid normal+ectopic morphology.",
    4: "Unknown (Q): Diffuse attention pattern — morphology doesn't fit standard categories.",
}


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
    from src.explainability.gradcam import GradCAM1D
    from src.explainability.heatmap_overlay import overlay_heatmap
    from src.data.preprocessor import normalize

    st.subheader("Model Explainability — Grad-CAM")
    st.markdown(
        "Grad-CAM shows **which parts of the ECG signal the model attended to** "
        "when making its prediction. Red = high attention, Blue = low attention."
    )

    cls_idx = predictions["class_idx"]
    signal_hash = hash(ecg_signal.tobytes())

    # Show cached heatmap if already computed for this signal
    if st.session_state.get("gradcam_signal_hash") == signal_hash and "gradcam_fig" in st.session_state:
        st.pyplot(st.session_state["gradcam_fig"])
        note = CLASS_ATTENTION_NOTES.get(cls_idx, "")
        st.info(f"**Expected attention pattern:** {note}")
        return

    # Not yet computed — show button
    if st.button("Compute Grad-CAM Heatmap"):
        with st.spinner("Computing Grad-CAM..."):
            try:
                x = normalize(ecg_signal.reshape(1, -1))          # (1, 187)
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 1, 187)
                device = next(model.parameters()).device
                x_tensor = x_tensor.to(device)

                gradcam = GradCAM1D(model, model.get_gradcam_target_layer())
                heatmap = gradcam.compute(x_tensor, target_class=cls_idx)

                fig = overlay_heatmap(
                    ecg_signal=ecg_signal,
                    heatmap=heatmap,
                    predicted_class=cls_idx,
                    true_class=cls_idx,
                    confidence=predictions["confidence"],
                )
                # Cache result and rerun to display it
                st.session_state["gradcam_fig"] = fig
                st.session_state["gradcam_signal_hash"] = signal_hash
                st.rerun()

            except Exception as e:
                st.error(f"Grad-CAM computation failed: {e}")
