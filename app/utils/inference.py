"""
Module: app/utils/inference.py

Responsibility:
    Model loading and inference pipeline for the Streamlit application.

Functions:
    load_model()     — Load trained ResNet1D from checkpoint file.
    run_inference()  — Preprocess a raw ECG signal and run model prediction.

Inference Pipeline (single sample):
    Raw numpy (187,)
    → Z-score normalization (same as training preprocessing)
    → Reshape to (1, 187, 1)
    → Convert to float32 tensor
    → Move to device
    → model.forward() → logits (1, 5)
    → Softmax → probabilities (1, 5)
    → argmax → predicted class index

Design Notes:
    - Z-score normalization uses PER-SAMPLE statistics (mean/std of the 187 values)
      matching the training preprocessing — no stored training statistics needed.
    - Model loaded via torch.load with weights_only=True for security.
    - Returns structured predictions dict for use by UI components.
    - Inference runs on CPU in the Streamlit app (no GPU assumption).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CHECKPOINT = str(_PROJECT_ROOT / "results/checkpoints/resnet1d_adam_best.pth")


def load_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> nn.Module:
    """Load trained ResNet1D from checkpoint.

    Returns:
        model: ResNet1D in eval mode on the specified device.
    """
    from src.models.resnet1d import ResNet1D
    from src.utils.device import get_device

    if device is None:
        device = get_device(verbose=False)

    model = ResNet1D()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model: nn.Module, ecg_signal: np.ndarray) -> dict:
    """Preprocess and classify a single ECG beat.

    Args:
        model: Trained ResNet1D in eval mode.
        ecg_signal: Raw ECG values, shape (187,).

    Returns:
        dict with keys:
            'class_idx'    : int (0-4)
            'class_name'   : str
            'confidence'   : float (max softmax probability)
            'probabilities': np.ndarray shape (5,)
    """
    from src.data.preprocessor import normalize

    # Z-score normalization (per-sample, same as training)
    x = normalize(ecg_signal.reshape(1, -1))        # (1, 187)

    # Reshape to (1, 1, 187) for Conv1d: (batch, channels, length)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 1, 187)

    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # (5,)

    idx = int(probs.argmax())
    return {
        "class_idx":    idx,
        "class_name":   CLASS_NAMES[idx],
        "confidence":   float(probs[idx]),
        "probabilities": probs,
    }
