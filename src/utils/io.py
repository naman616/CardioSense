"""
Module: src/utils/io.py

Responsibility:
    Save and load experiment artifacts (model weights, training histories,
    preprocessed arrays) in a consistent, versioned manner.

Artifacts Managed:
    - Model checkpoints: torch.save / torch.load (.pth)
    - Training histories: JSON (loss/F1 curves per epoch per optimizer)
    - Preprocessed data arrays: numpy .npz (compressed)
    - Evaluation reports: text / CSV

Design Notes:
    - All save paths are relative to the project root.
    - Directories are created automatically on save.
    - Timestamps are added to filenames for versioning when requested.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def save_artifact(obj, path: str, fmt: str = "auto") -> None:
    """Save an artifact to disk.

    Args:
        obj: Object to save (nn.Module, dict, np.ndarray).
        path: Target file path.
        fmt: 'torch', 'json', 'numpy', or 'auto' (inferred from extension).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() if fmt == "auto" else f".{fmt}"
    if ext == ".pth":
        torch.save(obj, p)
    elif ext == ".json":
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)
    elif ext == ".npz":
        if isinstance(obj, dict):
            np.savez_compressed(p, **obj)
        else:
            np.savez_compressed(p, data=obj)
    elif ext == ".npy":
        np.save(p, obj)
    else:
        raise ValueError(f"Unsupported format: {ext!r}. Use .pth, .json, .npz, or .npy.")


def load_artifact(path: str, fmt: str = "auto"):
    """Load an artifact from disk.

    Args:
        path: Source file path.
        fmt: 'torch', 'json', 'numpy', or 'auto' (inferred from extension).
    """
    p = Path(path)
    ext = p.suffix.lower() if fmt == "auto" else f".{fmt}"
    if ext == ".pth":
        return torch.load(p, map_location="cpu")
    elif ext == ".json":
        with open(p) as f:
            return json.load(f)
    elif ext == ".npz":
        return np.load(p, allow_pickle=True)
    elif ext == ".npy":
        return np.load(p, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported format: {ext!r}. Use .pth, .json, .npz, or .npy.")
