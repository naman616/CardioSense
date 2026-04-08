"""
Module: src/optimizers/optimizer_factory.py

Responsibility:
    Factory function that builds any of the 5 optimizers used in the
    comparison study (§8.7) from a name string and config dict.

Optimizers Supported:
    1. "sgd"              — SGD (no momentum)
       Prediction (Adam §6.1): stalls on sparse minority-class gradients;
       no adaptive mechanism to handle class imbalance.

    2. "sgd_momentum"     — SGD + Momentum (β=0.9)
       Prediction: improvement over vanilla SGD but lacks per-parameter
       adaptation; all parameters share one global learning rate.

    3. "adagrad"          — AdaGrad
       Prediction (Adam §5): handles sparsity via accumulated squared gradients
       but monotonically decaying learning rates stall late in training;
       cannot recover if a rare class gradient direction shifts.

    4. "rmsprop"          — RMSProp
       Prediction (Adam §5, §6.4): handles non-stationarity but diverges
       at high β2 without bias correction; worse than Adam on sparse ECG data.

    5. "adam"             — Adam (Proposed)
       Paper defaults: α=0.001, β1=0.9, β2=0.999, ε=1e-8
       Prediction (Adam §6): combines all advantages; expected fastest
       convergence and highest minority-class F1.

Design Notes:
    - All optimizers are initialized with the same weight_decay (λ=1e-4)
      for a fair comparison (applies L2 regularization equally).
    - The same random seed and model initialization are used across all runs.
    - Only the optimizer changes between comparison runs — everything else
      (model, data, loss, LR schedule) stays identical.
"""

import torch
import torch.nn as nn
from torch.optim import SGD, Adagrad, RMSprop, Adam


OPTIMIZER_CONFIGS = {
    # use_scheduler: True → apply CosineAnnealingLR to all optimizers for a fair comparison
    "sgd":          {"lr": 0.01,  "use_scheduler": True},
    "sgd_momentum": {"lr": 0.01,  "momentum": 0.9, "use_scheduler": True},
    "adagrad":      {"lr": 0.01,  "use_scheduler": True},
    "rmsprop":      {"lr": 0.001, "alpha": 0.99,   "use_scheduler": True},
    "adam":         {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "use_scheduler": True},
}


def optimizer_uses_scheduler(name: str) -> bool:
    """Return whether the named optimizer should use a global LR schedule."""
    if name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"Unknown optimizer: {name!r}")
    return OPTIMIZER_CONFIGS[name].get("use_scheduler", False)


def build_optimizer(
    model: nn.Module,
    name: str,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build an optimizer by name using paper-default hyperparameters.

    Args:
        model: The neural network whose parameters to optimize.
        name: One of 'sgd', 'sgd_momentum', 'adagrad', 'rmsprop', 'adam'.
        weight_decay: L2 regularization coefficient (same for all optimizers).

    Returns:
        Configured torch.optim.Optimizer instance.
    """
    if name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"Unknown optimizer: {name!r}. Choose from {list(OPTIMIZER_CONFIGS)}")

    cfg = dict(OPTIMIZER_CONFIGS[name])  # copy to avoid mutation
    cfg.pop("use_scheduler", None)       # not an optimizer kwarg — consumed by callers
    params = model.parameters()

    if name in ("sgd", "sgd_momentum"):
        return SGD(params, weight_decay=weight_decay, **cfg)
    elif name == "adagrad":
        return Adagrad(params, weight_decay=weight_decay, **cfg)
    elif name == "rmsprop":
        return RMSprop(params, weight_decay=weight_decay, **cfg)
    elif name == "adam":
        return Adam(params, weight_decay=weight_decay, **cfg)
