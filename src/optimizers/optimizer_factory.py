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


OPTIMIZER_CONFIGS = {
    "sgd": {"lr": 0.01},
    "sgd_momentum": {"lr": 0.01, "momentum": 0.9},
    "adagrad": {"lr": 0.01},
    "rmsprop": {"lr": 0.001, "alpha": 0.99},
    "adam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8},
}


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
    raise NotImplementedError
