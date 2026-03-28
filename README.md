# CardioSense

**Deep Learning-Based ECG Arrhythmia Classification Using the Adam Optimization Algorithm**

> Kingma & Ba (2015) — Adam: A Method for Stochastic Optimization (ICLR 2015, arXiv:1412.6980)

---

## Overview

CardioSense is an end-to-end deep learning system that classifies ECG heartbeat signals into five clinically defined arrhythmia categories using a **1D Residual Neural Network (ResNet)**. The system is trained with the **Adam optimizer** and empirically validates Adam's superiority over four competing optimizers, grounding every observed result in the theoretical predictions of the Kingma & Ba (2015) paper.

**Target Performance:** ≥98% accuracy, ≥93% macro F1 on MIT-BIH held-out test set.

---

## Folder Structure

```
CardioSense/
├── src/
│   ├── data/              # M1: Data Pipeline
│   │   ├── loader.py          — Load MIT-BIH and PTB-DB CSV files
│   │   ├── preprocessor.py    — Z-score → IQR → SMOTE → reshape → split
│   │   ├── dataset.py         — PyTorch Dataset / DataLoader wrappers
│   │   └── splitter.py        — Stratified 85/15 train/val split
│   ├── eda/               # M2: Exploratory Data Analysis
│   │   ├── class_distribution.py  — Class imbalance plots
│   │   ├── waveform_visualizer.py — Representative ECG waveforms per class
│   │   └── statistics.py          — Per-class amplitude/variance statistics
│   ├── models/            # M3: Neural Network Architectures
│   │   ├── residual_block.py  — Conv1D → BN → ReLU → Conv1D → BN → (+skip) → ReLU
│   │   ├── baseline_cnn.py    — 3-layer CNN baseline
│   │   └── resnet1d.py        — Full 1D ResNet (~1.2M params, 5-class)
│   ├── training/          # M4: Training Infrastructure
│   │   ├── focal_loss.py      — FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)
│   │   ├── trainer.py         — Optimizer-agnostic training loop
│   │   └── callbacks.py       — EarlyStopping, ModelCheckpoint
│   ├── optimizers/        # M5: Optimizer Comparison Study
│   │   ├── optimizer_factory.py   — Build SGD/AdaGrad/RMSProp/Adam by name
│   │   ├── comparison.py          — Train ResNet with all 5 optimizers
│   │   └── convergence_plots.py   — Loss/F1 curves, per-class F1 bar charts
│   ├── evaluation/        # M6: Model Evaluation
│   │   ├── metrics.py         — Accuracy, F1, ROC-AUC (MetricsResult dataclass)
│   │   ├── confusion_matrix.py — 5×5 normalized heatmap
│   │   ├── roc_curves.py      — One-vs-rest ROC per class
│   │   └── report.py          — Full evaluation report
│   ├── explainability/    # M7: Grad-CAM Explainability
│   │   ├── gradcam.py         — Captum-based 1D Grad-CAM on final conv layer
│   │   ├── heatmap_overlay.py — Color heatmap overlaid on ECG waveform
│   │   └── examples.py        — 1 correct + 1 misclassified example per class
│   └── utils/             # M8: Shared Utilities
│       ├── seed.py            — Global random seed (reproducibility)
│       ├── device.py          — Auto-detect CUDA > MPS > CPU
│       └── io.py              — Save/load checkpoints, histories, arrays
├── app/                   # M9: Streamlit Web Application
│   ├── main.py                — App entry point, layout, model caching
│   ├── components/
│   │   ├── upload.py          — CSV upload panel (187 ECG values)
│   │   ├── waveform.py        — Interactive Plotly ECG viewer
│   │   ├── prediction.py      — Class name, confidence, probability bar chart
│   │   ├── gradcam_view.py    — Live Grad-CAM heatmap overlay
│   │   └── reference_gallery.py — Canonical waveforms for 5 classes
│   └── utils/
│       └── inference.py       — Load model, preprocess + predict single ECG
├── notebooks/             # One notebook per pipeline step
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_resnet_training.ipynb
│   ├── 05_optimizer_comparison.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_gradcam_explainability.ipynb
│   └── 08_ptbdb_generalization.ipynb
├── configs/
│   ├── data_config.yaml       — Dataset paths, class frequencies, split ratios
│   ├── model_config.yaml      — Architecture specs for CNN and ResNet
│   ├── training_config.yaml   — Adam hyperparams, epochs, callbacks
│   └── optimizer_config.yaml  — All 5 optimizer configs + Adam paper predictions
├── data/
│   ├── raw/               — MIT-BIH + PTB-DB CSV files (not committed)
│   └── processed/         — Reference waveforms, preprocessed arrays
├── results/
│   ├── checkpoints/       — Trained model weights (.pth)
│   ├── plots/
│   │   ├── training_curves/   — Loss/F1 convergence plots
│   │   ├── confusion_matrices/
│   │   ├── roc_curves/
│   │   └── gradcam/           — Heatmap overlay figures (10 examples)
│   ├── logs/              — Per-optimizer training histories (JSON)
│   └── reports/           — Evaluation reports (text/CSV)
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_focal_loss.py
│   └── test_metrics.py
├── requirements.txt
└── .gitignore
```

---

## Module Design

### M1 — Data Pipeline (`src/data/`)

**Key Design Decisions:**
- SMOTE applied **after** train/val split (prevents data leakage into validation)
- Z-score normalization is **per-sample** — removes patient-specific amplitude differences; ensures uniform gradient magnitudes that Adam's adaptive updates handle well
- IQR outlier removal eliminates Holter digitization artifacts before SMOTE
- Final tensor shape: `(N, 187, 1)` for PyTorch `Conv1D`

---

### M2 — EDA (`src/eda/`)

**Clinical Motivation Documented Here:**
- Class 3 (Fusion Beat) at 0.7% → **sparse gradients** → motivates Adam (§6.1)
- Amplitude variance across patients → **Z-score normalization**
- Class imbalance 83%/0.7% → **Focal Loss** + **SMOTE**

---

### M3 — Model Architectures (`src/models/`)

**ResNet1D Architecture:**
```
Input (187, 1)
→ Conv1D(32, k=7) → BN → ReLU
→ ResidualBlock(32) × 2              [low-level: P-wave, QRS shape, T-wave]
→ ResidualBlock(64, stride=2) × 2   [mid-level: beat intervals, rhythm]
→ ResidualBlock(128, stride=2) × 2  [high-level: arrhythmia class patterns]
→ Global Average Pooling
→ Dense(128) → ReLU → Dropout(0.4)
→ Dense(5) → Softmax
~1.2M parameters
```

---

### M4 — Training Infrastructure (`src/training/`)

**Adam Hyperparameters (paper defaults):**

| Param | Value | Reference |
|---|---|---|
| α | 0.001 | Adam Algorithm 1 |
| β₁ | 0.9 | Adam Algorithm 1 |
| β₂ | 0.999 | Adam Algorithm 1 |
| ε | 1e-8 | Adam Algorithm 1 |
| γ (Focal) | 2.0 | Lin et al. (2017) |

---

### M5 — Optimizer Comparison (`src/optimizers/`)

| Optimizer | Adam Paper Prediction | Expected Macro F1 |
|---|---|---|
| SGD | Stalls on sparse minority gradients (§6.1) | ~72% |
| SGD+Momentum | No per-parameter adaptation (§5) | ~73% |
| AdaGrad | Monotonically decaying LR stalls late (§5) | ~78% |
| RMSProp | Diverges without bias correction (§6.4) | ~84% |
| **Adam** | **O(√T) regret bound; all advantages (§4, §6)** | **≥93%** |

---

### M6 — Evaluation (`src/evaluation/`)

**Expected Final Results (1D ResNet + Adam):**

| Class | Arrhythmia | Dataset % | Expected F1 |
|---|---|---|---|
| 0 | Normal (N) | 83.0% | ~99% |
| 1 | Supraventricular (S) | 3.0% | ~88% |
| 2 | Ventricular Ectopic (V) | 6.0% | ~95% |
| 3 | Fusion Beat (F) | 0.7% | ~82% |
| 4 | Unknown (Q) | 7.3% | ~91% |
| **Overall** | | | **≥98% acc, ≥93% macro F1** |

---

### M7 — Grad-CAM (`src/explainability/`)

**Clinical Verification (§8.9):**
- Supraventricular (S): heatmap peaks at **P-wave** (~samples 10-30)
- Ventricular (V): heatmap peaks at **QRS complex** (~samples 50-80)

---

### M9 — Streamlit App (`app/`)

```
streamlit run app/main.py
```

Accepts a CSV row of 187 ECG signal values → returns predicted arrhythmia class,
confidence scores, and Grad-CAM heatmap overlay.

---

## Datasets

| Dataset | Source | Size | Task |
|---|---|---|---|
| MIT-BIH Arrhythmia (primary) | Kaggle: shayanfazeli/heartbeat | 109,446 samples | 5-class |
| PTB Diagnostic ECG (secondary) | Same Kaggle page | 14,552 samples | Binary (healthy/MI) |

Place CSV files in `data/raw/` (see `data/raw/.gitkeep`).

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch 2.x | ResNet, training loop, Adam optimizer |
| Captum | Grad-CAM on 1D ECG signals |
| imbalanced-learn | SMOTE oversampling |
| scikit-learn | F1, ROC-AUC, confusion matrix |
| Matplotlib / Seaborn / Plotly | Visualization |
| Streamlit | Web application |

---

## Team

Purva Pote · Nikunj Rathi · Gurkirat Kaur — Division A, Batch 3

---

## References

1. Kingma & Ba (2015). Adam: A Method for Stochastic Optimization. ICLR 2015. arXiv:1412.6980
2. Moody & Mark (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE EMBS
3. Lin et al. (2017). Focal Loss for Dense Object Detection. IEEE ICCV
4. He et al. (2016). Deep Residual Learning for Image Recognition. IEEE CVPR
5. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR
6. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. IEEE ICCV
