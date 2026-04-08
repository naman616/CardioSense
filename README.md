---
title: CardioSense
emoji: 🫀
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: ECG arrhythmia classifier — 5-class ResNet1D with Grad-CAM
---

# CardioSense

**Deep Learning-Based ECG Arrhythmia Classification Using the Adam Optimization Algorithm**

> Kingma & Ba (2015) — Adam: A Method for Stochastic Optimization (ICLR 2015, arXiv:1412.6980)

🔗 **[Live Demo](https://huggingface.co/spaces/inikunjrathi/CardioSense)**

---

## Overview

CardioSense is an end-to-end deep learning system that classifies ECG heartbeat signals into five clinically defined arrhythmia categories using a **1D Residual Neural Network (ResNet1D)**. The system is trained with the **Adam optimizer** and empirically compares it against four competing optimizers, grounding every result in the theoretical predictions of Kingma & Ba (2015).

The web app accepts a CSV row of 187 ECG signal values and returns the predicted arrhythmia class, confidence scores, probability breakdown, and a **Grad-CAM heatmap** highlighting which ECG regions drove the prediction.

---

## Achieved Results

### Overall Metrics — Test Set (15% stratified split, ~16,000 samples)

| Metric | Score | Target |
|---|---|---|
| **Accuracy** | **98.20%** | ≥ 98% ✓ |
| **Macro F1** | **91.40%** | ≥ 93% (close) |
| **Weighted F1** | **98.23%** | — |
| **Mean ROC-AUC** | **98.92%** | — |

### Per-Class Performance

| Class | Arrhythmia | Dataset % | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 0 — N | Normal | 82.8% | 99.31% | 98.79% | **99.05%** | 99.46% |
| 1 — S | Supraventricular | 2.5% | 77.74% | 84.17% | **80.83%** | 97.48% |
| 2 — V | Ventricular | 6.6% | 94.47% | 96.75% | **95.60%** | 99.36% |
| 3 — F | Fusion Beat | 0.7% | 80.23% | 85.19% | **82.63%** | 98.37% |
| 4 — Q | Unknown/Paced | 7.3% | 98.76% | 99.07% | **99.91%** | 99.91% |

### Optimizer Comparison — Best Validation Macro F1

| Optimizer | Best Val F1 | Epochs |
|---|---|---|
| SGD | 89.24% | 62 |
| SGD + Momentum | **92.17%** ← best | 48 |
| Adagrad | 91.68% | 47 |
| RMSProp | 90.61% | 100 |
| Adam | 90.75% (91.40% test) | 100 |

> **Key finding:** SGD+Momentum outperformed Adam on validation F1 under CosineAnnealingLR scheduling. Adam's adaptive steps interact differently with cosine restarts. However, Adam's test-set F1 (91.40%) confirms strong generalisation.

### Cross-Dataset Generalisation — PTB Diagnostic ECG Database

| Task | F1 | Dataset |
|---|---|---|
| Healthy vs. Myocardial Infarction | **99.64%** | PTB-DB (14,552 samples) |

The model fine-tuned on PTB-DB achieves 99.64% F1 on an entirely different dataset, validating that the learnt features generalise beyond MIT-BIH rhythm artefacts.

---

## Folder Structure

```
CardioSense/
├── api/                        # FastAPI backend (production web app)
│   ├── main.py                     — Routes: /, /research, /api/predict, /api/gradcam, /api/reference
│   ├── schemas.py                  — Pydantic request/response models
│   └── deps.py                     — Model singleton (lru_cache)
│
├── app/
│   └── utils/
│       └── inference.py            — load_model() + run_inference() shared by API
│
├── src/
│   ├── models/                 # Neural network architectures
│   │   ├── resnet1d.py             — Primary model (~790K params, 5-class)
│   │   ├── residual_block.py       — Conv1D residual block with skip connection
│   │   └── baseline_cnn.py         — 3-layer CNN baseline for comparison
│   ├── data/                   # Data pipeline
│   │   ├── loader.py               — Load MIT-BIH and PTB-DB CSV files
│   │   ├── preprocessor.py         — Z-score → IQR → SMOTE → reshape
│   │   ├── dataset.py              — PyTorch ECGDataset / DataLoader
│   │   └── splitter.py             — Stratified 85/15 train/val split
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py              — Optimizer-agnostic training loop
│   │   ├── focal_loss.py           — FocalLoss (γ=2.0, α=inverse class freq)
│   │   └── callbacks.py            — EarlyStopping, ModelCheckpoint
│   ├── optimizers/             # Optimizer comparison study
│   │   ├── optimizer_factory.py    — Build SGD/Adagrad/RMSProp/Adam by name
│   │   ├── comparison.py           — Run all 5 optimizers through identical Trainer
│   │   └── convergence_plots.py    — Loss/F1 convergence curves
│   ├── evaluation/             # Model evaluation
│   │   ├── metrics.py              — Accuracy, F1, ROC-AUC
│   │   ├── confusion_matrix.py     — 5×5 normalised heatmap
│   │   ├── roc_curves.py           — One-vs-rest ROC per class
│   │   └── report.py               — Full evaluation report generator
│   ├── explainability/         # Grad-CAM explainability
│   │   ├── gradcam.py              — Captum LayerGradCam on layer3[-1].conv2
│   │   └── heatmap_overlay.py      — RdYlBu_r heatmap overlaid on ECG waveform
│   ├── eda/                    # Exploratory data analysis utilities
│   │   ├── class_distribution.py
│   │   ├── waveform_visualizer.py
│   │   └── statistics.py
│   └── utils/
│       ├── device.py               — Auto-detect CUDA > MPS > CPU
│       ├── seed.py                 — Global random seed
│       └── io.py                   — Checkpoint save/load helpers
│
├── frontend/                   # Static HTML served by FastAPI
│   ├── index.html                  — Analytics dashboard (ECG upload, inference, Grad-CAM)
│   └── research.html               — Research documentation with interactive charts
│
├── notebooks/                  # Run in sequence — each builds on the previous
│   ├── 01_eda.ipynb                — Class distribution, waveform morphology
│   ├── 02_preprocessing.ipynb      — Pipeline validation, leakage checks
│   ├── 03_baseline_cnn.ipynb       — Baseline CNN training
│   ├── 04_resnet_training.ipynb    — Primary ResNet1D training (saves best.pth)
│   ├── 05_optimizer_comparison.ipynb — SGD / SGD+M / Adagrad / RMSProp / Adam
│   ├── 06_evaluation.ipynb         — Confusion matrix, ROC, full report
│   ├── 07_gradcam_explainability.ipynb — Grad-CAM clinical verification
│   └── 08_ptbdb_generalization.ipynb   — Cross-dataset transfer test
│
├── configs/
│   ├── data_config.yaml            — Dataset paths, class names, split ratios
│   ├── model_config.yaml           — Architecture specs
│   ├── training_config.yaml        — Adam hyperparams, FocalLoss, callbacks
│   └── optimizer_config.yaml       — All 5 optimizer configs
│
├── data/
│   ├── raw/                        — MIT-BIH + PTB-DB CSVs (Git LFS, ~360 MB)
│   ├── processed/
│   │   └── reference_waveforms.npz — One canonical waveform per class (for gallery)
│   └── sample_ecg_class{0-4}.csv   — One sample beat per class for quick testing
│
├── results/
│   ├── checkpoints/
│   │   └── resnet1d_adam_best.pth  — Production model (used by web app)
│   └── plots/
│       ├── confusion_matrices/confusion_matrix.png
│       ├── roc_curves/roc_all_classes.png
│       └── ptbdb_roc_curve.png
│
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_focal_loss.py
│   └── test_metrics.py
│
├── Dockerfile                  # HF Spaces deployment (port 7860)
├── requirements.txt            # Production deps (9 packages)
└── .github/workflows/
    └── sync-to-hf.yml          # Auto-deploy to HF Spaces on push to main
```

---

## Architecture

### ResNet1D (~790K parameters)

```
Input (1 × 187)
→ Stem: Conv1d(1→32, k=7) → BatchNorm → ReLU
→ Layer 1: ResidualBlock(32, stride=1) × 2    ← P-wave, QRS shape, T-wave
→ Layer 2: ResidualBlock(64, stride=2) × 2    ← beat intervals, rhythm
→ Layer 3: ResidualBlock(128, stride=2) × 2   ← arrhythmia patterns  ← Grad-CAM target
→ Global Average Pooling → Dense(128) → ReLU → Dense(5) → Softmax
```

**Training config:** Adam (lr=0.001, β₁=0.9, β₂=0.999) · FocalLoss (γ=2.0) · CosineAnnealingLR · EarlyStopping (patience=15 on val macro F1) · 100 epochs · batch=128

---

## Data Pipeline

```
Raw CSV (187 values/row)
→ Z-score normalise (per-sample)
→ IQR outlier removal (train only)
→ Stratified 85/15 split
→ SMOTE oversample (train only — no leakage)
→ Reshape to (N, 1, 187) for Conv1d
```

**Critical leakage rule:** SMOTE and outlier removal are applied to training data only. `preprocessor.preprocess()` enforces this boundary.

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web app
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000

# Run notebooks (in order)
jupyter notebook notebooks/

# Run tests
pytest tests/ -v
```

---

## Datasets

| Dataset | Source | Samples | Task |
|---|---|---|---|
| MIT-BIH Arrhythmia | [Kaggle: shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) | 109,446 | 5-class arrhythmia |
| PTB Diagnostic ECG | Same Kaggle page | 14,552 | Binary (healthy vs. MI) |

Download CSVs and place them in `data/raw/`.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 2.0 | ResNet1D, training, inference |
| Captum | ≥ 0.6 | Grad-CAM via LayerGradCam |
| FastAPI | ≥ 0.110 | REST API + static file serving |
| Chart.js | 4 | Interactive frontend charts |
| scikit-learn | ≥ 1.2 | Metrics, train/val split |
| imbalanced-learn | ≥ 0.10 | SMOTE oversampling |
| Matplotlib | ≥ 3.7 | Grad-CAM heatmap generation |

---

## References

1. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. arXiv:1412.6980
2. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMBS*
3. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *IEEE ICCV*. arXiv:1708.02002
4. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *IEEE CVPR*. arXiv:1512.03385
5. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *IEEE ICCV*. arXiv:1610.02391
7. Kokhlikyan, N., et al. (2020). Captum: A unified model interpretability library for PyTorch. arXiv:2009.07896
