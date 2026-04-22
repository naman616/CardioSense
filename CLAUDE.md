# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup (venv already at .venv/)
pip install -r requirements.txt          # if adding new deps

# Run Streamlit app
.venv/bin/streamlit run app/main.py

# Run all tests
.venv/bin/pytest tests/ -v

# Run a single test file
.venv/bin/pytest tests/test_models.py

# Run a specific test
.venv/bin/pytest tests/test_models.py::TestResNet1D::test_output_shape

# Run notebooks (sequence matters — each builds on the previous)
.venv/bin/jupyter notebook notebooks/
```

## Architecture

CardioSense is a 5-class ECG arrhythmia classifier (MIT-BIH dataset) with a Streamlit web app for interactive inference and Grad-CAM explainability. The codebase is split into a research library (`src/`), notebooks (`notebooks/`), and a web app (`app/`).

### Data flow

Raw CSV (187 ECG values per row) → `src/data/loader.py` → `src/data/preprocessor.py` (Z-score normalize → IQR outlier removal → SMOTE on train only → reshape to `(N, 187, 1)`) → `src/data/dataset.py` (`ECGDataset` permutes to `(N, 1, 187)` for PyTorch Conv1d) → DataLoaders.

**Critical leakage rule**: outlier removal and SMOTE apply to training data only. `preprocessor.preprocess()` enforces this.

### Model architecture

Two models in `src/models/`:

- **`BaselineCNN`** — 3-block Conv1D → Dense. Used only as the optimizer comparison baseline.
- **`ResNet1D`** — Primary model (~790K params). Stem (Conv1d 32, k=7) → `layer1` (32 filters, stride 1) → `layer2` (64 filters, stride 2) → `layer3` (128 filters, stride 2) → Global Average Pooling → Dense(128) → Dense(5) → Softmax. `layer3[-1].conv2` is the Grad-CAM target layer, exposed via `model.get_gradcam_target_layer()`.

### Training (`src/training/`)

`Trainer` is optimizer-agnostic — it accepts any `torch.optim.Optimizer`. The optimizer comparison study (`src/optimizers/comparison.py`) runs 5 optimizers (SGD, SGD+momentum, Adagrad, RMSProp, Adam) through the identical `Trainer` loop.

Key choices: FocalLoss (γ=2.0, α=inverse class frequency) to handle severe class imbalance (Normal=83%, Fusion=0.7%); early stopping on **val macro F1** (not accuracy); CosineAnnealingLR scheduler; checkpoints saved to `results/checkpoints/`.

### Explainability (`src/explainability/`)

`GradCAM1D` wraps Captum's `LayerGradCam` against `model.layer3[-1].conv2`. It returns a `(187,)` heatmap upsampled from the feature map via linear interpolation. Clinical verification: Class 1 (Supraventricular) should peak at P-wave region; Class 2 (Ventricular) should peak at QRS complex.

### Streamlit app (`app/`)

Entry point: `app/main.py`. Model is loaded once via `@st.cache_resource`. Input: CSV with exactly 187 numeric columns (no label). Inference uses per-sample Z-score normalization (same as training — no stored training stats needed). The app runs on CPU; no GPU assumed.

### Notebooks (run in order)

`01_eda` → `02_preprocessing` → `03_baseline_cnn` → `04_resnet_training` → `05_optimizer_comparison` → `06_evaluation` → `07_gradcam_explainability` → `08_ptbdb_generalization`

### Configs (`configs/`)

YAML files drive all hyperparameters. Key values:
- `data_config.yaml`: dataset paths, class names, preprocessing settings
- `model_config.yaml`: architecture specs (Grad-CAM target layer: `layer3`)
- `training_config.yaml`: Adam (lr=0.001, β1=0.9, β2=0.999), FocalLoss γ=2.0, batch=128, patience=15
- `optimizer_config.yaml`: all 5 optimizer configs for the comparison study

### Results layout

```
results/
  checkpoints/   # saved .pth files (best val macro F1 per run)
  logs/          # per-epoch metric CSVs
```

Checkpoint filename format: `{model}_{optimizer}_epoch{epoch:03d}_f1{f1:.4f}.pth`. The app expects `results/checkpoints/resnet1d_adam_best.pth`.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
