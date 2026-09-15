
# Spillover Classifier

A machine learning pipeline for predicting viral host spillover events using XGBoost classifiers trained on ESM-2 protein embeddings. This repository supports two classification tasks:

- **Protein Host Tropism** — Classifies viral host family from ESM-2 sequence embeddings
- **Zoonotic Risk Prediction** — Classifies viral transmission type using per-segment family probability features derived from ESM-2 embeddings

---

## Overview

This repository contains two scripts:

1. **`esm2_embedding_merge.py`** — Loads ESM-2 protein language model embeddings (.pt files) and merges them with influenza metadata to produce a single CSV for downstream model training.

2. **`family_transition_spillover_model.py`** — Trains and evaluates XGBoost classifiers for two spillover prediction tasks:
    - **Protein Host Tropism Model** — Classifies protein host tropism using ESM-2 protein embeddings as features. Evaluated using 10-fold Stratified Group K-Fold cross-validation with out-of-fold predictions.
    - **Zoonotic Risk Prediction Risk** — Classifies zoonotic spillover risk using per-segment family probability features derived from Model 1 outputs. Evaluated using 3-fold Stratified Group K-Fold cross-validation with out-of-fold predictions.

Both models include SHAP feature importance analysis and probability calibration analysis (Brier scores and reliability curves).

---

## Requirements

Python 3.8+ with the following dependencies:

| Package | Min Version | Purpose |
|---|---|---|
| `pandas` | >=2.0.0 | Data loading and manipulation |
| `numpy` | >=1.24.0 | Numerical operations |
| `scikit-learn` | >=1.3.0 | Stratified Group K-Fold CV, label encoding, evaluation metrics |
| `xgboost` | >=2.0.0 | XGBoost classifier |
| `shap` | >=0.46.0 | SHAP feature importance analysis |
| `matplotlib` | >=3.7.0 | Figure generation and saving |
| `seaborn` | >=0.13.0 | Confusion matrix heatmap visualization |
| `joblib` | >=1.3.0 | Model serialization and loading |
| `torch` | — | Loading ESM-2 embedding .pt files (esm2_embedding_merge.py only) |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Merge ESM-2 Embeddings with Metadata

```bash
python esm2_embedding_merge.py \
    --metadata   <path_to_metadata_csv> \
    --embeddings <path_to_embedding_directory> \
    --output     <path_to_output_csv> \
    --layer <esm2_layer_number> (default: 33)
```

### Protein Host Tropism Model — Load and/or Evaluate Pre-Trained Model (10 Iterations)

```bash
python family_transition_spillover_model.py family_spillover \
    -c /path/to/merged_esm2_embeddings.csv \
    -m /path/to/family_spillover_model.pkl \
    -o /path/to/output_dir \
    -k 10
```

### Zoonotic Risk Prediction Model — Train and/or Evaluate New Model (10 Iterations)

```bash
python family_transition_spillover_model.py transition_spillover \
    -c /path/to/merged_esm2_embeddings_with_probs.csv \
    -m /path/to/transition_spillover_model.pkl \
    -o /path/to/output_dir \
    -k 3
```

---

## Input Data

### ESM-2 Embedding Merge
| Argument | Description |
|---|---|
| `--metadata` | Path to metadata CSV. Must contain a unique_ID column and strain-level metadata. |
| `--embeddgins` | Path to directory containing ESM-2 .pt embedding files. |
| `--output` | Path for the output merged CSV file. |
| `--layer` | ESM-2 model layer to extract mean representations from (default: 33). |

### Protein Host Tropism Model

| Flag | Argument | Description |
|---|---|---|
| `-c` | `--merged_esm2_embeddings_csv` | Path to merged ESM2 embeddings CSV. Must contain a `unique_ID` column and a `Host_Family` label column. |
| `-m` | `--family_spillover_model` | Path to the saved pre-trained family spillover model (`.pkl` or `.joblib`). |
| `-o` | `--output_path` | Output directory. |
| `-k` | `--kfold` | Number of cross-validation folds (default: 10). |

### Zoonotic Risk Prediction Model

| Flag | Argument | Description |
|---|---|---|
| `-c` | `--merged_esm2_embeddings_csv_with_probs` | Path to merged ESM2 embeddings + family probabilities CSV. Must contain a `Classification` label column. |
| `-m` | `--transition_spillover_model` | Path to the saved pre-trained transition spillover model (`.pkl` or `.joblib`). |
| `-o` | `--output_path` | Root output directory. |
| `-k` | `--kfold` | Number of cross-validation folds (default: 3). |

---

## Output Files

### ESM-2 Embedding Merge

output.csv — Merged CSV containing metadata columns and ESM-2 embedding features

### Protein Host Tropism Model

```
output_dir/
├── saved_models/
├── train-test_splits/
├── classification_reports/
├── confusion_matrices/
├── oof_predictions_long.csv/
├── oof_predictions_wids.csv/
├── shap_analysis/
└── calibration_analysis/
```

| File | Description |
|---|---|
| `saved_models/family_spillover_model_fold_{i}.pkl` | Saved XGBoost model per fold |
| `train-test_splits/train_fold_{i}.csv` | Full rows with all columns for training samples per fold |
| `train-test_splits/test_fold_{i}.csv` | Full rows with all columns for test samples per fold |
| `train-test_splits/summary.csv` | Sample and group counts per fold |
| `classification_reports/classification_report_{i}.csv` | Precision, Recall, and F1-score per fold |
| `confusion_matrices/confusion_matrix_fold_{i}.csv` | Raw count confusion matrix per fold |
| `oof_predictions_long.csv` | Out-of-fold predictions in long format (all folds combined) |
| `oof_predictions_wide.csv` | Out-of-fold predictions in wide format (all folds combined) |
| `shap_analysis/global_feature_importance.csv` | Global SHAP feature importance across all folds |
| `shap_analysis/feature_importance_{class}.csv` | Per-class SHAP feature importance |
| `calibration/brier_scores.csv` | One-vs-rest Brier scores per class |
| `calibration/calibration_curve_data.csv` | Reliability curve data per class |

### Zoonotic Risk Prediction Model

```
output_dir/
├── saved_models/
├── train-test_splits/
├── classification_reports/
├── confusion_matrices/
├── count_confusion_matrices/
├── percentage_confusion_matrices/
├── shap_analysis/
└── calibration_analysis/
```

| File | Description |
|---|---|
| `saved_models/transition_spillover_model_fold_{i}.pkl` | Saved XGBoost model per fold |
| `train-test_splits/train_test_split_combined_{i}.csv` | Full rows with predicted labels and probabilities per fold |
| `classification_reports/classification_report_{i}.csv` | Precision, Recall, and F1-score per fold |
| `confusion_matrices/confusion_matrix_fold_{i}.csv` | Raw count confusion matrix per fold |
| `count_confusion_matrices/transition_spillover_confusion_matrix_{i}.png` | Count-based heatmap figure per fold |
| `percentage_confusion_matrices/transition_spillover_confusion_matrix_{i}.png` | Percentage heatmap figure per fold |
| `shap_analysis/global_feature_importance.csv` | Global SHAP feature importance across all folds |
| `shap_analysis/feature_importance_{class}.csv` | Per-class SHAP feature importance |
| `calibration/brier_scores.csv` | One-vs-rest Brier scores per class |
| `calibration/calibration_curve_data.csv` | Reliability curve data per class |

---


## Cross-Validation Strategy

Both models use **Stratified Group K-Fold cross-validation** to prevent data leakage:

- Identical protein sequences within each genome segment are grouped by exact sequence identity.
- Groups are used as the splitting unit, ensuring that identical or near-identical sequences are never split across training and testing sets.
- Stratification maintains class balance across folds.
- The number of folds is configurable via the `-k` flag (default: 10 for Model 1, 3 for Model 2).
- Out-of-fold predictions are generated for every sample, providing unbiased performance estimates.

---

## Notes

- The **family spillover** mode supports both loading a pre-trained model for
  evaluation and training a new model from scratch — it is not limited to
  pre-trained model loading only.
- Both the **family spillover** and **transition spillover** modes train a new XGBClassifier by default. To use
  a pre-trained model instead, uncomment the corresponding `joblib.load()` line the proper function.
- These two modes do **not** use a fixed random seed per iteration — this is intentional to evaluate run-to-run variance across 10 independent splits.
- To make splits reproducible, add the `random_state=42` line in `train_test_split()` inside `family_spillover()` and/or `transition_spillover()`.
- The output subdirectories will be created if they are not manually created.
- Model 1 outputs (per-segment family probabilities in wide format) serve as input features for Model 2.
- SHAP analysis and calibration analysis are integrated into the model script and run automatically after cross-validation.
- Duplicate sequences are intentionally retained, as identical sequences across independent isolates reflect the natural prevalence of circulating viral lineages. The grouping strategy prevents data leakage while preserving this biological reality.

---

## Disclaimer

This code was developed at **Noblis** in support of the associated manuscript publication. Upon submission, this repository is provided as-is for reproducibility purposes.

> **Noblis is not responsible for maintaining or updating this codebase following manuscript submission.** Users who wish to build upon or adapt this code do so at their own discretion.

For questions related to the methodology described in the manuscript, please refer to the published paper and its supplementary materials.

---

## Citation

If you use this code, please cite the manuscript associated with this
repository.

---

## License

This project is licensed under the Apache License 2.0. See LICENSE for details.