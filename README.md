
# Spillover Classifier

A machine learning pipeline for predicting viral host spillover events using XGBoost classifiers trained on ESM2 protein embeddings. This repository supports two classification tasks:

- **Family Spillover** — Classifies viral host family from ESM2 sequence embeddings
- **Transition Spillover** — Classifies viral transmission type using per-segment family probability features derived from ESM2 embeddings

---

## Overview

This script trains or evaluates XGBoost classifiers for two spillover prediction tasks:
    1. Family Spillover     — Classifies protein host tropism using EMS2 protein embeddings. 
                              A new model can be trained from scratch, or a pre-trained model 
                              can be loaded for evaluation.
    2. Transition Spillover — Classifies zoonotic spillover risk using per-segment family probability 
                              features derived from ESM2 embeddings. A new model is trained by default; 
                              a pre-trained model can optionally be loaded instead.

---

## Requirements

Python 3.8+ with the following dependencies:

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Train-test splitting, label encoding, evaluation metrics |
| `xgboost` | XGBoost classifier |
| `seaborn` | Confusion matrix heatmap visualization |
| `matplotlib` | Figure generation and saving |
| `joblib` | Model serialization and loading |

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib joblib
```

---

## Usage

### Family Spillover — Load and/or Evaluate Pre-Trained Model (10 Iterations)

```bash
python family_transition_spillover_model.py family_spillover \
    -c /path/to/merged_esm2_embeddings.csv \
    -m /path/to/family_spillover_model.pkl \
    -o /path/to/output_dir
```

### Transition Spillover — Train and/or Evaluate New Model (10 Iterations)

```bash
python family_transition_spillover_model.py transition_spillover \
    -c /path/to/merged_esm2_embeddings_with_probs.csv \
    -m /path/to/transition_spillover_model.pkl \
    -o /path/to/output_dir
```

---

## Input Data

### Family Spillover

| Flag | Argument | Description |
|---|---|---|
| `-c` | `--merged_esm2_embeddings_csv` | Path to merged ESM2 embeddings CSV. Must contain a `unique_ID` column and a `Host_Family` label column. |
| `-m` | `--family_spillover_model` | Path to the saved pre-trained family spillover model (`.pkl` or `.joblib`). |
| `-o` | `--output_path` | Output directory. |

### Transition Spillover

| Flag | Argument | Description |
|---|---|---|
| `-c` | `--merged_esm2_embeddings_csv_with_probs` | Path to merged ESM2 embeddings + family probabilities CSV. Must contain a `Classification` label column. |
| `-m` | `--transition_spillover_model` | Path to the saved pre-trained transition spillover model (`.pkl` or `.joblib`). |
| `-o` | `--output_path` | Root output directory. |

---

## Output Files

The following output subdirectories will be created during the running of the transition spillover mode:

```
output_dir/
├── train-test_splits/
├── classification_reports/
├── confusion_matrices/
├── count_confusion_matrices/
└── percentage_confusion_matrices/
```

| File | Description |
|---|---|
| `train-test_splits/train_test_split_combined_{i}.csv` | Full rows with predicted labels and probabilities per iteration |
| `classification_reports/classification_report_{i}.csv` | Precision, Recall, and F1-score per iteration |
| `confusion_matrices/confusion_matrix_{i}.csv` | Raw count confusion matrix per iteration |
| `count_confusion_matrices/transition_spillover_confusion_matrix_{i}.png` | Count-based heatmap figure per iteration |
| `percentage_confusion_matrices/transition_spillover_confusion_matrix_{i}.png` | Percentage heatmap figure per iteration |

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
