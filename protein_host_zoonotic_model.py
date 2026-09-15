
"""
protein_host_zoonotic_model.py

This script trains and evaluates XGBoost classifiers for two spillover prediction tasks
using Stratified Group K-Fold Cross-Validation to prevent data leakage:
    1. Family Spillover     — Classifies protein host tropism using ESM2 protein embeddings.
                              Generates out-of-fold probability predictions and pivots them
                              to wide format (one row per isolate) for use by the lower model.
    2. Transition Spillover — Classifies zoonotic spillover risk using per-segment family
                              probability features derived from ESM2 embeddings.

Post-training analyses (run automatically after cross-validation):
    - Calibration Analysis  — One-vs-Rest Brier scores and reliability curves for both models.
    - SHAP Analysis         — Global and per-class feature importance for Model 2 only.

Cross-validation ensures:
    - No identical sequences appear in both training and testing folds (group-aware splitting)
    - Class balance is maintained across folds (stratification)
    - Every sample is tested exactly once
    - Out-of-fold predictions guarantee no leakage between upper and lower models

Usage:
    python protein_host_zoonotic_model.py family_spillover \
        -c /path/to/merged_esm2_embeddings.csv \
        -o /path/to/output_dir \
        -k 10

    python protein_host_zoonotic_model.py transition_spillover \
        -c /path/to/merged_esm2_embeddings_with_probs.csv \
        -o /path/to/output_dir \
        -k 3

Notes:
    - Both classifiers use StratifiedGroupKFold to prevent data leakage from
      duplicate sequences appearing in both train and test sets.
    - Groups are defined by sequence identity: rows with identical sequences
      within the same protein (upper model) or identical feature vectors
      (lower model) are assigned the same group and never split across folds.
    - The family spillover model generates out-of-fold predictions that are
      pivoted to wide format (one row per isolate, at least 90 probability columns)
      for use as input to the transition spillover model.
    - Output directories for CSVs, classification reports, confusion matrix
      figures, and saved models are auto-created.

Authors: Longest et al., 2026
Date: September 10, 2026
"""

import os
import joblib
import argparse

import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Helper function to safely index lists/arrays (avoids bracket corruption)
# ---------------------------------------------------------------------------

def get_item(obj, idx):
    """Safely index into a list, tuple, or array."""
    return obj[idx]


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def user_args():
    """
    Parse command-line arguments for selecting and configuring the spillover mode.

    Subcommands:
        family_spillover      -- Train and evaluate a family spillover classifier (Model 1)
                                 using Stratified Group K-Fold Cross-Validation.
        transition_spillover  -- Train and evaluate a transition spillover classifier (Model 2)
                                 using Stratified Group K-Fold Cross-Validation.

    Returns:
        argparse.Namespace: Parsed arguments containing mode and associated file paths.
    """
    parser = argparse.ArgumentParser(
        description="Trains and evaluates either a family or transition spillover XGBoost classifier "
                    "using Stratified Group K-Fold Cross-Validation."
    )

    sub_parser = parser.add_subparsers(
        title="Select model type to train or evaluate",
        dest="mode",
        required=True
    )

    # --- Family Spillover Subparser ---
    parser_family_spillover = sub_parser.add_parser(
        "family_spillover",
        help="Trains and evaluates a family spillover classifier (Model 1) using Stratified Group K-Fold CV."
    )
    parser_family_spillover.add_argument(
        "-c", "--merged_esm2_embeddings_csv",
        required=True,
        help='Path to the merged ESM2 embeddings CSV (e.g., "/data/merged_with_esm2.csv")'
    )
    parser_family_spillover.add_argument(
        "-m", "--family_spillover_model",
        required=False,
        help="Optional path to a saved pre-trained family spillover model (.pkl or .joblib). "
             "If not provided, a new model will be trained from scratch."
    )
    parser_family_spillover.add_argument(
        "-o", "--output_path",
        required=True,
        help="Directory path to all the outputs (reports & confusion matrices)."
    )
    parser_family_spillover.add_argument(
        "-k", "--n_folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)."
    )

    # --- Transition Spillover Subparser ---
    parser_transition_spillover = sub_parser.add_parser(
        "transition_spillover",
        help="Trains and evaluates a transition spillover classifier (Model 2) using Stratified Group K-Fold CV."
    )
    parser_transition_spillover.add_argument(
        "-c", "--merged_esm2_embeddings_csv_with_probs",
        required=True,
        help='Path to the merged ESM2 embeddings + family probabilities CSV.'
    )
    parser_transition_spillover.add_argument(
        "-m", "--transition_spillover_model",
        required=False,
        help="Optional path to a saved pre-trained transition spillover model (.pkl or .joblib). "
             "If not provided, a new model will be trained from scratch."
    )
    parser_transition_spillover.add_argument(
        "-o", "--output_path",
        required=True,
        help="Directory path to save all output files (reports, matrices, splits)."
    )
    parser_transition_spillover.add_argument(
        "-k", "--n_folds",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3)."
    )

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Calibration Analysis (used by both Model 1 and Model 2)
# ---------------------------------------------------------------------------

def calibration_analysis(oof_probabilities, y_true, class_names, prob_columns,
                         target_col_name, output_path, model_name, n_bins=10,
                         fold_data=None):
    """
    Compute one-vs-rest Brier scores and reliability curves from out-of-fold predictions.

    Args:
        oof_probabilities (np.ndarray): OOF probability matrix (n_samples x n_classes).
        y_true (pd.Series):            True labels (string class names).
        class_names (list):            Ordered list of class names.
        prob_columns (list):           Probability column names.
        target_col_name (str):         Name of the target column (e.g., "Host_Family" or "Classification").
        output_path (str):             Root output directory.
        model_name (str):              Model name for titles and filenames.
        n_bins (int):                  Number of bins for reliability curves.
        fold_data (list, optional):    List of dicts with fold-level data for per-fold stability check.

    Outputs:
        - CSV: calibration_analysis/brier_scores/brier_scores_per_class.csv
        - CSV: calibration_analysis/reliability_curves/calibration_curve_data.csv
        - PNG: calibration_analysis/reliability_curves/reliability_curves_all_classes.png
        - PNG: calibration_analysis/reliability_curves/reliability_curve_{class}.png (per class)
        - PNG: calibration_analysis/brier_scores/brier_scores.png
    """
    cal_path = os.path.join(output_path, "calibration_analysis")
    os.makedirs(os.path.join(cal_path, "reliability_curves"), exist_ok=True)
    os.makedirs(os.path.join(cal_path, "brier_scores"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Calibration Analysis — {model_name}")
    print(f"{'='*60}")

    # Build a DataFrame from the OOF probabilities for convenience
    df = pd.DataFrame(oof_probabilities, columns=prob_columns)
    df[target_col_name] = y_true.values

    # ---- 1. One-vs-Rest Brier Scores ----
    print(f"\n{'='*60}")
    print("Computing One-vs-Rest Brier Scores")
    print(f"{'='*60}")

    brier_results = []

    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_name).astype(int)
        y_prob = oof_probabilities[:, i]

        brier = brier_score_loss(y_true_binary, y_prob)
        prevalence = y_true_binary.mean()

        brier_results.append({
            "class": class_name,
            "brier_score": brier,
            "n_positive": int(y_true_binary.sum()),
            "n_total": len(y_true_binary),
            "prevalence": prevalence,
        })

        print(f"  {class_name:25s} | Brier: {brier:.6f} | Prevalence: {prevalence:.4f} | n={y_true_binary.sum()}")

    brier_df = pd.DataFrame(brier_results)
    brier_df = brier_df.sort_values("brier_score", ascending=True).reset_index(drop=True)
    brier_df.to_csv(os.path.join(cal_path, "brier_scores", "brier_scores_per_class.csv"), index=False)

    print(f"\n  Mean Brier Score: {brier_df['brier_score'].mean():.6f}")
    print(f"  Median Brier Score: {brier_df['brier_score'].median():.6f}")
    best_row = brier_df.iloc[0]
    worst_row = brier_df.iloc[-1]
    print(f"  Best calibrated: {best_row['class']} ({best_row['brier_score']:.6f})")
    print(f"  Worst calibrated: {worst_row['class']} ({worst_row['brier_score']:.6f})")
    print(f"\nBrier scores saved to: {cal_path}/brier_scores/brier_scores_per_class.csv")

    # ---- 2. Reliability Curves ----
    print(f"\n{'='*60}")
    print("Generating Reliability Curves")
    print(f"{'='*60}")

    # --- Individual reliability curves per class ---
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_name).astype(int)
        y_prob = oof_probabilities[:, i]

        if y_true_binary.sum() < 20:
            print(f"  Skipping {class_name}: only {y_true_binary.sum()} positive samples")
            continue

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob, n_bins=n_bins, strategy="uniform"
        )

        brier_val = float(brier_df[brier_df["class"] == class_name]["brier_score"].values[0])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                color="darkred", label=f"{class_name} (Brier={brier_val:.4f})",
                linewidth=2, markersize=8)

        ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title(f"Reliability Curve — {class_name}\n(One-vs-Rest, {model_name})",
                     fontsize=12, pad=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(cal_path, "reliability_curves", f"reliability_curve_{class_name}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  Saved: reliability_curve_{class_name}.png")

    # --- Combined reliability curve (all classes on one plot) ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    calibration_data = []

    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == class_name).astype(int)
        y_prob = oof_probabilities[:, i]

        if y_true_binary.sum() < 20:
            continue

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob, n_bins=n_bins, strategy="uniform"
        )

        brier_val = float(brier_df[brier_df["class"] == class_name]["brier_score"].values[0])
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                color=get_item(colors, i), label=f"{class_name} (Brier={brier_val:.4f})",
                linewidth=1.5, markersize=6)

        for bin_idx in range(len(fraction_of_positives)):
            calibration_data.append({
                "class": class_name,
                "bin": bin_idx + 1,
                "mean_predicted_probability": get_item(mean_predicted_value, bin_idx),
                "fraction_of_positives": get_item(fraction_of_positives, bin_idx),
            })

    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Reliability Curves — All Classes\n(One-vs-Rest, {model_name})",
                 fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=9, ncol=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(cal_path, "reliability_curves", "reliability_curves_all_classes.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"\n  Saved: reliability_curves_all_classes.png")

    # Save calibration curve data to CSV
    calibration_df = pd.DataFrame(calibration_data)
    calibration_df.to_csv(
        os.path.join(cal_path, "reliability_curves", "calibration_curve_data.csv"), index=False
    )

    # --- Brier Scores Bar Chart ---
    brier_sorted = brier_df.sort_values("brier_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(brier_sorted)), brier_sorted["brier_score"].values)
    ax.set_yticks(range(len(brier_sorted)))
    ax.set_yticklabels(brier_sorted["class"].values, fontsize=10)
    ax.set_xlabel("Brier Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Class", fontsize=12, fontweight="bold")
    ax.set_title(f"Brier Scores — One-vs-Rest ({model_name})", fontsize=14, fontweight="bold")

    for idx_val in range(len(brier_sorted)):
        val = get_item(brier_sorted["brier_score"].values, idx_val)
        ax.text(val + 0.0005, idx_val, f"{val:.4f}", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(cal_path, "brier_scores", "brier_scores.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: brier_scores.png")

    # ---- 3. Per-Fold Brier Score Stability (if fold data provided) ----
    if fold_data is not None:
        print(f"\n{'='*60}")
        print("Per-Fold Brier Score Stability")
        print(f"{'='*60}")

        fold_brier_results = []
        for fold_info in fold_data:
            fold_num = fold_info["fold"]
            fold_y_true = fold_info["y_true"]
            fold_y_prob = fold_info["y_prob"]

            for i, class_name in enumerate(class_names):
                y_binary = (fold_y_true == class_name).astype(int)
                y_p = fold_y_prob[:, i]
                brier = brier_score_loss(y_binary, y_p)
                fold_brier_results.append({
                    "fold": fold_num,
                    "class": class_name,
                    "brier_score": brier,
                    "n_samples": len(fold_y_true),
                })

        fold_brier_df = pd.DataFrame(fold_brier_results)
        fold_pivot = fold_brier_df.pivot(index="class", columns="fold", values="brier_score")
        fold_pivot["mean"] = fold_pivot.mean(axis=1)
        fold_pivot["std"] = fold_pivot.std(axis=1)

        print(fold_pivot.to_string())
        fold_pivot.to_csv(os.path.join(cal_path, "brier_scores", "brier_scores_per_fold.csv"))

    print(f"\n{'='*60}")
    print(f"Calibration Analysis Complete — {model_name}")
    print(f"{'='*60}")
    print(f"All outputs saved to: {cal_path}")


# ---------------------------------------------------------------------------
# SHAP Analysis (Model 2 only)
# ---------------------------------------------------------------------------

def shap_analysis(saved_model_dir, X_test_per_fold, x_columns, class_names,
                  output_path, model_name, n_folds, model_prefix="transition_spillover"):
    """
    Compute SHAP values across all folds using saved models and full test sets.

    Args:
        saved_model_dir (str):    Directory containing saved fold models.
        X_test_per_fold (list):   List of DataFrames, one X_test per fold.
        x_columns (list):         Feature column names.
        class_names (list):       Ordered class names.
        output_path (str):        Root output directory.
        model_name (str):         Model name for titles.
        n_folds (int):            Number of folds.
        model_prefix (str):       Prefix for saved model filenames.

    Outputs:
        - CSV: shap_analysis/global_feature_importance.csv
        - CSV: shap_analysis/feature_importance_{class_name}.csv per class
        - CSV: shap_analysis/importance_by_segment.csv
        - CSV: shap_analysis/importance_by_family.csv
        - PNG: shap_analysis/global_shap_bar_top30.png
        - PNG: shap_analysis/shap_multiclass_bar_top30.png
        - PNG: shap_analysis/shap_beeswarm_{class}.png per class
        - PNG: shap_analysis/shap_importance_by_segment.png
        - PNG: shap_analysis/shap_importance_by_family.png
    """
    shap_dir = os.path.join(output_path, "shap_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    n_classes = len(class_names)

    print(f"\n{'='*60}")
    print(f"SHAP Analysis — {model_name}")
    print(f"{'='*60}")

    all_shap_values = [[] for _ in range(n_classes)]
    all_X_samples = []

    for fold in range(1, n_folds + 1):
        print(f"\n--- Fold {fold} ---")
        model_path = os.path.join(saved_model_dir, f"{model_prefix}_model_fold_{fold}.pkl")

        if not os.path.exists(model_path):
            print(f"  WARNING: Model not found at {model_path}, skipping fold.")
            continue

        model = joblib.load(model_path)
        print(f"  Model loaded: {model_path}")

        X_test = get_item(X_test_per_fold, fold - 1)
        print(f"  Computing SHAP on {len(X_test)} test samples (full test set)...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        print(f"  SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"  List length: {len(shap_values)}, first element shape: {get_item(shap_values, 0).shape}")
        else:
            print(f"  Array shape: {shap_values.shape}")

        # Handle both old format (list of arrays) and new format (3D array)
        for i in range(n_classes):
            if isinstance(shap_values, list):
                all_shap_values[i].append(get_item(shap_values, i))
            else:
                all_shap_values[i].append(shap_values[:, :, i])

        all_X_samples.append(X_test)
        print(f"  Done. Fold {fold} complete.")

    # --- Concatenate SHAP values across all folds ---
    print(f"\n{'='*60}")
    print("Concatenating SHAP values across all folds...")
    combined_shap_values = [np.concatenate(get_item(all_shap_values, i), axis=0) for i in range(n_classes)]
    combined_X = pd.concat(all_X_samples, ignore_index=True)

    print(f"  Total SHAP samples: {get_item(combined_shap_values, 0).shape[0]}")
    print(f"  Features per sample: {get_item(combined_shap_values, 0).shape[1]}")
    print(f"  Number of classes: {len(combined_shap_values)}")

    # --- Global Feature Importance ---
    print("Computing global feature importance...")
    stacked_shap = np.stack(combined_shap_values, axis=0)  # (n_classes, n_samples, n_features)
    print(f"  Stacked SHAP shape: {stacked_shap.shape}")

    mean_abs_shap = np.abs(stacked_shap).mean(axis=(0, 1))  # (n_features,)
    print(f"  mean_abs_shap shape: {mean_abs_shap.shape}")
    print(f"  x_columns length: {len(x_columns)}")

    global_importance = pd.DataFrame({
        "feature": x_columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    global_importance.to_csv(os.path.join(shap_dir, "global_feature_importance.csv"), index=False)

    print(f"\nTop 20 features:")
    print(global_importance.head(20).to_string(index=False))

    # --- Figure 1: Global SHAP Bar Plot (top 30 features) ---
    print("Generating global SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    top_30 = global_importance.head(30)
    ax.barh(range(len(top_30)), top_30["mean_abs_shap"].values, color="darkred")
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(top_30["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} — Top 30 Global Feature Importance (SHAP, All Folds)", fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "global_shap_bar_top30.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: global_shap_bar_top30.png")

    # --- Figure 2: Multi-class SHAP bar plot (all classes stacked) ---
    print("Generating multi-class SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(14, 10))
    shap.summary_plot(
        combined_shap_values,
        combined_X,
        class_names=list(class_names),
        max_display=30,
        show=False,
        plot_type="bar"
    )
    plt.title(f"{model_name} — SHAP Feature Importance by Class (All Folds)", fontsize=14, pad=15)
    plt.savefig(os.path.join(shap_dir, "shap_multiclass_bar_top30.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_multiclass_bar_top30.png")

    # --- Figure 3: Per-Class SHAP Beeswarm Plots ---
    print("Generating per-class SHAP beeswarm plots...")
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            get_item(combined_shap_values, i),
            combined_X,
            max_display=20,
            show=False,
            plot_type="dot"
        )
        plt.title(f"{model_name} — SHAP Summary: {class_name} (All Folds)", fontsize=14, pad=15)
        plt.savefig(
            os.path.join(shap_dir, f"shap_beeswarm_{class_name}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  Saved: shap_beeswarm_{class_name}.png")

    # --- Figure 4: Importance by Segment (aggregated) ---
    print("Generating importance by segment figure...")

    PROB_COLS = [
        "anatidae_probability", "bovidae_probability", "canidae_probability",
        "equidae_probability", "felidae_probability", "hominidae_probability",
        "laridae_probability", "phasianidae_probability", "suidae_probability"
    ]
    SEGMENT_ORDER = ["NA", "HA", "PB1_1", "PB1_2", "PB2_1", "PA_1", "PA_2",
                     "NS_1", "NS_2", "M_1", "M_2", "NP"]

    segment_importance = {seg: 0 for seg in SEGMENT_ORDER}
    family_importance = {prob.replace("_probability", ""): 0 for prob in PROB_COLS}

    for _, row in global_importance.iterrows():
        feature = row["feature"]
        shap_val = row["mean_abs_shap"]
        for prob_col in PROB_COLS:
            if feature.startswith(prob_col):
                family = prob_col.replace("_probability", "")
                segment = feature.replace(f"{prob_col}_", "")
                segment_importance[segment] += shap_val
                family_importance[family] += shap_val
                break

    # Segment bar plot
    seg_df = pd.DataFrame(list(segment_importance.items()), columns=["segment", "total_shap"])
    seg_df = seg_df.sort_values("total_shap", ascending=False)
    seg_df.to_csv(os.path.join(shap_dir, "importance_by_segment.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(seg_df["segment"], seg_df["total_shap"], color="steelblue")
    ax.set_xlabel("Protein Segment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} — SHAP Importance by Protein Segment (All Folds)", fontsize=14, pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_importance_by_segment.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_importance_by_segment.png")

    # --- Figure 5: Importance by Host Family (aggregated) ---
    fam_df = pd.DataFrame(list(family_importance.items()), columns=["family", "total_shap"])
    fam_df = fam_df.sort_values("total_shap", ascending=False)
    fam_df.to_csv(os.path.join(shap_dir, "importance_by_family.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(fam_df["family"], fam_df["total_shap"], color="darkred")
    ax.set_xlabel("Host Family", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} — SHAP Importance by Host Family Probability (All Folds)", fontsize=14, pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_importance_by_family.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_importance_by_family.png")

    # --- Per-class feature importance CSVs ---
    print("Saving per-class feature importance CSVs...")
    for i, class_name in enumerate(class_names):
        class_shap = get_item(combined_shap_values, i)
        class_importance = pd.DataFrame({
            "feature": x_columns,
            "mean_abs_shap": np.abs(class_shap).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        class_importance.to_csv(os.path.join(shap_dir, f"feature_importance_{class_name}.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"SHAP Analysis Complete — {model_name}")
    print(f"{'='*60}")
    print(f"All outputs saved to: {shap_dir}")
    print(f"Figures:")
    print(f"  1. global_shap_bar_top30.png — Top 30 features by mean |SHAP|")
    print(f"  2. shap_multiclass_bar_top30.png — Feature importance stacked by class")
    print(f"  3. shap_beeswarm_{{class}}.png — Per-class beeswarm (dot) plots")
    print(f"  4. shap_importance_by_segment.png — Aggregated importance by genome segment")
    print(f"  5. shap_importance_by_family.png — Aggregated importance by host family")
    print(f"CSVs:")
    print(f"  - global_feature_importance.csv")
    print(f"  - importance_by_segment.csv")
    print(f"  - importance_by_family.csv")
    print(f"  - feature_importance_{{class}}.csv (one per class)")


# ---------------------------------------------------------------------------
# Family Spillover Classifier (with Stratified Group K-Fold CV)
# ---------------------------------------------------------------------------

def family_spillover(merged_esm2_embeddings_csv, family_spillover_model, output_path, n_folds):
    """
    Train and evaluate a family spillover XGBoost classifier using Stratified Group K-Fold
    Cross-Validation to prevent data leakage from duplicate sequences.

    Groups are defined by identical sequences within the same protein (refined_segment).
    All rows sharing the same sequence for the same protein are guaranteed to remain
    in the same fold (either all in train or all in test, never split).

    After cross-validation, out-of-fold predictions are generated and pivoted to wide
    format (one row per isolate, 108 probability columns) for use by the transition
    spillover model. Calibration analysis (Brier scores + reliability curves) is run
    automatically after predictions are generated.

    Args:
        merged_esm2_embeddings_csv (str): Path to the merged ESM2 embeddings CSV file.
        family_spillover_model (str):     Path to a saved pre-trained family spillover model
                                          file, or None to train a new model from scratch.
        output_path (str):                Directory path to save outputs.
        n_folds (int):                    Number of cross-validation folds.

    Outputs:
        - PKL:  saved_models/family_spillover_model_fold_{fold}.pkl per fold
        - CSV:  classification_reports/classification_report_fold_{fold}.csv per fold
        - CSV:  cross_validation_summary.csv with aggregated metrics
        - CSV:  out_of_fold_predictions_long.csv (one row per protein per isolate with probabilities)
        - CSV:  out_of_fold_predictions_wide.csv (one row per isolate, 108 probability columns)
        - CSV:  train_test_split/train_fold_{fold}.csv per fold
        - CSV:  train_test_split/test_fold_{fold}.csv per fold
        - CSV:  train_test_split/summary.csv (sample counts per fold)
        - PNG:  confusion_matrices/family_spillover_confusion_matrix_fold_{fold}.png per fold
        - CSV:  calibration_analysis/brier_scores/brier_scores_per_class.csv
        - PNG:  calibration_analysis/reliability_curves/reliability_curves_all_classes.png
        - PNG:  calibration_analysis/brier_scores/brier_scores.png
    """
    # --- Output Directory Setup ---
    subdirs = ["classification_reports", "confusion_matrices", "saved_models", "train_test_split"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    print("Reading CSV...")
    cleaned_df = pd.read_csv(merged_esm2_embeddings_csv)

    # Remove duplicate entries based on unique sample identifier
    cleaned_df = cleaned_df.drop_duplicates(subset=["unique_ID"])

    # Warn and drop samples with missing host family labels
    missing_labels = cleaned_df["Host_Family"].isnull().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} samples have missing labels and will be dropped.")
        cleaned_df = cleaned_df.dropna(subset=["Host_Family"])

    # Reset index for clean alignment with out-of-fold predictions
    cleaned_df = cleaned_df.reset_index(drop=True)

    # Define non-feature columns to exclude from the feature matrix
    non_numeric_cols = ['Isolate_Name', 'Host', 'Common_Name', 'Segment',
                        'Genotype', 'Country', 'Collection_Date', 'Sequence', 'Seq_Len',
                        'data_source', 'GISAID_Accession', 'NCBI_Accession', 'length_percent',
                        'isoform', 'refined_segment', 'unique_ID', 'Family_Group',
                        'Classification']

    print(cleaned_df.columns[0:20])

    # --- Group Assignment ---
    print("Assigning sequence groups for cross-validation...")
    cleaned_df["group_id"] = cleaned_df.groupby(["refined_segment", "Sequence"]).ngroup()
    print(f"Total unique groups: {cleaned_df['group_id'].nunique()}")

    # Build feature matrix (X) and label vector (y)
    X = cleaned_df.drop(columns=non_numeric_cols + ["Host_Family", "group_id"])
    y = cleaned_df["Host_Family"]
    groups = cleaned_df["group_id"]

    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    original_class_names = label_encoder.classes_

    # --- Out-of-Fold Prediction Storage ---
    oof_probabilities = np.zeros((len(X), len(original_class_names)))

    # --- Stratified Group K-Fold Cross-Validation ---
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_reports = []
    fold_summary = []
    fold_calibration_data = []

    print(f"\nRunning {n_folds}-Fold Stratified Group Cross-Validation...")
    print("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold + 1} of {n_folds} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Verify no group leakage
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Group overlap between train/test: {len(overlap)} (should be 0)")

        # --- Save Train/Test Split CSVs ---
        train_rows = cleaned_df.iloc[train_idx].copy()
        test_rows = cleaned_df.iloc[test_idx].copy()

        train_rows.to_csv(
            os.path.join(output_path, "train_test_split", f"train_fold_{fold + 1}.csv"),
            index=False
        )
        test_rows.to_csv(
            os.path.join(output_path, "train_test_split", f"test_fold_{fold + 1}.csv"),
            index=False
        )
        print(f"  Train/test split saved to {output_path}/train_test_split/")

        # Record fold summary
        fold_summary.append({
            "fold": fold + 1,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "total_samples": len(train_idx) + len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups)
        })

        # Encode labels
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # --- Load Pre-Trained Model or Train New Model ---
        if family_spillover_model is not None:
            print("  Loading pre-trained model...")
            model = joblib.load(os.path.abspath(family_spillover_model))
        else:
            print("  Training new model...")
            model = XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train_encoded)

            # Save trained model for this fold
            model_save_path = os.path.join(
                output_path, "saved_models", f"family_spillover_model_fold_{fold + 1}.pkl"
            )
            joblib.dump(model, model_save_path)
            print(f"  Model saved to {model_save_path}")

        # --- Prediction ---
        y_pred_proba = model.predict_proba(X_test)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # --- Store Out-of-Fold Predictions ---
        oof_probabilities[test_idx] = y_pred_proba

        # --- Store fold-level data for per-fold calibration stability ---
        fold_calibration_data.append({
            "fold": fold + 1,
            "y_true": y_test.reset_index(drop=True),
            "y_prob": y_pred_proba,
        })

        # --- Classification Report ---
        report = classification_report(y_test, y_pred, target_names=original_class_names, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=original_class_names))

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_path}/classification_reports/classification_report_fold_{fold + 1}.csv")

        all_reports.append(report_df)

        # --- Confusion Matrix Heatmap ---
        print("  Outputting confusion matrix...")
        cm = confusion_matrix(y_test, y_pred, labels=original_class_names)
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(
            cm_percent,
            xticklabels=original_class_names,
            yticklabels=original_class_names,
            cmap="YlOrRd",
            cbar_kws={"label": "Percentage (%)"},
            vmin=0,
            vmax=100,
            annot=False,
            ax=ax
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{cm_percent[i, j]:.1f}%",
                    ha="center", va="center",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("Percentage (%)", fontsize=10)

        ax.set_xlabel("Predicted Labels", fontsize=12, labelpad=10, fontweight="bold")
        ax.set_ylabel("True Labels", fontsize=12, labelpad=10, fontweight="bold")
        ax.set_xticklabels(original_class_names, rotation=45, ha="right", fontsize=9, fontstyle="italic")
        ax.set_yticklabels(original_class_names, rotation=0, fontsize=9, fontstyle="italic")

        plt.title(
            f"Family Spillover Classifier Confusion Matrix — Fold {fold + 1}",
            fontsize=12, pad=20
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, "confusion_matrices", f"family_spillover_confusion_matrix_fold_{fold + 1}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    # --- Train/Test Split Summary ---
    summary_split_df = pd.DataFrame(fold_summary)
    summary_split_df.to_csv(
        os.path.join(output_path, "train_test_split", "summary.csv"),
        index=False
    )
    print(f"\nTrain/test split summary saved to {output_path}/train_test_split/summary.csv")
    print(summary_split_df.to_string(index=False))

    # --- Cross-Validation Summary ---
    print("\n" + "=" * 50)
    print("Cross-Validation Summary")
    print("=" * 50)

    summary_metrics = {}
    for class_name in list(original_class_names) + ["accuracy", "macro avg", "weighted avg"]:
        class_metrics = []
        for report_df in all_reports:
            if class_name in report_df.index:
                class_metrics.append(report_df.loc[class_name])
        if class_metrics:
            class_metrics_df = pd.DataFrame(class_metrics)
            summary_metrics[class_name] = {
                "precision_mean": class_metrics_df["precision"].mean(),
                "precision_std": class_metrics_df["precision"].std(),
                "recall_mean": class_metrics_df["recall"].mean(),
                "recall_std": class_metrics_df["recall"].std(),
                "f1-score_mean": class_metrics_df["f1-score"].mean(),
                "f1-score_std": class_metrics_df["f1-score"].std(),
            }

    summary_df = pd.DataFrame(summary_metrics).transpose()
    summary_df.to_csv(os.path.join(output_path, "cross_validation_summary.csv"))
    print(summary_df.to_string())
    print(f"\nSummary saved to {output_path}/cross_validation_summary.csv")

    # --- Out-of-Fold Predictions: Long Format ---
    print("\n" + "=" * 50)
    print("Generating Out-of-Fold Predictions...")
    print("=" * 50)

    prob_columns = [f"{class_name}_probability" for class_name in original_class_names]
    oof_prob_df = pd.DataFrame(oof_probabilities, columns=prob_columns)

    oof_long_df = pd.concat([
        cleaned_df[["Isolate_Name", "refined_segment", "Host_Family", "Classification"]].reset_index(drop=True),
        oof_prob_df
    ], axis=1)

    oof_long_path = os.path.join(output_path, "out_of_fold_predictions_long.csv")
    oof_long_df.to_csv(oof_long_path, index=False)
    print(f"Out-of-fold predictions (long format) saved to {oof_long_path}")
    print(f"  Shape: {oof_long_df.shape}")

    # --- Out-of-Fold Predictions: Wide Format (Input for Lower Model) ---
    print("\nPivoting to wide format (one row per isolate)...")

    print(f"  All refined_segments found: {oof_long_df['refined_segment'].unique()}")

    SEGMENT_ORDER = ["NA", "HA", "PB1_1", "PB1_2", "PB2_1", "PA_1", "PA_2",
                     "NS_1", "NS_2", "M_1", "M_2", "NP"]
    oof_long_df = oof_long_df[oof_long_df["refined_segment"].isin(SEGMENT_ORDER)]
    print(f"  Filtered to {len(SEGMENT_ORDER)} segments: {len(oof_long_df)} rows remaining.")

    pivot_dfs = []
    for prob_col in prob_columns:
        pivot = oof_long_df.pivot_table(
            index="Isolate_Name",
            columns="refined_segment",
            values=prob_col,
            aggfunc="first"
        )
        pivot.columns = [f"{prob_col}_{segment}" for segment in pivot.columns]
        pivot_dfs.append(pivot)

    oof_wide_df = pd.concat(pivot_dfs, axis=1)
    oof_wide_df = oof_wide_df.reset_index()

    metadata_cols = ["Isolate_Name", "Host_Family", "Classification"]
    metadata_df = cleaned_df[metadata_cols].drop_duplicates(subset=["Isolate_Name"]).reset_index(drop=True)
    oof_wide_df = oof_wide_df.merge(metadata_df, on="Isolate_Name", how="left")

    missing_count = oof_wide_df.isnull().any(axis=1).sum()
    if missing_count > 0:
        print(f"  Warning: {missing_count} isolates have missing segments (incomplete genomes).")
        print(f"  These will have NaN values in some probability columns.")

    oof_wide_path = os.path.join(output_path, "out_of_fold_predictions_wide.csv")
    oof_wide_df.to_csv(oof_wide_path, index=False)
    print(f"Out-of-fold predictions (wide format) saved to {oof_wide_path}")
    print(f"  Shape: {oof_wide_df.shape}")
    print(f"  Columns: {len(oof_wide_df.columns)} ({len(prob_columns)} families x "
          f"{len(oof_long_df['refined_segment'].unique())} segments + metadata)")

    # --- Calibration Analysis ---
    calibration_analysis(
        oof_probabilities=oof_probabilities,
        y_true=y,
        class_names=list(original_class_names),
        prob_columns=prob_columns,
        target_col_name="Host_Family",
        output_path=output_path,
        model_name="Protein Host Tropism",
        n_bins=10,
        fold_data=fold_calibration_data
    )


# ---------------------------------------------------------------------------
# Transition Spillover Classifier (with Stratified Group K-Fold CV)
# ---------------------------------------------------------------------------

def transition_spillover(merged_esm2_embeddings_csv_with_probs, transition_spillover_model, output_path, n_folds):
    """
    Train and evaluate a transition spillover XGBoost classifier using Stratified Group
    K-Fold Cross-Validation to prevent data leakage from duplicate feature vectors.

    Groups are defined by identical probability feature vectors. All isolates that produce
    the same set of per-segment family probabilities are guaranteed to remain in the same
    fold (either all in train or all in test, never split).

    After cross-validation, calibration analysis (Brier scores + reliability curves) and
    SHAP analysis (global + per-class feature importance) are run automatically.

    Args:
        merged_esm2_embeddings_csv_with_probs (str): Path to the CSV with per-segment family
                                                      probabilities in wide format (one row per isolate).
        transition_spillover_model (str): Optional path to a saved pre-trained transition
                                          spillover model. If None, trains from scratch.
        output_path (str):                Root directory for all output files.
        n_folds (int):                    Number of cross-validation folds.

    Outputs:
        - PKL:  saved_models/transition_spillover_model_fold_{fold}.pkl per fold
        - CSV:  train-test_splits/train_test_split_combined_fold_{fold}.csv per fold
        - CSV:  train_test_split/train_fold_{fold}.csv per fold
        - CSV:  train_test_split/test_fold_{fold}.csv per fold
        - CSV:  train_test_split/summary.csv (sample counts per fold)
        - CSV:  classification_reports/classification_report_fold_{fold}.csv per fold
        - CSV:  confusion_matrices/confusion_matrix_fold_{fold}.csv per fold
        - CSV:  cross_validation_summary.csv with aggregated metrics
        - PNG:  count_confusion_matrices/transition_spillover_confusion_matrix_fold_{fold}.png
        - PNG:  percentage_confusion_matrices/transition_spillover_confusion_matrix_fold_{fold}.png
        - CSV:  calibration_analysis/brier_scores/brier_scores_per_class.csv
        - PNG:  calibration_analysis/reliability_curves/reliability_curves_all_classes.png
        - PNG:  calibration_analysis/brier_scores/brier_scores.png
        - CSV:  shap_analysis/global_feature_importance.csv
        - PNG:  shap_analysis/shap_importance_by_segment.png
        - PNG:  shap_analysis/shap_multiclass_bar_top30.png
    """
    # --- Output Directory Setup ---
    subdirs = [
        "train-test_splits",
        "train_test_split",
        "classification_reports",
        "confusion_matrices",
        "count_confusion_matrices",
        "percentage_confusion_matrices",
        "saved_models"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    print("Reading CSV...")
    cleaned_df = pd.read_csv(merged_esm2_embeddings_csv_with_probs)
    print(cleaned_df.columns)

    # --- Feature Column Definitions ---
    PROB_COLS = [
        "anatidae_probability", "bovidae_probability", "canidae_probability",
        "equidae_probability", "felidae_probability", "hominidae_probability",
        "laridae_probability", "phasianidae_probability", "suidae_probability"
    ]

    SEGMENT_ORDER = ["NA", "HA", "PB1_1", "PB1_2", "PB2_1", "PA_1", "PA_2",
                     "NS_1", "NS_2", "M_1", "M_2", "NP"]

    x_columns = [f"{prob}_{segment}" for prob in PROB_COLS for segment in SEGMENT_ORDER]

    # --- Label Filtering ---
    missing_labels = cleaned_df["Classification"].isnull().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} samples have missing labels and will be dropped.")
        cleaned_df = cleaned_df.dropna(subset=["Classification"])

    # --- Group Assignment ---
    print("Assigning feature-vector groups for cross-validation...")

    group_cols = [col for col in x_columns if col in cleaned_df.columns]
    cleaned_df["group_id"] = cleaned_df.groupby(
        cleaned_df[group_cols].round(6).apply(tuple, axis=1)
    ).ngroup()
    print(f"Total unique groups: {cleaned_df['group_id'].nunique()}")

    # --- Feature Matrix and Label Vector ---
    X = cleaned_df[x_columns]
    y = cleaned_df["Classification"]
    groups = cleaned_df["group_id"]

    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    original_class_names = label_encoder.classes_
    print("Original class names:", original_class_names)

    # --- Out-of-Fold Prediction Storage ---
    oof_probabilities = np.zeros((len(X), len(original_class_names)))

    # --- Stratified Group K-Fold Cross-Validation ---
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_reports = []
    fold_summary = []
    fold_calibration_data = []
    X_test_per_fold = []

    print(f"\nRunning {n_folds}-Fold Stratified Group Cross-Validation...")
    print("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold + 1} of {n_folds} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Store X_test for SHAP analysis
        X_test_per_fold.append(X_test.copy())

        # Verify no group leakage
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Group overlap between train/test: {len(overlap)} (should be 0)")

        # --- Save Train/Test Split CSVs ---
        train_rows = cleaned_df.iloc[train_idx].copy()
        test_rows = cleaned_df.iloc[test_idx].copy()

        train_rows.to_csv(
            os.path.join(output_path, "train_test_split", f"train_fold_{fold + 1}.csv"),
            index=False
        )
        test_rows.to_csv(
            os.path.join(output_path, "train_test_split", f"test_fold_{fold + 1}.csv"),
            index=False
        )
        print(f"  Train/test split saved to {output_path}/train_test_split/")

        # Record fold summary
        fold_summary.append({
            "fold": fold + 1,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "total_samples": len(train_idx) + len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups)
        })

        # Encode labels
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # --- Load Pre-Trained Model or Train New Model ---
        if transition_spillover_model is not None:
            print("  Loading pre-trained model...")
            model = joblib.load(os.path.abspath(transition_spillover_model))
        else:
            print("  Training new model...")
            model = XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train_encoded)

            # Save trained model for this fold
            model_save_path = os.path.join(
                output_path, "saved_models", f"transition_spillover_model_fold_{fold + 1}.pkl"
            )
            joblib.dump(model, model_save_path)
            print(f"  Model saved to {model_save_path}")

        # --- Prediction ---
        y_train_pred_proba = model.predict_proba(X_train)
        y_test_pred_proba = model.predict_proba(X_test)

        y_train_pred = label_encoder.inverse_transform(np.argmax(y_train_pred_proba, axis=1))
        y_test_pred = label_encoder.inverse_transform(np.argmax(y_test_pred_proba, axis=1))

        # --- Store Out-of-Fold Predictions ---
        oof_probabilities[test_idx] = y_test_pred_proba

        # --- Store fold-level data for per-fold calibration stability ---
        fold_calibration_data.append({
            "fold": fold + 1,
            "y_true": y_test.reset_index(drop=True),
            "y_prob": y_test_pred_proba,
        })

        # --- Save Full Predictions to CSV (legacy format) ---
        train_full_rows = cleaned_df.loc[X_train.index].copy()
        test_full_rows = cleaned_df.loc[X_test.index].copy()

        train_full_rows["split"] = "train"
        test_full_rows["split"] = "test"
        train_full_rows["predicted_classification"] = y_train_pred
        test_full_rows["predicted_classification"] = y_test_pred

        for i, class_name in enumerate(original_class_names):
            train_full_rows[f"predicted_prob_{class_name}"] = y_train_pred_proba[:, i]
            test_full_rows[f"predicted_prob_{class_name}"] = y_test_pred_proba[:, i]

        combined_df = pd.concat([train_full_rows, test_full_rows], axis=0)
        combined_df.to_csv(
            f"{output_path}/train-test_splits/train_test_split_combined_fold_{fold + 1}.csv",
            index=False
        )
        print(f"  Split saved to {output_path}/train-test_splits/train_test_split_combined_fold_{fold + 1}.csv")

        # --- Classification Report ---
        report = classification_report(y_test, y_test_pred, target_names=original_class_names, output_dict=True)
        print(classification_report(y_test, y_test_pred, target_names=original_class_names))

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_path}/classification_reports/classification_report_fold_{fold + 1}.csv")

        all_reports.append(report_df)

        # --- Confusion Matrix (Count) ---
        cm = confusion_matrix(y_test, y_test_pred, labels=original_class_names)

        cm_df = pd.DataFrame(cm, index=original_class_names, columns=original_class_names)
        cm_df.to_csv(f"{output_path}/confusion_matrices/confusion_matrix_fold_{fold + 1}.csv")

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            xticklabels=original_class_names,
            yticklabels=original_class_names,
            cmap="YlOrRd",
            cbar_kws={"label": "Count"},
            vmin=0,
            annot=False,
            ax=ax
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.5, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="black", fontsize=11, weight="bold"
                )
        ax.set_xlabel("Predicted Labels", fontsize=12, labelpad=10, fontweight="bold")
        ax.set_ylabel("True Labels", fontsize=12, labelpad=10, fontweight="bold")
        plt.title(f"Transition Spillover Confusion Matrix (Count) — Fold {fold + 1}", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"{output_path}/count_confusion_matrices/transition_spillover_confusion_matrix_fold_{fold + 1}.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close()

        # --- Confusion Matrix (Percentage) ---
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm_percent,
            xticklabels=original_class_names,
            yticklabels=original_class_names,
            cmap="YlOrRd",
            cbar_kws={"label": "Percentage (%)"},
            vmin=0,
            vmax=100,
            annot=False,
            ax=ax
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.5, f"{cm_percent[i, j]:.1f}%",
                    ha="center", va="center",
                    color="black", fontsize=11, weight="bold"
                )
        ax.set_xlabel("Predicted Labels", fontsize=12, labelpad=10, fontweight="bold")
        ax.set_ylabel("True Labels", fontsize=12, labelpad=10, fontweight="bold")
        plt.title(f"Transition Spillover Confusion Matrix (Percentage) — Fold {fold + 1}", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"{output_path}/percentage_confusion_matrices/transition_spillover_confusion_matrix_fold_{fold + 1}.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    # --- Train/Test Split Summary ---
    summary_split_df = pd.DataFrame(fold_summary)
    summary_split_df.to_csv(
        os.path.join(output_path, "train_test_split", "summary.csv"),
        index=False
    )
    print(f"\nTrain/test split summary saved to {output_path}/train_test_split/summary.csv")
    print(summary_split_df.to_string(index=False))

    # --- Cross-Validation Summary ---
    print("\n" + "=" * 50)
    print("Cross-Validation Summary")
    print("=" * 50)

    summary_metrics = {}
    for class_name in list(original_class_names) + ["accuracy", "macro avg", "weighted avg"]:
        class_metrics = []
        for report_df in all_reports:
            if class_name in report_df.index:
                class_metrics.append(report_df.loc[class_name])
        if class_metrics:
            class_metrics_df = pd.DataFrame(class_metrics)
            summary_metrics[class_name] = {
                "precision_mean": class_metrics_df["precision"].mean(),
                "precision_std": class_metrics_df["precision"].std(),
                "recall_mean": class_metrics_df["recall"].mean(),
                "recall_std": class_metrics_df["recall"].std(),
                "f1-score_mean": class_metrics_df["f1-score"].mean(),
                "f1-score_std": class_metrics_df["f1-score"].std(),
            }

    summary_df = pd.DataFrame(summary_metrics).transpose()
    summary_df.to_csv(os.path.join(output_path, "cross_validation_summary.csv"))
    print(summary_df.to_string())
    print(f"\nSummary saved to {output_path}/cross_validation_summary.csv")

    # --- Calibration Analysis ---
    prob_columns = [f"predicted_prob_{class_name}" for class_name in original_class_names]
    calibration_analysis(
        oof_probabilities=oof_probabilities,
        y_true=y,
        class_names=list(original_class_names),
        prob_columns=prob_columns,
        target_col_name="Classification",
        output_path=output_path,
        model_name="Zoonotic Risk Prediction Model",
        n_bins=10,
        fold_data=fold_calibration_data
    )

    # --- SHAP Analysis ---
    saved_model_dir = os.path.join(output_path, "saved_models")
    shap_analysis(
        saved_model_dir=saved_model_dir,
        X_test_per_fold=X_test_per_fold,
        x_columns=x_columns,
        class_names=list(original_class_names),
        output_path=output_path,
        model_name="Zoonotic Risk Prediction Model",
        n_folds=n_folds,
        model_prefix="transition_spillover"
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Parse arguments and dispatch to the appropriate spillover classifier function.

    Both modes use Stratified Group K-Fold Cross-Validation to ensure:
        - No duplicate sequences leak between train and test folds
        - Class balance is maintained across folds
        - Every sample is tested exactly once
    """
    args = user_args()

    if args.mode == "family_spillover":
        family_spillover(
            args.merged_esm2_embeddings_csv,
            args.family_spillover_model,
            args.output_path,
            args.n_folds
        )

    elif args.mode == "transition_spillover":
        transition_spillover(
            args.merged_esm2_embeddings_csv_with_probs,
            args.transition_spillover_model,
            args.output_path,
            args.n_folds
        )

    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Choose 'family_spillover' or 'transition_spillover'.")


if __name__ == "__main__":
    main()


