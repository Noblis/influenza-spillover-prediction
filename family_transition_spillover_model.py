
"""
family_transition_spillover_model.py

This script trains or evaluates XGBoost classifiers for two spillover prediction tasks:
    1. Family Spillover     — Classifies protein host tropism using EMS2 protein embeddings. 
                              A new model can be trained from scratch, or a pre-trained model 
                              can be loaded for evaluation.
    2. Transition Spillover — Classifies zoonotic spillover risk using per-segment family probability 
                              features derived from ESM2 embeddings. A new model is trained by default; 
                              a pre-trained model can optionally be loaded instead.

Usage:
    # Train and/or evaluate the family spillover model (new or pre-trained):
    python spillover_classifier.py family_spillover \
        -c /path/to/merged_esm2_embeddings.csv \
        -m /path/to/family_spillover_model.pkl \
        -o /path/to/output_dir

    # Train and/or evaluate the transition spillover model (new or pre-trained):
    python spillover_classifier.py transition_spillover \
        -c /path/to/merged_esm2_embeddings_with_probs.csv \
        -m /path/to/transition_spillover_model.pkl \
        -o /path/to/output_dir

Notes:
    - Both the family spillover and transition spillover classifiers support either
      training a new model from scratch or loading a previously saved pre-trained model.
    - Both modes run 10 independent iterations. The family spillover uses a fixed
      random seed (random_state=42) for reproducibility across iterations. The
      transition spillover does NOT fix the random seed per iteration — this is
      intentional to evaluate run-to-run variance across splits.
    - For the transition spillover mode, to load a pre-trained model instead of
      training a new one, uncomment the '-m' / '--transition_spillover_model'
      argument in user_args() and the corresponding joblib.load() line in
      transition_spillover().
    - Zoonotic-labeled samples are automatically excluded from transition spillover training.
    - Output directories for CSVs, classification reports, and confusion matrix figures
      must exist prior to running (subdirectories are not auto-created).

Authors: Longest et al., 2026
Date: May 4, 2026
"""

import os
import joblib
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def user_args():
    """
    Parse command-line arguments for selecting and configuring the spillover mode.

    Subcommands:
        family_spillover      -- Train a new or load a pre-trained family spillover
                                 classifier (Model 1) and evaluate on a held-out test set.
        transition_spillover  -- Train a new or load a pre-trained transition spillover
                                 classifier (Model 2) and evaluate on a held-out test set.

    Returns:
        argparse.Namespace: Parsed arguments containing mode and associated file paths.
    """
    parser = argparse.ArgumentParser(
        description="Trains and/or evaluates either a family or transition spillover XGBoost classifier."
    )

    sub_parser = parser.add_subparsers(
        title="Select model type to train or evaluate",
        dest="mode",
        required=True
    )

    # --- Family Spillover Subparser ---
    parser_family_spillover = sub_parser.add_parser(
        "family_spillover",
        help="Trains a new or loads a pre-trained family spillover classifier (Model 1) and evaluates it."
    )
    parser_family_spillover.add_argument(
        "-c", "--merged_esm2_embeddings_csv",
        required=True,
        help='Path to the merged ESM2 embeddings CSV (e.g., "/data/merged_with_esm2.csv")'
    )
    # Optional: uncomment to load a pre-trained family spillover model instead of training a new one
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

    # --- Transition Spillover Subparser ---
    parser_transition_spillover = sub_parser.add_parser(
        "transition_spillover",
        help="Trains a new or loads a pre-trained transition spillover classifier (Model 2) and evaluates it."
    )
    parser_transition_spillover.add_argument(
        "-c", "--merged_esm2_embeddings_csv_with_probs",
        required=True,
        help='Path to the merged ESM2 embeddings + family probabilities CSV (e.g., "/data/merged_esm2_with_probs.csv").'
    )
    # Optional: uncomment to load a pre-trained transition spillover model instead of training a new one
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

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Family Spillover Classifier
# ---------------------------------------------------------------------------

def family_spillover(merged_esm2_embeddings_csv, family_spillover_model, output_path, iteration):
    """
    Train a new or load a pre-trained family spillover XGBoost classifier and evaluate
    it on a held-out test set.

    This function:
        - Reads and deduplicates the input ESM2 embeddings CSV.
        - Drops samples with missing 'Host_Family' labels.
        - Splits data into 80/20 train/test sets (stratified by label, random seed not fixed).
        - If a pre-trained model path is provided, loads and evaluates it on the test set.
          Otherwise, trains a new XGBClassifier from scratch.
        - Prints a classification report and saves a confusion matrix heatmap.

    Args:
        merged_esm2_embeddings_csv (str): Path to the merged ESM2 embeddings CSV file.
        family_spillover_model (str):     Path to a saved pre-trained family spillover model
                                          file, or None to train a new model from scratch.
        output_path (str):                Directory path to save the confusion matrix PNG.
        iteration (int):                  Current iteration index (0-based). Used for output
                                          file naming and plot titles (displayed as iteration + 1).

    Outputs:
        - Console: Classification report (precision, recall, F1 per class).
        - File:    'family_spillover_confusion_matrix_{iteration}.png' saved to output_path.
    """
    print("Reading CSV...")
    cleaned_df = pd.read_csv(merged_esm2_embeddings_csv)

    # Remove duplicate entries based on unique sample identifier
    cleaned_df = cleaned_df.drop_duplicates(subset=["unique_ID"])

    # Warn and drop samples with missing host family labels
    missing_labels = cleaned_df["Host_Family"].isnull().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} samples have missing labels and will be dropped.")
        cleaned_df = cleaned_df.dropna(subset=["Host_Family"])

    # Define non-feature columns to exclude from the feature matrix
    # Note: These column names may need to be changed depending on data saved from ESM2
    #non_numeric_cols = [
    #    "Isolate_Name", "Host", "data_source",
    #    "Genotype", "Segment", "data_source",
    #    "GISAID_Accession", "refined_segment", "unique_ID",
    #    "Classification", "isoform"
    #]
    non_numeric_cols = ['Isolate_Name', 'Host', 'Common_Name', 'Segment',
                        'Genotype', 'Country', 'Collection_Date', 'Sequence', 'Seq_Len',
                        'data_source', 'GISAID_Accession', 'NCBI_Accession', 'length_percent',
                        'isoform', 'refined_segment', 'unique_ID', 'Family_Group',
                        'Classification']

    print(cleaned_df.columns[0:20])
    # Build feature matrix (X) and label vector (y)
    X = cleaned_df.drop(columns=non_numeric_cols + ["Host_Family"])
    y = cleaned_df["Host_Family"]

    # Stratified 80/20 train-test split (random seed not fixed — allows natural variability
    # across iterations so users can train their own models without forced reproducibility)
    # For reproducibility: add random seed and set it to 42 (e.g., random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # --- Label Encoding ---
    # XGBoost requires integer-encoded labels; LabelEncoder maps class names to indices
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(y_train)   # Fit on train labels to establish class ordering
    label_encoder.transform(y_test)        # Transform test labels for alignment
    original_class_names = label_encoder.classes_ # Save the original class names

    # --- Load Pre-Trained Model or Train New Model ---
    if family_spillover_model is not None:
        # Load a previously saved family spillover model from disk
        print("Loading pre-trained model...")
        model = joblib.load(os.path.abspath(family_spillover_model))
    else:
        # Train a new XGBClassifier from scratch on the training set
        print("Training new model...")
        model = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,   # Suppress deprecated internal label encoding
            eval_metric="logloss",     # Log loss for multi-class probability calibration
        )
        y_train_encoded = label_encoder.transform(y_train)
        model.fit(X_train, y_train_encoded)

        # Optional: Save the trained model to disk for future use or reproducibility (uncomment to use)
        #model_save_path = os.path.join(os.path.abspath(output_path), f"family_spillover_model_{iteration}.pkl")
        #joblib.dump(model, model_save_path)
        #print(f"Trained model saved to {model_save_path}")

    # Generate predicted class probabilities on the test set
    print("Predicting...")
    y_pred_proba = model.predict_proba(X_test)

    # Decode predicted class indices back to original string labels
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Print per-class precision, recall, and F1-score
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # --- Confusion Matrix Heatmap ---
    print("Outputting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=original_class_names)
    
    # --- Normalize Confusion Matrix by Row (True Label) ---
    # Each row sums to 100%, representing recall per class
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot count-based confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw percentage-normalized heatmap (annotations added manually below)
    sns.heatmap(
        cm_percent,
        xticklabels=original_class_names,
        yticklabels=original_class_names,
        cmap="YlOrRd",
        cbar_kws={"label": "Percentage (%)"},
        vmin=0,
        vmax=100,
        annot=False,   # Manual annotation used for full font/style control
        ax=ax
    )

    # --- Manually Annotate Each Cell with Percentage Values ---
    # Manual annotation (seaborn annot bypasses rcParams)
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

    # --- Colorbar Formatting ---
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Percentage (%)", fontsize=10)

    # --- Axis Labels ---
    ax.set_xlabel(
        "Predicted Labels",
        fontsize=11,
        labelpad=12,
        fontweight="bold"
    )
    ax.set_ylabel(
        "True Labels",
        fontsize=11,
        labelpad=12,
        fontweight="bold"
    )

    # --- Tick Labels ---
    # Italic style for class name tick labels; smaller than axis titles per journal hierarchy
    ax.set_xticklabels(
        original_class_names,
        rotation=45,
        ha="right",
        fontsize=9,
        fontstyle="italic"
    )
    ax.set_yticklabels(
        original_class_names,
        rotation=0,
        fontsize=9,
        fontstyle="italic"
    )

    plt.title(
        f"Family Spillover Classifier Confusion Matrix Heatmap — Iteration {iteration + 1}",
        fontsize=12, pad=20
    )
    plt.tight_layout()


    # Output filename includes iteration number for consistency with transition spillover outputs
    plt.savefig(os.path.join(
        os.path.abspath(output_path),
        f"family_spillover_confusion_matrix_{iteration}.png"
        ), dpi=300, bbox_inches="tight")

    plt.close()


# ---------------------------------------------------------------------------
# Transition Spillover Classifier
# ---------------------------------------------------------------------------

def transition_spillover(merged_esm2_embeddings_csv_with_probs, transition_spillover_model, output_path, iteration):
    """
    Train a new or load a pre-trained transition spillover XGBoost classifier and evaluate
    it on a held-out test set.

    This function:
        - Reads the input CSV containing ESM2-derived per-segment family probabilities.
        - Drops samples with missing 'Classification' labels.
        - Constructs the feature matrix from per-family, per-segment probability columns.
        - Trains a new XGBClassifier by default on an 80/20 stratified split (random seed
          not fixed to allow natural variability across iterations).
        - If a pre-trained model path is provided, loads and evaluates it on the test set
          instead of training a new model.
        - Saves predictions, classification reports, and confusion matrices (count + percentage).

    Args:
        merged_esm2_embeddings_csv_with_probs (str): Path to the CSV with ESM2 embeddings
                                                      and per-segment family probabilities.
        transition_spillover_model (str): Optional path to a saved pre-trained transition
                                          spillover model (.pkl or .joblib). If not provided,
                                          a new model will be trained from scratch.
        output_path (str):                Root directory for all output files. Expected subdirectories:
                                              - train-test_splits/
                                              - classification_reports/
                                              - confusion_matrices/
                                              - count_confusion_matrices/
                                              - percentage_confusion_matrices/
        iteration (int):                  Current iteration index (0-based). Used for output
                                          file naming and plot titles (displayed as iteration + 1).


    Outputs:
        - CSV:  train_test_split_combined_{iteration}.csv  — full rows with predictions + probabilities.
        - CSV:  classification_report_{iteration}.csv      — per-class precision, recall, F1.
        - CSV:  confusion_matrix_{iteration}.csv           — raw count confusion matrix.
        - PNG:  count_confusion_matrices/transition_spillover_confusion_matrix_{iteration}.png
        - PNG:  percentage_confusion_matrices/transition_spillover_confusion_matrix_{iteration}.png
    """

        # --- Output Directory Setup ---
    # Auto-create all required output subdirectories if they do not already exist
    subdirs = [
        "train-test_splits",
        "classification_reports",
        "confusion_matrices",
        "count_confusion_matrices",
        "percentage_confusion_matrices"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
    
    print("Reading CSV...")
    cleaned_df = pd.read_csv(merged_esm2_embeddings_csv_with_probs)
    print(cleaned_df.columns)

    # --- Feature Column Definitions ---
    # Per-family probability columns derived from the family spillover classifier (Model 1)
    PROB_COLS = [
        "anatidae_probability", "bovidae_probability", "canidae_probability",
        "equidae_probability", "felidae_probability", "hominidae_probability",
        "laridae_probability", "phasianidae_probability", "suidae_probability"
    ]

    # Influenza genome segments used as feature dimensions
    # Note: PB1_2 represents PB1-F2 and PA_2 represents PA-X
    SEGMENT_ORDER = ["N", "HA", "PB1_1", "PB1_2", "PB2_1", "PA_1", "PA_2", "NS_1", "NS_2", "M_1", "M_2", "NP"]

    # Build feature column names as all combinations of family probability x segment
    # e.g., "anatidae_probability_HA", "bovidae_probability_PB1_1", etc.
    x_columns = [f"{prob}_{segment}" for prob in PROB_COLS for segment in SEGMENT_ORDER]

    # --- Label Filtering ---
    # Drop samples with missing classification labels
    missing_labels = cleaned_df["Classification"].isnull().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} samples have missing labels and will be dropped.")
        cleaned_df = cleaned_df.dropna(subset=["Classification"])

    # Optional Run: Remove zoonotic-labeled samples — this model will then only target non-zoonotic transition events only
    #zoonotic_rows = cleaned_df[cleaned_df["Classification"].str.contains("zoonotic", na=False)].index
    #cleaned_df = cleaned_df.drop(zoonotic_rows)

    # --- Feature Matrix and Label Vector ---
    X = cleaned_df[x_columns]           # Per-segment family probability features
    y = cleaned_df["Classification"]    # Transmission classification label

    # Stratified 80/20 train-test split (random seed not fixed to assess run-to-run variability)
    # For reproducibility: add random seed and set it to 42 (e.g., random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # --- Label Encoding ---
    # XGBoost requires integer-encoded labels; LabelEncoder maps class names to indices
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Quality Check: Store original class names for decoding predictions and labeling outputs
    original_class_names = label_encoder.classes_
    print("Original class names:", original_class_names)

    # --- Load Pre-Trained Model or Train New Model ---
    if transition_spillover_model is not None:
        # Load a previously saved transition spillover model from disk
        print("Loading pre-trained model...")
        model = joblib.load(os.path.abspath(transition_spillover_model))
    else:
        # Train a new XGBClassifier from scratch on the training set (default behavior)
        print("Training new model...")
        model = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,   # Suppress deprecated internal label encoding
            eval_metric="logloss",     # Log loss for multi-class probability calibration
        )
        # Train on integer-encoded labels (required by XGBoost)
        model.fit(X_train, y_train_encoded)

        # Optional: Save the trained model to disk for future use or reproducibility (uncomment to use)
        #model_save_path = os.path.join(os.path.abspath(output_path), f"transition_spillover_model_{iteration}.pkl")
        #joblib.dump(model, model_save_path)
        #print(f"Trained model saved to {model_save_path}")

    # --- Prediction ---
    print("Predicting...")
    y_train_pred_proba = model.predict_proba(X_train)
    y_test_pred_proba = model.predict_proba(X_test)

    # Convert predicted probability arrays to class label strings
    y_train_pred = label_encoder.inverse_transform(np.argmax(y_train_pred_proba, axis=1))
    y_test_pred = label_encoder.inverse_transform(np.argmax(y_test_pred_proba, axis=1))

    # --- Save Full Predictions to CSV ---
    # Retrieve complete rows from the original DataFrame using train/test indices
    train_full_rows = cleaned_df.loc[X_train.index].copy()
    test_full_rows = cleaned_df.loc[X_test.index].copy()

    # Annotate rows with split assignment and predicted labels
    train_full_rows["split"] = "train"
    test_full_rows["split"] = "test"
    train_full_rows["predicted_classification"] = y_train_pred
    test_full_rows["predicted_classification"] = y_test_pred

    # Append per-class predicted probabilities as additional columns
    for i, class_name in enumerate(original_class_names):
        train_full_rows[f"predicted_prob_{class_name}"] = y_train_pred_proba[:, i]
        test_full_rows[f"predicted_prob_{class_name}"] = y_test_pred_proba[:, i]

    # Combine train and test rows and save to CSV
    combined_df = pd.concat([train_full_rows, test_full_rows], axis=0)
    combined_df.to_csv(f"{output_path}/train-test_splits/train_test_split_combined_{iteration}.csv", index=False)
    print(f"Combined train-test split saved to {output_path}/train-test_splits/train_test_split_combined_{iteration}.csv")

    # --- Classification Report ---
    print("Outputting results...")
    report = classification_report(y_test, y_test_pred, target_names=original_class_names, output_dict=True)
    print(classification_report(y_test, y_test_pred, target_names=original_class_names))

    # Save classification report as a CSV for downstream analysis
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{output_path}/classification_reports/classification_report_{iteration}.csv")
    print(f"Classification report saved to {output_path}/classification_reports/classification_report_{iteration}.csv")

    # --- Confusion Matrix (Count) ---
    cm = confusion_matrix(y_test, y_test_pred, labels=original_class_names)

    # Save raw count confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=original_class_names, columns=original_class_names)
    cm_df.to_csv(f"{output_path}/confusion_matrices/confusion_matrix_{iteration}.csv")
    print(f"Confusion matrix saved to {output_path}/confusion_matrices/confusion_matrix_{iteration}.csv")

    # Plot count-based confusion matrix heatmap
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

    # Manually annotate each cell with count values (bold black text for visibility)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.5, f"{cm[i, j]}",
                ha="center", va="center",
                color="black", fontsize=11, weight="bold"
            )

    plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
    plt.ylabel("True Labels", fontsize=12, labelpad=10)
    plt.title(
        f"Transition Spillover Classifier Confusion Matrix — Iteration {iteration + 1}",
        fontsize=14, pad=20
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        f"{output_path}/count_confusion_matrices/transition_spillover_confusion_matrix_{iteration}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # --- Confusion Matrix (Percentage) ---
    # Normalize confusion matrix by row (true label) to get per-class recall percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot percentage-normalized confusion matrix heatmap
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

    # Manually annotate each cell with percentage values (bold black text for visibility)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.5, f"{cm_percent[i, j]:.1f}%",
                ha="center", va="center",
                color="black", fontsize=11, weight="bold"
            )

    plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
    plt.ylabel("True Labels", fontsize=12, labelpad=10)
    plt.title(
        f"Transition Spillover Classifier Confusion Matrix — Iteration {iteration + 1}",
        fontsize=14, pad=20
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        f"{output_path}/percentage_confusion_matrices/transition_spillover_confusion_matrix_{iteration}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(iteration):
    """
    Parse arguments and dispatch to the appropriate spillover classifier function.

    Both modes support training a new model from scratch or loading a pre-trained model:
        - family_spillover:      Pass '-m' to load a pre-trained model; omit to train new.
        - transition_spillover:  Trains a new model by default. To load a pre-trained model,
                                 pass the '-m' argument in user_args() to load a pre-trained 
                                 model in transition_spillover().

    Args:
        iteration (int): Current iteration index (0-based), passed to transition_spillover
                         for output file naming and plot titles.

    Raises:
        ValueError: If an unrecognized mode is provided (should not occur with argparse enforcement).
    """
    args = user_args()

    if args.mode == "family_spillover":
        # Train a new or evaluate a pre-trained family spillover model (Model 1)
        family_spillover(
            args.merged_esm2_embeddings_csv,
            args.family_spillover_model,
            args.output_path,
            iteration
        )

    elif args.mode == "transition_spillover":
        # Train a new or evaluate a pre-trained transition spillover model (Model 2)
        transition_spillover(
            args.merged_esm2_embeddings_csv_with_probs,
            args.transition_spillover_model,
            args.output_path,
            iteration
        )

    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Please choose 'family_spillover' or 'transition_spillover'.")

if __name__ == "__main__":
    # Run 10 independent iterations for both family and transition spillover model stability
    # Note: Both family spillover and transition spillover functions do NOT fix the random seed —
    #       this is intentional to evaluate run-to-run variance across splits. However, this is changable.
    for i in range(10): # Change the numerical value inside the range() to change the iteration value
        print(f"{'='*50}")
        print(f"  Run {i + 1} of 10")
        print(f"{'='*50}")
        main(iteration=i)


