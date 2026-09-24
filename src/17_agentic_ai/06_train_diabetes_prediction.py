"""
HealthAI - Diabetes Risk Prediction Model
=========================================
Step 17.7

Creates a dedicated binary diabetes-risk classifier from the
existing healthcare dataset.

Target:
    Type 2 Diabetes = 1
    All other diagnoses = 0

Important:
    Primary_Diagnosis is NEVER used as an input feature.
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "advanced_healthcare.csv",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "classical_ml",
    "diabetes_prediction",
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    "Age",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Region",
    "Socioeconomic_Status",
    "Symptoms",
    "BMI_Category",
    "Age_Group",
    "High_Glucose",
    "High_Cholesterol",
    "HbA1c_Category",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_SOURCE = "Primary_Diagnosis"
TARGET_NAME = "Diabetes_Risk"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HEALTHAI - DIABETES RISK PREDICTION")
    print("STEP 17.7 - MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = FEATURES + [TARGET_SOURCE]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Create binary target
    # --------------------------------------------------------

    df[TARGET_NAME] = (
        df[TARGET_SOURCE]
        .astype(str)
        .str.strip()
        .eq("Type 2 Diabetes")
        .astype(int)
    )

    print("\nTarget distribution:")

    print(
        df[TARGET_NAME]
        .value_counts()
        .sort_index()
        .rename({
            0: "No Type 2 Diabetes",
            1: "Type 2 Diabetes",
        })
    )

    # --------------------------------------------------------
    # Clean invalid glucose values
    # --------------------------------------------------------

    invalid_glucose = (
        pd.to_numeric(
            df["Blood_Glucose_mg_dL"],
            errors="coerce"
        ) <= 0
    )

    invalid_count = int(invalid_glucose.sum())

    if invalid_count > 0:
        print(
            f"\nInvalid glucose values found: "
            f"{invalid_count}"
        )

        df.loc[
            invalid_glucose,
            "Blood_Glucose_mg_dL"
        ] = np.nan

        print(
            "Invalid glucose values converted to NaN."
        )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df[FEATURES].copy()
    y = df[TARGET_NAME].copy()

    print(
        f"\nFeatures used: {len(FEATURES)}"
    )

    print(FEATURES)

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nData split:")
    print(f"Training: {X_train.shape}")
    print(f"Testing : {X_test.shape}")

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
    }

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"TRAINING: {model_name}")
        print("-" * 70)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        trained_models[model_name] = pipeline

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
            }
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selected_row = results_df.sort_values(
        by=["f1", "roc_auc", "recall"],
        ascending=False
    ).iloc[0]

    selected_name = selected_row["model"]

    selected_model = trained_models[
        selected_name
    ]

    print(
        f"\nSelected production model: "
        f"{selected_name}"
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_predictions = selected_model.predict(
        X_test
    )

    final_probabilities = selected_model.predict_proba(
        X_test
    )[:, 1]

    print("\nFinal classification report:")
    print(
        classification_report(
            y_test,
            final_predictions,
            target_names=[
                "No Type 2 Diabetes",
                "Type 2 Diabetes",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save complete pipeline
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "diabetes_prediction_model.joblib",
    )

    joblib.dump(
        selected_model,
        model_path
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": selected_name,
        "task": "binary diabetes risk prediction",
        "target": TARGET_NAME,
        "target_definition": {
            "1": "Type 2 Diabetes",
            "0": "All other Primary_Diagnosis classes",
        },
        "source_dataset": (
            "data/processed/"
            "advanced_healthcare.csv"
        ),
        "source_target": TARGET_SOURCE,
        "features": FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_size": 0.20,
        "random_state": 42,
        "metrics": {
            "accuracy": float(
                selected_row["accuracy"]
            ),
            "precision": float(
                selected_row["precision"]
            ),
            "recall": float(
                selected_row["recall"]
            ),
            "f1": float(
                selected_row["f1"]
            ),
            "roc_auc": float(
                selected_row["roc_auc"]
            ),
        },
        "all_model_results":
            results_df.to_dict(
                orient="records"
            ),
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    evaluation_df = X_test.copy()

    evaluation_df["actual"] = y_test.values
    evaluation_df["predicted"] = final_predictions
    evaluation_df["probability"] = final_probabilities

    evaluation_path = os.path.join(
        MODEL_DIR,
        "test_predictions.csv",
    )

    evaluation_df.to_csv(
        evaluation_path,
        index=False
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 17.7 DIABETES MODEL COMPLETED")
    print("=" * 70)

    print("\nSaved:")
    print(model_path)
    print(metadata_path)
    print(evaluation_path)

    print("\nSTATUS: SUCCESS")


if __name__ == "__main__":
    main()