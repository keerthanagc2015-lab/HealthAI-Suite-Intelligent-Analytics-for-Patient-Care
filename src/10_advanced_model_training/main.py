# ==========================================================
# HealthAI Suite - Advanced Model Training
# Module : Main Training Pipeline
# ==========================================================

import json
from pathlib import Path

import joblib

from preprocessing import prepare_data
from random_forest import train_random_forest
from xgboost_model import train_xgboost
from evaluation import evaluate_model, compare_models


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = Path(
    "data/processed/advanced_healthcare.csv"
)

MODEL_DIR = Path(
    "models/classical_ml/primary_diagnosis"
)


# ==========================================================
# Save Production Artifacts
# ==========================================================

def save_artifacts(

    model,
    preprocessor,
    label_encoder,
    results

):

    # ------------------------------------------------------
    # Create model directory
    # ------------------------------------------------------

    MODEL_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ------------------------------------------------------
    # File Paths
    # ------------------------------------------------------

    model_path = (

        MODEL_DIR /

        "primary_diagnosis_model.joblib"

    )

    preprocessor_path = (

        MODEL_DIR /

        "primary_diagnosis_preprocessor.joblib"

    )

    label_encoder_path = (

        MODEL_DIR /

        "primary_diagnosis_label_encoder.joblib"

    )

    metadata_path = (

        MODEL_DIR /

        "metadata.json"

    )


    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    joblib.dump(

        model,

        model_path

    )


    # ------------------------------------------------------
    # Save Preprocessor
    # ------------------------------------------------------

    joblib.dump(

        preprocessor,

        preprocessor_path

    )


    # ------------------------------------------------------
    # Save Label Encoder
    # ------------------------------------------------------

    joblib.dump(

        label_encoder,

        label_encoder_path

    )


    # ------------------------------------------------------
    # Extract Model Performance
    # ------------------------------------------------------

    selected_result = None

    for result in results:

        if (

            result["Model"]

            ==

            "Random Forest + SMOTE + Feature Engineering"

        ):

            selected_result = result

            break


    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = {

        "project": "HealthAI-Intelligent-Platform",

        "task": "Multiclass Primary Diagnosis Classification",

        "target": "Primary_Diagnosis",

        "selected_model": "Random Forest",

        "model_type": "RandomForestClassifier",

        "training_method": [

            "Feature Engineering",

            "OneHotEncoding",

            "StandardScaler",

            "SMOTE"

        ],

        "training_dataset": str(DATA_PATH),

        "input_features": [

            "Age",
            "Gender",
            "Region",
            "Socioeconomic_Status",
            "Symptoms",
            "Blood_Glucose_mg_dL",
            "HbA1c_%",
            "Total_Cholesterol_mg_dL",
            "BMI",
            "BMI_Category",
            "Age_Group",
            "High_Glucose",
            "High_Cholesterol",
            "HbA1c_Category"

        ],

        "numerical_features": [

            "Age",
            "Blood_Glucose_mg_dL",
            "HbA1c_%",
            "Total_Cholesterol_mg_dL",
            "BMI"

        ],

        "categorical_features": [

            "Gender",
            "Region",
            "Socioeconomic_Status",
            "Symptoms",
            "BMI_Category",
            "Age_Group",
            "High_Glucose",
            "High_Cholesterol",
            "HbA1c_Category"

        ],

        "smote": {

            "enabled": True,

            "random_state": 42

        },

        "model_parameters": {

            "n_estimators": 100,

            "random_state": 42,

            "n_jobs": -1

        },

        "classes": (

            label_encoder.classes_.tolist()

        ),

        "performance": selected_result,

        "artifacts": {

            "model": str(model_path),

            "preprocessor": str(preprocessor_path),

            "label_encoder": str(label_encoder_path),

            "metadata": str(metadata_path)

        },

        "inference_note": (

            "SMOTE is used only during model training. "
            "It must not be applied to inference data."

        )

    }


    # ------------------------------------------------------
    # Save Metadata
    # ------------------------------------------------------

    with open(

        metadata_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metadata,

            file,

            indent=4

        )


    # ------------------------------------------------------
    # Print Saved Artifacts
    # ------------------------------------------------------

    print("\n" + "=" * 70)

    print("PRODUCTION ARTIFACTS SAVED")

    print("=" * 70)

    print(

        f"\nModel              : {model_path}"

    )

    print(

        f"Preprocessor       : {preprocessor_path}"

    )

    print(

        f"Label Encoder      : {label_encoder_path}"

    )

    print(

        f"Metadata            : {metadata_path}"

    )


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("\n" + "=" * 70)

    print("HEALTHAI - PRODUCTION MODEL TRAINING")

    print("=" * 70)


    # ------------------------------------------------------
    # Prepare Data
    # ------------------------------------------------------

    (

        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        preprocessor

    ) = prepare_data(

        str(DATA_PATH)

    )


    # ------------------------------------------------------
    # Results Container
    # ------------------------------------------------------

    results = []


    # ======================================================
    # Random Forest
    # ======================================================

    print("\n" + "=" * 70)

    print("TRAINING RANDOM FOREST")

    print("=" * 70)


    rf_model, rf_predictions = train_random_forest(

        X_train,

        y_train,

        X_test

    )


    rf_results = evaluate_model(

        "Random Forest + SMOTE + Feature Engineering",

        y_test,

        rf_predictions

    )


    results.append(

        rf_results

    )


    # ======================================================
    # XGBoost
    # ======================================================

    print("\n" + "=" * 70)

    print("TRAINING XGBOOST")

    print("=" * 70)


    xgb_model, xgb_predictions = train_xgboost(

        X_train,

        y_train,

        X_test

    )


    xgb_results = evaluate_model(

        "XGBoost + SMOTE + Feature Engineering",

        y_test,

        xgb_predictions

    )


    results.append(

        xgb_results

    )


    # ======================================================
    # Model Comparison
    # ======================================================

    print("\n" + "=" * 70)

    print("MODEL COMPARISON")

    print("=" * 70)


    compare_models(

        results

    )


    # ======================================================
    # Select Production Model
    # ======================================================

    # Random Forest is selected because it achieved better
    # Macro F1, Macro Recall and Weighted F1 in our evaluation.

    selected_model = rf_model


    print("\n" + "=" * 70)

    print("SELECTED PRODUCTION MODEL")

    print("=" * 70)

    print("\nModel : Random Forest")

    print(

        "Reason: Better Macro F1, Macro Recall and Weighted F1"

    )


    # ======================================================
    # Save Production Artifacts
    # ======================================================

    save_artifacts(

        model=selected_model,

        preprocessor=preprocessor,

        label_encoder=label_encoder,

        results=results

    )


    # ======================================================
    # Final Status
    # ======================================================

    print("\n" + "=" * 70)

    print("STEP 17.6 COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(

        "\nThe production Random Forest model, "

        "preprocessor and label encoder are ready "

        "for the Agentic AI adapter."

    )


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":

    main()