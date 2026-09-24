"""
HealthAI - Diabetes Prediction Adapter
======================================
Step 17.8

Loads the trained diabetes-risk pipeline and exposes
a standard agent interface.
"""

import os
import json
from typing import Any

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classical_ml",
    "diabetes_prediction",
    "diabetes_prediction_model.joblib",
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classical_ml",
    "diabetes_prediction",
    "metadata.json",
)


# ============================================================
# LAZY MODEL
# ============================================================

_MODEL = None
_METADATA = None


def load_model():
    global _MODEL

    if _MODEL is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Diabetes model not found: {MODEL_PATH}"
            )

        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL


def load_metadata():
    global _METADATA

    if _METADATA is None:

        if os.path.exists(METADATA_PATH):

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                _METADATA = json.load(file)

        else:
            _METADATA = {}

    return _METADATA


# ============================================================
# PREDICTION
# ============================================================

def predict_diabetes_risk(
    patient_data: dict[str, Any],
) -> dict[str, Any]:

    model = load_model()
    metadata = load_metadata()

    required_features = metadata.get(
        "features",
        [
            "Age",
            "Blood_Glucose_mg_dL",
            "HbA1c_%",
            "Total_Cholesterol_mg_dL",
            "BMI",
            "Gender",
            "Region",
            "Socioeconomic_Status",
            "Symptoms",
            "BMI_Category",
            "Age_Group",
            "High_Glucose",
            "High_Cholesterol",
            "HbA1c_Category",
        ],
    )

    missing = [
        feature
        for feature in required_features
        if feature not in patient_data
    ]

    if missing:
        return {
            "status": "missing_input",
            "tool": "diabetes_prediction",
            "message": (
                "Required patient features are missing."
            ),
            "missing_features": missing,
        }

    input_df = pd.DataFrame(
        [patient_data]
    )[required_features]

    prediction = int(
        model.predict(input_df)[0]
    )

    probabilities = model.predict_proba(
        input_df
    )[0]

    diabetes_probability = float(
        probabilities[1]
    )

    non_diabetes_probability = float(
        probabilities[0]
    )

    risk_label = (
        "High diabetes risk"
        if prediction == 1
        else "Low diabetes risk"
    )

    return {
        "status": "success",
        "tool": "diabetes_prediction",
        "prediction": prediction,
        "risk_label": risk_label,
        "diabetes_probability": round(
            diabetes_probability,
            4,
        ),
        "no_diabetes_probability": round(
            non_diabetes_probability,
            4,
        ),
        "confidence": round(
            max(probabilities),
            4,
        ),
        "model": metadata.get(
            "model_name",
            "Random Forest",
        ),
    }


# ============================================================
# AGENT INTERFACE
# ============================================================

def execute_diabetes_prediction(
    query: str = "",
    patient_data: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:

    if patient_data is None:

        return {
            "status": "missing_input",
            "tool": "diabetes_prediction",
            "message": (
                "Diabetes prediction requires "
                "patient clinical features."
            ),
        }

    try:

        return predict_diabetes_risk(
            patient_data
        )

    except Exception as exc:

        return {
            "status": "execution_error",
            "tool": "diabetes_prediction",
            "message": (
                "Diabetes prediction failed."
            ),
            "error": str(exc),
        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HEALTHAI DIABETES PREDICTION ADAPTER")
    print("=" * 70)

    print("\nLoading model...")

    model = load_model()

    print("Model loaded successfully.")
    print(
        f"Model: {load_metadata().get('model_name', 'Unknown')}"
    )

    test_patient = {
        "Age": 52,
        "Blood_Glucose_mg_dL": 165,
        "HbA1c_%": 7.8,
        "Total_Cholesterol_mg_dL": 220,
        "BMI": 29.5,
        "Gender": "Male",
        "Region": "Tamil Nadu",
        "Socioeconomic_Status": "Middle",
        "Symptoms": "Frequent urination, excessive thirst",
        "BMI_Category": "Overweight",
        "Age_Group": "Middle Age",
        "High_Glucose": 1,
        "High_Cholesterol": 1,
        "HbA1c_Category": "High",
    }

    print("\nRunning test prediction...")

    result = execute_diabetes_prediction(
        query="Predict diabetes risk",
        patient_data=test_patient,
    )

    print("\nPrediction:")
    print(json.dumps(
        result,
        indent=2
    ))

    print("\n" + "=" * 70)

    if result.get("status") == "success":
        print("STEP 17.8 DIABETES ADAPTER")
        print("STATUS: SUCCESS")
    else:
        print("STATUS: FAILED")

    print("=" * 70)
    