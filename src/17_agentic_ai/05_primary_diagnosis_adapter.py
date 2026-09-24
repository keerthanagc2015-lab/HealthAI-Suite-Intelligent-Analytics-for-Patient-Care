"""
======================================================================
HealthAI - Primary Diagnosis Prediction Adapter
======================================================================

Module : 17 Agentic AI
Step   : Primary Diagnosis Adapter

Purpose
-------
Loads the trained primary-diagnosis model and exposes a clean
prediction interface for the HealthAI Agent Executor and FastAPI.

Training-time feature schema
----------------------------
Original features:
    Age
    Gender
    Region
    Socioeconomic_Status
    Symptoms
    Blood_Glucose_mg_dL
    HbA1c_%
    Total_Cholesterol_mg_dL
    BMI

Engineered features:
    BMI_Category
    Age_Group
    High_Glucose
    High_Cholesterol
    HbA1c_Category

Target:
    Primary_Diagnosis

The preprocessing used during training treats the engineered binary
features High_Glucose and High_Cholesterol as CATEGORICAL features.
Therefore this adapter preserves them as categorical values ("Yes"/"No")
before calling the saved OneHotEncoder.
======================================================================
"""

from pathlib import Path
import json
import joblib
import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classical_ml"
    / "primary_diagnosis"
)

MODEL_PATH = MODEL_DIR / "primary_diagnosis_model.joblib"
PREPROCESSOR_PATH = (
    MODEL_DIR / "primary_diagnosis_preprocessor.joblib"
)
LABEL_ENCODER_PATH = (
    MODEL_DIR / "primary_diagnosis_label_encoder.joblib"
)
METADATA_PATH = MODEL_DIR / "metadata.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agentic_ai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# EXPECTED FEATURE SCHEMA
# =====================================================================

ORIGINAL_FEATURES = [
    "Age",
    "Gender",
    "Region",
    "Socioeconomic_Status",
    "Symptoms",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI",
]

ENGINEERED_FEATURES = [
    "BMI_Category",
    "Age_Group",
    "High_Glucose",
    "High_Cholesterol",
    "HbA1c_Category",
]

FEATURES = (
    ORIGINAL_FEATURES
    + ENGINEERED_FEATURES
)


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


NUMERICAL_FEATURES = [
    "Age",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI",
]


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_artifacts():
    """
    Load the trained model, preprocessing pipeline,
    label encoder and metadata.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Primary diagnosis model not found:\n{MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Primary diagnosis preprocessor not found:\n"
            f"{PREPROCESSOR_PATH}"
        )

    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Primary diagnosis label encoder not found:\n"
            f"{LABEL_ENCODER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    metadata = {}

    if METADATA_PATH.exists():
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

    return (
        model,
        preprocessor,
        label_encoder,
        metadata,
    )


# =====================================================================
# FEATURE ENGINEERING HELPERS
# =====================================================================

def get_bmi_category(bmi: float) -> str:
    """
    Create BMI category.

    Categories match the feature-engineering logic used
    by the HealthAI model pipeline.
    """

    bmi = float(bmi)

    if bmi < 18.5:
        return "Underweight"

    if bmi < 25:
        return "Normal"

    if bmi < 30:
        return "Overweight"

    return "Obese"


def get_age_group(age: float) -> str:
    """
    Create age group.
    """

    age = float(age)

    if age < 18:
        return "Child"

    if age < 35:
        return "Young Adult"

    if age < 55:
        return "Middle Age"

    return "Senior"


def get_high_glucose(glucose: float) -> str:
    """
    Create categorical high-glucose feature.

    IMPORTANT:
    Training treated High_Glucose as categorical.
    Therefore return "Yes"/"No", not integer 1/0.
    """

    glucose = float(glucose)

    if glucose >= 126:
        return "Yes"

    return "No"


def get_high_cholesterol(cholesterol: float) -> str:
    """
    Create categorical high-cholesterol feature.

    IMPORTANT:
    Training treated High_Cholesterol as categorical.
    Therefore return "Yes"/"No", not integer 1/0.
    """

    cholesterol = float(cholesterol)

    if cholesterol >= 200:
        return "Yes"

    return "No"


def get_hba1c_category(hba1c: float) -> str:
    """
    Create HbA1c category.
    """

    hba1c = float(hba1c)

    if hba1c < 5.7:
        return "Normal"

    if hba1c < 6.5:
        return "Prediabetes"

    return "High"


# =====================================================================
# VALUE NORMALIZATION
# =====================================================================

def normalize_binary_categorical_value(
    value,
    feature_name: str
) -> str:
    """
    Normalize High_Glucose / High_Cholesterol values into the
    categorical representation expected by the saved encoder.

    Accepted inputs:
        Yes
        No
        True
        False
        1
        0
        High
        Low

    Returned values:
        "Yes"
        "No"
    """

    if isinstance(value, str):

        normalized = value.strip().lower()

        if normalized in {
            "yes",
            "true",
            "1",
            "high",
        }:
            return "Yes"

        if normalized in {
            "no",
            "false",
            "0",
            "low",
        }:
            return "No"

    elif isinstance(value, bool):

        return "Yes" if value else "No"

    elif isinstance(value, (int, float)):

        if int(value) == 1:
            return "Yes"

        if int(value) == 0:
            return "No"

    raise ValueError(
        f"Invalid value for {feature_name}: {value!r}. "
        "Expected Yes/No or 1/0."
    )


# =====================================================================
# INPUT NORMALIZATION
# =====================================================================

def normalize_patient_data(
    patient_data: dict
) -> dict:
    """
    Convert incoming patient data into the exact feature schema
    expected by the saved preprocessing pipeline.

    Missing engineered features are automatically generated.
    """

    if not isinstance(patient_data, dict):
        raise TypeError(
            "patient_data must be a dictionary."
        )

    normalized = dict(patient_data)

    # -------------------------------------------------------------
    # Validate original required fields
    # -------------------------------------------------------------

    required_original = [
        "Age",
        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI",
    ]

    missing = [
        feature
        for feature in required_original
        if feature not in normalized
    ]

    if missing:

        raise ValueError(
            "Missing required patient fields: "
            + ", ".join(missing)
        )

    # -------------------------------------------------------------
    # Numeric normalization
    # -------------------------------------------------------------

    normalized["Age"] = float(
        normalized["Age"]
    )

    normalized["Blood_Glucose_mg_dL"] = float(
        normalized["Blood_Glucose_mg_dL"]
    )

    normalized["HbA1c_%"] = float(
        normalized["HbA1c_%"]
    )

    normalized["Total_Cholesterol_mg_dL"] = float(
        normalized["Total_Cholesterol_mg_dL"]
    )

    normalized["BMI"] = float(
        normalized["BMI"]
    )

    # -------------------------------------------------------------
    # String normalization
    # -------------------------------------------------------------

    normalized["Gender"] = str(
        normalized["Gender"]
    ).strip()

    normalized["Region"] = str(
        normalized["Region"]
    ).strip()

    normalized["Socioeconomic_Status"] = str(
        normalized["Socioeconomic_Status"]
    ).strip()

    normalized["Symptoms"] = str(
        normalized["Symptoms"]
    ).strip()

    # -------------------------------------------------------------
    # BMI category
    # -------------------------------------------------------------

    if not normalized.get("BMI_Category"):

        normalized["BMI_Category"] = (
            get_bmi_category(
                normalized["BMI"]
            )
        )

    else:

        normalized["BMI_Category"] = str(
            normalized["BMI_Category"]
        ).strip()

    # -------------------------------------------------------------
    # Age group
    # -------------------------------------------------------------

    if not normalized.get("Age_Group"):

        normalized["Age_Group"] = (
            get_age_group(
                normalized["Age"]
            )
        )

    else:

        normalized["Age_Group"] = str(
            normalized["Age_Group"]
        ).strip()

    # -------------------------------------------------------------
    # High glucose
    # -------------------------------------------------------------

    if "High_Glucose" not in normalized:

        normalized["High_Glucose"] = (
            get_high_glucose(
                normalized["Blood_Glucose_mg_dL"]
            )
        )

    else:

        normalized["High_Glucose"] = (
            normalize_binary_categorical_value(
                normalized["High_Glucose"],
                "High_Glucose"
            )
        )

    # -------------------------------------------------------------
    # High cholesterol
    # -------------------------------------------------------------

    if "High_Cholesterol" not in normalized:

        normalized["High_Cholesterol"] = (
            get_high_cholesterol(
                normalized["Total_Cholesterol_mg_dL"]
            )
        )

    else:

        normalized["High_Cholesterol"] = (
            normalize_binary_categorical_value(
                normalized["High_Cholesterol"],
                "High_Cholesterol"
            )
        )

    # -------------------------------------------------------------
    # HbA1c category
    # -------------------------------------------------------------

    if not normalized.get("HbA1c_Category"):

        normalized["HbA1c_Category"] = (
            get_hba1c_category(
                normalized["HbA1c_%"]
            )
        )

    else:

        normalized["HbA1c_Category"] = str(
            normalized["HbA1c_Category"]
        ).strip()

    return normalized


# =====================================================================
# BUILD MODEL INPUT
# =====================================================================

def build_model_input(
    patient_data: dict
) -> pd.DataFrame:
    """
    Build a one-row DataFrame with the exact training feature
    names and order.
    """

    normalized = normalize_patient_data(
        patient_data
    )

    row = {}

    for feature in FEATURES:

        if feature not in normalized:

            raise ValueError(
                f"Required feature '{feature}' "
                "is missing after normalization."
            )

        row[feature] = normalized[feature]

    dataframe = pd.DataFrame(
        [row],
        columns=FEATURES
    )

    # -------------------------------------------------------------
    # Explicit dtype protection
    # -------------------------------------------------------------
    #
    # This is important because sklearn's OneHotEncoder expects
    # consistent categorical values.
    #

    for feature in CATEGORICAL_FEATURES:

        dataframe[feature] = (
            dataframe[feature]
            .astype(str)
        )

    for feature in NUMERICAL_FEATURES:

        dataframe[feature] = pd.to_numeric(
            dataframe[feature],
            errors="raise"
        )

    return dataframe


# =====================================================================
# PREDICTION
# =====================================================================

def predict_primary_diagnosis(
    patient_data=None,
    query=None,
    **kwargs
) -> dict:
    """
    Predict the patient's primary diagnosis.

    Parameters
    ----------
    patient_data : dict
        Patient clinical information.

    query : str, optional
        Natural-language query supplied by the agent executor.

    Returns
    -------
    dict
        Standardized HealthAI prediction response.
    """

    # -------------------------------------------------------------
    # Accept patient_data from kwargs if necessary
    # -------------------------------------------------------------

    if patient_data is None:

        patient_data = kwargs.get(
            "patient_data"
        )

    # -------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------

    if patient_data is None:

        return {
            "status": "missing_input",
            "tool": "primary_diagnosis",
            "message": (
                "Patient data is required for "
                "primary diagnosis prediction."
            )
        }

    try:

        # ---------------------------------------------------------
        # Load artifacts
        # ---------------------------------------------------------

        (
            model,
            preprocessor,
            label_encoder,
            metadata,
        ) = load_artifacts()

        # ---------------------------------------------------------
        # Prepare input
        # ---------------------------------------------------------

        model_input = build_model_input(
            patient_data
        )

        # ---------------------------------------------------------
        # Transform
        # ---------------------------------------------------------

        processed_input = (
            preprocessor.transform(
                model_input
            )
        )

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------

        encoded_prediction = (
            model.predict(
                processed_input
            )
        )

        encoded_value = encoded_prediction[0]

        # ---------------------------------------------------------
        # Decode prediction
        # ---------------------------------------------------------

        try:

            prediction = (
                label_encoder.inverse_transform(
                    [encoded_value]
                )[0]
            )

        except Exception:

            prediction = str(
                encoded_value
            )

        # ---------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------

        confidence = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    processed_input
                )[0]
            )

            confidence = float(
                max(probabilities)
            )

        # ---------------------------------------------------------
        # Build response
        # ---------------------------------------------------------

        response = {
            "status": "success",
            "tool": "primary_diagnosis",
            "prediction_type": "primary_diagnosis",
            "predicted_primary_diagnosis": str(
                prediction
            ),
            "model": (
                metadata.get(
                    "selected_model",
                    "Random Forest"
                )
                if isinstance(
                    metadata,
                    dict
                )
                else "Random Forest"
            ),
            "target": "Primary_Diagnosis",
        }

        if confidence is not None:

            response["confidence"] = confidence

        # ---------------------------------------------------------
        # Optional query
        # ---------------------------------------------------------

        if query is not None:

            response["query"] = str(
                query
            )

        return response

    except Exception as exc:

        return {
            "status": "execution_error",
            "tool": "primary_diagnosis",
            "message": (
                "Primary diagnosis prediction failed."
            ),
            "error": str(exc),
        }


# =====================================================================
# ADAPTER COMPATIBILITY INTERFACE
# =====================================================================

def execute_primary_diagnosis(
    query: str = "",
    patient_data=None,
    **kwargs
) -> dict:
    """
    Compatibility wrapper for the Agent Executor.
    """

    return predict_primary_diagnosis(
        patient_data=patient_data,
        query=query,
        **kwargs
    )


# =====================================================================
# STANDALONE TEST PATIENT
# =====================================================================

def build_test_patient():
    """
    Test patient.

    Engineered categorical features intentionally use both the
    generated representation and the values expected by the
    training-time OneHotEncoder.
    """

    return {
        "Age": 52,
        "Gender": "Male",
        "Region": "Tamil Nadu",
        "Socioeconomic_Status": "Middle",
        "Symptoms": (
            "frequent urination, "
            "excessive thirst"
        ),
        "Blood_Glucose_mg_dL": 165,
        "HbA1c_%": 7.8,
        "Total_Cholesterol_mg_dL": 220,
        "BMI": 29.5,

        "BMI_Category": "Overweight",
        "Age_Group": "Middle Age",

        # IMPORTANT:
        # These are categorical.
        "High_Glucose": "Yes",
        "High_Cholesterol": "Yes",

        "HbA1c_Category": "High",
    }


# =====================================================================
# STANDALONE TEST
# =====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "HEALTHAI - PRIMARY DIAGNOSIS ADAPTER"
    )
    print("=" * 70)

    print("\nSTEP 17.5")

    try:

        # ---------------------------------------------------------
        # Check artifacts
        # ---------------------------------------------------------

        print("\nChecking model artifacts...")

        print(
            "Model       :",
            "AVAILABLE"
            if MODEL_PATH.exists()
            else "MISSING"
        )

        print(
            "Preprocessor:",
            "AVAILABLE"
            if PREPROCESSOR_PATH.exists()
            else "MISSING"
        )

        print(
            "Label encoder:",
            "AVAILABLE"
            if LABEL_ENCODER_PATH.exists()
            else "MISSING"
        )

        # ---------------------------------------------------------
        # Load test patient
        # ---------------------------------------------------------

        test_patient = (
            build_test_patient()
        )

        print(
            "\nTest patient:"
        )

        for key, value in test_patient.items():

            print(
                f"{key}: {value}"
            )

        # ---------------------------------------------------------
        # Normalize
        # ---------------------------------------------------------

        normalized = (
            normalize_patient_data(
                test_patient
            )
        )

        print(
            "\nNormalized categorical fields:"
        )

        print(
            "High_Glucose     :",
            normalized["High_Glucose"]
        )

        print(
            "High_Cholesterol:",
            normalized["High_Cholesterol"]
        )

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------

        result = predict_primary_diagnosis(
            patient_data=test_patient,
            query=(
                "What is the patient's "
                "primary diagnosis?"
            )
        )

        print(
            "\nPrediction result:"
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=str
            )
        )

        # ---------------------------------------------------------
        # Save test result
        # ---------------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / "primary_diagnosis_adapter_test.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                default=str
            )

        print(
            "\nTest result saved:"
        )

        print(output_path)

        # ---------------------------------------------------------
        # Final status
        # ---------------------------------------------------------

        if result.get("status") == "success":

            print(
                "\nSTEP 17.5 "
                "PRIMARY DIAGNOSIS ADAPTER TEST PASSED"
            )

        else:

            print(
                "\nSTEP 17.5 "
                "PRIMARY DIAGNOSIS ADAPTER TEST FAILED"
            )

    except Exception as exc:

        print(
            "\nSTEP 17.5 "
            "PRIMARY DIAGNOSIS ADAPTER TEST FAILED"
        )

        print(
            "Error:",
            str(exc)
        )