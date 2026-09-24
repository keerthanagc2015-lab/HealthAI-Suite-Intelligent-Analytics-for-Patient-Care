"""
HealthAI - Hospital Length of Stay Agent Adapter
STEP 17.11

Loads the production LOS pipeline saved by STEP 17.10 and exposes a
standard agent-tool interface.

Required patient fields follow the Module 13 regression feature design:
    rcount, gender, dialysisrenalendstage, asthma, irondef, pneum,
    substancedependence, psychologicaldisordermajor, depress, psychother,
    fibrosisandother, malnutrition, hemo, hematocrit, neutrophils, sodium,
    glucose, bloodureanitro, creatinine, bmi, pulse, respiration,
    secondarydiagnosisnonicd9, facid

The saved artifact contains preprocessing + XGBoost, so the adapter passes
raw feature values directly to the fitted Pipeline.
"""

from pathlib import Path
import json
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classical_ml"
    / "hospital_los"
    / "hospital_los_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "classical_ml"
    / "hospital_los"
    / "metadata.json"
)

REQUIRED_FEATURES = [
    "rcount",
    "gender",
    "dialysisrenalendstage",
    "asthma",
    "irondef",
    "pneum",
    "substancedependence",
    "psychologicaldisordermajor",
    "depress",
    "psychother",
    "fibrosisandother",
    "malnutrition",
    "hemo",
    "hematocrit",
    "neutrophils",
    "sodium",
    "glucose",
    "bloodureanitro",
    "creatinine",
    "bmi",
    "pulse",
    "respiration",
    "secondarydiagnosisnonicd9",
    "facid",
]

_model = None
_metadata = None


def load_hospital_los_model():
    """Lazy-load the fitted production pipeline."""
    global _model, _metadata

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Hospital LOS model not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    if _metadata is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        else:
            _metadata = {}

    return _model, _metadata


def predict_hospital_los(patient_data: dict):
    """
    Predict hospital length of stay in days.

    Returns a structured dictionary suitable for the Agent Executor.
    """
    if not isinstance(patient_data, dict):
        return {
            "status": "error",
            "error": "patient_data must be a dictionary",
        }

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in patient_data
    ]

    if missing:
        return {
            "status": "missing_input",
            "missing_features": missing,
            "message": (
                "Hospital LOS prediction requires all configured "
                "patient features."
            ),
        }

    try:
        model, metadata = load_hospital_los_model()

        row = {feature: patient_data[feature] for feature in REQUIRED_FEATURES}
        df = pd.DataFrame([row])

        # Match the training-time rcount cleaning.
        df["rcount"] = (
            df["rcount"]
            .astype(str)
            .str.strip()
            .str.replace("+", "", regex=False)
        )
        df["rcount"] = pd.to_numeric(df["rcount"], errors="coerce")

        prediction = float(model.predict(df)[0])

        # LOS cannot be negative.
        prediction = max(0.0, prediction)

        model_name = metadata.get("selected_model", "XGBoost")

        return {
            "status": "success",
            "prediction_type": "hospital_length_of_stay",
            "predicted_length_of_stay_days": round(prediction, 2),
            "model": model_name,
            "target": metadata.get("target", "lengthofstay"),
            "mae": metadata.get("metrics", {}).get("MAE"),
            "rmse": metadata.get("metrics", {}).get("RMSE"),
            "r2": metadata.get("metrics", {}).get("R2"),
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def execute_hospital_los_prediction(
    query: str = "",
    patient_data: dict | None = None,
    **kwargs,
):
    """
    Standard Agent Executor interface.

    The query is retained for compatibility with the agent architecture.
    Actual LOS prediction requires structured patient_data.
    """
    if patient_data is None:
        patient_data = kwargs.get("patient_data")

    if patient_data is None:
        return {
            "status": "missing_input",
            "message": (
                "Hospital LOS prediction requires structured "
                "patient_data."
            ),
            "prediction_type": "hospital_length_of_stay",
        }

    return predict_hospital_los(patient_data)


def build_test_patient():
    """Create a valid test record using the Module 13 feature schema."""
    return {
        "rcount": 3,
        "gender": "M",
        "dialysisrenalendstage": 0,
        "asthma": 0,
        "irondef": 0,
        "pneum": 0,
        "substancedependence": 0,
        "psychologicaldisordermajor": 0,
        "depress": 0,
        "psychother": 0,
        "fibrosisandother": 0,
        "malnutrition": 0,
        "hemo": 0,
        "hematocrit": 40.0,
        "neutrophils": 65.0,
        "sodium": 139.0,
        "glucose": 110.0,
        "bloodureanitro": 15.0,
        "creatinine": 1.0,
        "bmi": 25.0,
        "pulse": 78.0,
        "respiration": 18.0,
        "secondarydiagnosisnonicd9": 1,
        "facid": "A",
    }


def main():
    print("=" * 70)
    print("HEALTHAI - HOSPITAL LENGTH OF STAY AGENT ADAPTER")
    print("STEP 17.11")
    print("=" * 70)

    print("\nModel path:")
    print(MODEL_PATH)

    print("\nLoading production artifact...")

    try:
        model, metadata = load_hospital_los_model()

        print("Model loaded successfully.")
        print(
            f"Selected model: "
            f"{metadata.get('selected_model', 'Unknown')}"
        )
        print(
            f"Model metrics: "
            f"MAE={metadata.get('metrics', {}).get('MAE')}, "
            f"RMSE={metadata.get('metrics', {}).get('RMSE')}, "
            f"R2={metadata.get('metrics', {}).get('R2')}"
        )

        test_patient = build_test_patient()

        print("\nRunning standalone prediction test...")
        result = execute_hospital_los_prediction(
            query="Predict hospital length of stay",
            patient_data=test_patient,
        )

        print("\nPrediction result:")
        for key, value in result.items():
            print(f"{key}: {value}")

        if result.get("status") != "success":
            raise RuntimeError(
                f"Adapter test failed: {result}"
            )

        output_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "agentic_ai"
            / "hospital_los_adapter_test.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_patient": test_patient,
                    "result": result,
                },
                f,
                indent=2,
            )

        print("\nTest result saved to:")
        print(output_path)

        print("\n" + "=" * 70)
        print("STEP 17.11 HOSPITAL LOS ADAPTER TEST PASSED")
        print("=" * 70)

    except Exception as exc:
        print("\n" + "=" * 70)
        print("STEP 17.11 FAILED")
        print("=" * 70)
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
