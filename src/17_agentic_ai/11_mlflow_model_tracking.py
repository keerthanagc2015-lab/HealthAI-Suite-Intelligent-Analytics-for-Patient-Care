"""
HealthAI - MLflow Model Tracking
Module 17.16

Purpose
-------
Track the already-trained HealthAI models in MLflow.

No model retraining is performed.

The script:
1. Creates a SQLite MLflow tracking database.
2. Creates MLflow experiments.
3. Records the actual evaluation metrics already obtained.
4. Logs existing model artifacts when available.
5. Saves a centralized MLflow tracking summary.

Output
------
mlflow.db
data/processed/agentic_ai/mlflow_tracking_summary.json
"""

from pathlib import Path
import json
from datetime import datetime

import mlflow


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# SQLite database for modern MLflow versions
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

SUMMARY_DIR = DATA_PROCESSED / "agentic_ai"

SUMMARY_FILE = SUMMARY_DIR / "mlflow_tracking_summary.json"


# =====================================================================
# MLFLOW SETUP
# =====================================================================

# Current MLflow versions no longer accept the old filesystem
# tracking backend by default.
#
# SQLite provides a supported local tracking backend.

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def relative_path(path):
    """
    Return a project-relative path for reporting.
    """

    path = Path(path)

    try:
        return str(path.relative_to(PROJECT_ROOT))

    except ValueError:
        return str(path)


def log_artifact_if_exists(path):
    """
    Log an artifact if it exists.

    Returns:
        True  -> artifact successfully logged
        False -> artifact does not exist
    """

    path = Path(path)

    if not path.exists():
        return False

    if path.is_file():

        mlflow.log_artifact(
            str(path),
            artifact_path="model_artifacts",
        )

        return True

    if path.is_dir():

        mlflow.log_artifacts(
            str(path),
            artifact_path="model_artifacts",
        )

        return True

    return False


def create_model_run(
    model_name,
    task,
    framework,
    metrics,
    model_artifacts,
    parameters=None,
):
    """
    Create one MLflow run for an already-trained model.
    """

    experiment_name = f"HealthAI/{task}"

    # ---------------------------------------------------------------
    # Get or create experiment
    # ---------------------------------------------------------------

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:

        mlflow.create_experiment(
            experiment_name
        )

    mlflow.set_experiment(
        experiment_name
    )

    # ---------------------------------------------------------------
    # Start MLflow run
    # ---------------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # -----------------------------------------------------------
        # Tags
        # -----------------------------------------------------------

        mlflow.set_tags(
            {
                "project": "HealthAI-Intelligent-Platform",
                "module": "17_agentic_ai",
                "step": "17.16",
                "task": task,
                "framework": framework,
                "tracking_type": "existing_trained_model",
                "clinical_use": "portfolio_only",
            }
        )

        # -----------------------------------------------------------
        # Parameters
        # -----------------------------------------------------------

        if parameters:

            for key, value in parameters.items():

                mlflow.log_param(
                    key,
                    str(value),
                )

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        valid_metrics = {}

        for key, value in metrics.items():

            if value is None:
                continue

            try:

                numeric_value = float(value)

                mlflow.log_metric(
                    key,
                    numeric_value,
                )

                valid_metrics[key] = numeric_value

            except (TypeError, ValueError):

                print(
                    f"    Warning: could not log metric "
                    f"{key}={value}"
                )

        # -----------------------------------------------------------
        # Model artifacts
        # -----------------------------------------------------------

        logged_artifacts = []

        missing_artifacts = []

        for artifact in model_artifacts:

            artifact = Path(artifact)

            if log_artifact_if_exists(
                artifact
            ):

                logged_artifacts.append(
                    relative_path(artifact)
                )

            else:

                missing_artifacts.append(
                    relative_path(artifact)
                )

        # -----------------------------------------------------------
        # Run summary
        # -----------------------------------------------------------

        result = {

            "model_name": model_name,

            "task": task,

            "framework": framework,

            "run_id": run.info.run_id,

            "experiment": experiment_name,

            "metrics": valid_metrics,

            "logged_artifacts": logged_artifacts,

            "missing_artifacts": missing_artifacts,
        }

        return result


# =====================================================================
# MODEL DEFINITIONS
# =====================================================================

def build_model_definitions():
    """
    Define the models using metrics already obtained during the project.

    IMPORTANT:
    These models are NOT retrained here.
    """

    return [

        # =============================================================
        # 1. DIABETES PREDICTION
        # =============================================================

        {
            "model_name": "Diabetes_RandomForest",

            "task": "diabetes_prediction",

            "framework": "scikit-learn",

            "metrics": {
                "accuracy": 0.9276,
                "precision": 0.8661,
                "recall": 0.8795,
                "f1": 0.8728,
                "roc_auc": 0.9662,
            },

            "parameters": {
                "model_type": "RandomForestClassifier",
                "target": "Type 2 Diabetes",
            },

            "artifacts": [

                MODELS_DIR
                / "classical_ml"
                / "diabetes_prediction"
                / "diabetes_prediction_model.joblib",

                MODELS_DIR
                / "classical_ml"
                / "diabetes_prediction"
                / "metadata.json",

                MODELS_DIR
                / "classical_ml"
                / "diabetes_prediction"
                / "test_predictions.csv",
            ],
        },


        # =============================================================
        # 2. PRIMARY DIAGNOSIS
        # =============================================================

        {
            "model_name": "Primary_Diagnosis_RandomForest",

            "task": "primary_diagnosis",

            "framework": "scikit-learn",

            "metrics": {
                "accuracy": 0.4346,
                "macro_precision": 0.2283,
                "macro_recall": 0.2278,
                "macro_f1": 0.2255,
                "weighted_f1": 0.4181,
            },

            "parameters": {
                "model_type": "RandomForestClassifier",
                "target": "Primary_Diagnosis",
            },

            "artifacts": [

                MODELS_DIR
                / "classical_ml"
                / "primary_diagnosis"
                / "primary_diagnosis_model.joblib",

                MODELS_DIR
                / "classical_ml"
                / "primary_diagnosis"
                / "primary_diagnosis_preprocessor.joblib",

                MODELS_DIR
                / "classical_ml"
                / "primary_diagnosis"
                / "primary_diagnosis_label_encoder.joblib",

                MODELS_DIR
                / "classical_ml"
                / "primary_diagnosis"
                / "metadata.json",
            ],
        },


        # =============================================================
        # 3. HOSPITAL LENGTH OF STAY
        # =============================================================

        {
            "model_name": "Hospital_LOS_XGBoost",

            "task": "hospital_los_prediction",

            "framework": "XGBoost",

            "metrics": {
                "mae": 0.3201228380203247,
                "rmse": 0.43285540702027536,
                "r2": 0.9658543467521667,
            },

            "parameters": {
                "model_type": "XGBRegressor",
                "target": "lengthofstay",
            },

            "artifacts": [

                MODELS_DIR
                / "classical_ml"
                / "hospital_los"
                / "hospital_los_model.joblib",

                MODELS_DIR
                / "classical_ml"
                / "hospital_los"
                / "metadata.json",

                DATA_PROCESSED
                / "length_of_stay"
                / "hospital_los_test_predictions.csv",

                DATA_PROCESSED
                / "length_of_stay"
                / "regression_model_comparison.csv",
            ],
        },


        # =============================================================
        # 4. CHEST X-RAY CNN
        # =============================================================

        {
            "model_name": "Chest_XRay_CNN",

            "task": "xray_analysis",

            "framework": "TensorFlow/Keras",

            "metrics": {
                "accuracy": 0.9917,
                "precision": 0.9917,
                "recall": 0.9917,
                "f1": 0.9917,
            },

            "parameters": {
                "model_type": "CNN",
                "task_type": "image_classification",
                "target": "pneumonia",
            },

            "artifacts": [

                DATA_PROCESSED
                / "deep_learning"
                / "cnn"
                / "cnn_model.keras",

                DATA_PROCESSED
                / "deep_learning"
                / "cnn"
                / "xray_gradcam_explanation.png",

                DATA_PROCESSED
                / "deep_learning"
                / "cnn"
                / "xray_gradcam_heatmap.png",
            ],
        },


        # =============================================================
        # 5. PATIENT DETERIORATION RNN
        # =============================================================

        {
            "model_name": "Patient_Deterioration_RNN",

            "task": "deterioration_prediction",

            "framework": "TensorFlow/Keras",

            "metrics": {
                "accuracy": 0.8453,
                "precision": 0.3270,
                "recall": 0.6150,
                "f1": 0.4270,
                "roc_auc": 0.8199,
            },

            "parameters": {
                "model_type": "RNN",
                "sequence_length": 24,
                "features": 28,
                "prediction_horizon": "12 hours",
            },

            "artifacts": [

                DATA_PROCESSED
                / "deep_learning"
                / "rnn"
                / "rnn_model.keras",
            ],
        },


        # =============================================================
        # 6. BIOBERT MEDICAL NER
        # =============================================================

        {
            "model_name": "BioBERT_Medical_NER",

            "task": "medical_ner",

            "framework": "HuggingFace Transformers",

            "metrics": {
                "token_accuracy": 0.7816,
                "entity_precision": 0.4101,
                "entity_recall": 0.4602,
                "entity_f1": 0.4337,
            },

            "parameters": {
                "model_type": "BioBERT",
                "base_model": "dmis-lab/biobert-v1.1",
                "epochs": 3,
                "learning_rate": 0.00002,
            },

            "artifacts": [

                MODELS_DIR
                / "nlp"
                / "biobert_ner",
            ],
        },


        # =============================================================
        # 7. CLINICALBERT MEDICAL NER
        # =============================================================

        {
            "model_name": "ClinicalBERT_Medical_NER",

            "task": "medical_ner",

            "framework": "HuggingFace Transformers",

            "metrics": {
                "token_accuracy": 0.7714,
                "entity_precision": 0.3687,
                "entity_recall": 0.4408,
                "entity_f1": 0.4016,
            },

            "parameters": {
                "model_type": "ClinicalBERT",
                "base_model": "emilyalsentzer/Bio_ClinicalBERT",
                "epochs": 3,
                "learning_rate": 0.00002,
            },

            "artifacts": [

                MODELS_DIR
                / "nlp"
                / "clinicalbert_ner",
            ],
        },


        # =============================================================
        # 8. SENTIMENT ANALYSIS
        # =============================================================

        {
            "model_name": "Sentiment_TFIDF_LogisticRegression",

            "task": "sentiment_analysis",

            "framework": "scikit-learn",

            "metrics": {
                "test_accuracy": 0.50,
                "test_precision": 0.50,
                "test_recall": 1.00,
                "test_f1": 0.6667,
            },

            "parameters": {
                "model_type": "LogisticRegression",
                "representation": "TF-IDF",
                "evaluation_note": "20 unique feedback texts",
            },

            "artifacts": [

                MODELS_DIR
                / "nlp"
                / "sentiment"
                / "tfidf_logistic_regression"
                / "tfidf_vectorizer.pkl",

                MODELS_DIR
                / "nlp"
                / "sentiment"
                / "tfidf_logistic_regression"
                / "logistic_regression_model.pkl",
            ],
        },
    ]


# =====================================================================
# MAIN
# =====================================================================

def run_mlflow_tracking():

    print("=" * 72)
    print("HEALTHAI - MLFLOW MODEL TRACKING")
    print("STEP 17.16")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Build model list
    # ---------------------------------------------------------------

    definitions = build_model_definitions()

    results = []

    # ---------------------------------------------------------------
    # Track every model
    # ---------------------------------------------------------------

    for index, model in enumerate(
        definitions,
        start=1,
    ):

        print(
            f"\n[{index}/{len(definitions)}] "
            f"{model['model_name']}"
        )

        try:

            result = create_model_run(
                model_name=model["model_name"],
                task=model["task"],
                framework=model["framework"],
                metrics=model["metrics"],
                model_artifacts=model["artifacts"],
                parameters=model["parameters"],
            )

            results.append(result)

            print(
                f"    Experiment : "
                f"{result['experiment']}"
            )

            print(
                f"    Run ID     : "
                f"{result['run_id']}"
            )

            print(
                f"    Artifacts  : "
                f"{len(result['logged_artifacts'])}"
            )

            print(
                f"    Missing    : "
                f"{len(result['missing_artifacts'])}"
            )

        except Exception as error:

            print(
                f"    ERROR      : "
                f"{type(error).__name__}: {error}"
            )

            results.append(
                {
                    "model_name": model["model_name"],
                    "task": model["task"],
                    "framework": model["framework"],
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )


    # =================================================================
    # SUMMARY
    # =================================================================

    total_models = len(definitions)

    successful_runs = sum(
        1
        for result in results
        if result.get("run_id")
    )

    total_artifacts = sum(
        len(result.get("logged_artifacts", []))
        for result in results
    )

    total_missing = sum(
        len(result.get("missing_artifacts", []))
        for result in results
    )

    total_errors = sum(
        1
        for result in results
        if result.get("status") == "error"
    )


    summary = {

        "project": "HealthAI-Intelligent-Platform",

        "module": "17_agentic_ai",

        "step": "17.16",

        "tracking_system": "MLflow",

        "tracking_backend": "SQLite",

        "tracking_database": "mlflow.db",

        "tracking_type": "existing_trained_models",

        "generated_at": datetime.now().isoformat(),

        "summary": {

            "models_defined": total_models,

            "successful_mlflow_runs": successful_runs,

            "failed_mlflow_runs": total_errors,

            "artifacts_logged": total_artifacts,

            "missing_artifacts": total_missing,
        },

        "models": results,

        "notes": [

            "Models were not retrained during this step.",

            "Previously obtained evaluation metrics were logged.",

            "MLflow is used for experiment tracking and artifact tracking.",

            "SQLite is used instead of the deprecated filesystem backend.",

            "Model explanations describe model behavior and are not clinical causality.",

            "HealthAI models are portfolio models and are not clinically validated.",

        ],
    }


    # =================================================================
    # SAVE SUMMARY
    # =================================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


    # =================================================================
    # FINAL OUTPUT
    # =================================================================

    print("\n" + "-" * 72)

    print(
        f"Models defined       : "
        f"{total_models}"
    )

    print(
        f"Successful runs      : "
        f"{successful_runs}"
    )

    print(
        f"Failed runs          : "
        f"{total_errors}"
    )

    print(
        f"Artifacts logged     : "
        f"{total_artifacts}"
    )

    print(
        f"Missing artifacts    : "
        f"{total_missing}"
    )

    print(
        f"\nMLflow database      : "
        f"{MLFLOW_DB}"
    )

    print(
        f"Summary saved        : "
        f"{SUMMARY_FILE}"
    )

    print("\n" + "-" * 72)


    # ---------------------------------------------------------------
    # Success criteria
    # ---------------------------------------------------------------

    if (
        successful_runs == total_models
        and total_errors == 0
    ):

        print(
            "STEP 17.16 MLFLOW MODEL TRACKING PASSED"
        )

        return True


    print(
        "STEP 17.16 MLFLOW MODEL TRACKING FAILED"
    )

    return False


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    success = run_mlflow_tracking()

    if not success:

        raise SystemExit(1)