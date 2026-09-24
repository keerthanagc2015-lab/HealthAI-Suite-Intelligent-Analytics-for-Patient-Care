"""
HealthAI - Hospital Length of Stay Regression
STEP 17.10

Production-ready training script based on the existing Module 13 design.
SVR is intentionally excluded because it is computationally expensive for
80,000 training rows and is not necessary for the production comparison.

Run from project root:
python src/13_length_of_stay_regression/10_train_hospital_los_model.py
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "LengthOfStay.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "classical_ml" / "hospital_los"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "length_of_stay"

MODEL_PATH = MODEL_DIR / "hospital_los_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
COMPARISON_PATH = OUTPUT_DIR / "regression_model_comparison.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "hospital_los_test_predictions.csv"
PROCESSED_PATH = OUTPUT_DIR / "processed_length_of_stay.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Existing Module 13 feature design
FEATURES = [
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

TARGET = "lengthofstay"

CATEGORICAL_FEATURES = ["gender", "facid"]
NUMERICAL_FEATURES = [
    feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES
]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(random_state=RANDOM_STATE),
        "ElasticNet Regression": ElasticNet(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("WARNING: xgboost is not installed. XGBoost will be skipped.")

    return models


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    start_time = time.time()

    print("=" * 70)
    print("HEALTHAI - HOSPITAL LENGTH OF STAY MODEL TRAINING")
    print("STEP 17.10 - SVR SKIPPED FOR COMPUTATIONAL EFFICIENCY")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully")
    print(f"Shape: {df.shape}")

    required_columns = FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n" + "\n".join(missing)
        )

    # ---------------------------------------------------------------
    # CLEAN TARGET / RCOUNT
    # ---------------------------------------------------------------
    df = df.copy()

    # Existing Module 13 cleaning: rcount may contain '+'
    df["rcount"] = (
        df["rcount"]
        .astype(str)
        .str.strip()
        .str.replace("+", "", regex=False)
    )
    df["rcount"] = pd.to_numeric(df["rcount"], errors="coerce")
    df["rcount"] = df["rcount"].fillna(df["rcount"].median())

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    removed = before - len(df)

    print(f"\nRows removed due to invalid target: {removed}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print(f"\nInput shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nTarget summary:")
    print(y.describe())

    # Save processed dataset
    processed_df = df[FEATURES + [TARGET]].copy()
    processed_df.to_csv(PROCESSED_PATH, index=False)

    # ---------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # ---------------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------------
    models = build_models()

    results = []
    fitted_models = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    for name, model in models.items():
        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)
        print("Training...")

        model_start = time.time()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        model_rmse = rmse(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        elapsed = time.time() - model_start

        print("Training completed.")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {model_rmse:.4f}")
        print(f"R2   : {r2:.4f}")
        print(f"Time : {elapsed:.2f} seconds")

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": model_rmse,
                "R2": r2,
                "Training_Time_Seconds": elapsed,
            }
        )

        fitted_models[name] = pipeline

    # ---------------------------------------------------------------
    # MODEL COMPARISON
    # ---------------------------------------------------------------
    comparison_df = (
        pd.DataFrame(results)
        .sort_values("MAE", ascending=True)
        .reset_index(drop=True)
    )

    comparison_df["Rank"] = np.arange(1, len(comparison_df) + 1)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # ---------------------------------------------------------------
    # SELECT BEST MODEL BY LOWEST MAE
    # ---------------------------------------------------------------
    best_name = comparison_df.iloc[0]["Model"]
    best_pipeline = fitted_models[best_name]

    best_pred = best_pipeline.predict(X_test)

    best_mae = mean_absolute_error(y_test, best_pred)
    best_rmse = rmse(y_test, best_pred)
    best_r2 = r2_score(y_test, best_pred)

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(f"Selected model : {best_name}")
    print(f"MAE            : {best_mae:.4f}")
    print(f"RMSE           : {best_rmse:.4f}")
    print(f"R2             : {best_r2:.4f}")

    # ---------------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # ---------------------------------------------------------------
    predictions_df = X_test.reset_index(drop=True).copy()
    predictions_df["actual_lengthofstay"] = y_test.reset_index(drop=True)
    predictions_df["predicted_lengthofstay"] = best_pred
    predictions_df["absolute_error"] = (
        predictions_df["actual_lengthofstay"]
        - predictions_df["predicted_lengthofstay"]
    ).abs()

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    # ---------------------------------------------------------------
    # SAVE PRODUCTION ARTIFACT
    # Full pipeline = preprocessing + trained model
    # ---------------------------------------------------------------
    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "project": "HealthAI-Intelligent-Platform",
        "module": "17_agentic_ai",
        "step": "17.10",
        "task": "Hospital Length of Stay Regression",
        "target": TARGET,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "selected_model": best_name,
        "selection_metric": "MAE",
        "selection_rule": "Lowest test MAE",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {
            "MAE": float(best_mae),
            "RMSE": float(best_rmse),
            "R2": float(best_r2),
        },
        "all_model_results": comparison_df.to_dict(orient="records"),
        "svr_skipped": True,
        "svr_reason": (
            "Standard SVR was skipped because it scales poorly with "
            "80,000 training rows and caused excessive runtime."
        ),
        "artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 70)
    print("ARTIFACTS SAVED")
    print("=" * 70)
    print(f"Model     : {MODEL_PATH}")
    print(f"Metadata  : {METADATA_PATH}")
    print(f"Comparison: {COMPARISON_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Processed : {PROCESSED_PATH}")

    print("\n" + "=" * 70)
    print(f"TOTAL TRAINING TIME: {elapsed_total:.2f} seconds")
    print("STEP 17.10 HOSPITAL LOS MODEL TRAINING PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
