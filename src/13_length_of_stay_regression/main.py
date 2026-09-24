# ==========================================================
# HealthAI Suite - Length of Stay Regression
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/LengthOfStay.csv"

OUTPUT_DIR = "data/processed/length_of_stay"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("HealthAI Suite - Length of Stay Regression")
    print("=" * 70)

    # ======================================================
    # 1. Load Dataset
    # ======================================================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # ======================================================
    # 2. Basic Dataset Information
    # ======================================================

    print("\n" + "=" * 70)
    print("Dataset Information")
    print("=" * 70)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(df["lengthofstay"].describe())

    # ======================================================
    # 3. Clean Readmission Count
    # ======================================================

    print("\n" + "=" * 70)
    print("Data Cleaning")
    print("=" * 70)

    # rcount contains values such as 0, 1, 2, 3, 4, 5+
    # Convert it into a numeric feature.

    df["rcount"] = (
        df["rcount"]
        .astype(str)
        .str.replace("+", "", regex=False)
    )

    df["rcount"] = pd.to_numeric(
        df["rcount"],
        errors="coerce"
    )

    # Fill any conversion-generated missing values
    df["rcount"] = df["rcount"].fillna(
        df["rcount"].median()
    )

    print("\nrcount converted to numeric.")

    print(
        "Missing values after cleaning:",
        df.isnull().sum().sum()
    )

    # ======================================================
    # 4. Feature Selection
    # ======================================================

    selected_features = [

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
        "facid"
    ]

    target = "lengthofstay"

    X = df[selected_features].copy()

    y = df[target].copy()

    print("\n" + "=" * 70)
    print("Feature Selection")
    print("=" * 70)

    print("\nSelected Features:")

    for feature in selected_features:
        print(feature)

    print("\nTarget:")
    print(target)

    print("\nInput Shape:", X.shape)
    print("Target Shape:", y.shape)

    # ======================================================
    # 5. Train-Test Split
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42

    )

    print("\n" + "=" * 70)
    print("Train-Test Split")
    print("=" * 70)

    print("\nX_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\ny_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # ======================================================
    # 6. Identify Feature Types
    # ======================================================

    categorical_features = [

        "gender",
        "facid"

    ]

    numerical_features = [

        "rcount",

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

        "secondarydiagnosisnonicd9"

    ]

    print("\n" + "=" * 70)
    print("Feature Types")
    print("=" * 70)

    print("\nCategorical Features:")
    print(categorical_features)

    print("\nNumerical Features:")
    print(numerical_features)

    # ======================================================
    # 7. Preprocessing
    # ======================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                StandardScaler(),
                numerical_features
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )

        ]

    )

    print("\n" + "=" * 70)
    print("Preprocessing")
    print("=" * 70)

    print(
        "\nNumerical features will be standardized."
    )

    print(
        "Categorical features will be one-hot encoded."
    )

    # ======================================================
    # 8. Define Regression Models
    # ======================================================

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(alpha=1.0),

        "Lasso Regression":
            Lasso(alpha=0.001),

        "ElasticNet Regression":
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.5
            ),

        "SVR":
            SVR(
                kernel="rbf",
                C=10,
                epsilon=0.1
            ),

        "Decision Tree Regressor":
            DecisionTreeRegressor(
                max_depth=12,
                random_state=42
            ),

        "Random Forest Regressor":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost Regressor":
            XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )

    }

    # ======================================================
    # 9. Train and Evaluate Models
    # ======================================================

    results = []

    print("\n" + "=" * 70)
    print("Regression Model Training")
    print("=" * 70)

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(model_name)
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
                )

            ]

        )

        print("Training...")

        pipeline.fit(
            X_train,
            y_train
        )

        print("Training completed.")

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        y_pred = pipeline.predict(X_test)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        print("\nEvaluation:")

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )

        results.append({

            "Model": model_name,

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2

        })

    # ======================================================
    # 10. Model Comparison
    # ======================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by="MAE",

        ascending=True

    )

    print("\n" + "=" * 70)
    print("FINAL REGRESSION MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ======================================================
    # 11. Select Best Model
    # ======================================================

    best_model_name = results_df.iloc[0]["Model"]

    print("\n" + "=" * 70)
    print("BEST REGRESSION MODEL")
    print("=" * 70)

    print(
        "\nBest Model:",
        best_model_name
    )

    print(
        "Selected primarily based on lowest MAE."
    )

    # ======================================================
    # 12. Save Results
    # ======================================================

    results_path = os.path.join(

        OUTPUT_DIR,

        "regression_model_comparison.csv"

    )

    results_df.to_csv(

        results_path,

        index=False

    )

    print(
        "\nResults saved to:",
        results_path
    )

    # ======================================================
    # 13. Save Cleaned Dataset
    # ======================================================

    processed_path = os.path.join(

        OUTPUT_DIR,

        "processed_length_of_stay.csv"

    )

    df.to_csv(

        processed_path,

        index=False

    )

    print(
        "Processed dataset saved to:",
        processed_path
    )

    print("\n" + "=" * 70)
    print("Length of Stay Regression Completed")
    print("=" * 70)


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()