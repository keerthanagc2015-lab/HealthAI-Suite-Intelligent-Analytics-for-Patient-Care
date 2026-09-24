# ==========================================================
# HealthAI Suite
# Length of Stay - Feature Engineered Model Comparison
# SVR is intentionally handled separately
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

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# 1. PATHS
# ==========================================================

DATA_PATH = (
    "data/processed/length_of_stay/"
    "feature_engineering/length_of_stay_engineered.csv"
)

OUTPUT_DIR = (
    "data/processed/length_of_stay/"
    "feature_engineered_model"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print("=" * 70)
print("HEALTHAI SUITE - LENGTH OF STAY MODEL COMPARISON")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================================
# 3. TARGET
# ==========================================================

target = "lengthofstay"

y = df[target].copy()


# ==========================================================
# 4. EXCLUDE FEATURES
# ==========================================================

excluded_features = [
    "eid",
    "vdate",
    "discharged"
]

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_features + [target]
]

X = df[feature_columns].copy()

print("\n" + "=" * 70)
print("FEATURE SET")
print("=" * 70)

print("\nExcluded:")
for feature in excluded_features:
    print(f"  {feature}")

print("\nNumber of Features:", len(feature_columns))

print("\nFeatures:")
for feature in feature_columns:
    print(f"  {feature}")


# ==========================================================
# 5. CLEAN rcount
# ==========================================================

if "rcount" in X.columns:

    X["rcount"] = (
        X["rcount"]
        .astype(str)
        .str.replace("+", "", regex=False)
    )

    X["rcount"] = pd.to_numeric(
        X["rcount"],
        errors="coerce"
    )

    X["rcount"] = X["rcount"].fillna(
        X["rcount"].median()
    )


# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================================
# 7. FEATURE TYPES
# ==========================================================

categorical_features = [
    "gender",
    "facid"
]

numerical_features = [
    feature
    for feature in X.columns
    if feature not in categorical_features
]

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ==========================================================
# 8. PREPROCESSOR
# ==========================================================

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


# ==========================================================
# 9. MODELS
# ==========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(
            alpha=1.0
        ),

    "Lasso Regression":
        Lasso(
            alpha=0.001,
            max_iter=10000
        ),

    "ElasticNet Regression":
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000
        ),

    "Decision Tree Regressor":
        DecisionTreeRegressor(
            max_depth=12,
            random_state=42
        ),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost Regressor":
        XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

}


# ==========================================================
# 10. TRAIN MODELS ONE BY ONE
# ==========================================================

results = []

print("\n" + "=" * 70)
print("MODEL TRAINING")
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

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    print("Predicting...")

    y_pred = pipeline.predict(
        X_test
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

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

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })


# ==========================================================
# 11. MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# 12. BEST MODEL
# ==========================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "\nModel:",
    best_model["Model"]
)

print(
    "MAE:",
    round(best_model["MAE"], 4)
)

print(
    "RMSE:",
    round(best_model["RMSE"], 4)
)

print(
    "R²:",
    round(best_model["R2"], 4)
)


# ==========================================================
# 13. SAVE RESULTS
# ==========================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "feature_engineered_model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    "\nResults saved to:",
    results_path
)


# ==========================================================
# 14. SAVE BEST MODEL INFORMATION
# ==========================================================

best_model_df = pd.DataFrame([best_model])

best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_model_results.csv"
)

best_model_df.to_csv(
    best_model_path,
    index=False
)

print(
    "Best model information saved to:",
    best_model_path
)


# ==========================================================
# 15. COMPLETION
# ==========================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

print(
    "\nSVR is intentionally NOT included here."
)

print(
    "SVR will be trained separately because it is"
    " computationally expensive for this dataset."
)