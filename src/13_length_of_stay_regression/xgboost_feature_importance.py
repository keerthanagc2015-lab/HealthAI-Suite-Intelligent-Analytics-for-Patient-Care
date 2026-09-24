# ==========================================================
# HealthAI Suite - LOS XGBoost Feature Importance
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = (
    "data/processed/length_of_stay/"
    "feature_engineering/length_of_stay_engineered.csv"
)

OUTPUT_DIR = (
    "data/processed/length_of_stay/"
    "feature_importance"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Dataset
# ==========================================================

print("=" * 70)
print("LOS - XGBOOST FEATURE IMPORTANCE")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. Target
# ==========================================================

target = "lengthofstay"

y = df[target].copy()


# ==========================================================
# 3. Exclude Features
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


# ==========================================================
# 4. Clean rcount
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
# 5. Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)


# ==========================================================
# 6. Feature Types
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


# ==========================================================
# 7. Preprocessing
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            "passthrough",
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
# 8. XGBoost
# ==========================================================

model = XGBRegressor(

    n_estimators=200,

    max_depth=8,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1

)


# ==========================================================
# 9. Pipeline
# ==========================================================

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


# ==========================================================
# 10. Train
# ==========================================================

print("\n" + "=" * 70)
print("Training XGBoost")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("XGBoost trained successfully.")


# ==========================================================
# 11. Get Processed Feature Names
# ==========================================================

fitted_preprocessor = pipeline.named_steps[
    "preprocessor"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


# ==========================================================
# 12. Get Feature Importance
# ==========================================================

xgb_model = pipeline.named_steps[
    "model"
]

importance_values = (
    xgb_model.feature_importances_
)


# ==========================================================
# 13. Create Importance DataFrame
# ==========================================================

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

).reset_index(drop=True)


# ==========================================================
# 14. Display Top 20
# ==========================================================

print("\n" + "=" * 70)
print("TOP 20 XGBOOST FEATURES")
print("=" * 70)

print(

    importance_df
    .head(20)
    .to_string(index=False)

)


# ==========================================================
# 15. Save Full Feature Importance
# ==========================================================

csv_path = os.path.join(

    OUTPUT_DIR,

    "xgboost_feature_importance.csv"

)

importance_df.to_csv(

    csv_path,

    index=False

)


# ==========================================================
# 16. Plot Top 15 Features
# ==========================================================

top_features = (

    importance_df
    .head(15)
    .sort_values(
        by="Importance"
    )

)

plt.figure(
    figsize=(10, 7)
)

plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)

plt.xlabel(
    "XGBoost Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features for Length of Stay Prediction"
)

plt.tight_layout()

plot_path = os.path.join(

    OUTPUT_DIR,

    "xgboost_feature_importance.png"

)

plt.savefig(

    plot_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ==========================================================
# 17. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nCSV:",
    csv_path
)

print(
    "Plot:",
    plot_path
)

print("\nXGBoost feature importance completed successfully.")