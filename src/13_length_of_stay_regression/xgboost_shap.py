# ==========================================================
# HealthAI Suite - LOS XGBoost SHAP Explainability
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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
print("LOS - XGBOOST SHAP EXPLAINABILITY")
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
# 3. Exclude Identifier / Unavailable Features
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
# 7. Preprocessor
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
# 8. XGBoost Model
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
# 9. Preprocess Training Data
# ==========================================================

print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    "Processed Training Shape:",
    X_train_processed.shape
)


# ==========================================================
# 10. Train XGBoost
# ==========================================================

print("\n" + "=" * 70)
print("Training XGBoost")
print("=" * 70)

model.fit(
    X_train_processed,
    y_train
)

print("XGBoost trained successfully.")


# ==========================================================
# 11. Take SHAP Sample
# ==========================================================

SAMPLE_SIZE = 2000

sample_size = min(
    SAMPLE_SIZE,
    X_test_processed.shape[0]
)

X_shap = X_test_processed[
    :sample_size
]

print("\nSHAP Sample Shape:")
print(X_shap.shape)


# ==========================================================
# 12. SHAP Explainer
# ==========================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_shap
)

print("SHAP calculation completed.")


# ==========================================================
# 13. Calculate Mean Absolute SHAP
# ==========================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

shap_importance = pd.DataFrame({

    "Feature": feature_names,

    "Mean_Absolute_SHAP": mean_abs_shap

})

shap_importance = (
    shap_importance
    .sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ==========================================================
# 14. Display Top 20
# ==========================================================

print("\n" + "=" * 70)
print("TOP 20 SHAP FEATURES")
print("=" * 70)

print(

    shap_importance
    .head(20)
    .to_string(index=False)

)


# ==========================================================
# 15. Save SHAP Importance
# ==========================================================

csv_path = os.path.join(

    OUTPUT_DIR,

    "xgboost_shap_importance.csv"

)

shap_importance.to_csv(

    csv_path,

    index=False

)


# ==========================================================
# 16. SHAP Bar Plot
# ==========================================================

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(

    shap_values,

    X_shap,

    feature_names=feature_names,

    plot_type="bar",

    max_display=15,

    show=False

)

plt.title(
    "Top 15 SHAP Features - Length of Stay"
)

plt.tight_layout()

bar_plot_path = os.path.join(

    OUTPUT_DIR,

    "xgboost_shap_importance.png"

)

plt.savefig(

    bar_plot_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ==========================================================
# 17. SHAP Beeswarm Plot
# ==========================================================

shap.summary_plot(

    shap_values,

    X_shap,

    feature_names=feature_names,

    max_display=15,

    show=False

)

beeswarm_path = os.path.join(

    OUTPUT_DIR,

    "xgboost_shap_beeswarm.png"

)

plt.savefig(

    beeswarm_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ==========================================================
# 18. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nCSV:",
    csv_path
)

print(
    "SHAP Bar Plot:",
    bar_plot_path
)

print(
    "SHAP Beeswarm Plot:",
    beeswarm_path
)

print("\nSHAP analysis completed successfully.")