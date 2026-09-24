# ==========================================================
# HealthAI Suite - Classification Feature Importance + SHAP
# ==========================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/processed/advanced_healthcare.csv"

OUTPUT_DIR = "data/processed/feature_importance"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 70)
    print("HealthAI Suite - Classification Feature Importance + SHAP")
    print("=" * 70)

    # ======================================================
    # 1. Load Dataset
    # ======================================================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    # ======================================================
    # 2. Select Features
    # ======================================================

    selected_features = [

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

    ]

    target = "Primary_Diagnosis"

    X = df[selected_features].copy()

    y = df[target].copy()

    print("\nSelected Features")
    print("-" * 50)

    for feature in selected_features:
        print(feature)

    print("\nTarget")
    print(target)

    # ======================================================
    # 3. Train-Test Split
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\n" + "=" * 70)
    print("Train-Test Split")
    print("=" * 70)

    print("\nX_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # ======================================================
    # 4. Encode Target
    # ======================================================

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)

    y_test_encoded = label_encoder.transform(y_test)

    print("\n" + "=" * 70)
    print("Target Encoding")
    print("=" * 70)

    for index, class_name in enumerate(label_encoder.classes_):

        print(f"{index} --> {class_name}")

    # ======================================================
    # 5. Identify Feature Types
    # ======================================================

    categorical_features = [

        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",
        "BMI_Category",
        "Age_Group",
        "High_Glucose",
        "High_Cholesterol",
        "HbA1c_Category"

    ]

    numerical_features = [

        "Age",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI"

    ]

    # ======================================================
    # 6. Preprocessing
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
                    handle_unknown="ignore",
                    sparse_output=False
                ),

                categorical_features
            )

        ]

    )

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # ======================================================
    # 7. Get Feature Names
    # ======================================================

    feature_names = preprocessor.get_feature_names_out()

    print("\n" + "=" * 70)
    print("Feature Preprocessing")
    print("=" * 70)

    print("\nOriginal Feature Count:", len(selected_features))

    print(
        "Processed Feature Count:",
        len(feature_names)
    )

    # ======================================================
    # 8. Apply SMOTE
    # ======================================================

    print("\n" + "=" * 70)
    print("Applying SMOTE")
    print("=" * 70)

    print("\nBefore SMOTE:")

    print(
        pd.Series(y_train_encoded).value_counts().sort_index()
    )

    smote = SMOTE(

        random_state=42

    )

    X_train_smote, y_train_smote = smote.fit_resample(

        X_train_processed,

        y_train_encoded

    )

    print("\nAfter SMOTE:")

    print(
        pd.Series(y_train_smote).value_counts().sort_index()
    )

    print("\nTraining Shape After SMOTE:")

    print(X_train_smote.shape)

    # ======================================================
    # 9. Random Forest
    # ======================================================

    print("\n" + "=" * 70)
    print("Random Forest Feature Importance")
    print("=" * 70)

    rf_model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        n_jobs=-1

    )

    rf_model.fit(

        X_train_smote,

        y_train_smote

    )

    print("\nRandom Forest trained successfully.")

    # ======================================================
    # 10. Random Forest Feature Importance
    # ======================================================

    rf_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": rf_model.feature_importances_

    })

    rf_importance = rf_importance.sort_values(

        by="Importance",

        ascending=False

    )

    print("\nTop 15 Random Forest Features")

    print(
        rf_importance.head(15).to_string(index=False)
    )

    # ======================================================
    # 11. Save Random Forest Importance
    # ======================================================

    rf_importance_path = os.path.join(

        OUTPUT_DIR,

        "random_forest_feature_importance.csv"

    )

    rf_importance.to_csv(

        rf_importance_path,

        index=False

    )

    # ======================================================
    # 12. Plot Random Forest Importance
    # ======================================================

    top_rf = rf_importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(

        top_rf["Feature"][::-1],

        top_rf["Importance"][::-1]

    )

    plt.xlabel("Feature Importance")

    plt.ylabel("Feature")

    plt.title(
        "Random Forest - Top 15 Feature Importance"
    )

    plt.tight_layout()

    rf_plot_path = os.path.join(

        OUTPUT_DIR,

        "random_forest_feature_importance.png"

    )

    plt.savefig(

        rf_plot_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(
        "\nRandom Forest importance saved successfully."
    )

    # ======================================================
    # 13. Train XGBoost
    # ======================================================

    print("\n" + "=" * 70)
    print("XGBoost SHAP Analysis")
    print("=" * 70)

    xgb_model = XGBClassifier(

        n_estimators=200,

        max_depth=8,

        learning_rate=0.1,

        subsample=0.8,

        objective="multi:softprob",

        num_class=len(label_encoder.classes_),

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1

    )

    xgb_model.fit(

        X_train_smote,

        y_train_smote

    )

    print("\nXGBoost trained successfully.")

    # ======================================================
    # 14. SHAP Sample
    # ======================================================

    # SHAP on the entire 225k-row SMOTE dataset can be
    # computationally expensive.
    #
    # Therefore we use a representative sample.

    sample_size = min(

        2000,

        X_test_processed.shape[0]

    )

    X_shap = X_test_processed[

        :sample_size

    ]

    print("\nSHAP Sample Shape:")

    print(X_shap.shape)

    # ======================================================
    # 15. SHAP Explainer
    # ======================================================

    explainer = shap.TreeExplainer(

        xgb_model

    )

    shap_values = explainer.shap_values(

        X_shap

    )

    # ======================================================
    # 16. Handle Multiclass SHAP
    # ======================================================

    if isinstance(shap_values, list):

        # One SHAP matrix per class

        shap_array = np.stack(

            shap_values,

            axis=0

        )

        mean_abs_shap = np.mean(

            np.abs(shap_array),

            axis=(0, 1)

        )

    else:

        # Newer SHAP versions may return:

        # samples × features × classes

        if shap_values.ndim == 3:

            mean_abs_shap = np.mean(

                np.abs(shap_values),

                axis=(0, 2)

            )

        else:

            mean_abs_shap = np.mean(

                np.abs(shap_values),

                axis=0

            )

    # ======================================================
    # 17. Create SHAP Importance Table
    # ======================================================

    shap_importance = pd.DataFrame({

        "Feature": feature_names,

        "Mean_Absolute_SHAP": mean_abs_shap

    })

    shap_importance = shap_importance.sort_values(

        by="Mean_Absolute_SHAP",

        ascending=False

    )

    print("\nTop 15 SHAP Features")

    print(
        shap_importance.head(15).to_string(index=False)
    )

    # ======================================================
    # 18. Save SHAP Importance
    # ======================================================

    shap_csv_path = os.path.join(

        OUTPUT_DIR,

        "xgboost_shap_importance.csv"

    )

    shap_importance.to_csv(

        shap_csv_path,

        index=False

    )

    # ======================================================
    # 19. SHAP Bar Plot
    # ======================================================

    plt.figure(figsize=(10, 7))

    top_shap = shap_importance.head(15)

    plt.barh(

        top_shap["Feature"][::-1],

        top_shap["Mean_Absolute_SHAP"][::-1]

    )

    plt.xlabel("Mean Absolute SHAP Value")

    plt.ylabel("Feature")

    plt.title(

        "XGBoost - SHAP Feature Importance"

    )

    plt.tight_layout()

    shap_plot_path = os.path.join(

        OUTPUT_DIR,

        "xgboost_shap_importance.png"

    )

    plt.savefig(

        shap_plot_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # ======================================================
    # 20. Final Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("Feature Importance Analysis Completed")
    print("=" * 70)

    print("\nFiles Created:")

    print(
        rf_importance_path
    )

    print(
        rf_plot_path
    )

    print(
        shap_csv_path
    )

    print(
        shap_plot_path
    )

    print("\nTop 10 SHAP Features")

    print(
        shap_importance.head(10).to_string(index=False)
    )

    print("\nFeature Importance Analysis Completed Successfully.")


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()