# ==========================================================
# HealthAI Suite - Length of Stay Feature Selection
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/LengthOfStay.csv"

OUTPUT_DIR = "data/processed/length_of_stay/feature_selection"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 70)
    print("HealthAI Suite - Length of Stay Feature Selection")
    print("=" * 70)

    # ======================================================
    # 1. Load Dataset
    # ======================================================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    # ======================================================
    # 2. Define Target
    # ======================================================

    target = "lengthofstay"

    y = df[target]

    # ======================================================
    # 3. Remove Identifier / Potential Leakage Features
    # ======================================================

    excluded_features = [

        "eid",
        "vdate",
        "discharged"

    ]

    candidate_features = [

        column

        for column in df.columns

        if column not in excluded_features + [target]

    ]

    print("\n" + "=" * 70)
    print("Excluded Features")
    print("=" * 70)

    for feature in excluded_features:
        print(feature)

    print("\nReason:")
    print("eid        -> Identifier; excluded")
    print("vdate      -> Visit date; not used in our current feature set")
    print("discharged -> Discharge date; unavailable at admission; excluded")
    print("rcount     -> Previous 180-day readmission count; retained as a valid predictor")

    # ======================================================
    # 4. Candidate Features
    # ======================================================

    X = df[candidate_features].copy()

    print("\n" + "=" * 70)
    print("Candidate Features")
    print("=" * 70)

    print("\nNumber of Candidate Features:", len(candidate_features))

    for feature in candidate_features:
        print(feature)

    # ======================================================
    # 5. Convert rcount
    # ======================================================

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

    # ======================================================
    # 6. Identify Numerical Features
    # ======================================================

    numerical_features = X.select_dtypes(

        include=np.number

    ).columns.tolist()

    print("\n" + "=" * 70)
    print("Numerical Features")
    print("=" * 70)

    print(numerical_features)

    # ======================================================
    # 7. Correlation Analysis
    # ======================================================

    correlation_scores = (

        X[numerical_features]

        .corrwith(y)

        .sort_values(
            key=abs,
            ascending=False
        )

    )

    correlation_df = (

        correlation_scores

        .reset_index()

    )

    correlation_df.columns = [

        "Feature",
        "Correlation"

    ]

    correlation_df["Absolute_Correlation"] = (

        correlation_df["Correlation"].abs()

    )

    print("\n" + "=" * 70)
    print("Correlation With Length of Stay")
    print("=" * 70)

    print(

        correlation_df.to_string(
            index=False
        )

    )

    correlation_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "correlation_feature_ranking.csv"
        ),

        index=False

    )

    # ======================================================
    # 8. Mutual Information
    # ======================================================

    print("\n" + "=" * 70)
    print("Mutual Information Analysis")
    print("=" * 70)

    # Use a sample to make MI computation faster
    sample_size = min(20000, len(X))

    sample_indices = (

        X.sample(
            sample_size,
            random_state=42
        ).index

    )

    X_mi = X.loc[
        sample_indices,
        numerical_features
    ].copy()

    y_mi = y.loc[
        sample_indices
    ]

    # Fill missing values just in case
    X_mi = X_mi.fillna(
        X_mi.median()
    )

    mi_scores = mutual_info_regression(

        X_mi,
        y_mi,

        random_state=42

    )

    mi_df = pd.DataFrame({

        "Feature":
        numerical_features,

        "Mutual_Information":
        mi_scores

    })

    mi_df = mi_df.sort_values(

        "Mutual_Information",

        ascending=False

    )

    print(
        mi_df.to_string(
            index=False
        )
    )

    mi_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "mutual_information_ranking.csv"
        ),

        index=False

    )

    # ======================================================
    # 9. Combined Ranking
    # ======================================================

    combined_df = correlation_df.merge(

        mi_df,

        on="Feature",

        how="outer"

    )

    # Rank based on absolute correlation
    combined_df["Correlation_Rank"] = (

        combined_df[
            "Absolute_Correlation"
        ]

        .rank(
            ascending=False,
            method="min"
        )

    )

    # Rank based on mutual information
    combined_df["MI_Rank"] = (

        combined_df[
            "Mutual_Information"
        ]

        .rank(
            ascending=False,
            method="min"
        )

    )

    # Average the two ranks
    combined_df["Combined_Rank"] = (

        (
            combined_df["Correlation_Rank"]

            +

            combined_df["MI_Rank"]
        )

        / 2

    )

    combined_df = combined_df.sort_values(

        "Combined_Rank"

    )

    # ======================================================
    # 10. Display Combined Ranking
    # ======================================================

    print("\n" + "=" * 70)
    print("Combined Feature Ranking")
    print("=" * 70)

    print(

        combined_df[
            [
                "Feature",
                "Correlation",
                "Mutual_Information",
                "Combined_Rank"
            ]
        ]

        .to_string(
            index=False
        )

    )

    # ======================================================
    # 11. Save Final Ranking
    # ======================================================

    final_path = os.path.join(

        OUTPUT_DIR,

        "combined_feature_ranking.csv"

    )

    combined_df.to_csv(

        final_path,

        index=False

    )

    print(
        "\nFeature ranking saved to:"
    )

    print(final_path)

    # ======================================================
    # 12. Final Candidate Feature List
    # ======================================================

    print("\n" + "=" * 70)
    print("Feature Selection Summary")
    print("=" * 70)

    print(
        "\nTotal Original Features:",
        len(df.columns)
    )

    print(
        "Excluded Features:",
        len(excluded_features)
    )

    print(
        "Candidate Features:",
        len(candidate_features)
    )

    print("\nImportant Note:")
    print(
        "Correlation measures linear relationships."
    )

    print(
        "Mutual Information can detect broader dependencies."
    )

    print(
        "Final selection will also consider domain relevance,"
        " leakage and model-based feature importance."
    )

    print("\nFeature selection analysis completed successfully.")


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()