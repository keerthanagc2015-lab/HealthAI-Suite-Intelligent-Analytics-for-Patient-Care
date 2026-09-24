# ==========================================================
# HealthAI Suite - Length of Stay Feature Engineering
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/LengthOfStay.csv"

OUTPUT_DIR = "data/processed/length_of_stay/feature_engineering"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 70)
    print("HealthAI Suite - Length of Stay Feature Engineering")
    print("=" * 70)

    # ======================================================
    # 1. Load Dataset
    # ======================================================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully")
    print("Original Shape:", df.shape)

    # ======================================================
    # 2. Define Medical Condition Features
    # ======================================================

    condition_features = [

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
        "hemo"

    ]

    print("\n" + "=" * 70)
    print("Medical Condition Features")
    print("=" * 70)

    for feature in condition_features:
        print(feature)

    # ======================================================
    # 3. Verify Features Exist
    # ======================================================

    missing_features = [

        feature

        for feature in condition_features

        if feature not in df.columns

    ]

    if missing_features:

        print("\nERROR: Missing condition features:")

        for feature in missing_features:
            print(feature)

        return

    # ======================================================
    # 4. Create Total Health Issues
    # ======================================================

    df["Total_Health_Issues"] = (

        df[condition_features]

        .sum(axis=1)

    )

    print("\n" + "=" * 70)
    print("Feature Engineering")
    print("=" * 70)

    print(
        "\nCreated Feature: Total_Health_Issues"
    )

    # ======================================================
    # 5. Display Distribution
    # ======================================================

    print("\nTotal Health Issues Distribution:")

    print(

        df["Total_Health_Issues"]

        .value_counts()

        .sort_index()

    )

    # ======================================================
    # 6. Summary Statistics
    # ======================================================

    print("\nTotal Health Issues Statistics:")

    print(

        df["Total_Health_Issues"]

        .describe()

    )

    # ======================================================
    # 7. Compare With Length of Stay
    # ======================================================

    print("\n" + "=" * 70)
    print("Total Health Issues vs Length of Stay")
    print("=" * 70)

    issue_vs_los = (

        df.groupby(
            "Total_Health_Issues"
        )["lengthofstay"]

        .agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        )

    )

    print(issue_vs_los)

    # ======================================================
    # 8. Correlation With Target
    # ======================================================

    correlation = (

        df[
            [
                "Total_Health_Issues",
                "lengthofstay"
            ]
        ]

        .corr()

        .loc[
            "Total_Health_Issues",
            "lengthofstay"
        ]

    )

    print(
        "\nCorrelation with Length of Stay:",
        round(correlation, 4)
    )

    # ======================================================
    # 9. Save Feature Engineering Summary
    # ======================================================

    summary_path = os.path.join(

        OUTPUT_DIR,

        "total_health_issues_summary.csv"

    )

    issue_vs_los.to_csv(

        summary_path

    )

    # ======================================================
    # 10. Save Engineered Dataset
    # ======================================================

    output_path = os.path.join(

        OUTPUT_DIR,

        "length_of_stay_engineered.csv"

    )

    df.to_csv(

        output_path,

        index=False

    )

    print("\n" + "=" * 70)
    print("Files Created")
    print("=" * 70)

    print(
        "\nSummary:",
        summary_path
    )

    print(
        "Engineered Dataset:",
        output_path
    )

    print("\n" + "=" * 70)
    print("Feature Engineering Completed Successfully")
    print("=" * 70)


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()