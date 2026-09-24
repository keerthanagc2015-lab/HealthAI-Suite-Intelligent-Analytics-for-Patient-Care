# ==========================================================
# HealthAI Suite - Length of Stay EDA
# ==========================================================

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/LengthOfStay.csv"

OUTPUT_DIR = "data/processed/length_of_stay/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 70)
    print("HealthAI Suite - Length of Stay EDA")
    print("=" * 70)

    # ======================================================
    # 1. Load Dataset
    # ======================================================

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully")

    print("Dataset Shape:")
    print(df.shape)

    # ======================================================
    # 2. Basic Information
    # ======================================================

    print("\n" + "=" * 70)
    print("Basic Dataset Information")
    print("=" * 70)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Info:")
    print(df.info())

    # ======================================================
    # 3. Duplicate Check
    # ======================================================

    print("\n" + "=" * 70)
    print("Duplicate Check")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print(
        f"\nDuplicate Rows: {duplicate_count}"
    )

    # ======================================================
    # 4. Missing Values
    # ======================================================

    print("\n" + "=" * 70)
    print("Missing Value Analysis")
    print("=" * 70)

    missing_values = df.isnull().sum()

    missing_values = missing_values.sort_values(
        ascending=False
    )

    print("\nMissing Values:")
    print(missing_values)

    # ======================================================
    # 5. Unique Values
    # ======================================================

    print("\n" + "=" * 70)
    print("Unique Value Analysis")
    print("=" * 70)

    unique_values = pd.DataFrame({

        "Feature": df.columns,

        "Unique_Values":
        [
            df[column].nunique()
            for column in df.columns
        ]

    })

    print(
        unique_values.to_string(index=False)
    )

    unique_values.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "unique_values.csv"
        ),

        index=False

    )

    # ======================================================
    # 6. Target Analysis
    # ======================================================

    print("\n" + "=" * 70)
    print("Target Analysis - Length of Stay")
    print("=" * 70)

    target = "lengthofstay"

    print("\nTarget Statistics:")

    print(
        df[target].describe()
    )

    print("\nTarget Value Counts:")

    print(
        df[target]
        .value_counts()
        .sort_index()
    )

    # ======================================================
    # 7. Target Distribution Plot
    # ======================================================

    plt.figure(
        figsize=(10, 6)
    )

    sns.histplot(

        df[target],

        bins=17,

        kde=True

    )

    plt.title(
        "Distribution of Length of Stay"
    )

    plt.xlabel(
        "Length of Stay (Days)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "length_of_stay_distribution.png"
        ),

        dpi=300

    )

    plt.close()

    # ======================================================
    # 8. Length of Stay Boxplot
    # ======================================================

    plt.figure(
        figsize=(10, 5)
    )

    sns.boxplot(
        x=df[target]
    )

    plt.title(
        "Length of Stay Boxplot"
    )

    plt.xlabel(
        "Length of Stay (Days)"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "length_of_stay_boxplot.png"
        ),

        dpi=300

    )

    plt.close()

    # ======================================================
    # 9. Numerical Features
    # ======================================================

    numerical_columns = df.select_dtypes(

        include=np.number

    ).columns.tolist()

    print("\n" + "=" * 70)
    print("Numerical Features")
    print("=" * 70)

    print(
        numerical_columns
    )

    numerical_summary = df[
        numerical_columns
    ].describe().T

    print(
        numerical_summary
    )

    numerical_summary.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "numerical_summary.csv"
        )

    )

    # ======================================================
    # 10. Correlation with Target
    # ======================================================

    print("\n" + "=" * 70)
    print("Correlation With Length of Stay")
    print("=" * 70)

    correlation_with_target = (

        df[numerical_columns]

        .corr()["lengthofstay"]

        .drop("lengthofstay")

        .sort_values(
            key=abs,
            ascending=False
        )

    )

    print(
        correlation_with_target
    )

    correlation_df = (

        correlation_with_target

        .reset_index()

    )

    correlation_df.columns = [

        "Feature",
        "Correlation"

    ]

    correlation_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "correlation_with_length_of_stay.csv"
        ),

        index=False

    )

    # ======================================================
    # 11. Correlation Heatmap
    # ======================================================

    plt.figure(
        figsize=(16, 12)
    )

    correlation_matrix = df[
        numerical_columns
    ].corr()

    sns.heatmap(

        correlation_matrix,

        annot=True,

        fmt=".2f",

        cmap="coolwarm",

        center=0

    )

    plt.title(
        "Numerical Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "correlation_heatmap.png"
        ),

        dpi=300

    )

    plt.close()

    # ======================================================
    # 12. Readmission Count vs LOS
    # ======================================================

    if "rcount" in df.columns:

        plt.figure(
            figsize=(10, 6)
        )

        sns.boxplot(

            data=df,

            x="rcount",

            y="lengthofstay"

        )

        plt.title(
            "Readmission Count vs Length of Stay"
        )

        plt.xlabel(
            "Readmission Count"
        )

        plt.ylabel(
            "Length of Stay (Days)"
        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                OUTPUT_DIR,
                "readmission_vs_length_of_stay.png"
            ),

            dpi=300

        )

        plt.close()

    # ======================================================
    # 13. Clinical Feature Relationships
    # ======================================================

    clinical_features = [

        "hematocrit",
        "neutrophils",
        "sodium",
        "glucose",
        "bloodureanitro",
        "creatinine",
        "bmi",
        "pulse",
        "respiration"

    ]

    available_clinical_features = [

        feature

        for feature in clinical_features

        if feature in df.columns

    ]

    for feature in available_clinical_features:

        plt.figure(
            figsize=(8, 5)
        )

        sns.scatterplot(

            data=df.sample(
                min(10000, len(df)),
                random_state=42
            ),

            x=feature,

            y="lengthofstay",

            alpha=0.4

        )

        plt.title(

            f"{feature} vs Length of Stay"

        )

        plt.xlabel(feature)

        plt.ylabel(
            "Length of Stay (Days)"
        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                OUTPUT_DIR,

                f"{feature}_vs_length_of_stay.png"

            ),

            dpi=300

        )

        plt.close()

    # ======================================================
    # 14. Binary Health Conditions
    # ======================================================

    binary_features = [

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

    available_binary_features = [

        feature

        for feature in binary_features

        if feature in df.columns

    ]

    binary_summary = []

    for feature in available_binary_features:

        grouped_mean = (

            df.groupby(feature)["lengthofstay"]

            .mean()

        )

        row = {

            "Feature": feature,

            "LOS_Mean_When_0":
            grouped_mean.get(0, np.nan),

            "LOS_Mean_When_1":
            grouped_mean.get(1, np.nan)

        }

        binary_summary.append(row)

    binary_summary_df = pd.DataFrame(
        binary_summary
    )

    print("\n" + "=" * 70)
    print("Binary Health Condition vs LOS")
    print("=" * 70)

    print(
        binary_summary_df.to_string(
            index=False
        )
    )

    binary_summary_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "binary_conditions_vs_los.csv"
        ),

        index=False

    )

    # ======================================================
    # 15. Gender vs Length of Stay
    # ======================================================

    if "gender" in df.columns:

        gender_summary = (

            df.groupby("gender")["lengthofstay"]

            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std"
                ]
            )

        )

        print("\n" + "=" * 70)
        print("Gender vs Length of Stay")
        print("=" * 70)

        print(
            gender_summary
        )

        gender_summary.to_csv(

            os.path.join(
                OUTPUT_DIR,
                "gender_vs_length_of_stay.csv"
            )

        )

    # ======================================================
    # 16. Facility vs Length of Stay
    # ======================================================

    if "facid" in df.columns:

        facility_summary = (

            df.groupby("facid")["lengthofstay"]

            .agg(
                [
                    "count",
                    "mean",
                    "median"
                ]
            )

            .sort_values(
                "mean",
                ascending=False
            )

        )

        print("\n" + "=" * 70)
        print("Facility vs Length of Stay")
        print("=" * 70)

        print(
            facility_summary.head(20)
        )

        facility_summary.to_csv(

            os.path.join(
                OUTPUT_DIR,
                "facility_vs_length_of_stay.csv"
            )

        )

    # ======================================================
    # 17. Outlier Analysis
    # ======================================================

    print("\n" + "=" * 70)
    print("Outlier Analysis")
    print("=" * 70)

    outlier_summary = []

    for column in numerical_columns:

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        outliers = (

            (df[column] < lower_bound)

            |

            (df[column] > upper_bound)

        ).sum()

        outlier_summary.append({

            "Feature": column,

            "Q1": Q1,

            "Q3": Q3,

            "IQR": IQR,

            "Lower_Bound": lower_bound,

            "Upper_Bound": upper_bound,

            "Outlier_Count": outliers

        })

    outlier_df = pd.DataFrame(
        outlier_summary
    )

    print(
        outlier_df.to_string(
            index=False
        )
    )

    outlier_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "outlier_analysis.csv"
        ),

        index=False

    )

    # ======================================================
    # 18. Final EDA Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("EDA COMPLETED")
    print("=" * 70)

    print("\nEDA files saved to:")

    print(
        OUTPUT_DIR
    )

    print("\nGenerated analyses:")

    print("✓ Dataset structure")
    print("✓ Missing value analysis")
    print("✓ Duplicate analysis")
    print("✓ Unique value analysis")
    print("✓ Target distribution")
    print("✓ Target boxplot")
    print("✓ Numerical feature summary")
    print("✓ Correlation analysis")
    print("✓ Correlation heatmap")
    print("✓ Readmission vs LOS")
    print("✓ Clinical feature relationships")
    print("✓ Binary condition analysis")
    print("✓ Gender analysis")
    print("✓ Facility analysis")
    print("✓ Outlier analysis")

    print("\nLength of Stay EDA completed successfully.")


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()