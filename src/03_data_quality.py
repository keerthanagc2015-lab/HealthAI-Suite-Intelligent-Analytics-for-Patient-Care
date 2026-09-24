import pandas as pd
from data_loader import load_data


def analyze_data_quality(df):

    print("=" * 70)
    print("HealthAI Suite - Data Quality Analysis")
    print("=" * 70)

    # -------------------------------------------------
    # 1. Imaging missingness investigation
    # -------------------------------------------------

    print("\n1. IMAGING MISSINGNESS")

    missing_imaging = df[df["Imaging_Type"].isna()]

    print(f"Missing Imaging_Type records: {len(missing_imaging):,}")

    print("\nImaging findings when Imaging_Type is missing:")
    print(
        missing_imaging["Imaging_Findings"]
        .value_counts(dropna=False)
    )

    # -------------------------------------------------
    # 2. Numerical range validation
    # -------------------------------------------------

    print("\n2. CLINICAL NUMERICAL RANGES")

    numerical_columns = [
        "Age",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI"
    ]

    for column in numerical_columns:

        print(f"\n{column}")

        print(f"Minimum : {df[column].min()}")
        print(f"Maximum : {df[column].max()}")

    # -------------------------------------------------
    # 3. Invalid blood glucose
    # -------------------------------------------------

    print("\n3. INVALID BLOOD GLUCOSE VALUES")

    invalid_glucose = df[df["Blood_Glucose_mg_dL"] <= 0]

    print(f"Records with glucose <= 0: {len(invalid_glucose):,}")

    if len(invalid_glucose) > 0:
        print(
            invalid_glucose[
                ["Patient_ID", "Blood_Glucose_mg_dL"]
            ].head(10)
        )

    # -------------------------------------------------
    # 4. Categorical distributions
    # -------------------------------------------------

    print("\n4. CATEGORICAL DISTRIBUTIONS")

    categorical_columns = [
        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",
        "Primary_Diagnosis",
        "Treatment_Type",
        "Treatment_Outcome",
        "Imaging_Type",
        "Imaging_Findings",
        "Hospital_Type",
        "Insurance_Covered"
    ]

    for column in categorical_columns:

        print("\n" + "-" * 50)
        print(column)
        print("-" * 50)

        print(
            df[column]
            .value_counts(dropna=False)
        )

    # -------------------------------------------------
    # 5. Date validation
    # -------------------------------------------------

    print("\n5. VISIT DATE")

    converted_dates = pd.to_datetime(
        df["Visit_Date"],
        errors="coerce"
    )

    print(f"Invalid dates: {converted_dates.isna().sum():,}")
    print(f"Earliest visit: {converted_dates.min()}")
    print(f"Latest visit  : {converted_dates.max()}")

    # -------------------------------------------------
    # 6. Patient ID validation
    # -------------------------------------------------

    print("\n6. PATIENT ID")

    print(f"Total records      : {len(df):,}")
    print(f"Unique Patient IDs : {df['Patient_ID'].nunique():,}")

    print("\n" + "=" * 70)
    print("Data Quality Analysis Completed")
    print("=" * 70)


if __name__ == "__main__":

    dataframe = load_data()

    analyze_data_quality(dataframe)