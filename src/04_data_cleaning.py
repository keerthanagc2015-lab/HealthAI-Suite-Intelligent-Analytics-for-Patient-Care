import pandas as pd

from data_loader import load_data


def clean_data(df):
    """
    Clean and prepare the HealthAI patient dataset.
    """

    # Work on a copy to preserve the original dataframe
    cleaned_df = df.copy()

    print("=" * 70)
    print("HealthAI Suite - Data Cleaning")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Handle structural imaging missingness
    # --------------------------------------------------

    cleaned_df["Imaging_Type"] = (
        cleaned_df["Imaging_Type"]
        .fillna("Not_Performed")
    )

    # --------------------------------------------------
    # 2. Handle invalid blood glucose values
    # --------------------------------------------------

    invalid_glucose_mask = (
        cleaned_df["Blood_Glucose_mg_dL"] <= 0
    )

    invalid_count = invalid_glucose_mask.sum()

    print(f"\nInvalid glucose values found: {invalid_count}")

    cleaned_df.loc[
        invalid_glucose_mask,
        "Blood_Glucose_mg_dL"
    ] = pd.NA

    glucose_median = cleaned_df[
        "Blood_Glucose_mg_dL"
    ].median()

    cleaned_df["Blood_Glucose_mg_dL"] = (
        cleaned_df["Blood_Glucose_mg_dL"]
        .fillna(glucose_median)
    )

    # --------------------------------------------------
    # 3. Convert Visit_Date to datetime
    # --------------------------------------------------

    cleaned_df["Visit_Date"] = pd.to_datetime(
        cleaned_df["Visit_Date"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 4. Validation after cleaning
    # --------------------------------------------------

    print("\nMissing values after cleaning:")
    print(cleaned_df.isna().sum())

    print(
        "\nNon-positive glucose values remaining:",
        (cleaned_df["Blood_Glucose_mg_dL"] <= 0).sum()
    )

    print(
        "Missing Imaging_Type values remaining:",
        cleaned_df["Imaging_Type"].isna().sum()
    )

    print(
        "Invalid Visit_Date values:",
        cleaned_df["Visit_Date"].isna().sum()
    )

    print("\nDataset shape after cleaning:")
    print(cleaned_df.shape)

    # Save cleaned dataset
    output_path = "data/processed/cleaned_healthcare.csv"

    cleaned_df.to_csv(
       output_path,
       index=False
    )

    print(f"\nCleaned dataset saved to: {output_path}")

    print("\nData Cleaning Completed")

    return cleaned_df


if __name__ == "__main__":

    dataframe = load_data()

    cleaned_dataframe = clean_data(dataframe)