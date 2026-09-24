"""
======================================================================
HealthAI Suite - Advanced Feature Engineering
======================================================================

Author      : Keerthana
Description :
Creates clinically meaningful features to improve
machine learning model performance.

New Features:
1. BMI_Category
2. Age_Group
3. High_Glucose
4. High_Cholesterol
5. HbA1c_Category
======================================================================
"""

import pandas as pd
import numpy as np


# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------

def load_dataset(file_path):
    """
    Load cleaned healthcare dataset.
    """
    df = pd.read_csv(file_path)

    print("=" * 70)
    print("HealthAI Suite - Advanced Feature Engineering")
    print("=" * 70)

    print("\nDataset Loaded Successfully.")
    print(f"Dataset Shape : {df.shape}")

    return df


# -------------------------------------------------------------------
# BMI Category
# -------------------------------------------------------------------

def create_bmi_category(df):

    conditions = [
        df["BMI"] < 18.5,
        (df["BMI"] >= 18.5) & (df["BMI"] < 25),
        (df["BMI"] >= 25) & (df["BMI"] < 30),
        df["BMI"] >= 30
    ]

    categories = [
        "Underweight",
        "Normal",
        "Overweight",
        "Obese"
    ]

    df["BMI_Category"] = np.select(
        conditions,
        categories,
        default="Unknown"
    )

    print("\n✓ BMI_Category Created")
    print(df["BMI_Category"].value_counts())

    return df


# -------------------------------------------------------------------
# Age Group
# -------------------------------------------------------------------

def create_age_group(df):

    conditions = [
        df["Age"] <= 18,
        (df["Age"] > 18) & (df["Age"] <= 35),
        (df["Age"] > 35) & (df["Age"] <= 50),
        (df["Age"] > 50) & (df["Age"] <= 65),
        df["Age"] > 65
    ]

    groups = [
        "Child",
        "Young Adult",
        "Adult",
        "Middle Age",
        "Senior"
    ]

    df["Age_Group"] = np.select(
        conditions,
        groups,
        default="Unknown"
    )

    print("\n✓ Age_Group Created")
    print(df["Age_Group"].value_counts())

    return df


# -------------------------------------------------------------------
# High Glucose Flag
# -------------------------------------------------------------------

def create_high_glucose(df):

    df["High_Glucose"] = np.where(
        df["Blood_Glucose_mg_dL"] > 126,
        "Yes",
        "No"
    )

    print("\n✓ High_Glucose Created")
    print(df["High_Glucose"].value_counts())

    return df


# -------------------------------------------------------------------
# High Cholesterol Flag
# -------------------------------------------------------------------

def create_high_cholesterol(df):

    df["High_Cholesterol"] = np.where(
        df["Total_Cholesterol_mg_dL"] > 200,
        "Yes",
        "No"
    )

    print("\n✓ High_Cholesterol Created")
    print(df["High_Cholesterol"].value_counts())

    return df


# -------------------------------------------------------------------
# HbA1c Category
# -------------------------------------------------------------------

def create_hba1c_category(df):

    conditions = [
        df["HbA1c_%"] < 5.7,
        (df["HbA1c_%"] >= 5.7) & (df["HbA1c_%"] < 6.5),
        df["HbA1c_%"] >= 6.5
    ]

    categories = [
        "Normal",
        "Prediabetes",
        "Diabetes"
    ]

    df["HbA1c_Category"] = np.select(
        conditions,
        categories,
        default="Unknown"
    )

    print("\n✓ HbA1c_Category Created")
    print(df["HbA1c_Category"].value_counts())

    return df


# -------------------------------------------------------------------
# Display Summary
# -------------------------------------------------------------------

def display_summary(df):

    print("\n" + "=" * 70)
    print("Advanced Feature Engineering Summary")
    print("=" * 70)

    print("\nNew Features Added:")

    new_features = [
        "BMI_Category",
        "Age_Group",
        "High_Glucose",
        "High_Cholesterol",
        "HbA1c_Category"
    ]

    for feature in new_features:
        print(f"✓ {feature}")

    print("\nDataset Shape After Feature Engineering:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())


# -------------------------------------------------------------------
# Save Dataset
# -------------------------------------------------------------------

def save_dataset(df, output_path):

    df.to_csv(output_path, index=False)

    print("\nAdvanced dataset saved successfully.")

    print(f"\nLocation:\n{output_path}")


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------

def main():

    input_path = "data/processed/cleaned_healthcare.csv"

    output_path = "data/processed/advanced_healthcare.csv"

    df = load_dataset(input_path)

    df = create_bmi_category(df)

    df = create_age_group(df)

    df = create_high_glucose(df)

    df = create_high_cholesterol(df)

    df = create_hba1c_category(df)

    display_summary(df)

    save_dataset(df, output_path)


# -------------------------------------------------------------------
# Driver Code
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()