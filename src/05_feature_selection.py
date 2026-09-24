import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

from data_loader import load_data
from data_cleaning import clean_data


# ============================================================
# 1. CHI-SQUARE ANALYSIS FOR CATEGORICAL FEATURES
# ============================================================

def chi_square_analysis(df):

    print("=" * 70)
    print("HealthAI Suite - Categorical Feature Selection")
    print("=" * 70)

    categorical_features = [
        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",
        "Occupation"
    ]

    target = "Primary_Diagnosis"

    print("\nCHI-SQUARE TEST")
    print("-" * 70)

    for feature in categorical_features:

        # Create contingency table
        contingency_table = pd.crosstab(
            df[feature],
            df[target]
        )

        # Perform Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(
            contingency_table
        )

        print(f"\nFeature: {feature}")
        print(f"Chi-square Value: {chi2:.2f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Degrees of Freedom: {dof}")

        if p_value < 0.05:
            print(
                "Result: Statistically significant association "
                "with Primary Diagnosis"
            )
        else:
            print(
                "Result: No statistically significant association "
                "with Primary Diagnosis"
            )

    print("\n" + "=" * 70)
    print("Chi-Square Analysis Completed")
    print("=" * 70)


# ============================================================
# 2. ANOVA ANALYSIS FOR NUMERICAL FEATURES
# ============================================================

def anova_analysis(df):

    print("\n" + "=" * 70)
    print("HealthAI Suite - Numerical Feature Selection")
    print("=" * 70)

    numerical_features = [
        "Age",
        "BMI",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL"
    ]

    target = "Primary_Diagnosis"

    print("\nANOVA TEST")
    print("-" * 70)

    for feature in numerical_features:

        # Create one numerical group for each diagnosis
        groups = [
            group[feature].dropna().values
            for _, group in df.groupby(target)
        ]

        # Perform one-way ANOVA
        f_statistic, p_value = f_oneway(*groups)

        print(f"\nFeature: {feature}")
        print(f"F-statistic: {f_statistic:.2f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print(
                "Result: Statistically significant difference "
                "across Primary Diagnosis groups"
            )
        else:
            print(
                "Result: No statistically significant difference "
                "across Primary Diagnosis groups"
            )

    print("\n" + "=" * 70)
    print("ANOVA Analysis Completed")
    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Load raw dataset
    dataframe = load_data()

    # Apply our existing data-cleaning process
    dataframe = clean_data(dataframe)

    # Categorical feature vs target analysis
    chi_square_analysis(dataframe)

    # Numerical feature vs target analysis
    anova_analysis(dataframe)