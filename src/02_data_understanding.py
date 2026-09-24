import pandas as pd
from data_loader import load_data


def understand_data(df):
    """Perform initial understanding of the HealthAI dataset."""

    print("=" * 70)
    print("HealthAI Suite - Data Understanding")
    print("=" * 70)

    # 1. Dataset dimensions
    print("\n1. DATASET SHAPE")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # 2. Data types and non-null counts
    print("\n2. DATASET INFORMATION")
    df.info()

    # 3. Missing values
    print("\n3. MISSING VALUES")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing_Count": missing_values,
        "Missing_Percentage": missing_percentage
    })

    print(missing_report)

    # 4. Duplicate records
    print("\n4. DUPLICATE RECORDS")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    # 5. Unique values
    print("\n5. UNIQUE VALUES")
    print(df.nunique())

    # 6. Numerical statistics
    print("\n6. NUMERICAL SUMMARY")
    print(df.describe().T)

    # 7. Categorical statistics
    print("\n7. CATEGORICAL SUMMARY")
    categorical_columns = df.select_dtypes(include=["object", "bool"]).columns
    print(df[categorical_columns].describe().T)

    print("\n" + "=" * 70)
    print("Data Understanding Completed")
    print("=" * 70)


if __name__ == "__main__":
    df = load_data()
    understand_data(df)

