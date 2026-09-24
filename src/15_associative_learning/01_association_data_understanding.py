import os
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/Indian_healthcare.csv"

OUTPUT_DIR = "data/processed/associative_learning"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Dataset
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - DATA UNDERSTANDING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. Display Columns
# ==========================================================

print("\nAvailable Columns:")

for column in df.columns:
    print(column)


# ==========================================================
# 3. Data Types
# ==========================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ==========================================================
# 4. Unique Values
# ==========================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:

    print(
        f"\n{column}: "
        f"{df[column].nunique()} unique values"
    )


# ==========================================================
# 5. Missing Values
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ==========================================================
# 6. Candidate Association Features
# ==========================================================

candidate_features = [
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


print("\n" + "=" * 70)
print("CANDIDATE ASSOCIATION FEATURES")
print("=" * 70)

for feature in candidate_features:

    if feature in df.columns:

        print(
            f"{feature}: "
            f"{df[feature].nunique()} unique values"
        )


# ==========================================================
# 7. Save Candidate Feature Information
# ==========================================================

feature_info = []

for feature in candidate_features:

    if feature in df.columns:

        feature_info.append({

            "Feature": feature,

            "Data_Type": str(
                df[feature].dtype
            ),

            "Unique_Values": (
                df[feature].nunique()
            )

        })


feature_info_df = pd.DataFrame(
    feature_info
)

output_path = os.path.join(
    OUTPUT_DIR,
    "association_candidate_features.csv"
)

feature_info_df.to_csv(
    output_path,
    index=False
)


# ==========================================================
# 8. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nCandidate Feature Information:",
    output_path
)

print(
    "\nAssociation data understanding completed successfully."
)