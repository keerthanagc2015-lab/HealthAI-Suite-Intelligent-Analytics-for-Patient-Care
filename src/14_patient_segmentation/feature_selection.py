import os
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/Indian_healthcare.csv"

OUTPUT_DIR = "data/processed/patient_segmentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("PATIENT SEGMENTATION - FEATURE SELECTION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================================
# Display Columns
# ==========================================================

print("\nAvailable Columns:")

for column in df.columns:
    print(column)


# ==========================================================
# Features Considered for Clustering
# ==========================================================

candidate_features = [
    "Age",
    "Gender",
    "Region",
    "Socioeconomic_Status",
    "Symptoms",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI"
]


# ==========================================================
# Check Missing Columns
# ==========================================================

missing_features = [
    feature
    for feature in candidate_features
    if feature not in df.columns
]


if missing_features:

    print("\nMissing Candidate Features:")

    for feature in missing_features:
        print(feature)

    print(
        "\nPlease check the actual column names "
        "before continuing."
    )

    raise SystemExit


# ==========================================================
# Excluded Features
# ==========================================================

excluded_features = []

if "Patient_ID" in df.columns:
    excluded_features.append("Patient_ID")

if "Primary_Diagnosis" in df.columns:
    excluded_features.append("Primary_Diagnosis")


# ==========================================================
# Final Clustering Features
# ==========================================================

clustering_features = [
    feature
    for feature in candidate_features
    if feature not in excluded_features
]


print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print("\nExcluded Features:")

for feature in excluded_features:
    print(
        f"{feature} -> "
        "Excluded from clustering"
    )


print("\nSelected Clustering Features:")

for feature in clustering_features:
    print(feature)


# ==========================================================
# Create Clustering Dataset
# ==========================================================

clustering_df = df[clustering_features].copy()


print("\nClustering Dataset Shape:")
print(clustering_df.shape)


# ==========================================================
# Missing Values
# ==========================================================

print("\nMissing Values:")

print(
    clustering_df.isnull().sum()
)


# ==========================================================
# Save Feature Selection Information
# ==========================================================

feature_info = pd.DataFrame({

    "Feature": clustering_features,

    "Selected_For_Clustering": True

})


feature_info_path = os.path.join(
    OUTPUT_DIR,
    "clustering_feature_selection.csv"
)

feature_info.to_csv(
    feature_info_path,
    index=False
)


# ==========================================================
# Save Clustering Dataset
# ==========================================================

clustering_data_path = os.path.join(
    OUTPUT_DIR,
    "clustering_selected_features.csv"
)

clustering_df.to_csv(
    clustering_data_path,
    index=False
)


# ==========================================================
# Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nFeature Selection:",
    feature_info_path
)

print(
    "Clustering Dataset:",
    clustering_data_path
)

print(
    "\nClustering feature selection completed successfully."
)