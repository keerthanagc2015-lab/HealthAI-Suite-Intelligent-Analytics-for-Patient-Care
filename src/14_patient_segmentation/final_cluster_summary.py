import os
import pandas as pd


ORIGINAL_DATA_PATH = "data/raw/Indian_healthcare.csv"

CLUSTER_DATA_PATH = (
    "data/processed/patient_segmentation/"
    "clustering_results/patient_clusters.csv"
)

OUTPUT_DIR = (
    "data/processed/patient_segmentation/"
    "final_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Load data
# ==========================================================

df = pd.read_csv(ORIGINAL_DATA_PATH)
clustered_data = pd.read_csv(CLUSTER_DATA_PATH)

df["Cluster"] = clustered_data["Cluster"]


# ==========================================================
# Cluster size
# ==========================================================

cluster_size = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_percentage = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)


# ==========================================================
# Numerical profile
# ==========================================================

numerical_features = [
    "Age",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI"
]

profile = (
    df.groupby("Cluster")[numerical_features]
    .mean()
)


# ==========================================================
# Type 2 Diabetes percentage
# ==========================================================

diabetes_percentage = (
    df.assign(
        Type2Diabetes=(
            df["Primary_Diagnosis"]
            == "Type 2 Diabetes"
        )
    )
    .groupby("Cluster")["Type2Diabetes"]
    .mean()
    * 100
)


# ==========================================================
# Create final summary
# ==========================================================

summary = profile.copy()

summary.insert(
    0,
    "Patient_Count",
    cluster_size
)

summary.insert(
    1,
    "Percentage",
    cluster_percentage
)

summary["Type2_Diabetes_Percentage"] = (
    diabetes_percentage
)


summary = summary.reset_index()


# ==========================================================
# Display
# ==========================================================

print("=" * 70)
print("FINAL PATIENT CLUSTER SUMMARY")
print("=" * 70)

print(
    summary.round(2).to_string(index=False)
)


# ==========================================================
# Save
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "final_cluster_summary.csv"
)

summary.to_csv(
    output_path,
    index=False
)


print("\nFinal summary saved to:")
print(output_path)

print("\nClustering module completed successfully.")