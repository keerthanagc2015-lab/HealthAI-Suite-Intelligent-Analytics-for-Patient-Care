import os
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

ORIGINAL_DATA_PATH = (
    "data/raw/Indian_healthcare.csv"
)

CLUSTER_DATA_PATH = (
    "data/processed/patient_segmentation/"
    "clustering_results/patient_clusters.csv"
)

OUTPUT_DIR = (
    "data/processed/patient_segmentation/"
    "cluster_profiles"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Original Dataset
# ==========================================================

print("=" * 70)
print("PATIENT SEGMENTATION - CLUSTER PROFILING")
print("=" * 70)

df = pd.read_csv(
    ORIGINAL_DATA_PATH
)

print("\nOriginal Dataset Loaded")
print("Shape:", df.shape)


# ==========================================================
# 2. Load Cluster Labels
# ==========================================================

clustered_data = pd.read_csv(
    CLUSTER_DATA_PATH
)

print("\nCluster Data Loaded")
print("Shape:", clustered_data.shape)


# ==========================================================
# 3. Attach Cluster Labels
# ==========================================================

df["Cluster"] = clustered_data["Cluster"]

print("\nCluster labels attached successfully.")


# ==========================================================
# 4. Cluster Size
# ==========================================================

print("\n" + "=" * 70)
print("CLUSTER SIZE")
print("=" * 70)

cluster_size = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_size)


# ==========================================================
# 5. Numerical Cluster Profile
# ==========================================================

numerical_features = [
    "Age",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI"
]

numerical_profile = (
    df.groupby("Cluster")[numerical_features]
    .mean()
)

print("\n" + "=" * 70)
print("NUMERICAL CLUSTER PROFILE")
print("=" * 70)

print(
    numerical_profile.round(2)
)


# ==========================================================
# 6. Median Profile
# ==========================================================

numerical_median = (
    df.groupby("Cluster")[numerical_features]
    .median()
)

print("\n" + "=" * 70)
print("MEDIAN CLUSTER PROFILE")
print("=" * 70)

print(
    numerical_median.round(2)
)


# ==========================================================
# 7. Categorical Profiles
# ==========================================================

categorical_features = [
    "Gender",
    "Region",
    "Socioeconomic_Status"
]


for feature in categorical_features:

    print("\n" + "=" * 70)

    print(
        f"{feature.upper()} DISTRIBUTION BY CLUSTER"
    )

    print("=" * 70)

    distribution = pd.crosstab(
        df["Cluster"],
        df[feature],
        normalize="index"
    ) * 100

    print(
        distribution.round(2)
    )


# ==========================================================
# 8. Symptoms by Cluster
# ==========================================================

print("\n" + "=" * 70)
print("SYMPTOMS BY CLUSTER")
print("=" * 70)

symptoms_distribution = pd.crosstab(
    df["Cluster"],
    df["Symptoms"],
    normalize="index"
) * 100

print(
    symptoms_distribution.round(2)
)


# ==========================================================
# 9. Primary Diagnosis by Cluster
# ==========================================================

print("\n" + "=" * 70)
print("PRIMARY DIAGNOSIS BY CLUSTER")
print("=" * 70)

diagnosis_distribution = pd.crosstab(
    df["Cluster"],
    df["Primary_Diagnosis"],
    normalize="index"
) * 100

print(
    diagnosis_distribution.round(2)
)


# ==========================================================
# 10. Save Numerical Profile
# ==========================================================

numerical_profile_path = os.path.join(
    OUTPUT_DIR,
    "numerical_cluster_profile.csv"
)

numerical_profile.to_csv(
    numerical_profile_path
)


# ==========================================================
# 11. Save Diagnosis Profile
# ==========================================================

diagnosis_path = os.path.join(
    OUTPUT_DIR,
    "diagnosis_by_cluster.csv"
)

diagnosis_distribution.to_csv(
    diagnosis_path
)


# ==========================================================
# 12. Save Symptoms Profile
# ==========================================================

symptoms_path = os.path.join(
    OUTPUT_DIR,
    "symptoms_by_cluster.csv"
)

symptoms_distribution.to_csv(
    symptoms_path
)


# ==========================================================
# 13. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nNumerical Profile:",
    numerical_profile_path
)

print(
    "Diagnosis Profile:",
    diagnosis_path
)

print(
    "Symptoms Profile:",
    symptoms_path
)

print(
    "\nCluster profiling completed successfully."
)