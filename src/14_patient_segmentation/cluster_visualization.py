import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    "cluster_visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Data
# ==========================================================

print("=" * 70)
print("PATIENT SEGMENTATION - VISUALIZATION")
print("=" * 70)

df = pd.read_csv(
    ORIGINAL_DATA_PATH
)

clustered_data = pd.read_csv(
    CLUSTER_DATA_PATH
)

print("\nOriginal Dataset:", df.shape)
print("Cluster Dataset:", clustered_data.shape)


# ==========================================================
# 2. Attach Cluster Labels
# ==========================================================

df["Cluster"] = clustered_data["Cluster"]

print("\nCluster labels attached successfully.")


# ==========================================================
# 3. Cluster Distribution
# ==========================================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Patients")
plt.title("Patient Distribution by Cluster")

plt.tight_layout()

distribution_path = os.path.join(
    OUTPUT_DIR,
    "cluster_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 4. Glucose vs HbA1c
# ==========================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Blood_Glucose_mg_dL",
    y="HbA1c_%",
    hue="Cluster",
    alpha=0.4
)

plt.title(
    "Blood Glucose vs HbA1c by Cluster"
)

plt.xlabel(
    "Blood Glucose (mg/dL)"
)

plt.ylabel(
    "HbA1c (%)"
)

plt.tight_layout()

glucose_hba1c_path = os.path.join(
    OUTPUT_DIR,
    "glucose_vs_hba1c_clusters.png"
)

plt.savefig(
    glucose_hba1c_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


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

profile = (
    df.groupby("Cluster")[numerical_features]
    .mean()
)


# ==========================================================
# 6. Normalize Profile for Visualization
# ==========================================================

profile_normalized = (
    profile - profile.mean()
) / profile.std()


profile_plot = profile_normalized.T


plt.figure(figsize=(10, 6))

profile_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.axhline(
    0,
    linewidth=0.8
)

plt.xlabel("Feature")
plt.ylabel("Standardized Mean")
plt.title(
    "Numerical Feature Profile by Cluster"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

profile_path = os.path.join(
    OUTPUT_DIR,
    "numerical_cluster_profile.png"
)

plt.savefig(
    profile_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 7. Primary Diagnosis Distribution
# ==========================================================

diagnosis_distribution = pd.crosstab(
    df["Primary_Diagnosis"],
    df["Cluster"],
    normalize="columns"
) * 100


diagnosis_distribution.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.xlabel(
    "Primary Diagnosis"
)

plt.ylabel(
    "Percentage of Patients"
)

plt.title(
    "Primary Diagnosis Distribution by Cluster"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

diagnosis_path = os.path.join(
    OUTPUT_DIR,
    "diagnosis_distribution_by_cluster.png"
)

plt.savefig(
    diagnosis_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 8. Save Visualization Data
# ==========================================================

profile.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "numerical_profile_for_visualization.csv"
    )
)

diagnosis_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "diagnosis_distribution_for_visualization.csv"
    )
)


# ==========================================================
# 9. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nCluster Distribution:",
    distribution_path
)

print(
    "Glucose vs HbA1c:",
    glucose_hba1c_path
)

print(
    "Numerical Profile:",
    profile_path
)

print(
    "Diagnosis Distribution:",
    diagnosis_path
)

print(
    "\nCluster visualization completed successfully."
)