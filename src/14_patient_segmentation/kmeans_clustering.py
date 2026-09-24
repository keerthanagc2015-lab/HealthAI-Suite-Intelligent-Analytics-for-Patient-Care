import os
import pandas as pd
from sklearn.cluster import KMeans


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = (
    "data/processed/patient_segmentation/"
    "clustering_analysis/processed_clustering_features.csv"
)

OUTPUT_DIR = (
    "data/processed/patient_segmentation/"
    "clustering_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Processed Data
# ==========================================================

print("=" * 70)
print("PATIENT SEGMENTATION - K-MEANS CLUSTERING")
print("=" * 70)

X = pd.read_csv(DATA_PATH)

print("\nProcessed Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)


# ==========================================================
# 2. Set Number of Clusters
# ==========================================================

K = 2

print("\nNumber of Clusters (K):", K)


# ==========================================================
# 3. Train K-Means
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING K-MEANS")
print("=" * 70)

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

print("\nK-Means clustering completed successfully.")


# ==========================================================
# 4. Add Cluster Labels
# ==========================================================

clustered_data = X.copy()

clustered_data["Cluster"] = labels


# ==========================================================
# 5. Cluster Distribution
# ==========================================================

print("\n" + "=" * 70)
print("CLUSTER DISTRIBUTION")
print("=" * 70)

cluster_counts = (
    clustered_data["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)

print("\nCluster Percentages:")

cluster_percentages = (
    clustered_data["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(
    cluster_percentages.round(2)
)


# ==========================================================
# 6. Cluster Centers
# ==========================================================

print("\n" + "=" * 70)
print("CLUSTER CENTERS")
print("=" * 70)

cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns
)

cluster_centers.index.name = "Cluster"

print(cluster_centers)


# ==========================================================
# 7. Save Clustered Dataset
# ==========================================================

clustered_path = os.path.join(
    OUTPUT_DIR,
    "patient_clusters.csv"
)

clustered_data.to_csv(
    clustered_path,
    index=False
)


# ==========================================================
# 8. Save Cluster Distribution
# ==========================================================

distribution_df = pd.DataFrame({

    "Cluster": cluster_counts.index,

    "Patient_Count": cluster_counts.values,

    "Percentage": cluster_percentages.values

})

distribution_path = os.path.join(
    OUTPUT_DIR,
    "cluster_distribution.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False
)


# ==========================================================
# 9. Save Cluster Centers
# ==========================================================

centers_path = os.path.join(
    OUTPUT_DIR,
    "cluster_centers.csv"
)

cluster_centers.to_csv(
    centers_path
)


# ==========================================================
# 10. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nClustered Dataset:",
    clustered_path
)

print(
    "Cluster Distribution:",
    distribution_path
)

print(
    "Cluster Centers:",
    centers_path
)

print(
    "\nK-Means clustering completed successfully."
)