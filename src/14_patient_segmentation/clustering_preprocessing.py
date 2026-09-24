import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = (
    "data/processed/patient_segmentation/"
    "clustering_selected_features.csv"
)

OUTPUT_DIR = (
    "data/processed/patient_segmentation/"
    "clustering_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Selected Features
# ==========================================================

print("=" * 70)
print("PATIENT SEGMENTATION - PREPROCESSING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. Define Feature Types
# ==========================================================

categorical_features = [
    "Gender",
    "Region",
    "Socioeconomic_Status",
    "Symptoms"
]

numerical_features = [
    "Age",
    "Blood_Glucose_mg_dL",
    "HbA1c_%",
    "Total_Cholesterol_mg_dL",
    "BMI"
]


print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ==========================================================
# 3. Preprocessing
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )

    ]

)


# ==========================================================
# 4. Transform Dataset
# ==========================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

X_processed = preprocessor.fit_transform(df)

print(
    "\nProcessed Dataset Shape:",
    X_processed.shape
)


# ==========================================================
# 5. Get Processed Feature Names
# ==========================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    "\nNumber of Processed Features:",
    len(feature_names)
)


# ==========================================================
# 6. Elbow Method
# ==========================================================

print("\n" + "=" * 70)
print("ELBOW METHOD")
print("=" * 70)

inertias = []

k_values = range(2, 11)

for k in k_values:

    print(f"Testing K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_processed)

    inertias.append(
        kmeans.inertia_
    )


# ==========================================================
# 7. Save Elbow Results
# ==========================================================

elbow_results = pd.DataFrame({

    "K": list(k_values),

    "Inertia": inertias

})

elbow_path = os.path.join(
    OUTPUT_DIR,
    "elbow_results.csv"
)

elbow_results.to_csv(
    elbow_path,
    index=False
)


# ==========================================================
# 8. Elbow Plot
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method - Patient Segmentation"
)

plt.xticks(
    list(k_values)
)

plt.tight_layout()

elbow_plot_path = os.path.join(
    OUTPUT_DIR,
    "elbow_method.png"
)

plt.savefig(
    elbow_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 9. Silhouette Score
# ==========================================================

print("\n" + "=" * 70)
print("SILHOUETTE SCORE")
print("=" * 70)

silhouette_results = []

for k in k_values:

    print(f"Calculating silhouette score for K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_processed
    )

    score = silhouette_score(
        X_processed,
        labels
    )

    silhouette_results.append(
        score
    )

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


# ==========================================================
# 10. Save Silhouette Results
# ==========================================================

silhouette_df = pd.DataFrame({

    "K": list(k_values),

    "Silhouette_Score": silhouette_results

})

silhouette_path = os.path.join(
    OUTPUT_DIR,
    "silhouette_scores.csv"
)

silhouette_df.to_csv(
    silhouette_path,
    index=False
)


# ==========================================================
# 11. Best K Based on Silhouette
# ==========================================================

best_index = (
    silhouette_df[
        "Silhouette_Score"
    ].idxmax()
)

best_k = int(
    silhouette_df.loc[
        best_index,
        "K"
    ]
)

best_score = (
    silhouette_df.loc[
        best_index,
        "Silhouette_Score"
    ]
)


print("\n" + "=" * 70)
print("BEST K")
print("=" * 70)

print(
    f"\nBest K based on Silhouette Score: {best_k}"
)

print(
    f"Best Silhouette Score: {best_score:.4f}"
)


# ==========================================================
# 12. Silhouette Plot
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    list(k_values),
    silhouette_results,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score - Patient Segmentation"
)

plt.xticks(
    list(k_values)
)

plt.tight_layout()

silhouette_plot_path = os.path.join(
    OUTPUT_DIR,
    "silhouette_scores.png"
)

plt.savefig(
    silhouette_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 13. Save Processed Dataset
# ==========================================================

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

processed_path = os.path.join(
    OUTPUT_DIR,
    "processed_clustering_features.csv"
)

processed_df.to_csv(
    processed_path,
    index=False
)


# ==========================================================
# 14. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nProcessed Data:",
    processed_path
)

print(
    "Elbow Results:",
    elbow_path
)

print(
    "Elbow Plot:",
    elbow_plot_path
)

print(
    "Silhouette Results:",
    silhouette_path
)

print(
    "Silhouette Plot:",
    silhouette_plot_path
)

print(
    "\nClustering preprocessing completed successfully."
)