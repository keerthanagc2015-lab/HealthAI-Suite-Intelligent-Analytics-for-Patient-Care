import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


print("=" * 70)
print("HEALTHAI - RNN FEATURE IMPORTANCE")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = (
    "data/processed/deep_learning/rnn/"
    "rnn_model.keras"
)

DATA_DIR = (
    "data/processed/deep_learning/rnn_lstm"
)

FEATURE_MAP_PATH = (
    "data/processed/deep_learning/rnn/"
    "explainability/rnn_feature_map.csv"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/rnn/"
    "explainability"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

# We don't need all 21,981 sequences for every permutation.
# Using 2,000 gives us a practical explanation while keeping
# the runtime manageable on CPU.

SAMPLE_SIZE = 2000

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading trained RNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✓ RNN model loaded successfully.")


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

print("\nLoading test sequences...")

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test.npy"
    )
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)


# ============================================================
# 5. LOAD FEATURE MAP
# ============================================================

print("\nLoading feature map...")

feature_map = pd.read_csv(
    FEATURE_MAP_PATH
)

feature_names = (
    feature_map[
        "Feature_Name"
    ].tolist()
)

print(
    "Number of feature names:",
    len(feature_names)
)


# ============================================================
# 6. VERIFY FEATURE COUNT
# ============================================================

number_of_features = X_test.shape[2]

if len(feature_names) != number_of_features:

    raise ValueError(
        "Feature name count does not match "
        "RNN input feature count."
    )

print(
    "\n✓ Feature count verified:",
    number_of_features
)


# ============================================================
# 7. SELECT REPRESENTATIVE TEST SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SELECTING TEST SAMPLE")
print("=" * 70)

actual_sample_size = min(
    SAMPLE_SIZE,
    len(X_test)
)

sample_indices = np.random.choice(
    len(X_test),
    size=actual_sample_size,
    replace=False
)

X_sample = X_test[
    sample_indices
]

y_sample = y_test[
    sample_indices
]

print(
    "\nSequences selected:",
    actual_sample_size
)

print(
    "Sample X shape:",
    X_sample.shape
)

print(
    "Sample y shape:",
    y_sample.shape
)

print(
    "\nSample target distribution:"
)

print(
    pd.Series(
        y_sample
    ).value_counts()
)


# ============================================================
# 8. BASELINE PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING BASELINE PERFORMANCE")
print("=" * 70)

baseline_probability = (
    model.predict(
        X_sample,
        verbose=0
    ).reshape(-1)
)

baseline_auc = roc_auc_score(
    y_sample,
    baseline_probability
)

print(
    "\nBaseline ROC-AUC:",
    f"{baseline_auc:.4f}"
)


# ============================================================
# 9. PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING FEATURE IMPORTANCE")
print("=" * 70)

print(
    "\nEach feature will be shuffled across "
    "the selected patient sequences."
)

print(
    "A larger ROC-AUC decrease means "
    "the feature is more important."
)


importance_results = []


for feature_index in range(
    number_of_features
):

    feature_name = (
        feature_names[
            feature_index
        ]
    )

    print(
        f"\n[{feature_index + 1}/"
        f"{number_of_features}] "
        f"Testing: {feature_name}"
    )

    # --------------------------------------------------------
    # Copy sample
    # --------------------------------------------------------

    X_permuted = X_sample.copy()

    # --------------------------------------------------------
    # Shuffle the selected feature across patients.
    #
    # IMPORTANT:
    # We keep all 24 hours together for that feature.
    # This preserves the temporal structure within each
    # patient's 24-hour sequence.
    # --------------------------------------------------------

    permutation = np.random.permutation(
        actual_sample_size
    )

    X_permuted[:, :, feature_index] = (
        X_permuted[
            permutation,
            :,
            feature_index
        ]
    )

    # --------------------------------------------------------
    # Predict with shuffled feature
    # --------------------------------------------------------

    permuted_probability = (
        model.predict(
            X_permuted,
            verbose=0
        ).reshape(-1)
    )

    # --------------------------------------------------------
    # Calculate ROC-AUC
    # --------------------------------------------------------

    permuted_auc = roc_auc_score(
        y_sample,
        permuted_probability
    )

    # --------------------------------------------------------
    # Importance
    # --------------------------------------------------------

    importance = (
        baseline_auc
        - permuted_auc
    )

    importance_results.append({

        "Feature_Index":
            feature_index,

        "Feature_Name":
            feature_name,

        "Baseline_ROC_AUC":
            baseline_auc,

        "Permuted_ROC_AUC":
            permuted_auc,

        "Importance":
            importance

    })

    print(
        "Baseline AUC:",
        f"{baseline_auc:.4f}"
    )

    print(
        "Permuted AUC:",
        f"{permuted_auc:.4f}"
    )

    print(
        "Importance:",
        f"{importance:.4f}"
    )


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

importance_df = pd.DataFrame(
    importance_results
)

# Sort from highest importance to lowest

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RNN FEATURE IMPORTANCE RESULTS")
print("=" * 70)

print(
    "\n"
)

print(
    importance_df[
        [
            "Feature_Name",
            "Importance"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 12. TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 RNN FEATURES")
print("=" * 70)

top_features = (
    importance_df
    .head(10)
)

for rank, (_, row) in enumerate(
    top_features.iterrows(),
    start=1
):

    print(
        f"{rank}. "
        f"{row['Feature_Name']} "
        f"→ "
        f"{row['Importance']:.4f}"
    )


# ============================================================
# 13. SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "rnn_feature_importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)

print(
    "\nFeature importance saved:"
)

print(
    csv_path
)


# ============================================================
# 14. CREATE TOP-10 PLOT
# ============================================================

print(
    "\nCreating feature importance plot..."
)

plot_data = (
    importance_df
    .head(10)
    .sort_values(
        "Importance",
        ascending=True
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    plot_data["Feature_Name"],
    plot_data["Importance"]
)

plt.xlabel(
    "Decrease in ROC-AUC"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 RNN Feature Importance"
)

plt.tight_layout()


plot_path = os.path.join(
    OUTPUT_DIR,
    "rnn_feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Feature importance plot saved:"
)

print(
    plot_path
)


# ============================================================
# 15. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Permutation importance works by changing one feature
while keeping the rest of the patient sequence unchanged.

If the model's ROC-AUC decreases substantially,
that feature was important to the model's predictions.

Higher importance:
    → Greater influence on model performance

Lower importance:
    → Smaller influence on model performance
""")


# ============================================================
# 16. IMPORTANT CAUTION
# ============================================================

print(
    "IMPORTANT:"
)

print(
    "Feature importance describes model behavior."
)

print(
    "It does NOT prove that a feature causes "
    "patient deterioration."
)


# ============================================================
# 17. FINAL
# ============================================================

print("\n" + "=" * 70)
print("RNN FEATURE IMPORTANCE COMPLETED")
print("=" * 70)