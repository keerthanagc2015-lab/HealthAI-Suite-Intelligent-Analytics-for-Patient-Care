import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

print("=" * 70)
print("HEALTHAI - RNN THRESHOLD ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD RNN PREDICTIONS
# ============================================================

prediction_path = (
    "data/processed/deep_learning/rnn/"
    "rnn_predictions.csv"
)

print("\nLoading RNN predictions...")

df = pd.read_csv(prediction_path)

print("Predictions loaded successfully!")

print("\nRows:", len(df))

print(
    "Columns:",
    df.columns.tolist()
)

# ============================================================
# 2. EXTRACT ACTUAL VALUES AND PROBABILITIES
# ============================================================

y_true = df["Actual"].values

probabilities = df["Probability"].values

print("\nActual class distribution:")

print(
    pd.Series(y_true)
    .value_counts()
    .sort_index()
)

print("\nProbability range:")

print(
    "Minimum:",
    probabilities.min()
)

print(
    "Maximum:",
    probabilities.max()
)

# ============================================================
# 3. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    positive_predictions = (
        predictions.sum()
    )

    positive_percentage = (
        positive_predictions
        / len(predictions)
        * 100
    )

    results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Positive_Predictions":
            positive_predictions,

        "Positive_Percentage":
            positive_percentage
    })

results_df = pd.DataFrame(results)

print("\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "Positive_Percentage":
                "{:.2f}".format
        }
    )
)

# ============================================================
# 4. BEST F1 THRESHOLD
# ============================================================

best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    "\nThreshold:",
    best_f1_row["Threshold"]
)

print(
    "Accuracy:",
    f"{best_f1_row['Accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_f1_row['Precision']:.4f}"
)

print(
    "Recall:",
    f"{best_f1_row['Recall']:.4f}"
)

print(
    "F1:",
    f"{best_f1_row['F1']:.4f}"
)

# ============================================================
# 5. BEST RECALL THRESHOLD
# ============================================================

best_recall_row = results_df.loc[
    results_df["Recall"].idxmax()
]

print("\n" + "=" * 70)
print("BEST RECALL THRESHOLD")
print("=" * 70)

print(
    "\nThreshold:",
    best_recall_row["Threshold"]
)

print(
    "Accuracy:",
    f"{best_recall_row['Accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_recall_row['Precision']:.4f}"
)

print(
    "Recall:",
    f"{best_recall_row['Recall']:.4f}"
)

print(
    "F1:",
    f"{best_recall_row['F1']:.4f}"
)

# ============================================================
# 6. BEST PRECISION THRESHOLD
# ============================================================

best_precision_row = results_df.loc[
    results_df["Precision"].idxmax()
]

print("\n" + "=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(
    "\nThreshold:",
    best_precision_row["Threshold"]
)

print(
    "Accuracy:",
    f"{best_precision_row['Accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_precision_row['Precision']:.4f}"
)

print(
    "Recall:",
    f"{best_precision_row['Recall']:.4f}"
)

print(
    "F1:",
    f"{best_precision_row['F1']:.4f}"
)

# ============================================================
# 7. RECOMMENDED MEDICAL WARNING THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDED WARNING THRESHOLD")
print("=" * 70)

# For deterioration detection,
# prioritize recall while maintaining
# reasonable precision.

candidate_results = results_df[
    results_df["Recall"] >= 0.70
]

if len(candidate_results) > 0:

    recommended_row = candidate_results.loc[
        candidate_results["F1"].idxmax()
    ]

else:

    recommended_row = best_f1_row

print(
    "\nRecommended threshold:",
    recommended_row["Threshold"]
)

print(
    "Accuracy:",
    f"{recommended_row['Accuracy']:.4f}"
)

print(
    "Precision:",
    f"{recommended_row['Precision']:.4f}"
)

print(
    "Recall:",
    f"{recommended_row['Recall']:.4f}"
)

print(
    "F1:",
    f"{recommended_row['F1']:.4f}"
)

print(
    "\nReason:"
)

print(
    "For hospital deterioration prediction, "
    "recall is important because missing a true "
    "deterioration case can be more serious than "
    "generating an additional warning."
)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

output_dir = (
    "data/processed/deep_learning/rnn/"
    "threshold_analysis"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "rnn_threshold_analysis.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("FILE CREATED")
print("=" * 70)

print(
    "\nThreshold analysis:",
    output_path
)

# ============================================================
# 9. SAVE RECOMMENDATION
# ============================================================

recommendation_df = pd.DataFrame({

    "Metric": [
        "Best F1 Threshold",
        "Best Recall Threshold",
        "Best Precision Threshold",
        "Recommended Threshold"
    ],

    "Threshold": [
        best_f1_row["Threshold"],
        best_recall_row["Threshold"],
        best_precision_row["Threshold"],
        recommended_row["Threshold"]
    ]
})

recommendation_path = os.path.join(
    output_dir,
    "rnn_threshold_recommendation.csv"
)

recommendation_df.to_csv(
    recommendation_path,
    index=False
)

print(
    "Recommendation:",
    recommendation_path
)

print("\n" + "=" * 70)
print("RNN THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)