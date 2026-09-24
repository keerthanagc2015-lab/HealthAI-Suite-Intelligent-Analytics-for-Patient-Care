import os
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

METRICS_FILE = (
    "data/processed/deep_learning/cnn/"
    "cnn_metrics.csv"
)

CONFUSION_FILE = (
    "data/processed/deep_learning/cnn/"
    "cnn_confusion_matrix.csv"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/cnn/"
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("HEALTHAI - CNN EVALUATION")
print("=" * 70)


# ================================================================
# 1. LOAD METRICS
# ================================================================

metrics = pd.read_csv(
    METRICS_FILE
)

print("\nCNN Metrics Loaded")

print(metrics)


# ================================================================
# 2. DISPLAY METRICS
# ================================================================

row = metrics.iloc[0]

accuracy = row["Accuracy"]
precision = row["Precision"]
recall = row["Recall"]
f1 = row["F1"]


print("\n" + "=" * 70)
print("FINAL CNN METRICS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ================================================================
# 3. LOAD CONFUSION MATRIX
# ================================================================

cm = pd.read_csv(
    CONFUSION_FILE,
    index_col=0
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ================================================================
# 4. CREATE CONFUSION MATRIX PLOT
# ================================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm.values
)

plt.title(
    "CNN Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(2),
    [
        "NORMAL",
        "PNEUMONIA"
    ]
)

plt.yticks(
    range(2),
    [
        "NORMAL",
        "PNEUMONIA"
    ]
)


# Display numbers inside cells

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm.values[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()


confusion_plot = os.path.join(
    OUTPUT_DIR,
    "cnn_confusion_matrix.png"
)

plt.savefig(
    confusion_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 5. METRICS BAR CHART
# ================================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1
]


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    metric_names,
    metric_values
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Score"
)

plt.title(
    "CNN Performance Metrics"
)


for i, value in enumerate(
    metric_values
):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )


plt.tight_layout()


metrics_plot = os.path.join(
    OUTPUT_DIR,
    "cnn_performance_metrics.png"
)

plt.savefig(
    metrics_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 6. SAVE EVALUATION SUMMARY
# ================================================================

summary = pd.DataFrame(
    [
        {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    ]
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "cnn_evaluation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ================================================================
# 7. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nConfusion Matrix Plot:",
    confusion_plot
)

print(
    "\nPerformance Metrics Plot:",
    metrics_plot
)

print(
    "\nEvaluation Summary:",
    summary_path
)

print("\n" + "=" * 70)
print("CNN EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)