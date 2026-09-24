import os
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("HEALTHAI - FINAL DEEP LEARNING MODEL COMPARISON")
print("=" * 70)

# ============================================================
# 1. FILE PATHS
# ============================================================

cnn_path = (
    "data/processed/deep_learning/cnn/"
    "cnn_metrics.csv"
)

rnn_path = (
    "data/processed/deep_learning/rnn/"
    "rnn_metrics.csv"
)

lstm_path = (
    "data/processed/deep_learning/lstm/"
    "lstm_metrics.csv"
)

# ============================================================
# 2. LOAD METRICS
# ============================================================

print("\nLoading model metrics...")

cnn = pd.read_csv(cnn_path)
rnn = pd.read_csv(rnn_path)
lstm = pd.read_csv(lstm_path)

print("CNN metrics loaded.")
print("CNN columns:", cnn.columns.tolist())

print("\nRNN metrics loaded.")
print("RNN columns:", rnn.columns.tolist())

print("\nLSTM metrics loaded.")
print("LSTM columns:", lstm.columns.tolist())

# ============================================================
# 3. FUNCTION TO READ METRICS SAFELY
# ============================================================

def extract_metrics(df, model_name):

    result = {}

    # --------------------------------------------------------
    # Format 1:
    # Metric | Value
    # --------------------------------------------------------

    if (
        "Metric" in df.columns
        and "Value" in df.columns
    ):

        for _, row in df.iterrows():

            result[str(row["Metric"]).strip()] = float(
                row["Value"]
            )

        return result

    # --------------------------------------------------------
    # Format 2:
    # lowercase metric names
    # --------------------------------------------------------

    column_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
        "roc-auc": "ROC-AUC"
    }

    for column in df.columns:

        column_lower = str(column).lower().strip()

        if column_lower in column_map:

            value = df[column].iloc[0]

            result[column_map[column_lower]] = float(
                value
            )

    if len(result) == 0:

        raise ValueError(
            f"Could not understand the format of "
            f"{model_name} metrics file.\n"
            f"Columns found: {df.columns.tolist()}"
        )

    return result


# ============================================================
# 4. EXTRACT METRICS
# ============================================================

print("\n" + "=" * 70)
print("READING METRICS")
print("=" * 70)

cnn_values = extract_metrics(
    cnn,
    "CNN"
)

rnn_values = extract_metrics(
    rnn,
    "RNN"
)

lstm_values = extract_metrics(
    lstm,
    "LSTM"
)

print("\nCNN metrics found:")
print(cnn_values)

print("\nRNN metrics found:")
print(rnn_values)

print("\nLSTM metrics found:")
print(lstm_values)

# ============================================================
# 5. CREATE COMPARISON TABLE
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

comparison = pd.DataFrame({

    "Metric": metrics,

    "CNN": [
        cnn_values.get(metric, None)
        for metric in metrics
    ],

    "RNN": [
        rnn_values.get(metric, None)
        for metric in metrics
    ],

    "LSTM": [
        lstm_values.get(metric, None)
        for metric in metrics
    ]
})

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 6. MODEL PURPOSE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PURPOSE")
print("=" * 70)

print("""
CNN:
Chest X-ray image classification
→ Normal / Pneumonia

RNN:
24-hour patient sequence
→ Deterioration within next 12 hours

LSTM:
24-hour patient sequence
→ Deterioration within next 12 hours
""")

# ============================================================
# 7. BEST MODEL BY METRIC
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL BY METRIC")
print("=" * 70)

for _, row in comparison.iterrows():

    metric = row["Metric"]

    values = {}

    if pd.notna(row["CNN"]):
        values["CNN"] = row["CNN"]

    if pd.notna(row["RNN"]):
        values["RNN"] = row["RNN"]

    if pd.notna(row["LSTM"]):
        values["LSTM"] = row["LSTM"]

    if len(values) == 0:
        continue

    best_model = max(
        values,
        key=values.get
    )

    print(
        f"{metric}: "
        f"{best_model} "
        f"({values[best_model]:.4f})"
    )

# ============================================================
# 8. RNN VS LSTM
# ============================================================

print("\n" + "=" * 70)
print("RNN vs LSTM")
print("=" * 70)

print(
    f"\nRNN Recall : "
    f"{rnn_values['Recall']:.4f}"
)

print(
    f"LSTM Recall: "
    f"{lstm_values['Recall']:.4f}"
)

print(
    f"\nRNN F1     : "
    f"{rnn_values['F1 Score']:.4f}"
)

print(
    f"LSTM F1    : "
    f"{lstm_values['F1 Score']:.4f}"
)

print(
    f"\nRNN ROC-AUC : "
    f"{rnn_values['ROC-AUC']:.4f}"
)

print(
    f"LSTM ROC-AUC: "
    f"{lstm_values['ROC-AUC']:.4f}"
)

# ============================================================
# 9. SELECT SEQUENTIAL MODEL
# ============================================================

if (
    rnn_values["Recall"]
    > lstm_values["Recall"]
    and
    rnn_values["F1 Score"]
    > lstm_values["F1 Score"]
):

    sequential_winner = "RNN"

else:

    sequential_winner = "LSTM"

print(
    "\nSelected sequential model:",
    sequential_winner
)

print(
    "\nReason:"
)

print(
    "For deterioration prediction, "
    "Recall and F1 are given greater importance "
    "than accuracy alone."
)

# ============================================================
# 10. SAVE OUTPUT DIRECTORY
# ============================================================

output_dir = (
    "data/processed/deep_learning/"
    "final_comparison"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# ============================================================
# 11. SAVE COMPARISON CSV
# ============================================================

comparison_path = os.path.join(
    output_dir,
    "cnn_rnn_lstm_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print(
    "\nComparison saved:",
    comparison_path
)

# ============================================================
# 12. CREATE COMPARISON PLOT
# ============================================================

print("\nCreating comparison plot...")

x = list(range(len(metrics)))

width = 0.25

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    [i - width for i in x],
    comparison["CNN"].fillna(0),
    width=width,
    label="CNN"
)

plt.bar(
    x,
    comparison["RNN"].fillna(0),
    width=width,
    label="RNN"
)

plt.bar(
    [i + width for i in x],
    comparison["LSTM"].fillna(0),
    width=width,
    label="LSTM"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.title(
    "HealthAI Deep Learning Model Comparison"
)

plt.legend()

plt.tight_layout()

plot_path = os.path.join(
    output_dir,
    "cnn_rnn_lstm_comparison.png"
)

plt.savefig(
    plot_path
)

plt.close()

print(
    "Comparison plot saved:",
    plot_path
)

# ============================================================
# 13. PROJECT SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Component": [
        "CNN",
        "RNN",
        "LSTM"
    ],

    "Task": [
        "Chest X-ray classification",
        "Patient deterioration prediction",
        "Patient deterioration prediction"
    ],

    "Role": [
        "Primary image model",
        "Primary sequential model",
        "Sequential comparison model"
    ]
})

summary_path = os.path.join(
    output_dir,
    "deep_learning_project_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    "Project summary saved:",
    summary_path
)

# ============================================================
# 14. FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL DEEP LEARNING COMPARISON COMPLETED")
print("=" * 70)

print("\nDeep-learning pipeline:")

print("✓ CNN - Chest X-ray pneumonia classification")
print("✓ RNN - Patient deterioration prediction")
print("✓ LSTM - Sequential model comparison")
print("✓ Final metrics comparison")
print("✓ Comparison visualization")
print("✓ Primary sequential model selected")

print("\n" + "=" * 70)