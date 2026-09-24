import os
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("HEALTHAI - LSTM TRAINING ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD LSTM TRAINING HISTORY
# ============================================================

history_path = (
    "data/processed/deep_learning/lstm/"
    "lstm_training_history.csv"
)

print("\nLoading LSTM training history...")

history = pd.read_csv(history_path)

print("Training history loaded successfully!")

print("\nDataset shape:")
print(history.shape)

print("\nAvailable columns:")
print(history.columns.tolist())

print("\nFirst 5 epochs:")
print(history.head())

# ============================================================
# 2. BEST VALIDATION PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION PERFORMANCE")
print("=" * 70)

best_val_loss_row = history.loc[
    history["val_loss"].idxmin()
]

best_val_accuracy_row = history.loc[
    history["val_accuracy"].idxmax()
]

print(
    "\nBest Validation Loss Epoch:",
    int(best_val_loss_row["Epoch"])
)

print(
    "Best Validation Loss:",
    f"{best_val_loss_row['val_loss']:.4f}"
)

print(
    "\nBest Validation Accuracy Epoch:",
    int(best_val_accuracy_row["Epoch"])
)

print(
    "Best Validation Accuracy:",
    f"{best_val_accuracy_row['val_accuracy']:.4f}"
)

# ============================================================
# 3. FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING PERFORMANCE")
print("=" * 70)

last_row = history.iloc[-1]

print(
    "\nFinal Training Loss:",
    f"{last_row['loss']:.4f}"
)

print(
    "Final Validation Loss:",
    f"{last_row['val_loss']:.4f}"
)

print(
    "Final Training Accuracy:",
    f"{last_row['accuracy']:.4f}"
)

print(
    "Final Validation Accuracy:",
    f"{last_row['val_accuracy']:.4f}"
)

# ============================================================
# 4. OVERFITTING CHECK
# ============================================================

print("\n" + "=" * 70)
print("OVERFITTING ANALYSIS")
print("=" * 70)

accuracy_gap = (
    last_row["accuracy"]
    - last_row["val_accuracy"]
)

loss_gap = (
    last_row["val_loss"]
    - last_row["loss"]
)

print(
    "\nAccuracy gap:",
    f"{accuracy_gap:.4f}"
)

print(
    "Loss gap:",
    f"{loss_gap:.4f}"
)

if abs(accuracy_gap) < 0.05:

    print(
        "\n✓ No major accuracy gap detected."
    )

else:

    print(
        "\n⚠ Possible overfitting detected."
    )

# ============================================================
# 5. TRAINING LOSS PLOT
# ============================================================

output_dir = (
    "data/processed/deep_learning/lstm/"
    "training_analysis"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

print("\n" + "=" * 70)
print("CREATING TRAINING PLOTS")
print("=" * 70)

plt.figure(figsize=(8, 5))

plt.plot(
    history["Epoch"],
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["Epoch"],
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "LSTM Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    output_dir,
    "lstm_training_vs_validation_loss.png"
)

plt.savefig(loss_path)

plt.close()

print(
    "\nLoss plot saved:",
    loss_path
)

# ============================================================
# 6. TRAINING ACCURACY PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["Epoch"],
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["Epoch"],
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "LSTM Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    output_dir,
    "lstm_training_vs_validation_accuracy.png"
)

plt.savefig(
    accuracy_path
)

plt.close()

print(
    "Accuracy plot saved:",
    accuracy_path
)

# ============================================================
# 7. SAVE LSTM SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Best Validation Loss",
        "Best Validation Loss Epoch",
        "Best Validation Accuracy",
        "Best Validation Accuracy Epoch",
        "Final Training Loss",
        "Final Validation Loss",
        "Final Training Accuracy",
        "Final Validation Accuracy",
        "Accuracy Gap"
    ],

    "Value": [
        best_val_loss_row["val_loss"],
        best_val_loss_row["Epoch"],
        best_val_accuracy_row["val_accuracy"],
        best_val_accuracy_row["Epoch"],
        last_row["loss"],
        last_row["val_loss"],
        last_row["accuracy"],
        last_row["val_accuracy"],
        accuracy_gap
    ]
})

summary_path = os.path.join(
    output_dir,
    "lstm_training_analysis_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    "Summary saved:",
    summary_path
)

# ============================================================
# 8. FINAL RNN vs LSTM COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("RNN vs LSTM FINAL COMPARISON")
print("=" * 70)

rnn_metrics_path = (
    "data/processed/deep_learning/rnn/"
    "rnn_metrics.csv"
)

lstm_metrics_path = (
    "data/processed/deep_learning/lstm/"
    "lstm_metrics.csv"
)

rnn_metrics = pd.read_csv(
    rnn_metrics_path
)

lstm_metrics = pd.read_csv(
    lstm_metrics_path
)

rnn_values = dict(
    zip(
        rnn_metrics["Metric"],
        rnn_metrics["Value"]
    )
)

lstm_values = dict(
    zip(
        lstm_metrics["Metric"],
        lstm_metrics["Value"]
    )
)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "RNN": [
        rnn_values["Accuracy"],
        rnn_values["Precision"],
        rnn_values["Recall"],
        rnn_values["F1 Score"],
        rnn_values["ROC-AUC"]
    ],

    "LSTM": [
        lstm_values["Accuracy"],
        lstm_values["Precision"],
        lstm_values["Recall"],
        lstm_values["F1 Score"],
        lstm_values["ROC-AUC"]
    ]
})

print("\n")
print(
    comparison.to_string(
        index=False,
        formatters={
            "RNN": "{:.4f}".format,
            "LSTM": "{:.4f}".format
        }
    )
)

# ============================================================
# 9. DETERMINE BEST MODEL BY METRIC
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL BY METRIC")
print("=" * 70)

for _, row in comparison.iterrows():

    metric = row["Metric"]

    rnn_score = row["RNN"]

    lstm_score = row["LSTM"]

    if rnn_score > lstm_score:

        winner = "RNN"

    elif lstm_score > rnn_score:

        winner = "LSTM"

    else:

        winner = "TIE"

    print(
        f"{metric}: {winner}"
    )

# ============================================================
# 10. PRIMARY MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY MODEL SELECTION")
print("=" * 70)

# For deterioration detection,
# recall and F1 are more important than
# accuracy alone.

if (
    rnn_values["Recall"]
    > lstm_values["Recall"]
    and
    rnn_values["F1 Score"]
    > lstm_values["F1 Score"]
):

    primary_model = "RNN"

else:

    primary_model = "LSTM"

print(
    "\nPrimary model:",
    primary_model
)

print(
    "\nReason:"
)

print(
    "The primary model is selected using "
    "deterioration detection performance, "
    "with emphasis on Recall and F1 Score."
)

# ============================================================
# 11. SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    output_dir,
    "rnn_vs_lstm_comparison.csv"
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
# 12. SAVE MODEL DECISION
# ============================================================

decision = pd.DataFrame({

    "Decision": [
        "Primary Model"
    ],

    "Selected_Model": [
        primary_model
    ],

    "Reason": [
        "Selected based on deterioration "
        "Recall and F1 performance"
    ]
})

decision_path = os.path.join(
    output_dir,
    "primary_model_decision.csv"
)

decision.to_csv(
    decision_path,
    index=False
)

print(
    "Model decision saved:",
    decision_path
)

# ============================================================
# 13. FINAL
# ============================================================

print("\n" + "=" * 70)
print("LSTM TRAINING ANALYSIS COMPLETED")
print("=" * 70)