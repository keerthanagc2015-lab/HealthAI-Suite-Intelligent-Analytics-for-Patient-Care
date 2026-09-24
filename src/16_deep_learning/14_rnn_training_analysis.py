import os
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("HEALTHAI - RNN TRAINING ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD TRAINING HISTORY
# ============================================================

history_path = (
    "data/processed/deep_learning/rnn/"
    "rnn_training_history.csv"
)

print("\nLoading RNN training history...")

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
    "data/processed/deep_learning/rnn/"
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
    "RNN Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    output_dir,
    "rnn_training_vs_validation_loss.png"
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
    "RNN Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    output_dir,
    "rnn_training_vs_validation_accuracy.png"
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
# 7. SAVE SUMMARY
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
    "rnn_training_analysis_summary.csv"
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
# 8. FINAL
# ============================================================

print("\n" + "=" * 70)
print("RNN TRAINING ANALYSIS COMPLETED")
print("=" * 70)