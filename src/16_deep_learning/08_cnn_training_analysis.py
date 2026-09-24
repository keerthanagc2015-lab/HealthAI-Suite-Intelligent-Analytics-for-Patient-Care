import os
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

HISTORY_FILE = (
    "data/processed/deep_learning/cnn/"
    "cnn_training_history.csv"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/cnn/"
    "training_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("HEALTHAI - CNN TRAINING ANALYSIS")
print("=" * 70)


# ================================================================
# 1. LOAD TRAINING HISTORY
# ================================================================

history = pd.read_csv(HISTORY_FILE)

print("\nTraining History Loaded Successfully")

print(
    "\nDataset Shape:",
    history.shape
)

print(
    "\nAvailable Columns:"
)

print(
    history.columns.tolist()
)

print(
    "\nFirst 5 Epochs:"
)

print(
    history.head()
)


# ================================================================
# 2. BEST VALIDATION PERFORMANCE
# ================================================================

best_loss_index = history[
    "val_loss"
].idxmin()

best_accuracy_index = history[
    "val_accuracy"
].idxmax()


best_loss_epoch = int(
    history.loc[
        best_loss_index,
        "Epoch"
    ]
)

best_val_loss = history.loc[
    best_loss_index,
    "val_loss"
]

best_accuracy_epoch = int(
    history.loc[
        best_accuracy_index,
        "Epoch"
    ]
)

best_val_accuracy = history.loc[
    best_accuracy_index,
    "val_accuracy"
]


print("\n" + "=" * 70)
print("BEST VALIDATION PERFORMANCE")
print("=" * 70)

print(
    f"\nBest Validation Loss Epoch : "
    f"{best_loss_epoch}"
)

print(
    f"Best Validation Loss       : "
    f"{best_val_loss:.4f}"
)

print(
    f"\nBest Validation Accuracy Epoch : "
    f"{best_accuracy_epoch}"
)

print(
    f"Best Validation Accuracy       : "
    f"{best_val_accuracy:.4f}"
)


# ================================================================
# 3. TRAINING VS VALIDATION LOSS
# ================================================================

plt.figure(figsize=(10, 6))

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
    "CNN Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

loss_plot = os.path.join(
    OUTPUT_DIR,
    "cnn_training_vs_validation_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 4. TRAINING VS VALIDATION ACCURACY
# ================================================================

plt.figure(figsize=(10, 6))

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
    "CNN Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

accuracy_plot = os.path.join(
    OUTPUT_DIR,
    "cnn_training_vs_validation_accuracy.png"
)

plt.savefig(
    accuracy_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 5. OVERFITTING CHECK
# ================================================================

final_train_loss = history[
    "loss"
].iloc[-1]

final_val_loss = history[
    "val_loss"
].iloc[-1]

final_train_accuracy = history[
    "accuracy"
].iloc[-1]

final_val_accuracy = history[
    "val_accuracy"
].iloc[-1]


print("\n" + "=" * 70)
print("OVERFITTING ANALYSIS")
print("=" * 70)

print(
    f"\nFinal Training Loss     : "
    f"{final_train_loss:.4f}"
)

print(
    f"Final Validation Loss   : "
    f"{final_val_loss:.4f}"
)

print(
    f"Final Training Accuracy : "
    f"{final_train_accuracy:.4f}"
)

print(
    f"Final Validation Accuracy : "
    f"{final_val_accuracy:.4f}"
)


if final_train_accuracy > final_val_accuracy + 0.05:

    print(
        "\nPossible overfitting detected."
    )

else:

    print(
        "\nNo major accuracy gap detected."
    )


# ================================================================
# 6. SAVE SUMMARY
# ================================================================

summary = pd.DataFrame(
    [
        {
            "Best_Validation_Loss_Epoch":
                best_loss_epoch,

            "Best_Validation_Loss":
                best_val_loss,

            "Best_Validation_Accuracy_Epoch":
                best_accuracy_epoch,

            "Best_Validation_Accuracy":
                best_val_accuracy,

            "Final_Training_Loss":
                final_train_loss,

            "Final_Validation_Loss":
                final_val_loss,

            "Final_Training_Accuracy":
                final_train_accuracy,

            "Final_Validation_Accuracy":
                final_val_accuracy
        }
    ]
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "cnn_training_analysis_summary.csv"
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
    "\nLoss Plot:",
    loss_plot
)

print(
    "\nAccuracy Plot:",
    accuracy_plot
)

print(
    "\nSummary:",
    summary_path
)

print("\n" + "=" * 70)
print("CNN TRAINING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)