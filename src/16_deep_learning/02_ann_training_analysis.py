import os
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 70)
print("HEALTHAI - ANN TRAINING ANALYSIS")
print("=" * 70)


# ================================================================
# 1. PATHS
# ================================================================

HISTORY_PATH = (
    "data/processed/deep_learning/"
    "ann_training_history.csv"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/"
    "training_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 2. LOAD TRAINING HISTORY
# ================================================================

history_df = pd.read_csv(HISTORY_PATH)

print("\nTraining History Loaded Successfully")

print(
    "Dataset Shape:",
    history_df.shape
)

print("\nAvailable Columns:")

print(
    history_df.columns.tolist()
)


# ================================================================
# 3. DISPLAY FIRST FEW EPOCHS
# ================================================================

print("\n" + "=" * 70)
print("TRAINING HISTORY - FIRST 5 EPOCHS")
print("=" * 70)

print(
    history_df.head()
)


# ================================================================
# 4. ADD EPOCH NUMBER
# ================================================================

history_df["Epoch"] = (
    range(
        1,
        len(history_df) + 1
    )
)


# ================================================================
# 5. FINAL TRAINING VALUES
# ================================================================

print("\n" + "=" * 70)
print("FINAL TRAINING RESULTS")
print("=" * 70)

final_row = history_df.iloc[-1]

print(
    f"Final Training Loss : "
    f"{final_row['loss']:.4f}"
)

print(
    f"Final Training MAE  : "
    f"{final_row['mae']:.4f}"
)

print(
    f"Final Validation Loss : "
    f"{final_row['val_loss']:.4f}"
)

print(
    f"Final Validation MAE  : "
    f"{final_row['val_mae']:.4f}"
)


# ================================================================
# 6. BEST VALIDATION EPOCH
# ================================================================

best_epoch_index = (
    history_df["val_loss"].idxmin()
)

best_epoch = (
    history_df.loc[
        best_epoch_index,
        "Epoch"
    ]
)

best_val_loss = (
    history_df.loc[
        best_epoch_index,
        "val_loss"
    ]
)

best_val_mae = (
    history_df.loc[
        best_epoch_index,
        "val_mae"
    ]
)


print("\n" + "=" * 70)
print("BEST VALIDATION PERFORMANCE")
print("=" * 70)

print(
    f"Best Epoch       : "
    f"{best_epoch}"
)

print(
    f"Best Val Loss    : "
    f"{best_val_loss:.4f}"
)

print(
    f"Val MAE at Best Epoch : "
    f"{best_val_mae:.4f}"
)


# ================================================================
# 7. TRAINING LOSS VS VALIDATION LOSS
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["Epoch"],
    history_df["loss"],
    label="Training Loss"
)

plt.plot(
    history_df["Epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "ANN Training Loss vs Validation Loss"
)

plt.legend()

plt.grid(True)

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "training_vs_validation_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ================================================================
# 8. TRAINING MAE VS VALIDATION MAE
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["Epoch"],
    history_df["mae"],
    label="Training MAE"
)

plt.plot(
    history_df["Epoch"],
    history_df["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.title(
    "ANN Training MAE vs Validation MAE"
)

plt.legend()

plt.grid(True)

mae_plot_path = os.path.join(
    OUTPUT_DIR,
    "training_vs_validation_mae.png"
)

plt.savefig(
    mae_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ================================================================
# 9. OVERFITTING ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("OVERFITTING ANALYSIS")
print("=" * 70)


initial_val_loss = (
    history_df["val_loss"].iloc[0]
)

final_val_loss = (
    history_df["val_loss"].iloc[-1]
)


if final_val_loss < initial_val_loss:

    print(
        "Validation loss decreased during training."
    )

    print(
        "The model learned from the training data."
    )

else:

    print(
        "Validation loss did not improve."
    )


# Check whether validation loss increased
# near the end of training.

minimum_val_loss = (
    history_df["val_loss"].min()
)

last_val_loss = (
    history_df["val_loss"].iloc[-1]
)


if last_val_loss > minimum_val_loss:

    print(
        "Validation loss increased after reaching "
        "its minimum."
    )

    print(
        "This may indicate some overfitting."
    )

else:

    print(
        "Validation loss did not increase "
        "after its minimum."
    )


# ================================================================
# 10. TRAINING SUMMARY
# ================================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Initial Training Loss",
        "Final Training Loss",
        "Initial Validation Loss",
        "Final Validation Loss",
        "Best Validation Loss",
        "Best Validation Epoch",
        "Final Training MAE",
        "Final Validation MAE"
    ],

    "Value": [
        history_df["loss"].iloc[0],
        history_df["loss"].iloc[-1],
        history_df["val_loss"].iloc[0],
        history_df["val_loss"].iloc[-1],
        best_val_loss,
        best_epoch,
        history_df["mae"].iloc[-1],
        history_df["val_mae"].iloc[-1]
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "ann_training_analysis_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ================================================================
# 11. SAVE UPDATED HISTORY
# ================================================================

history_output_path = os.path.join(
    OUTPUT_DIR,
    "ann_training_history_with_epoch.csv"
)

history_df.to_csv(
    history_output_path,
    index=False
)


# ================================================================
# 12. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nTraining Loss Plot:"
)

print(
    loss_plot_path
)

print(
    "\nTraining MAE Plot:"
)

print(
    mae_plot_path
)

print(
    "\nTraining Analysis Summary:"
)

print(
    summary_path
)

print(
    "\nUpdated Training History:"
)

print(
    history_output_path
)


print("\n" + "=" * 70)
print("ANN TRAINING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)