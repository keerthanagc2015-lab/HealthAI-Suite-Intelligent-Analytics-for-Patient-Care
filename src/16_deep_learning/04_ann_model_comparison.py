import os
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 70)
print("HEALTHAI - ANN MODEL COMPARISON")
print("=" * 70)


# ================================================================
# 1. FILE PATHS
# ================================================================

TRADITIONAL_MODELS_PATH = (
    "data/processed/length_of_stay/"
    "regression_model_comparison.csv"
)

ANN_METRICS_PATH = (
    "data/processed/deep_learning/"
    "ann_metrics.csv"
)

TUNING_RESULTS_PATH = (
    "data/processed/deep_learning/"
    "hyperparameter_tuning/"
    "ann_hyperparameter_results.csv"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/"
    "model_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 2. LOAD TRADITIONAL MODEL RESULTS
# ================================================================

print("\nLoading traditional ML results...")

traditional_df = pd.read_csv(
    TRADITIONAL_MODELS_PATH
)

print(
    "Traditional model results shape:",
    traditional_df.shape
)

print("\nTraditional model columns:")

print(
    traditional_df.columns.tolist()
)

print("\nTraditional model results:")

print(
    traditional_df.to_string(
        index=False
    )
)


# ================================================================
# 3. LOAD BASELINE ANN RESULTS
# ================================================================

print("\n" + "=" * 70)
print("LOADING BASELINE ANN")
print("=" * 70)

ann_df = pd.read_csv(
    ANN_METRICS_PATH
)

print("\nANN Metrics:")

print(
    ann_df.to_string(
        index=False
    )
)


# ================================================================
# 4. LOAD HYPERPARAMETER TUNING RESULTS
# ================================================================

print("\n" + "=" * 70)
print("LOADING TUNED ANN RESULTS")
print("=" * 70)

tuning_df = pd.read_csv(
    TUNING_RESULTS_PATH
)

print(
    "\nNumber of ANN configurations:",
    len(tuning_df)
)


# Sort by Test MAE
tuning_df = tuning_df.sort_values(
    by="Test_MAE",
    ascending=True
)


best_tuned = tuning_df.iloc[0]


print("\nBest Tuned ANN:")

print(
    "Experiment:",
    best_tuned["Experiment"]
)

print(
    "Architecture:",
    best_tuned["Architecture"]
)

print(
    "Learning Rate:",
    best_tuned["Learning_Rate"]
)

print(
    "Batch Size:",
    best_tuned["Batch_Size"]
)

print(
    "Dropout:",
    best_tuned["Dropout"]
)

print(
    "Test MAE:",
    best_tuned["Test_MAE"]
)


# ================================================================
# 5. STANDARDIZE TRADITIONAL MODEL COLUMNS
# ================================================================

print("\n" + "=" * 70)
print("PREPARING COMPARISON TABLE")
print("=" * 70)


# Show available columns so we can identify them
print(
    "\nTraditional model columns:"
)

print(
    traditional_df.columns.tolist()
)


# Try to identify model column
model_column = None

for column in [
    "Model",
    "model",
    "Model_Name",
    "model_name"
]:

    if column in traditional_df.columns:

        model_column = column

        break


if model_column is None:

    raise ValueError(
        "Could not find model-name column "
        "in regression_model_comparison.csv"
    )


# Identify metric columns
def find_column(
    dataframe,
    possible_names
):

    for name in possible_names:

        if name in dataframe.columns:

            return name

    return None


mae_column = find_column(
    traditional_df,
    [
        "MAE",
        "mae",
        "Test_MAE",
        "test_mae"
    ]
)

rmse_column = find_column(
    traditional_df,
    [
        "RMSE",
        "rmse",
        "Test_RMSE",
        "test_rmse"
    ]
)

r2_column = find_column(
    traditional_df,
    [
        "R2",
        "r2",
        "R²",
        "Test_R2",
        "test_r2"
    ]
)


print("\nDetected columns:")

print(
    "Model:",
    model_column
)

print(
    "MAE:",
    mae_column
)

print(
    "RMSE:",
    rmse_column
)

print(
    "R2:",
    r2_column
)


if mae_column is None:

    raise ValueError(
        "MAE column not found."
    )

if rmse_column is None:

    raise ValueError(
        "RMSE column not found."
    )

if r2_column is None:

    raise ValueError(
        "R2 column not found."
    )


# ================================================================
# 6. CREATE TRADITIONAL MODEL TABLE
# ================================================================

traditional_comparison = pd.DataFrame({

    "Model":
        traditional_df[model_column],

    "MAE":
        traditional_df[mae_column],

    "RMSE":
        traditional_df[rmse_column],

    "R2":
        traditional_df[r2_column]

})


# ================================================================
# 7. CREATE BASELINE ANN ROW
# ================================================================

print("\nPreparing baseline ANN result...")


# ann_metrics.csv is expected to contain
# MAE, RMSE and R2.

ann_mae_column = find_column(
    ann_df,
    [
        "MAE",
        "mae",
        "Test_MAE",
        "test_mae"
    ]
)

ann_rmse_column = find_column(
    ann_df,
    [
        "RMSE",
        "rmse",
        "Test_RMSE",
        "test_rmse"
    ]
)

ann_r2_column = find_column(
    ann_df,
    [
        "R2",
        "r2",
        "R²",
        "Test_R2",
        "test_r2"
    ]
)


if ann_mae_column is None:

    raise ValueError(
        "ANN MAE column not found."
    )

if ann_rmse_column is None:

    raise ValueError(
        "ANN RMSE column not found."
    )

if ann_r2_column is None:

    raise ValueError(
        "ANN R2 column not found."
    )


# If multiple rows exist, use the first/final
# saved ANN metrics row.

baseline_ann_row = pd.DataFrame({

    "Model": [
        "ANN Baseline"
    ],

    "MAE": [
        ann_df.iloc[0][ann_mae_column]
    ],

    "RMSE": [
        ann_df.iloc[0][ann_rmse_column]
    ],

    "R2": [
        ann_df.iloc[0][ann_r2_column]
    ]

})


# ================================================================
# 8. CREATE TUNED ANN ROW
# ================================================================

# Our tuning script currently saved Test MAE
# and Test Loss.
#
# RMSE and R2 were not calculated in that script.
# Therefore we do NOT invent them.

tuned_ann_row = pd.DataFrame({

    "Model": [
        "ANN Tuned"
    ],

    "MAE": [
        best_tuned["Test_MAE"]
    ],

    "RMSE": [
        float("nan")
    ],

    "R2": [
        float("nan")
    ]

})


# ================================================================
# 9. COMBINE ALL MODELS
# ================================================================

comparison_df = pd.concat(
    [
        traditional_comparison,
        baseline_ann_row,
        tuned_ann_row
    ],
    ignore_index=True
)


# ================================================================
# 10. SORT BY MAE
# ================================================================

comparison_df = comparison_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(
    drop=True
)


# ================================================================
# 11. DISPLAY FINAL COMPARISON
# ================================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


# ================================================================
# 12. FIND BEST MODEL BY MAE
# ================================================================

best_model = comparison_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL BY MAE")
print("=" * 70)

print(
    "Model:",
    best_model["Model"]
)

print(
    f"MAE: {best_model['MAE']:.4f}"
)


# ================================================================
# 13. FIND BEST MODEL BY RMSE
# ================================================================

rmse_comparison = (
    comparison_df
    .dropna(subset=["RMSE"])
    .sort_values(
        by="RMSE"
    )
)

if len(rmse_comparison) > 0:

    best_rmse_model = (
        rmse_comparison.iloc[0]
    )

    print(
        "\nBest Model by RMSE:",
        best_rmse_model["Model"]
    )

    print(
        f"RMSE: "
        f"{best_rmse_model['RMSE']:.4f}"
    )


# ================================================================
# 14. FIND BEST MODEL BY R2
# ================================================================

r2_comparison = (
    comparison_df
    .dropna(subset=["R2"])
    .sort_values(
        by="R2",
        ascending=False
    )
)

if len(r2_comparison) > 0:

    best_r2_model = (
        r2_comparison.iloc[0]
    )

    print(
        "\nBest Model by R2:",
        best_r2_model["Model"]
    )

    print(
        f"R2: "
        f"{best_r2_model['R2']:.4f}"
    )


# ================================================================
# 15. SAVE COMPARISON TABLE
# ================================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ================================================================
# 16. MAE VISUALIZATION
# ================================================================

plt.figure(
    figsize=(12, 7)
)

plt.bar(
    comparison_df["Model"],
    comparison_df["MAE"]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Model Comparison - MAE"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

mae_plot_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison_mae.png"
)

plt.savefig(
    mae_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ================================================================
# 17. RMSE VISUALIZATION
# ================================================================

rmse_plot_df = (
    comparison_df
    .dropna(subset=["RMSE"])
)

if len(rmse_plot_df) > 0:

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        rmse_plot_df["Model"],
        rmse_plot_df["RMSE"]
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "Model Comparison - RMSE"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    rmse_plot_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison_rmse.png"
    )

    plt.savefig(
        rmse_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ================================================================
# 18. R2 VISUALIZATION
# ================================================================

r2_plot_df = (
    comparison_df
    .dropna(subset=["R2"])
)

if len(r2_plot_df) > 0:

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        r2_plot_df["Model"],
        r2_plot_df["R2"]
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "R²"
    )

    plt.title(
        "Model Comparison - R²"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    r2_plot_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison_r2.png"
    )

    plt.savefig(
        r2_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ================================================================
# 19. FINAL
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nFinal Comparison:"
)

print(
    comparison_path
)

print(
    "\nMAE Plot:"
)

print(
    mae_plot_path
)

if len(rmse_plot_df) > 0:

    print(
        "\nRMSE Plot:"
    )

    print(
        rmse_plot_path
    )

if len(r2_plot_df) > 0:

    print(
        "\nR2 Plot:"
    )

    print(
        r2_plot_path
    )


print("\n" + "=" * 70)
print("ANN MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)