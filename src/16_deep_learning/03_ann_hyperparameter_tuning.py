import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ================================================================
# 1. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 70)
print("HEALTHAI - ANN HYPERPARAMETER TUNING")
print("=" * 70)


# ================================================================
# 2. PATHS
# ================================================================

DATA_PATH = (
    "data/processed/length_of_stay/"
    "feature_engineering/"
    "length_of_stay_engineered.csv"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/"
    "hyperparameter_tuning"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 3. LOAD DATA
# ================================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ================================================================
# 4. CLEAN RCOUNT
# ================================================================

df["rcount"] = (
    df["rcount"]
    .astype(str)
    .str.replace("+", "", regex=False)
)

df["rcount"] = pd.to_numeric(
    df["rcount"],
    errors="coerce"
)

df["rcount"] = df["rcount"].fillna(
    df["rcount"].median()
)


# ================================================================
# 5. FEATURES AND TARGET
# ================================================================

selected_features = [
    "rcount",
    "gender",
    "dialysisrenalendstage",
    "asthma",
    "irondef",
    "pneum",
    "substancedependence",
    "psychologicaldisordermajor",
    "depress",
    "psychother",
    "fibrosisandother",
    "malnutrition",
    "hemo",
    "hematocrit",
    "neutrophils",
    "sodium",
    "glucose",
    "bloodureanitro",
    "creatinine",
    "bmi",
    "pulse",
    "respiration",
    "secondarydiagnosisnonicd9",
    "facid"
]

target = "lengthofstay"

X = df[selected_features]
y = df[target]


# ================================================================
# 6. TRAIN / TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)

print("\nTrain/Test Split")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ================================================================
# 7. FEATURE TYPES
# ================================================================

categorical_features = [
    "gender",
    "facid"
]

numerical_features = [
    feature
    for feature in selected_features
    if feature not in categorical_features
]


# ================================================================
# 8. PREPROCESSING
# ================================================================

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


print("\nPreprocessing data...")

X_train_processed = (
    preprocessor.fit_transform(X_train)
)

X_test_processed = (
    preprocessor.transform(X_test)
)

print(
    "Processed Training Shape:",
    X_train_processed.shape
)

print(
    "Processed Testing Shape:",
    X_test_processed.shape
)


# ================================================================
# 9. TRAINING / VALIDATION SPLIT
# ================================================================

X_train_dl, X_val, y_train_dl, y_val = train_test_split(
    X_train_processed,
    y_train,
    test_size=0.20,
    random_state=SEED
)


# ================================================================
# 10. MODEL BUILDER
# ================================================================

def build_model(
    architecture,
    learning_rate,
    dropout_rate
):

    model = Sequential()

    # Input layer + first hidden layer
    model.add(
        Dense(
            architecture[0],
            activation="relu",
            input_shape=(
                X_train_processed.shape[1],
            )
        )
    )

    if dropout_rate > 0:
        model.add(
            Dropout(dropout_rate)
        )

    # Remaining hidden layers
    for units in architecture[1:]:

        model.add(
            Dense(
                units,
                activation="relu"
            )
        )

        if dropout_rate > 0:
            model.add(
                Dropout(dropout_rate)
            )

    # Regression output
    model.add(
        Dense(1)
    )

    optimizer = Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


# ================================================================
# 11. EXPERIMENT CONFIGURATIONS
# ================================================================

experiments = [

    {
        "name": "Baseline",
        "architecture": [128, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 256,
        "dropout": 0.0
    },

    {
        "name": "Larger_Network",
        "architecture": [256, 128, 64],
        "learning_rate": 0.001,
        "batch_size": 256,
        "dropout": 0.0
    },

    {
        "name": "Lower_Learning_Rate",
        "architecture": [128, 64, 32],
        "learning_rate": 0.0005,
        "batch_size": 256,
        "dropout": 0.0
    },

    {
        "name": "Dropout_0.1",
        "architecture": [128, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 256,
        "dropout": 0.1
    },

    {
        "name": "Dropout_0.2",
        "architecture": [128, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 256,
        "dropout": 0.2
    },

    {
        "name": "Larger_Lower_LR",
        "architecture": [256, 128, 64],
        "learning_rate": 0.0005,
        "batch_size": 256,
        "dropout": 0.1
    },

    {
        "name": "Smaller_Batch",
        "architecture": [128, 64, 32],
        "learning_rate": 0.001,
        "batch_size": 128,
        "dropout": 0.1
    }
]


# ================================================================
# 12. RUN EXPERIMENTS
# ================================================================

results = []

histories = {}

for experiment in experiments:

    print("\n" + "=" * 70)

    print(
        "EXPERIMENT:",
        experiment["name"]
    )

    print("=" * 70)

    print(
        "Architecture:",
        experiment["architecture"]
    )

    print(
        "Learning Rate:",
        experiment["learning_rate"]
    )

    print(
        "Batch Size:",
        experiment["batch_size"]
    )

    print(
        "Dropout:",
        experiment["dropout"]
    )

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------

    model = build_model(
        architecture=experiment["architecture"],
        learning_rate=experiment["learning_rate"],
        dropout_rate=experiment["dropout"]
    )

    # ------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    history = model.fit(
        X_train_dl,
        y_train_dl,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=40,
        batch_size=experiment["batch_size"],
        callbacks=[early_stopping],
        verbose=0
    )

    histories[
        experiment["name"]
    ] = history.history

    # ------------------------------------------------------------
    # Best validation performance
    # ------------------------------------------------------------

    best_epoch = (
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    best_val_mae = min(
        history.history["val_mae"]
    )

    # ------------------------------------------------------------
    # Test performance
    # ------------------------------------------------------------

    test_loss, test_mae = model.evaluate(
        X_test_processed,
        y_test,
        verbose=0
    )

    # ------------------------------------------------------------
    # Store result
    # ------------------------------------------------------------

    results.append(
        {
            "Experiment": experiment["name"],
            "Architecture": str(
                experiment["architecture"]
            ),
            "Learning_Rate":
                experiment["learning_rate"],
            "Batch_Size":
                experiment["batch_size"],
            "Dropout":
                experiment["dropout"],
            "Best_Epoch":
                best_epoch,
            "Best_Validation_Loss":
                best_val_loss,
            "Best_Validation_MAE":
                best_val_mae,
            "Test_Loss":
                test_loss,
            "Test_MAE":
                test_mae
        }
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best Validation MAE: "
        f"{best_val_mae:.4f}"
    )

    print(
        f"Test MAE: "
        f"{test_mae:.4f}"
    )


# ================================================================
# 13. RESULTS DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Best_Validation_MAE"
)


# ================================================================
# 14. DISPLAY RESULTS
# ================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# 15. BEST MODEL
# ================================================================

best_result = (
    results_df.iloc[0]
)

print("\n" + "=" * 70)
print("BEST ANN CONFIGURATION")
print("=" * 70)

print(
    "Experiment:",
    best_result["Experiment"]
)

print(
    "Architecture:",
    best_result["Architecture"]
)

print(
    "Learning Rate:",
    best_result["Learning_Rate"]
)

print(
    "Batch Size:",
    best_result["Batch_Size"]
)

print(
    "Dropout:",
    best_result["Dropout"]
)

print(
    "Best Epoch:",
    best_result["Best_Epoch"]
)

print(
    "Best Validation MAE:",
    f"{best_result['Best_Validation_MAE']:.4f}"
)

print(
    "Test MAE:",
    f"{best_result['Test_MAE']:.4f}"
)


# ================================================================
# 16. SAVE RESULTS
# ================================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "ann_hyperparameter_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ================================================================
# 17. SAVE HISTORIES
# ================================================================

for name, history in histories.items():

    history_df = pd.DataFrame(history)

    history_df.insert(
        0,
        "Epoch",
        range(
            1,
            len(history_df) + 1
        )
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )


# ================================================================
# 18. FINAL
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nResults:"
)

print(
    results_path
)

print(
    "\nHyperparameter tuning completed successfully."
)

print("=" * 70)