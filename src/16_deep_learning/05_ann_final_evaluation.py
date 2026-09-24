import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
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
print("HEALTHAI - FINAL TUNED ANN EVALUATION")
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
    "final_ann"
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


print("\nPreprocessing...")

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
# 9. TRAIN / VALIDATION SPLIT
# ================================================================

X_train_dl, X_val, y_train_dl, y_val = train_test_split(
    X_train_processed,
    y_train,
    test_size=0.20,
    random_state=SEED
)

print("\nTraining/Validation Split")

print(
    "Training:",
    X_train_dl.shape
)

print(
    "Validation:",
    X_val.shape
)


# ================================================================
# 10. WINNING ANN ARCHITECTURE
# ================================================================

print("\n" + "=" * 70)
print("WINNING ANN CONFIGURATION")
print("=" * 70)

print("Architecture : 128 → 64 → 32")
print("Learning Rate: 0.001")
print("Batch Size   : 128")
print("Dropout      : 0.1")


# ================================================================
# 11. BUILD MODEL
# ================================================================

model = Sequential(
    [
        Input(
            shape=(
                X_train_processed.shape[1],
            )
        ),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.1
        ),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.1
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(
            0.1
        ),

        Dense(
            1
        )
    ]
)


# ================================================================
# 12. COMPILE
# ================================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


# ================================================================
# 13. DISPLAY MODEL
# ================================================================

print("\n" + "=" * 70)
print("ANN ARCHITECTURE")
print("=" * 70)

model.summary()


# ================================================================
# 14. EARLY STOPPING
# ================================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ================================================================
# 15. TRAIN
# ================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL ANN")
print("=" * 70)

history = model.fit(
    X_train_dl,
    y_train_dl,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=40,
    batch_size=128,
    callbacks=[
        early_stopping
    ],
    verbose=1
)


# ================================================================
# 16. PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_processed,
    verbose=0
).flatten()


# ================================================================
# 17. CALCULATE METRICS
# ================================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ================================================================
# 18. DISPLAY FINAL RESULTS
# ================================================================

print("\n" + "=" * 70)
print("FINAL TUNED ANN TEST RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ================================================================
# 19. BEST VALIDATION RESULTS
# ================================================================

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

print("\n" + "=" * 70)
print("BEST VALIDATION PERFORMANCE")
print("=" * 70)

print(
    "Best Epoch:",
    best_epoch
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best Validation MAE: "
    f"{best_val_mae:.4f}"
)


# ================================================================
# 20. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame(
    [
        {
            "Model": "ANN Tuned",
            "Architecture": "128-64-32",
            "Learning_Rate": 0.001,
            "Batch_Size": 128,
            "Dropout": 0.1,
            "Best_Epoch": best_epoch,
            "Best_Validation_Loss": best_val_loss,
            "Best_Validation_MAE": best_val_mae,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    ]
)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "final_tuned_ann_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ================================================================
# 21. SAVE PREDICTIONS
# ================================================================

predictions_df = pd.DataFrame(
    {
        "Actual_Length_of_Stay": y_test.values,
        "Predicted_Length_of_Stay": y_pred
    }
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "final_tuned_ann_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ================================================================
# 22. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "Epoch",
    range(
        1,
        len(history_df) + 1
    )
)

history_path = os.path.join(
    OUTPUT_DIR,
    "final_tuned_ann_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ================================================================
# 23. SAVE MODEL
# ================================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "final_tuned_ann.keras"
)

model.save(
    model_path
)


# ================================================================
# 24. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nMetrics:",
    metrics_path
)

print(
    "\nPredictions:",
    predictions_path
)

print(
    "\nTraining History:",
    history_path
)

print(
    "\nTrained Model:",
    model_path
)

print("\n" + "=" * 70)
print("FINAL ANN EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)