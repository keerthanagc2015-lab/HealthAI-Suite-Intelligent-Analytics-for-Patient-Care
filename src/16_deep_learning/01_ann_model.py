import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


print("=" * 70)
print("HEALTHAI - DEEP LEARNING - ANN / MLP")
print("=" * 70)


# ================================================================
# 1. PATHS
# ================================================================

DATA_PATH = "data/raw/LengthOfStay.csv"

OUTPUT_DIR = "data/processed/deep_learning"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 2. LOAD DATA
# ================================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ================================================================
# 3. DATA CLEANING
# ================================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# rcount contains values such as:
# 0, 1, 2, 3, 4, 5+
#
# Remove "+" so that 5+ becomes 5.

df["rcount"] = (
    df["rcount"]
    .astype(str)
    .str.replace("+", "", regex=False)
)

# Convert rcount to numeric
df["rcount"] = pd.to_numeric(
    df["rcount"],
    errors="coerce"
)

# Fill any values that could not be converted
# using the median of rcount.

df["rcount"] = df["rcount"].fillna(
    df["rcount"].median()
)

print("rcount converted to numeric.")

print(
    "Missing values after cleaning:",
    df.isnull().sum().sum()
)


# ================================================================
# 4. FEATURE SELECTION
# ================================================================

FEATURES = [
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

TARGET = "lengthofstay"


X = df[FEATURES].copy()

y = df[TARGET].copy()


print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print("Number of Input Features:", len(FEATURES))
print("Input Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget:")
print(TARGET)


# ================================================================
# 5. TRAIN / TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ================================================================
# 6. IDENTIFY FEATURE TYPES
# ================================================================

categorical_features = [
    "gender",
    "facid"
]

numerical_features = [
    column
    for column in FEATURES
    if column not in categorical_features
]


print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ================================================================
# 7. CREATE PREPROCESSOR
# ================================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

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


# ================================================================
# 8. FIT PREPROCESSOR ON TRAINING DATA
# ================================================================

print("\nFitting preprocessing on training data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

print("Training preprocessing completed.")


# ================================================================
# 9. TRANSFORM TEST DATA
# ================================================================

print("Transforming test data...")

X_test_processed = preprocessor.transform(
    X_test
)

print("Test preprocessing completed.")


# ================================================================
# 10. CONVERT TO NUMPY
# ================================================================

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

y_train = np.asarray(
    y_train,
    dtype=np.float32
)

y_test = np.asarray(
    y_test,
    dtype=np.float32
)


print("\n" + "=" * 70)
print("PROCESSED DATA")
print("=" * 70)

print(
    "Processed Training Shape:",
    X_train_processed.shape
)

print(
    "Processed Testing Shape:",
    X_test_processed.shape
)

print(
    "Number of ANN Input Features:",
    X_train_processed.shape[1]
)


# ================================================================
# 11. BUILD ANN / MLP
# ================================================================

print("\n" + "=" * 70)
print("BUILDING ANN / MLP")
print("=" * 70)


model = Sequential([
    
    Input(
        shape=(X_train_processed.shape[1],)
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1
    )
])


# ================================================================
# 12. COMPILE ANN
# ================================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


print("\nModel Architecture:")

model.summary()


# ================================================================
# 13. TRAIN ANN
# ================================================================

print("\n" + "=" * 70)
print("TRAINING ANN")
print("=" * 70)

history = model.fit(
    X_train_processed,
    y_train,

    validation_split=0.20,

    epochs=30,

    batch_size=256,

    verbose=1
)


# ================================================================
# 14. PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_processed,
    verbose=0
).flatten()


# ================================================================
# 15. MODEL EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("ANN MODEL EVALUATION")
print("=" * 70)


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


print("\nANN Performance:")

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
# 16. SAVE MODEL
# ================================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "ann_length_of_stay.keras"
)

model.save(
    model_path
)


# ================================================================
# 17. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(
    history.history
)

history_path = os.path.join(
    OUTPUT_DIR,
    "ann_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ================================================================
# 18. SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({
    "Actual_Length_of_Stay": y_test,
    "Predicted_Length_of_Stay": y_pred
})


prediction_path = os.path.join(
    OUTPUT_DIR,
    "ann_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ================================================================
# 19. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame({
    "Model": ["ANN_MLP"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})


metrics_path = os.path.join(
    OUTPUT_DIR,
    "ann_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ================================================================
# 20. SAVE PROCESSED FEATURE NAMES
# ================================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


feature_names_df = pd.DataFrame({
    "Feature": feature_names
})


feature_names_path = os.path.join(
    OUTPUT_DIR,
    "ann_processed_feature_names.csv"
)


feature_names_df.to_csv(
    feature_names_path,
    index=False
)


# ================================================================
# 21. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nANN Model:"
)

print(model_path)

print(
    "\nTraining History:"
)

print(history_path)

print(
    "\nPredictions:"
)

print(prediction_path)

print(
    "\nMetrics:"
)

print(metrics_path)

print(
    "\nProcessed Feature Names:"
)

print(feature_names_path)


print("\n" + "=" * 70)
print("ANN / MLP TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)