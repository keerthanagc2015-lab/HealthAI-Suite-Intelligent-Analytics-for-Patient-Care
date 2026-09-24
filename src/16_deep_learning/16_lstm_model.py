import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("HEALTHAI - CORRECTED LSTM MODEL")
print("=" * 70)

# ============================================================
# 1. LOAD CLEAN SEQUENTIAL DATA
# ============================================================

DATA_DIR = "data/processed/deep_learning/rnn_lstm"

print("\nLoading clean sequential data...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

print("\nData loaded successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print(
    "\nTraining NaN:",
    np.isnan(X_train).sum()
)

print(
    "Testing NaN :",
    np.isnan(X_test).sum()
)

print(
    "Training Inf:",
    np.isinf(X_train).sum()
)

print(
    "Testing Inf :",
    np.isinf(X_test).sum()
)

if (
    np.isnan(X_train).any()
    or np.isnan(X_test).any()
    or np.isinf(X_train).any()
    or np.isinf(X_test).any()
):

    raise ValueError(
        "Input data contains NaN/Inf. "
        "Stop training."
    )

print("\n✓ DATA QUALITY PASSED")

# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nTesting:")

print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

# ============================================================
# 4. CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")

for class_id, weight in class_weights.items():

    if class_id == 0:
        name = "NO_DETERIORATION"
    else:
        name = "DETERIORATION"

    print(
        f"{class_id} ({name}): {weight:.4f}"
    )

# ============================================================
# 5. BUILD LSTM
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LSTM")
print("=" * 70)

timesteps = X_train.shape[1]

features = X_train.shape[2]

print(
    "\nInput shape:",
    (timesteps, features)
)

model = Sequential([

    LSTM(
        64,
        activation="tanh",
        input_shape=(
            timesteps,
            features
        ),
        return_sequences=False
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

model.summary()

# ============================================================
# 6. TRAIN LSTM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LSTM")
print("=" * 70)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=20,

    batch_size=256,

    class_weight=class_weights,

    callbacks=[
        early_stopping
    ],

    verbose=1
)

# ============================================================
# 7. GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.50
).astype(int)

# ============================================================
# 8. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()

# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED LSTM TEST RESULTS")
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

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print(
    "\nTrue Negative :",
    tn
)

print(
    "False Positive:",
    fp
)

print(
    "False Negative:",
    fn
)

print(
    "True Positive :",
    tp
)

# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(

    y_test,

    predictions,

    target_names=[
        "NO_DETERIORATION",
        "DETERIORATION"
    ],

    zero_division=0
)

print(report)

# ============================================================
# 12. PROBABILITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION PROBABILITY ANALYSIS")
print("=" * 70)

print(
    "\nMinimum probability:",
    probabilities.min()
)

print(
    "Maximum probability:",
    probabilities.max()
)

print(
    "Mean probability:",
    probabilities.mean()
)

print(
    "Positive predictions:",
    predictions.sum()
)

# ============================================================
# 13. SAVE MODEL
# ============================================================

output_dir = (
    "data/processed/deep_learning/lstm"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

model_path = os.path.join(
    output_dir,
    "lstm_model.keras"
)

model.save(
    model_path
)

# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_path = os.path.join(
    output_dir,
    "lstm_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

# ============================================================
# 15. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual_0",
        "Actual_1"
    ],

    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)

cm_path = os.path.join(
    output_dir,
    "lstm_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)

# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

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
    output_dir,
    "lstm_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "Actual": y_test,

    "Probability": probabilities,

    "Predicted": predictions

})

predictions_path = os.path.join(
    output_dir,
    "lstm_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nLSTM Model:",
    model_path
)

print(
    "Metrics:",
    metrics_path
)

print(
    "Confusion Matrix:",
    cm_path
)

print(
    "Training History:",
    history_path
)

print(
    "Predictions:",
    predictions_path
)

print("\n" + "=" * 70)
print("CORRECTED LSTM TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)