import os
import numpy as np
import pandas as pd
import tensorflow as tf

print("=" * 70)
print("HEALTHAI - PATIENT DETERIORATION PREDICTION")
print("=" * 70)

# ============================================================
# 1. LOAD TRAINED RNN
# ============================================================

MODEL_PATH = (
    "data/processed/deep_learning/rnn/"
    "rnn_model.keras"
)

DATA_DIR = (
    "data/processed/deep_learning/rnn_lstm"
)

print("\nLoading trained RNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("RNN model loaded successfully!")

# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test sequences...")

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test.npy"
    )
)

print(
    "Test data shape:",
    X_test.shape
)

print(
    "Test target shape:",
    y_test.shape
)

# ============================================================
# 3. SELECT A SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SELECT PATIENT SEQUENCE")
print("=" * 70)

print(
    "\nThe test dataset contains",
    len(X_test),
    "patient sequences."
)

print(
    "\nEnter a sequence number "
    "(0 to",
    len(X_test) - 1,
    ")"
)

user_input = input(
    "\nSequence number: "
).strip()

try:

    sample_index = int(
        user_input
    )

except ValueError:

    print(
        "\nInvalid input."
    )

    print(
        "Using sequence 0."
    )

    sample_index = 0

# Keep index within range

if (
    sample_index < 0
    or sample_index >= len(X_test)
):

    print(
        "\nInvalid sequence number."
    )

    print(
        "Using sequence 0."
    )

    sample_index = 0

# ============================================================
# 4. GET SAMPLE
# ============================================================

sample = X_test[
    sample_index
]

actual_label = y_test[
    sample_index
]

print(
    "\nSelected sequence:",
    sample_index
)

print(
    "Input shape:",
    sample.shape
)

print(
    "Actual outcome:",
    "DETERIORATION"
    if actual_label == 1
    else
    "NO DETERIORATION"
)

# ============================================================
# 5. PREPARE INPUT
# ============================================================

sample_input = np.expand_dims(
    sample,
    axis=0
)

print(
    "\nModel input shape:",
    sample_input.shape
)

# ============================================================
# 6. PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTION")
print("=" * 70)

probability = model.predict(
    sample_input,
    verbose=0
)[0][0]

# ============================================================
# 7. CLASSIFICATION
# ============================================================

threshold = 0.50

if probability >= threshold:

    prediction = 1

    result = "DETERIORATION"

else:

    prediction = 0

    result = "NO DETERIORATION"

# ============================================================
# 8. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("HEALTHAI PREDICTION")
print("=" * 70)

print(
    "\nDeterioration Probability:",
    f"{probability * 100:.2f}%"
)

print(
    "No Deterioration Probability:",
    f"{(1 - probability) * 100:.2f}%"
)

print(
    "\nPrediction:",
    result
)

print(
    "\nClassification threshold:",
    f"{threshold:.2f}"
)

print(
    "\nActual outcome:",
    "DETERIORATION"
    if actual_label == 1
    else
    "NO DETERIORATION"
)

# ============================================================
# 9. CHECK WHETHER PREDICTION IS CORRECT
# ============================================================

if prediction == actual_label:

    print(
        "\nPrediction status: CORRECT ✓"
    )

else:

    print(
        "\nPrediction status: INCORRECT"
    )

# ============================================================
# 10. RISK LEVEL
# ============================================================

print("\n" + "=" * 70)
print("RISK INTERPRETATION")
print("=" * 70)

if probability >= 0.70:

    risk_level = "HIGH RISK"

elif probability >= 0.40:

    risk_level = "MODERATE RISK"

else:

    risk_level = "LOW RISK"

print(
    "\nRisk Level:",
    risk_level
)

print(
    "\nNote:"
)

print(
    "This prediction is a machine-learning "
    "risk estimate and is not a medical diagnosis."
)

# ============================================================
# 11. SAVE DEMO RESULT
# ============================================================

output_dir = (
    "data/processed/deep_learning/rnn/"
    "prediction_demo"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

result_df = pd.DataFrame({

    "Sequence_Index": [
        sample_index
    ],

    "Actual": [
        actual_label
    ],

    "Probability": [
        probability
    ],

    "Predicted": [
        prediction
    ],

    "Risk_Level": [
        risk_level
    ],

    "Threshold": [
        threshold
    ]
})

output_path = os.path.join(
    output_dir,
    "rnn_prediction_demo.csv"
)

result_df.to_csv(
    output_path,
    index=False
)

print(
    "\nPrediction saved to:",
    output_path
)

print("\n" + "=" * 70)
print("PREDICTION DEMO COMPLETED")
print("=" * 70)