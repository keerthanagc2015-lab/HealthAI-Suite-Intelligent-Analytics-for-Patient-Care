import os
import numpy as np
import pandas as pd

print("=" * 70)
print("HEALTHAI - RNN EXPLAINABILITY")
print("=" * 70)

# ============================================================
# 1. LOAD PROCESSED SEQUENTIAL DATA
# ============================================================

DATA_DIR = (
    "data/processed/deep_learning/rnn_lstm"
)

print("\nLoading processed RNN data...")

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

print("\nData loaded successfully.")

print(
    "X_train shape:",
    X_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "y_test shape:",
    y_test.shape
)


# ============================================================
# 2. VERIFY FEATURE COUNT
# ============================================================

number_of_features = X_train.shape[2]

number_of_time_steps = X_train.shape[1]

print("\n" + "=" * 70)
print("SEQUENCE INFORMATION")
print("=" * 70)

print(
    "\nTime steps:",
    number_of_time_steps
)

print(
    "Number of features:",
    number_of_features
)


# ============================================================
# 3. DEFINE ORIGINAL NUMERIC FEATURES
# ============================================================

numeric_features = [

    "heart_rate",

    "respiratory_rate",

    "spo2_pct",

    "temperature_c",

    "systolic_bp",

    "diastolic_bp",

    "oxygen_flow",

    "mobility_score",

    "nurse_alert",

    "wbc_count",

    "lactate",

    "creatinine",

    "crp_level",

    "hemoglobin",

    "sepsis_risk_score",

    "age",

    "comorbidity_index",

    "baseline_risk_score"
]


# ============================================================
# 4. ORIGINAL CATEGORICAL FEATURES
# ============================================================

categorical_features = [

    "oxygen_device",

    "gender",

    "admission_type"
]


# ============================================================
# 5. ONE-HOT ENCODED FEATURE NAMES
# ============================================================

# These are the categories expected from the dataset.
# We create names matching the preprocessing concept.

categorical_encoded_names = [

    "oxygen_device_encoded_0",
    "oxygen_device_encoded_1",

    "gender_encoded_0",
    "gender_encoded_1",

    "admission_type_encoded_0",
    "admission_type_encoded_1",
    "admission_type_encoded_2",
    "admission_type_encoded_3",
    "admission_type_encoded_4",
    "admission_type_encoded_5"

]


# ============================================================
# 6. COMBINE FEATURE NAMES
# ============================================================

feature_names = (
    numeric_features
    + categorical_encoded_names
)


# ============================================================
# 7. VERIFY FEATURE COUNT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE NAME CHECK")
print("=" * 70)

print(
    "\nNumeric features:",
    len(numeric_features)
)

print(
    "Encoded categorical features:",
    len(categorical_encoded_names)
)

print(
    "Total expected features:",
    len(feature_names)
)

print(
    "Actual model features:",
    number_of_features
)


# ============================================================
# 8. SAFETY CHECK
# ============================================================

if len(feature_names) != number_of_features:

    print("\n⚠ FEATURE COUNT DOES NOT MATCH")

    print(
        "\nWe will NOT calculate feature importance yet."
    )

    print(
        "The exact preprocessing feature names "
        "must first be recovered."
    )

    raise ValueError(
        "Feature name count does not match "
        "the RNN input feature count."
    )


print(
    "\n✓ Feature count matches."
)


# ============================================================
# 9. DISPLAY FEATURE MAP
# ============================================================

print("\n" + "=" * 70)
print("RNN FEATURE MAP")
print("=" * 70)

feature_map = pd.DataFrame({

    "Feature_Index": range(
        number_of_features
    ),

    "Feature_Name": feature_names

})

print(
    feature_map.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE FEATURE MAP
# ============================================================

output_dir = (
    "data/processed/deep_learning/rnn/"
    "explainability"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

feature_map_path = os.path.join(
    output_dir,
    "rnn_feature_map.csv"
)

feature_map.to_csv(
    feature_map_path,
    index=False
)

print(
    "\nFeature map saved:"
)

print(
    feature_map_path
)


# ============================================================
# 11. BASIC FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

# Flatten time dimension while preserving feature identity.

feature_data = X_train.reshape(
    -1,
    number_of_features
)

statistics = []

for index, name in enumerate(
    feature_names
):

    values = feature_data[:, index]

    valid_values = values[
        np.isfinite(values)
    ]

    statistics.append({

        "Feature_Index":
            index,

        "Feature_Name":
            name,

        "Mean":
            float(
                np.mean(valid_values)
            ),

        "Std":
            float(
                np.std(valid_values)
            ),

        "Min":
            float(
                np.min(valid_values)
            ),

        "Max":
            float(
                np.max(valid_values)
            )

    })


statistics_df = pd.DataFrame(
    statistics
)

statistics_path = os.path.join(
    output_dir,
    "rnn_feature_statistics.csv"
)

statistics_df.to_csv(
    statistics_path,
    index=False
)

print(
    "\nFeature statistics saved:"
)

print(
    statistics_path
)


# ============================================================
# 12. FINAL
# ============================================================

print("\n" + "=" * 70)
print("RNN FEATURE MAPPING COMPLETED")
print("=" * 70)

print(
    "\nThe next explainability step can now "
    "use the verified feature names."
)

print(
    "\nNo model was retrained."
)

print(
    "No predictions were changed."
)

print("=" * 70)