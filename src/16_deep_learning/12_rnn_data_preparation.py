import os
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("=" * 70)
print("HEALTHAI - CORRECTED RNN/LSTM SEQUENTIAL DATA PREPARATION")
print("=" * 70)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading Hospital Deterioration dataset...")

dataset = load_dataset(
    "tarekmasryo/hospital-deterioration-dataset"
)

df = dataset["train"].to_pandas()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 2. SORT PATIENTS CHRONOLOGICALLY
# ============================================================

print("\nSorting patients chronologically...")

df["hour_from_admission"] = pd.to_numeric(
    df["hour_from_admission"],
    errors="coerce"
)

df = df.sort_values(
    ["patient_id", "hour_from_admission"]
).reset_index(drop=True)

print("Sorting completed.")

# ============================================================
# 3. DEFINE FEATURES
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

categorical_features = [
    "oxygen_device",
    "gender",
    "admission_type"
]

target_column = "deterioration_next_12h"

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# ============================================================
# 4. RAW DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("RAW DATA QUALITY")
print("=" * 70)

print("\nMissing values in selected features:")

print(
    df[
        numeric_features + categorical_features
    ].isna().sum()
)

print("\nTarget missing values:")

print(
    df[target_column].isna().sum()
)

# ============================================================
# 5. REMOVE ROWS WITH MISSING TARGET
# ============================================================

print("\n" + "=" * 70)
print("REMOVING MISSING TARGET ROWS")
print("=" * 70)

rows_before = len(df)

df = df[
    df[target_column].notna()
].copy()

rows_removed = rows_before - len(df)

print(
    "\nRows removed:",
    rows_removed
)

print(
    "Rows remaining:",
    len(df)
)

print(
    "Target missing after removal:",
    df[target_column].isna().sum()
)

# ============================================================
# 6. PATIENT-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("PATIENT-LEVEL TRAIN / TEST SPLIT")
print("=" * 70)

patient_ids = df[
    "patient_id"
].unique()

train_patients, test_patients = train_test_split(
    patient_ids,
    test_size=0.20,
    random_state=42
)

train_df = df[
    df["patient_id"].isin(train_patients)
].copy()

test_df = df[
    df["patient_id"].isin(test_patients)
].copy()

print(
    "\nTotal patients:",
    len(patient_ids)
)

print(
    "Training patients:",
    len(train_patients)
)

print(
    "Testing patients:",
    len(test_patients)
)

# ============================================================
# 7. CLEAN NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CLEANING NUMERIC FEATURES")
print("=" * 70)

for column in numeric_features:

    train_df[column] = pd.to_numeric(
        train_df[column],
        errors="coerce"
    )

    test_df[column] = pd.to_numeric(
        test_df[column],
        errors="coerce"
    )

    # IMPORTANT:
    # Median is calculated ONLY from training data

    median_value = train_df[
        column
    ].median()

    if pd.isna(median_value):

        median_value = 0.0

    train_df[column] = train_df[
        column
    ].fillna(median_value)

    test_df[column] = test_df[
        column
    ].fillna(median_value)

print(
    "Numeric missing values handled."
)

# ============================================================
# 8. CLEAN CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CLEANING CATEGORICAL FEATURES")
print("=" * 70)

for column in categorical_features:

    train_df[column] = (
        train_df[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

    test_df[column] = (
        test_df[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

print(
    "Categorical missing values handled."
)

# ============================================================
# 9. NORMALIZE NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING NUMERIC FEATURES")
print("=" * 70)

scaler = StandardScaler()

train_numeric = scaler.fit_transform(
    train_df[numeric_features]
)

test_numeric = scaler.transform(
    test_df[numeric_features]
)

print(
    "Numeric normalization completed."
)

# ============================================================
# 10. ONE-HOT ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 70)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

train_categorical = encoder.fit_transform(
    train_df[categorical_features]
)

test_categorical = encoder.transform(
    test_df[categorical_features]
)

print(
    "Categorical encoding completed."
)

print(
    "Encoded categorical features:",
    train_categorical.shape[1]
)

# ============================================================
# 11. COMBINE FEATURES
# ============================================================

train_features = np.concatenate(
    [
        train_numeric,
        train_categorical
    ],
    axis=1
)

test_features = np.concatenate(
    [
        test_numeric,
        test_categorical
    ],
    axis=1
)

print("\nFinal feature count:")

print(
    train_features.shape[1]
)

# ============================================================
# 12. PREPROCESSING QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING QUALITY CHECK")
print("=" * 70)

train_nan = np.isnan(
    train_features
).sum()

test_nan = np.isnan(
    test_features
).sum()

train_inf = np.isinf(
    train_features
).sum()

test_inf = np.isinf(
    test_features
).sum()

print(
    "\nTraining NaN:",
    train_nan
)

print(
    "Testing NaN:",
    test_nan
)

print(
    "Training Inf:",
    train_inf
)

print(
    "Testing Inf:",
    test_inf
)

if (
    train_nan > 0
    or test_nan > 0
    or train_inf > 0
    or test_inf > 0
):

    raise ValueError(
        "Preprocessing still contains NaN or Inf values."
    )

print(
    "\n✓ PREPROCESSING QUALITY CHECK PASSED"
)

# ============================================================
# 13. CREATE TARGET ARRAYS
# ============================================================

train_targets = train_df[
    target_column
].astype(int).values

test_targets = test_df[
    target_column
].astype(int).values

print("\nTarget arrays created successfully.")

# ============================================================
# 14. SEQUENCE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 24

PREDICTION_HORIZON = 12

print("\n" + "=" * 70)
print("SEQUENCE SETTINGS")
print("=" * 70)

print(
    "\nInput history:",
    SEQUENCE_LENGTH,
    "hours"
)

print(
    "Prediction horizon:",
    PREDICTION_HORIZON,
    "hours"
)

# ============================================================
# 15. CREATE SEQUENCES FUNCTION
# ============================================================

def create_sequences(
    dataframe,
    features,
    targets
):

    X_sequences = []

    y_sequences = []

    patient_values = dataframe[
        "patient_id"
    ].values

    hours = dataframe[
        "hour_from_admission"
    ].values

    unique_patients = (
        dataframe[
            "patient_id"
        ].unique()
    )

    for patient in unique_patients:

        positions = np.where(
            patient_values == patient
        )[0]

        # Sort patient's rows by hour

        positions = positions[
            np.argsort(
                hours[positions]
            )
        ]

        patient_hours = hours[
            positions
        ]

        patient_features = features[
            positions
        ]

        patient_targets = targets[
            positions
        ]

        n = len(positions)

        # Need at least:
        #
        # 24 input hours
        # +
        # 12 future hours
        #
        # = 36 continuous hours

        if n < (
            SEQUENCE_LENGTH
            + PREDICTION_HORIZON
        ):

            continue

        # ----------------------------------------------------
        # Sliding window
        # ----------------------------------------------------

        max_start = (
            n
            - SEQUENCE_LENGTH
            - PREDICTION_HORIZON
            + 1
        )

        for start in range(
            max_start
        ):

            input_end = (
                start
                + SEQUENCE_LENGTH
            )

            future_end = (
                input_end
                + PREDICTION_HORIZON
            )

            # ------------------------------------------------
            # INPUT HOURS
            # ------------------------------------------------

            input_hours = patient_hours[
                start:input_end
            ]

            # ------------------------------------------------
            # FUTURE HOURS
            # ------------------------------------------------

            future_hours = patient_hours[
                input_end:future_end
            ]

            # ------------------------------------------------
            # EXPECTED HOURS
            # ------------------------------------------------

            expected_input = np.arange(
                input_hours[0],
                input_hours[0]
                + SEQUENCE_LENGTH
            )

            expected_future = np.arange(
                input_hours[-1] + 1,
                input_hours[-1]
                + 1
                + PREDICTION_HORIZON
            )

            # ------------------------------------------------
            # CHECK CONTINUITY
            # ------------------------------------------------

            if not np.array_equal(
                input_hours,
                expected_input
            ):

                continue

            if not np.array_equal(
                future_hours,
                expected_future
            ):

                continue

            # ------------------------------------------------
            # CREATE INPUT
            # ------------------------------------------------

            sequence = patient_features[
                start:input_end
            ]

            # ------------------------------------------------
            # CREATE TARGET
            # ------------------------------------------------

            future_target = patient_targets[
                input_end:future_end
            ]

            target = int(
                np.any(
                    future_target == 1
                )
            )

            X_sequences.append(
                sequence
            )

            y_sequences.append(
                target
            )

    return (
        np.asarray(
            X_sequences,
            dtype=np.float32
        ),
        np.asarray(
            y_sequences,
            dtype=np.int32
        )
    )

# ============================================================
# 16. CREATE TRAINING SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAINING SEQUENCES")
print("=" * 70)

X_train, y_train = create_sequences(
    train_df,
    train_features,
    train_targets
)

print(
    "\nTraining sequences created:",
    len(X_train)
)

print(
    "Training X shape:",
    X_train.shape
)

print(
    "Training y shape:",
    y_train.shape
)

# ============================================================
# 17. CREATE TEST SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TESTING SEQUENCES")
print("=" * 70)

X_test, y_test = create_sequences(
    test_df,
    test_features,
    test_targets
)

print(
    "\nTesting sequences created:",
    len(X_test)
)

print(
    "Testing X shape:",
    X_test.shape
)

print(
    "Testing y shape:",
    y_test.shape
)

# ============================================================
# 18. FINAL NaN / INF CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL SEQUENCE QUALITY CHECK")
print("=" * 70)

train_nan = np.isnan(
    X_train
).sum()

test_nan = np.isnan(
    X_test
).sum()

train_inf = np.isinf(
    X_train
).sum()

test_inf = np.isinf(
    X_test
).sum()

print(
    "\nTraining NaN:",
    train_nan
)

print(
    "Testing NaN:",
    test_nan
)

print(
    "Training Inf:",
    train_inf
)

print(
    "Testing Inf:",
    test_inf
)

if (
    train_nan > 0
    or test_nan > 0
    or train_inf > 0
    or test_inf > 0
):

    raise ValueError(
        "FINAL SEQUENCES CONTAIN NaN/Inf. "
        "DO NOT TRAIN RNN/LSTM."
    )

print(
    "\n✓ FINAL SEQUENCE QUALITY CHECK PASSED"
)

# ============================================================
# 19. CHECK TARGET QUALITY
# ============================================================

print("\n" + "=" * 70)
print("TARGET QUALITY")
print("=" * 70)

print("\nTraining target distribution:")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nTesting target distribution:")

print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

print(
    "\nTraining target unique values:",
    np.unique(y_train)
)

print(
    "Testing target unique values:",
    np.unique(y_test)
)

# ============================================================
# 20. CHECK FINAL SHAPE
# ============================================================

print("\n" + "=" * 70)
print("FINAL SHAPE CHECK")
print("=" * 70)

print(
    "\nTraining X:",
    X_train.shape
)

print(
    "Training y:",
    y_train.shape
)

print(
    "Testing X:",
    X_test.shape
)

print(
    "Testing y:",
    y_test.shape
)

print(
    "\nNumber of time steps:",
    X_train.shape[1]
)

print(
    "Number of input features:",
    X_train.shape[2]
)

# ============================================================
# 21. SAVE ARRAYS
# ============================================================

output_dir = (
    "data/processed/deep_learning/"
    "rnn_lstm"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

print("\n" + "=" * 70)
print("SAVING PROCESSED SEQUENTIAL DATA")
print("=" * 70)

np.save(
    os.path.join(
        output_dir,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        output_dir,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        output_dir,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        output_dir,
        "y_test.npy"
    ),
    y_test
)

print(
    "\nSaved:"
)

print(
    os.path.join(
        output_dir,
        "X_train.npy"
    )
)

print(
    os.path.join(
        output_dir,
        "X_test.npy"
    )
)

print(
    os.path.join(
        output_dir,
        "y_train.npy"
    )
)

print(
    os.path.join(
        output_dir,
        "y_test.npy"
    )
)

# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SEQUENTIAL DATA SUMMARY")
print("=" * 70)

print(
    "\nPatients:",
    len(patient_ids)
)

print(
    "Training patients:",
    len(train_patients)
)

print(
    "Testing patients:",
    len(test_patients)
)

print(
    "\nInput history:",
    SEQUENCE_LENGTH,
    "hours"
)

print(
    "Prediction horizon:",
    PREDICTION_HORIZON,
    "hours"
)

print(
    "\nTraining X:",
    X_train.shape
)

print(
    "Training y:",
    y_train.shape
)

print(
    "\nTesting X:",
    X_test.shape
)

print(
    "Testing y:",
    y_test.shape
)

print(
    "\nNumber of features:",
    X_train.shape[2]
)

print(
    "\nData saved to:",
    output_dir
)

print("\n" + "=" * 70)
print("CORRECTED RNN/LSTM DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)