"""
HealthAI - Patient Feedback Train / Validation / Test Split

Step 31C

Important:
    The dataset contains many repeated feedback texts.
    Therefore, splitting is performed at the unique-feedback level
    to prevent text leakage between train, validation, and test sets.

Split:
    60% Train
    20% Validation
    20% Test
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "medical_sentiment"
    / "patient_feedback_clean_unique.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "medical_sentiment"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("HEALTHAI - PATIENT FEEDBACK DATA SPLIT")
print("=" * 75)

df = pd.read_csv(
    INPUT_PATH
)

print()
print("✓ Dataset loaded.")

print(
    f"Unique feedback records: "
    f"{len(df)}"
)


# ============================================================
# 3. VALIDATE INPUT
# ============================================================

required_columns = [
    "Theme",
    "Feedback",
    "Sentiment",
    "sentiment_label",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


if df["Feedback"].duplicated().any():

    raise ValueError(
        "Input dataset still contains "
        "duplicate feedback texts."
    )


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

class_distribution = (
    df["sentiment_label"]
    .value_counts()
)

print(
    class_distribution.to_string()
)


# ============================================================
# 5. STRATIFIED TRAIN / TEMP SPLIT
# ============================================================

# 60% train
# 40% temporary set

train_df, temp_df = train_test_split(
    df,
    test_size=0.40,
    random_state=42,
    stratify=df["Sentiment"],
)


# ============================================================
# 6. STRATIFIED VALIDATION / TEST SPLIT
# ============================================================

# Split temporary 40% into:
# 20% validation
# 20% test

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Sentiment"],
)


# ============================================================
# 7. RESET INDEX
# ============================================================

train_df = train_df.reset_index(
    drop=True
)

validation_df = validation_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# 8. PRINT SPLIT SIZES
# ============================================================

print()
print("=" * 75)
print("SPLIT SIZES")
print("=" * 75)

print(
    f"Train:      {len(train_df)}"
)

print(
    f"Validation: {len(validation_df)}"
)

print(
    f"Test:       {len(test_df)}"
)

print(
    f"Total:      "
    f"{len(train_df) + len(validation_df) + len(test_df)}"
)


# ============================================================
# 9. CLASS DISTRIBUTION PER SPLIT
# ============================================================

print()
print("=" * 75)
print("CLASS DISTRIBUTION PER SPLIT")
print("=" * 75)


def print_distribution(
    name,
    split_df
):

    print()
    print(name)

    counts = (
        split_df["sentiment_label"]
        .value_counts()
    )

    percentages = (
        split_df["sentiment_label"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    distribution = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    print(
        distribution.to_string()
    )


print_distribution(
    "TRAIN",
    train_df
)

print_distribution(
    "VALIDATION",
    validation_df
)

print_distribution(
    "TEST",
    test_df
)


# ============================================================
# 10. VERIFY NO TEXT LEAKAGE
# ============================================================

print()
print("=" * 75)
print("TEXT LEAKAGE CHECK")
print("=" * 75)

train_texts = set(
    train_df["Feedback"]
)

validation_texts = set(
    validation_df["Feedback"]
)

test_texts = set(
    test_df["Feedback"]
)

train_validation_overlap = (
    train_texts
    & validation_texts
)

train_test_overlap = (
    train_texts
    & test_texts
)

validation_test_overlap = (
    validation_texts
    & test_texts
)

print(
    f"Train ∩ Validation: "
    f"{len(train_validation_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test_overlap)}"
)


if (
    len(train_validation_overlap) == 0
    and len(train_test_overlap) == 0
    and len(validation_test_overlap) == 0
):

    print()
    print(
        "✓ No feedback-text leakage detected."
    )

else:

    raise ValueError(
        "DATA LEAKAGE DETECTED."
    )


# ============================================================
# 11. VERIFY ALL RECORDS USED ONCE
# ============================================================

train_ids = set(
    train_df.index
)

validation_ids = set(
    validation_df.index
)

test_ids = set(
    test_df.index
)

# We cannot use reset indexes for this check,
# so use feedback texts instead.

all_split_texts = (
    train_texts
    | validation_texts
    | test_texts
)

original_texts = set(
    df["Feedback"]
)

if all_split_texts == original_texts:

    print(
        "✓ Every unique feedback text "
        "appears in exactly one split."
    )

else:

    raise ValueError(
        "Some feedback texts were lost "
        "during splitting."
    )


# ============================================================
# 12. SAVE SPLITS
# ============================================================

train_path = (
    OUTPUT_DIR
    / "train.csv"
)

validation_path = (
    OUTPUT_DIR
    / "validation.csv"
)

test_path = (
    OUTPUT_DIR
    / "test.csv"
)


train_df.to_csv(
    train_path,
    index=False,
    encoding="utf-8"
)

validation_df.to_csv(
    validation_path,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    test_path,
    index=False,
    encoding="utf-8"
)


print()
print(
    f"✓ Train saved: {train_path}"
)

print(
    f"✓ Validation saved: {validation_path}"
)

print(
    f"✓ Test saved: {test_path}"
)


# ============================================================
# 13. SAVE SPLIT SUMMARY
# ============================================================

split_summary = {
    "total_unique_feedback": int(len(df)),
    "train_records": int(len(train_df)),
    "validation_records": int(
        len(validation_df)
    ),
    "test_records": int(len(test_df)),
    "train_percentage": float(
        len(train_df) / len(df) * 100
    ),
    "validation_percentage": float(
        len(validation_df) / len(df) * 100
    ),
    "test_percentage": float(
        len(test_df) / len(df) * 100
    ),
    "random_state": 42,
    "stratified": True,
    "split_level": "unique feedback text",
    "train_validation_overlap": int(
        len(train_validation_overlap)
    ),
    "train_test_overlap": int(
        len(train_test_overlap)
    ),
    "validation_test_overlap": int(
        len(validation_test_overlap)
    ),
}


summary_path = (
    OUTPUT_DIR
    / "split_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        split_summary,
        file,
        indent=4
    )


print()
print(
    f"✓ Split summary saved: "
    f"{summary_path}"
)


# ============================================================
# 14. FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 31C - COMPLETED")
print("=" * 75)

print()
print(
    "Leakage-safe split created:"
)

print(
    "60% Train / 20% Validation / 20% Test"
)

print()
print(
    "Split unit:"
)

print(
    "Unique feedback text"
)

print()
print(
    "Next:"
)

print(
    "31D - TF-IDF + Logistic Regression baseline"
)

print()
print("=" * 75)