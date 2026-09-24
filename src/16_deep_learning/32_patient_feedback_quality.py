"""
HealthAI - Patient Feedback Quality & Label Validation

Step 31B

Purpose:
    Validate and prepare the patient feedback dataset for
    leakage-safe sentiment classification.

Important:
    The dataset contains 1,000 rows but only 20 unique
    feedback texts. Therefore, duplicate feedback texts
    must not be split across train/validation/test sets.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "medical_sentiment"
    / "patient_feedback_dataset.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "medical_sentiment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("HEALTHAI - PATIENT FEEDBACK QUALITY VALIDATION")
print("=" * 75)

df = pd.read_excel(
    DATA_PATH,
    sheet_name="patient_feedback_dataset"
)

print()
print("✓ Dataset loaded successfully.")

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

required_columns = [
    "Theme",
    "Feedback",
    "Sentiment",
    "Satisfaction",
    "Readmission",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# 4. BASIC CLEANING
# ============================================================

df["Theme"] = (
    df["Theme"]
    .astype(str)
    .str.strip()
)

df["Feedback"] = (
    df["Feedback"]
    .astype(str)
    .str.strip()
)

df["Sentiment"] = (
    pd.to_numeric(
        df["Sentiment"],
        errors="coerce"
    )
)

df["Satisfaction"] = (
    pd.to_numeric(
        df["Satisfaction"],
        errors="coerce"
    )
)

df["Readmission"] = (
    pd.to_numeric(
        df["Readmission"],
        errors="coerce"
    )
)


# ============================================================
# 5. REMOVE INVALID ROWS
# ============================================================

before_cleaning = len(df)

df = df.dropna(
    subset=[
        "Feedback",
        "Sentiment",
        "Satisfaction",
    ]
).copy()

df = df[
    df["Feedback"].str.len() > 0
].copy()

after_cleaning = len(df)

print()
print("=" * 75)
print("BASIC CLEANING")
print("=" * 75)

print(
    f"Rows before cleaning: "
    f"{before_cleaning:,}"
)

print(
    f"Rows after cleaning: "
    f"{after_cleaning:,}"
)

print(
    f"Rows removed: "
    f"{before_cleaning - after_cleaning:,}"
)


# ============================================================
# 6. VALIDATE SENTIMENT VALUES
# ============================================================

print()
print("=" * 75)
print("SENTIMENT LABEL VALIDATION")
print("=" * 75)

sentiment_values = sorted(
    df["Sentiment"]
    .unique()
    .tolist()
)

print(
    "Sentiment values:",
    sentiment_values
)

expected_sentiment = {
    0,
    1,
}

if set(sentiment_values) != expected_sentiment:

    raise ValueError(
        "Unexpected sentiment labels: "
        + str(sentiment_values)
    )

print(
    "✓ Binary sentiment labels confirmed."
)


# ============================================================
# 7. VERIFY SATISFACTION → SENTIMENT RELATIONSHIP
# ============================================================

print()
print("=" * 75)
print("SATISFACTION / SENTIMENT VALIDATION")
print("=" * 75)

relationship = pd.crosstab(
    df["Satisfaction"],
    df["Sentiment"]
)

print(
    relationship.to_string()
)

# Check:
# Satisfaction 1,2,3 → Sentiment 0
# Satisfaction 4,5   → Sentiment 1

expected_mapping = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
}

mapping_is_valid = True

for satisfaction, expected_label in (
    expected_mapping.items()
):

    observed = df.loc[
        df["Satisfaction"] == satisfaction,
        "Sentiment"
    ].unique()

    observed_set = set(
        observed.tolist()
    )

    if observed_set != {expected_label}:

        mapping_is_valid = False

        print(
            f"WARNING: Satisfaction "
            f"{satisfaction} has labels "
            f"{observed_set}"
        )

if mapping_is_valid:

    print()
    print(
        "✓ Sentiment labels are perfectly "
        "consistent with satisfaction groups."
    )


# ============================================================
# 8. FEEDBACK-LEVEL CONSISTENCY
# ============================================================

print()
print("=" * 75)
print("FEEDBACK-LEVEL CONSISTENCY")
print("=" * 75)

feedback_stats = (
    df.groupby("Feedback")
    .agg(
        row_count=("Feedback", "size"),
        sentiment_count=(
            "Sentiment",
            "nunique"
        ),
        satisfaction_count=(
            "Satisfaction",
            "nunique"
        ),
    )
    .reset_index()
)

conflicting_sentiment = (
    feedback_stats[
        feedback_stats["sentiment_count"] > 1
    ]
)

print(
    f"Unique feedback texts: "
    f"{len(feedback_stats)}"
)

print(
    f"Feedback texts with conflicting "
    f"sentiment: "
    f"{len(conflicting_sentiment)}"
)

if len(conflicting_sentiment) == 0:

    print(
        "✓ No conflicting sentiment labels."
    )

else:

    print(
        conflicting_sentiment.to_string(
            index=False
        )
    )


# ============================================================
# 9. CREATE UNIQUE FEEDBACK DATASET
# ============================================================

print()
print("=" * 75)
print("CREATING UNIQUE FEEDBACK DATASET")
print("=" * 75)

# Because multiple rows contain the same feedback text,
# keep one representative row per unique feedback.

unique_df = (
    df[
        [
            "Theme",
            "Feedback",
            "Sentiment",
        ]
    ]
    .drop_duplicates(
        subset=["Feedback"]
    )
    .reset_index(drop=True)
)

print(
    f"Original rows: "
    f"{len(df):,}"
)

print(
    f"Unique feedback records: "
    f"{len(unique_df):,}"
)


# ============================================================
# 10. VALIDATE UNIQUE FEEDBACK LABELS
# ============================================================

label_counts = (
    unique_df["Sentiment"]
    .value_counts()
    .sort_index()
)

print()
print(
    "Unique-feedback sentiment distribution:"
)

print(
    label_counts.to_string()
)


# ============================================================
# 11. ADD HUMAN-READABLE LABEL
# ============================================================

unique_df["sentiment_label"] = (
    unique_df["Sentiment"]
    .map(
        {
            0: "negative",
            1: "positive",
        }
    )
)

print()
print(
    "✓ Added human-readable sentiment labels:"
)

print(
    unique_df[
        [
            "Sentiment",
            "sentiment_label",
        ]
    ]
    .drop_duplicates()
    .sort_values("Sentiment")
    .to_string(index=False)
)


# ============================================================
# 12. SAVE CLEAN UNIQUE DATASET
# ============================================================

clean_path = (
    OUTPUT_DIR
    / "patient_feedback_clean_unique.csv"
)

unique_df.to_csv(
    clean_path,
    index=False,
    encoding="utf-8"
)

print()
print(
    f"✓ Saved clean dataset:"
)

print(
    clean_path
)


# ============================================================
# 13. SAVE FULL CLEAN DATASET
# ============================================================

full_clean_path = (
    OUTPUT_DIR
    / "patient_feedback_clean_all_rows.csv"
)

df.to_csv(
    full_clean_path,
    index=False,
    encoding="utf-8"
)

print()
print(
    f"✓ Saved full cleaned dataset:"
)

print(
    full_clean_path
)


# ============================================================
# 14. DATASET QUALITY METRICS
# ============================================================

duplicate_percentage = (
    df.duplicated().sum()
    / len(df)
    * 100
)

feedback_duplicate_percentage = (
    df["Feedback"].duplicated().sum()
    / len(df)
    * 100
)

quality_report = {
    "original_rows": int(before_cleaning),
    "clean_rows": int(after_cleaning),
    "unique_feedback_texts": int(
        len(unique_df)
    ),
    "duplicate_complete_rows": int(
        df.duplicated().sum()
    ),
    "duplicate_complete_percentage": float(
        duplicate_percentage
    ),
    "duplicate_feedback_rows": int(
        df["Feedback"].duplicated().sum()
    ),
    "duplicate_feedback_percentage": float(
        feedback_duplicate_percentage
    ),
    "sentiment_classes": int(
        unique_df["Sentiment"].nunique()
    ),
    "sentiment_labels": {
        "0": "negative",
        "1": "positive",
    },
    "conflicting_sentiment_feedback": int(
        len(conflicting_sentiment)
    ),
    "satisfaction_sentiment_consistent": bool(
        mapping_is_valid
    ),
    "data_leakage_risk": (
        "HIGH if randomly splitting all rows; "
        "MITIGATED by splitting unique feedback texts"
    ),
}


quality_path = (
    OUTPUT_DIR
    / "patient_feedback_quality_report.json"
)

with open(
    quality_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        quality_report,
        file,
        indent=4
    )


print()
print(
    f"✓ Saved quality report:"
)

print(
    quality_path
)


# ============================================================
# 15. FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 31B - COMPLETED")
print("=" * 75)

print()
print(
    "Final modeling dataset:"
)

print(
    f"Unique feedback texts: "
    f"{len(unique_df)}"
)

print(
    f"Sentiment classes: "
    f"{unique_df['Sentiment'].nunique()}"
)

print()
print(
    "Sentiment mapping:"
)

print(
    "0 = Negative"
)

print(
    "1 = Positive"
)

print()
print(
    "IMPORTANT:"
)

print(
    "Train/validation/test must be split "
    "at the unique-feedback level."
)

print()
print("=" * 75)