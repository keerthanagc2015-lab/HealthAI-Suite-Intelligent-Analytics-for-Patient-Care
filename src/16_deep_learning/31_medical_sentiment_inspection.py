"""
HealthAI - Patient Feedback Sentiment Dataset Inspection

Step 31A

Purpose:
    Inspect the healthcare patient-feedback dataset before
    cleaning, label preparation, splitting, and model training.

Checks:
    - Excel sheets
    - Dataset shape
    - Column names
    - Data types
    - Missing values
    - Duplicate records
    - Unique values
    - Candidate text columns
    - Candidate sentiment/rating columns
    - Text length statistics
    - Value distributions
    - Dataset preview
    - Numeric summary
    - Quality summary
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
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
# 2. HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - PATIENT FEEDBACK DATASET INSPECTION")
print("=" * 75)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Dataset:")
print(DATA_PATH)


# ============================================================
# 3. VALIDATE FILE
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

print()
print("✓ Dataset file found.")


# ============================================================
# 4. INSPECT EXCEL SHEETS
# ============================================================

print()
print("=" * 75)
print("EXCEL SHEETS")
print("=" * 75)

excel_file = pd.ExcelFile(
    DATA_PATH
)

sheet_names = excel_file.sheet_names

for i, sheet in enumerate(
    sheet_names,
    start=1
):

    print(
        f"{i}. {sheet}"
    )


# ============================================================
# 5. LOAD FIRST SHEET
# ============================================================

print()
print("=" * 75)
print("LOADING DATASET")
print("=" * 75)

df = pd.read_excel(
    DATA_PATH,
    sheet_name=sheet_names[0]
)

print()
print("✓ Dataset loaded successfully.")

print(
    f"Rows: {df.shape[0]:,}"
)

print(
    f"Columns: {df.shape[1]:,}"
)


# ============================================================
# 6. CLEAN COLUMN NAMES FOR INSPECTION ONLY
# ============================================================

print()
print("=" * 75)
print("COLUMN NAMES")
print("=" * 75)

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:>3}. {column}"
    )


# ============================================================
# 7. DATA TYPES
# ============================================================

print()
print("=" * 75)
print("DATA TYPES")
print("=" * 75)

print(
    df.dtypes.to_string()
)


# ============================================================
# 8. MISSING VALUES
# ============================================================

print()
print("=" * 75)
print("MISSING VALUE ANALYSIS")
print("=" * 75)

missing = pd.DataFrame(
    {
        "column": df.columns,
        "missing_count": [
            int(
                df[column].isna().sum()
            )
            for column in df.columns
        ],
    }
)

missing["missing_percentage"] = (
    missing["missing_count"]
    / len(df)
    * 100
)

missing = missing.sort_values(
    "missing_count",
    ascending=False
)

print(
    missing.to_string(
        index=False
    )
)

missing_path = (
    OUTPUT_DIR
    / "patient_feedback_missing_report.csv"
)

missing.to_csv(
    missing_path,
    index=False
)

print()
print(
    f"✓ Saved: {missing_path}"
)


# ============================================================
# 9. DUPLICATE ANALYSIS
# ============================================================

print()
print("=" * 75)
print("DUPLICATE ANALYSIS")
print("=" * 75)

duplicate_count = int(
    df.duplicated().sum()
)

duplicate_percentage = (
    duplicate_count
    / len(df)
    * 100
)

print(
    f"Duplicate rows: "
    f"{duplicate_count:,}"
)

print(
    f"Duplicate percentage: "
    f"{duplicate_percentage:.2f}%"
)


# ============================================================
# 10. UNIQUE VALUES
# ============================================================

print()
print("=" * 75)
print("UNIQUE VALUE COUNTS")
print("=" * 75)

unique_report = pd.DataFrame(
    {
        "column": df.columns,
        "unique_values": [
            int(
                df[column].nunique(
                    dropna=True
                )
            )
            for column in df.columns
        ],
    }
)

print(
    unique_report.to_string(
        index=False
    )
)

unique_path = (
    OUTPUT_DIR
    / "patient_feedback_unique_report.csv"
)

unique_report.to_csv(
    unique_path,
    index=False
)

print()
print(
    f"✓ Saved: {unique_path}"
)


# ============================================================
# 11. IDENTIFY TEXT COLUMNS
# ============================================================

print()
print("=" * 75)
print("TEXT COLUMNS")
print("=" * 75)

text_columns = (
    df.select_dtypes(
        include=["object", "string"]
    )
    .columns
    .tolist()
)

if text_columns:

    for column in text_columns:

        print(
            f"✓ {column}"
        )

else:

    print(
        "No text columns detected."
    )


# ============================================================
# 12. TEXT LENGTH ANALYSIS
# ============================================================

print()
print("=" * 75)
print("TEXT LENGTH ANALYSIS")
print("=" * 75)

text_length_results = []

for column in text_columns:

    values = (
        df[column]
        .fillna("")
        .astype(str)
    )

    lengths = values.str.len()

    result = {
        "column": column,
        "min_length": int(
            lengths.min()
        ),
        "max_length": int(
            lengths.max()
        ),
        "mean_length": float(
            lengths.mean()
        ),
        "median_length": float(
            lengths.median()
        ),
    }

    text_length_results.append(
        result
    )

    print()
    print(
        f"Column: {column}"
    )

    print(
        f"  Minimum length: "
        f"{result['min_length']}"
    )

    print(
        f"  Maximum length: "
        f"{result['max_length']}"
    )

    print(
        f"  Mean length: "
        f"{result['mean_length']:.2f}"
    )

    print(
        f"  Median length: "
        f"{result['median_length']:.2f}"
    )


if text_length_results:

    text_length_df = pd.DataFrame(
        text_length_results
    )

    text_length_path = (
        OUTPUT_DIR
        / "patient_feedback_text_length_report.csv"
    )

    text_length_df.to_csv(
        text_length_path,
        index=False
    )

    print()
    print(
        f"✓ Saved: {text_length_path}"
    )


# ============================================================
# 13. CANDIDATE SENTIMENT / RATING COLUMNS
# ============================================================

print()
print("=" * 75)
print("POTENTIAL SENTIMENT / RATING COLUMNS")
print("=" * 75)

keywords = [
    "sentiment",
    "rating",
    "review",
    "feedback",
    "satisfaction",
    "comment",
    "text",
    "label",
    "polarity",
    "score",
    "emotion",
    "recommend",
]

candidate_columns = []

for column in df.columns:

    column_lower = str(
        column
    ).lower()

    if any(
        keyword in column_lower
        for keyword in keywords
    ):

        candidate_columns.append(
            column
        )


if candidate_columns:

    for column in candidate_columns:

        print(
            f"✓ {column}"
        )

else:

    print(
        "No obvious sentiment/rating "
        "columns detected."
    )


# ============================================================
# 14. VALUE DISTRIBUTIONS
# ============================================================

print()
print("=" * 75)
print("CANDIDATE COLUMN VALUE DISTRIBUTIONS")
print("=" * 75)

for column in candidate_columns:

    print()
    print(
        f"Column: {column}"
    )

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
        .head(30)
    )

    print(
        counts.to_string()
    )


# ============================================================
# 15. ALL LOW-CARDINALITY CATEGORICAL COLUMNS
# ============================================================

print()
print("=" * 75)
print("LOW-CARDINALITY CATEGORICAL COLUMNS")
print("=" * 75)

low_cardinality_columns = []

for column in df.columns:

    unique_count = df[column].nunique(
        dropna=True
    )

    if (
        unique_count <= 20
        and unique_count > 1
    ):

        low_cardinality_columns.append(
            column
        )

        print()
        print(
            f"Column: {column}"
        )

        print(
            f"Unique values: "
            f"{unique_count}"
        )

        print(
            df[column]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


# ============================================================
# 16. FIRST 10 RECORDS
# ============================================================

print()
print("=" * 75)
print("FIRST 10 RECORDS")
print("=" * 75)

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 17. RANDOM SAMPLE
# ============================================================

print()
print("=" * 75)
print("RANDOM SAMPLE")
print("=" * 75)

sample_size = min(
    10,
    len(df)
)

sample_df = df.sample(
    n=sample_size,
    random_state=42
)

print(
    sample_df.to_string(
        index=False
    )
)


# ============================================================
# 18. NUMERIC SUMMARY
# ============================================================

print()
print("=" * 75)
print("NUMERIC COLUMN SUMMARY")
print("=" * 75)

numeric_columns = (
    df.select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

if numeric_columns:

    numeric_summary = (
        df[numeric_columns]
        .describe()
        .T
    )

    print(
        numeric_summary.to_string()
    )

    numeric_path = (
        OUTPUT_DIR
        / "patient_feedback_numeric_summary.csv"
    )

    numeric_summary.to_csv(
        numeric_path
    )

    print()
    print(
        f"✓ Saved: {numeric_path}"
    )

else:

    print(
        "No numeric columns detected."
    )


# ============================================================
# 19. SAVE DATASET INSPECTION SUMMARY
# ============================================================

quality_summary = {
    "dataset_file": str(DATA_PATH),
    "sheet_names": sheet_names,
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "column_names": [
        str(column)
        for column in df.columns
    ],
    "duplicate_rows": duplicate_count,
    "duplicate_percentage": (
        float(duplicate_percentage)
    ),
    "text_columns": [
        str(column)
        for column in text_columns
    ],
    "candidate_sentiment_columns": [
        str(column)
        for column in candidate_columns
    ],
    "low_cardinality_columns": [
        str(column)
        for column in low_cardinality_columns
    ],
}


summary_path = (
    OUTPUT_DIR
    / "patient_feedback_inspection_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        quality_summary,
        file,
        indent=4
    )


print()
print(
    f"✓ Saved: {summary_path}"
)


# ============================================================
# 20. FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 31A - PATIENT FEEDBACK INSPECTION COMPLETED")
print("=" * 75)

print()
print(
    f"Dataset size: "
    f"{df.shape[0]:,} rows × "
    f"{df.shape[1]:,} columns"
)

print(
    f"Duplicate rows: "
    f"{duplicate_count:,}"
)

print(
    f"Text columns: "
    f"{len(text_columns)}"
)

print(
    f"Candidate sentiment/rating columns: "
    f"{len(candidate_columns)}"
)

print(
    f"Low-cardinality categorical columns: "
    f"{len(low_cardinality_columns)}"
)

print()
print(
    "Inspection artifacts saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 75)