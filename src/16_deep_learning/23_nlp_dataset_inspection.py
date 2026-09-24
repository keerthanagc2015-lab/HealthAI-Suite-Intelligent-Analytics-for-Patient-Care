import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset


# ============================================================
# HEALTHAI - MEDICAL NLP DATASET INSPECTION
# ============================================================

print("=" * 70)
print("HEALTHAI - MEDICAL NLP DATASET INSPECTION")
print("=" * 70)


# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME = "E3-JSI/synthetic-multi-med-notes-ner-v1"

OUTPUT_DIR = "data/processed/deep_learning/nlp"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading E3-JSI synthetic medical NER dataset...")

dataset = load_dataset(DATASET_NAME)

print("\n✓ Dataset loaded successfully!")

print("\nDataset structure:")
print(dataset)


# ============================================================
# DATASET SPLITS
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLITS")
print("=" * 70)

for split_name in dataset.keys():

    print(
        f"{split_name}: "
        f"{len(dataset[split_name])} records"
    )


# ============================================================
# CONVERT TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING TRAINING DATA")
print("=" * 70)

df = dataset["train"].to_pandas()

print("\nTraining dataframe shape:")
print(df.shape)


# ============================================================
# COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET COLUMNS")
print("=" * 70)

print(df.columns.tolist())


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# FIRST RECORD
# ============================================================

print("\n" + "=" * 70)
print("FIRST RECORD")
print("=" * 70)

first_record = df.iloc[0]

print("\nText:")
print(first_record["text"])

print("\nLanguage:")
print(first_record["language"])

print("\nEntities:")

entities = first_record["entities"]

if isinstance(entities, np.ndarray):
    entities = entities.tolist()

print(
    json.dumps(
        entities,
        indent=2,
        ensure_ascii=False,
        default=str
    )
)


# ============================================================
# TEXT QUALITY
# ============================================================

print("\n" + "=" * 70)
print("TEXT QUALITY CHECK")
print("=" * 70)

missing_text = df["text"].isna().sum()

empty_text = (
    df["text"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

duplicate_text = (
    df["text"]
    .duplicated()
    .sum()
)

print("Missing text:", missing_text)
print("Empty text:", empty_text)
print("Duplicate texts:", duplicate_text)


# ============================================================
# LANGUAGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LANGUAGE DISTRIBUTION")
print("=" * 70)

language_distribution = (
    df["language"]
    .value_counts()
)

print(language_distribution)


# ============================================================
# ENTITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ENTITY ANALYSIS")
print("=" * 70)

entity_counts = {}

total_entities = 0

for entities in df["entities"]:

    if entities is None:
        continue

    if isinstance(entities, np.ndarray):
        entities = entities.tolist()

    for entity in entities:

        if not isinstance(entity, dict):
            continue

        label = entity.get("label")

        if label is None:
            continue

        total_entities += 1

        entity_counts[label] = (
            entity_counts.get(label, 0) + 1
        )


print(
    "\nTotal annotated entities:",
    total_entities
)

entity_distribution = (
    pd.Series(entity_counts)
    .sort_values(ascending=False)
)

print("\nEntity type distribution:")
print(entity_distribution)


# ============================================================
# ENTITY EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("ENTITY EXAMPLES")
print("=" * 70)

shown_labels = set()

for entities in df["entities"]:

    if entities is None:
        continue

    if isinstance(entities, np.ndarray):
        entities = entities.tolist()

    for entity in entities:

        if not isinstance(entity, dict):
            continue

        label = entity.get("label")

        if label is None:
            continue

        if label not in shown_labels:

            print(
                f"\nEntity type: {label}"
            )

            print(
                "Example text:",
                entity.get("text")
            )

            print(
                "Start:",
                entity.get("start")
            )

            print(
                "End:",
                entity.get("end")
            )

            shown_labels.add(label)

        if len(shown_labels) >= 19:
            break

    if len(shown_labels) >= 19:
        break


# ============================================================
# ENTITY COUNT PER DOCUMENT
# ============================================================

print("\n" + "=" * 70)
print("ENTITIES PER DOCUMENT")
print("=" * 70)


def count_entities(value):

    if value is None:
        return 0

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        return len(value)

    return 0


df["entity_count"] = (
    df["entities"]
    .apply(count_entities)
)

print(
    df["entity_count"].describe()
)


# ============================================================
# TEXT LENGTH
# ============================================================

print("\n" + "=" * 70)
print("TEXT LENGTH ANALYSIS")
print("=" * 70)

df["character_count"] = (
    df["text"]
    .astype(str)
    .str.len()
)

df["word_count"] = (
    df["text"]
    .astype(str)
    .str.split()
    .str.len()
)

print("\nCharacter count:")
print(
    df["character_count"].describe()
)

print("\nWord count:")
print(
    df["word_count"].describe()
)


# ============================================================
# MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ROW CHECK")
print("=" * 70)

# Do NOT use df.duplicated() because entity columns
# contain NumPy arrays and are therefore unhashable.

duplicate_rows = (
    df[
        ["text", "language"]
    ]
    .duplicated()
    .sum()
)

print(
    "Duplicate text/language rows:",
    duplicate_rows
)


# ============================================================
# ENTITY ANNOTATION VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ENTITY ANNOTATION VALIDATION")
print("=" * 70)

invalid_entities = 0
checked_entities = 0

for index, row in df.iterrows():

    text = str(row["text"])

    entities = row["entities"]

    if entities is None:
        continue

    if isinstance(entities, np.ndarray):
        entities = entities.tolist()

    for entity in entities:

        if not isinstance(entity, dict):
            continue

        start = entity.get("start")
        end = entity.get("end")
        entity_text = entity.get("text")

        if (
            start is None
            or end is None
            or entity_text is None
        ):
            invalid_entities += 1
            continue

        checked_entities += 1

        try:

            start = int(start)
            end = int(end)

            extracted_text = text[start:end]

            if extracted_text != str(entity_text):

                invalid_entities += 1

                if invalid_entities <= 5:

                    print(
                        "\nInvalid entity found:"
                    )

                    print(
                        "Expected:",
                        entity_text
                    )

                    print(
                        "Extracted:",
                        extracted_text
                    )

                    print(
                        "Start:",
                        start
                    )

                    print(
                        "End:",
                        end
                    )

        except Exception:

            invalid_entities += 1


print(
    "\nEntities checked:",
    checked_entities
)

print(
    "Invalid entity annotations:",
    invalid_entities
)


# ============================================================
# LANGUAGE QUALITY
# ============================================================

print("\n" + "=" * 70)
print("LANGUAGE QUALITY CHECK")
print("=" * 70)

missing_language = (
    df["language"]
    .isna()
    .sum()
)

empty_language = (
    df["language"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print(
    "Missing language:",
    missing_language
)

print(
    "Empty language:",
    empty_language
)


# ============================================================
# ENGLISH DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("ENGLISH DATA CHECK")
print("=" * 70)

english_df = df[
    df["language"]
    .astype(str)
    .str.lower()
    .eq("english")
].copy()

print(
    "English records:",
    len(english_df)
)

print(
    "English entities:",
    english_df["entity_count"].sum()
)

print(
    "English percentage:",
    round(
        len(english_df) / len(df) * 100,
        2
    ),
    "%"
)


# ============================================================
# DATA QUALITY STATUS
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY STATUS")
print("=" * 70)

if (
    missing_text == 0
    and
    empty_text == 0
    and
    invalid_entities == 0
    and
    missing_language == 0
):

    print(
        "\n✓ NLP DATA QUALITY CHECK PASSED"
    )

else:

    print(
        "\n⚠ NLP DATA QUALITY CHECK REQUIRES REVIEW"
    )


# ============================================================
# SAVE ENTITY DISTRIBUTION
# ============================================================

entity_report = (
    entity_distribution
    .reset_index()
)

entity_report.columns = [
    "Entity_Type",
    "Count"
]

entity_report_path = os.path.join(
    OUTPUT_DIR,
    "nlp_entity_distribution.csv"
)

entity_report.to_csv(
    entity_report_path,
    index=False
)


# ============================================================
# SAVE LANGUAGE DISTRIBUTION
# ============================================================

language_report = (
    language_distribution
    .reset_index()
)

language_report.columns = [
    "Language",
    "Count"
]

language_report_path = os.path.join(
    OUTPUT_DIR,
    "nlp_language_distribution.csv"
)

language_report.to_csv(
    language_report_path,
    index=False
)


# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Total Records",
        "Total Columns",
        "Total Entities",
        "Number of Entity Types",
        "Number of Languages",
        "English Records",
        "English Entities",
        "Missing Text",
        "Empty Text",
        "Duplicate Texts",
        "Duplicate Text/Language Rows",
        "Invalid Entity Annotations"

    ],

    "Value": [

        len(df),
        len(df.columns),
        total_entities,
        len(entity_counts),
        df["language"].nunique(),
        len(english_df),
        int(english_df["entity_count"].sum()),
        missing_text,
        empty_text,
        duplicate_text,
        duplicate_rows,
        invalid_entities

    ]

})

summary_path = os.path.join(
    OUTPUT_DIR,
    "nlp_dataset_inspection_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# SAVE ENGLISH SAMPLE
# ============================================================

english_sample_path = os.path.join(
    OUTPUT_DIR,
    "nlp_english_dataset_sample.csv"
)

english_df[
    [
        "text",
        "language",
        "entity_count",
        "character_count",
        "word_count"
    ]
].head(100).to_csv(
    english_sample_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL NLP DATASET SUMMARY")
print("=" * 70)

print(
    "\nTotal records:",
    len(df)
)

print(
    "Total annotated entities:",
    total_entities
)

print(
    "Entity types:",
    len(entity_counts)
)

print(
    "Languages:",
    df["language"].nunique()
)

print(
    "English records:",
    len(english_df)
)

print(
    "English entities:",
    int(english_df["entity_count"].sum())
)

print("\nEntity types:")

for entity_type in entity_distribution.index:

    print(
        " -",
        entity_type
    )

print("\nLanguages:")

for language in language_distribution.index:

    print(
        " -",
        language
    )

print("\nReports saved:")

print(entity_report_path)
print(language_report_path)
print(summary_path)
print(english_sample_path)


print("\n" + "=" * 70)
print("NLP DATASET INSPECTION COMPLETED")
print("=" * 70)