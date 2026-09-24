import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset


# ============================================================
# HEALTHAI - NLP PREPROCESSING
# ============================================================

print("=" * 70)
print("HEALTHAI - MEDICAL NLP PREPROCESSING")
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

print("\nLoading dataset...")

dataset = load_dataset(DATASET_NAME)

print("✓ Dataset loaded successfully.")


# ============================================================
# LOAD TRAIN DATA
# ============================================================

df = dataset["train"].to_pandas()

print(
    "\nTotal records:",
    len(df)
)


# ============================================================
# SELECT ENGLISH DATA
# ============================================================

print("\n" + "=" * 70)
print("SELECTING ENGLISH DATA")
print("=" * 70)

english_df = df[
    df["language"]
    .astype(str)
    .str.lower()
    .eq("english")
].copy()

english_df = english_df.reset_index(drop=True)

print(
    "\nEnglish records:",
    len(english_df)
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def convert_entities(value):

    """
    Convert entity annotations into normal Python list format.
    """

    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, list):
        return []

    return value


# ============================================================
# ENTITY SPAN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING ENTITY SPANS")
print("=" * 70)

total_entities = 0
exact_matches = 0
case_insensitive_matches = 0
invalid_entities = 0

processed_records = []


for index, row in english_df.iterrows():

    text = str(row["text"])

    entities = convert_entities(
        row["entities"]
    )

    valid_entities = []

    for entity in entities:

        if not isinstance(entity, dict):
            continue

        entity_text = entity.get("text")
        label = entity.get("label")
        start = entity.get("start")
        end = entity.get("end")

        total_entities += 1

        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if (
            entity_text is None
            or label is None
            or start is None
            or end is None
        ):
            invalid_entities += 1
            continue

        try:

            start = int(start)
            end = int(end)

        except Exception:

            invalid_entities += 1
            continue

        # ----------------------------------------------------
        # BOUNDARY VALIDATION
        # ----------------------------------------------------

        if (
            start < 0
            or end > len(text)
            or start >= end
        ):
            invalid_entities += 1
            continue

        extracted_text = text[start:end]

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        if extracted_text == str(entity_text):

            exact_matches += 1

            valid_entities.append({
                "text": str(entity_text),
                "label": str(label),
                "start": start,
                "end": end
            })

        # ----------------------------------------------------
        # CASE-INSENSITIVE MATCH
        # ----------------------------------------------------

        elif (
            extracted_text.lower()
            == str(entity_text).lower()
        ):

            case_insensitive_matches += 1

            # Use the actual text from the document
            # while preserving the original label.

            valid_entities.append({
                "text": extracted_text,
                "label": str(label),
                "start": start,
                "end": end
            })

        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        else:

            invalid_entities += 1

    processed_records.append({
        "text": text,
        "language": "english",
        "entities": valid_entities
    })


print(
    "\nTotal entities checked:",
    total_entities
)

print(
    "Exact matches:",
    exact_matches
)

print(
    "Case-insensitive matches:",
    case_insensitive_matches
)

print(
    "Invalid entities:",
    invalid_entities
)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

processed_df = pd.DataFrame(
    processed_records
)


# ============================================================
# ENTITY COUNTS
# ============================================================

processed_df["entity_count"] = (
    processed_df["entities"]
    .apply(len)
)


print("\n" + "=" * 70)
print("PROCESSED DATA SUMMARY")
print("=" * 70)

print(
    "\nRecords:",
    len(processed_df)
)

print(
    "Valid entities:",
    processed_df["entity_count"].sum()
)


# ============================================================
# ENTITY LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ENTITY LABEL DISTRIBUTION")
print("=" * 70)

label_counts = {}

for entities in processed_df["entities"]:

    for entity in entities:

        label = entity["label"]

        label_counts[label] = (
            label_counts.get(label, 0) + 1
        )


label_distribution = (
    pd.Series(label_counts)
    .sort_values(
        ascending=False
    )
)

print(label_distribution)


# ============================================================
# CREATE BIO LABELS
# ============================================================

print("\n" + "=" * 70)
print("CREATING BIO LABELS")
print("=" * 70)


def create_bio_labels(text, entities):

    """
    Create character-level BIO labels.

    These are intermediate labels.
    They will later be aligned with
    Transformer tokenizer tokens.
    """

    labels = [
        "O"
        for _ in text
    ]

    # Sort entities by start position
    entities = sorted(
        entities,
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    for entity in entities:

        start = int(entity["start"])
        end = int(entity["end"])
        entity_label = entity["label"]

        if start >= end:
            continue

        # Beginning of entity
        labels[start] = (
            f"B-{entity_label}"
        )

        # Continuation of entity
        for position in range(
            start + 1,
            end
        ):

            if labels[position] == "O":

                labels[position] = (
                    f"I-{entity_label}"
                )

    return labels


processed_df["character_bio_labels"] = processed_df.apply(
    lambda row: create_bio_labels(
        row["text"],
        row["entities"]
    ),
    axis=1
)


print(
    "✓ BIO labels created."
)


# ============================================================
# CREATE TOKEN-LEVEL WHITESPACE REPRESENTATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING TOKEN-LEVEL PREVIEW")
print("=" * 70)


def create_token_preview(text, entities):

    """
    Create a simple whitespace-token BIO preview.

    This is only for inspection.

    The actual Transformer alignment will happen
    later using the BioBERT/ClinicalBERT tokenizer.
    """

    tokens = []
    labels = []

    entities = sorted(
        entities,
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    # --------------------------------------------------------
    # Simple whitespace tokenization
    # --------------------------------------------------------

    for match in text.split():

        token = match

        token_start = text.find(
            token,
            0 if not tokens else
            sum(
                len(t) + 1
                for t in tokens
            )
        )

        token_end = (
            token_start
            + len(token)
        )

        token_label = "O"

        for entity in entities:

            entity_start = entity["start"]
            entity_end = entity["end"]

            # Token overlaps entity
            if (
                token_start < entity_end
                and token_end > entity_start
            ):

                if (
                    token_start
                    <= entity_start
                ):

                    token_label = (
                        "B-"
                        + entity["label"]
                    )

                else:

                    token_label = (
                        "I-"
                        + entity["label"]
                    )

                break

        tokens.append(token)
        labels.append(token_label)

    return tokens, labels


# Preview only first 100 records
preview_count = min(
    100,
    len(processed_df)
)

preview_rows = []

for index in range(preview_count):

    row = processed_df.iloc[index]

    tokens, labels = create_token_preview(
        row["text"],
        row["entities"]
    )

    preview_rows.append({
        "record_id": index,
        "tokens": json.dumps(
            tokens,
            ensure_ascii=False
        ),
        "labels": json.dumps(
            labels,
            ensure_ascii=False
        )
    })


preview_df = pd.DataFrame(
    preview_rows
)


# ============================================================
# SHOW SAMPLE
# ============================================================

print("\nSample BIO representation:")

if len(preview_df) > 0:

    sample_tokens = json.loads(
        preview_df.iloc[0]["tokens"]
    )

    sample_labels = json.loads(
        preview_df.iloc[0]["labels"]
    )

    for token, label in zip(
        sample_tokens[:30],
        sample_labels[:30]
    ):

        print(
            f"{token:25s} -> {label}"
        )


# ============================================================
# CREATE LABEL LIST
# ============================================================

print("\n" + "=" * 70)
print("CREATING LABEL SET")
print("=" * 70)

unique_entity_labels = sorted(
    label_counts.keys()
)

bio_labels = ["O"]

for label in unique_entity_labels:

    bio_labels.append(
        f"B-{label}"
    )

    bio_labels.append(
        f"I-{label}"
    )


label2id = {
    label: index
    for index, label
    in enumerate(bio_labels)
}

id2label = {
    index: label
    for label, index
    in label2id.items()
}


print(
    "\nNumber of entity types:",
    len(unique_entity_labels)
)

print(
    "Number of BIO labels:",
    len(bio_labels)
)

print("\nLabels:")

for index, label in id2label.items():

    print(
        index,
        "->",
        label
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)


processed_json_path = os.path.join(
    OUTPUT_DIR,
    "english_ner_processed.json"
)

with open(
    processed_json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        processed_df[
            [
                "text",
                "language",
                "entities",
                "entity_count"
            ]
        ].to_dict(
            orient="records"
        ),
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE LABEL MAP
# ============================================================

label_map_path = os.path.join(
    OUTPUT_DIR,
    "ner_label_map.json"
)

with open(
    label_map_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "label2id": label2id,
            "id2label": id2label,
            "entity_labels": unique_entity_labels
        },
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE LABEL DISTRIBUTION
# ============================================================

label_distribution_path = os.path.join(
    OUTPUT_DIR,
    "english_ner_label_distribution.csv"
)

label_distribution.reset_index(
    name="Count"
).rename(
    columns={
        "index": "Entity_Type"
    }
).to_csv(
    label_distribution_path,
    index=False
)


# ============================================================
# SAVE PREVIEW
# ============================================================

preview_path = os.path.join(
    OUTPUT_DIR,
    "ner_bio_preview.csv"
)

preview_df.to_csv(
    preview_path,
    index=False
)


# ============================================================
# SAVE QUALITY REPORT
# ============================================================

quality_report = pd.DataFrame({

    "Metric": [

        "Total English Records",
        "Total Entities Checked",
        "Exact Entity Matches",
        "Case-Insensitive Entity Matches",
        "Invalid Entities",
        "Valid Entities",
        "Entity Types",
        "BIO Labels"

    ],

    "Value": [

        len(processed_df),
        total_entities,
        exact_matches,
        case_insensitive_matches,
        invalid_entities,
        int(
            processed_df[
                "entity_count"
            ].sum()
        ),
        len(unique_entity_labels),
        len(bio_labels)

    ]

})


quality_report_path = os.path.join(
    OUTPUT_DIR,
    "nlp_preprocessing_quality_report.csv"
)

quality_report.to_csv(
    quality_report_path,
    index=False
)


# ============================================================
# FINAL QUALITY STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING QUALITY")
print("=" * 70)

if (
    len(processed_df) > 0
    and
    processed_df["entity_count"].sum() > 0
    and
    invalid_entities == 0
):

    print(
        "\n✓ NLP PREPROCESSING QUALITY CHECK PASSED"
    )

elif len(processed_df) > 0:

    print(
        "\n⚠ NLP DATA PREPROCESSING COMPLETED"
    )

    print(
        "Some entity annotations require review."
    )

else:

    print(
        "\n✗ NLP PREPROCESSING FAILED"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL NLP PREPROCESSING SUMMARY")
print("=" * 70)

print(
    "\nEnglish records:",
    len(processed_df)
)

print(
    "Entities checked:",
    total_entities
)

print(
    "Valid entities:",
    int(
        processed_df[
            "entity_count"
        ].sum()
    )
)

print(
    "Invalid entities:",
    invalid_entities
)

print(
    "Entity types:",
    len(unique_entity_labels)
)

print(
    "BIO labels:",
    len(bio_labels)
)

print(
    "\nFiles created:"
)

print(
    processed_json_path
)

print(
    label_map_path
)

print(
    label_distribution_path
)

print(
    preview_path
)

print(
    quality_report_path
)


print("\n" + "=" * 70)
print("NLP PREPROCESSING COMPLETED")
print("=" * 70)