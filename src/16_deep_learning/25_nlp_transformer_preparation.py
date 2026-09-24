import os
import json
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


# ============================================================
# HEALTHAI - TRANSFORMER NER DATA PREPARATION
# ============================================================

print("=" * 70)
print("HEALTHAI - TRANSFORMER NER DATA PREPARATION")
print("=" * 70)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "data/processed/deep_learning/nlp/"
    "english_ner_processed.json"
)

LABEL_MAP_FILE = (
    "data/processed/deep_learning/nlp/"
    "ner_label_map.json"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/nlp/"
    "transformer"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# BioBERT tokenizer
BIOBERT_TOKENIZER = (
    "dmis-lab/biobert-v1.1"
)

MAX_LENGTH = 512

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

print("\nLoading processed English NER data...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    records = json.load(file)


print(
    "✓ Processed data loaded."
)

print(
    "Records:",
    len(records)
)


# ============================================================
# LOAD LABEL MAP
# ============================================================

print("\nLoading NER label map...")

with open(
    LABEL_MAP_FILE,
    "r",
    encoding="utf-8"
) as file:

    label_data = json.load(file)


label2id = {
    key: int(value)
    for key, value in
    label_data["label2id"].items()
}

id2label = {
    int(key): value
    for key, value in
    label_data["id2label"].items()
}

print(
    "✓ Label map loaded."
)

print(
    "Number of labels:",
    len(label2id)
)


# ============================================================
# LOAD BIOBERT TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING BIOBERT TOKENIZER")
print("=" * 70)

print(
    "\nTokenizer:",
    BIOBERT_TOKENIZER
)

tokenizer = AutoTokenizer.from_pretrained(
    BIOBERT_TOKENIZER
)

print(
    "✓ BioBERT tokenizer loaded."
)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


# ============================================================
# DISPLAY TOKENIZATION EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZATION EXAMPLE")
print("=" * 70)

example_text = records[0]["text"]

example_tokens = tokenizer.tokenize(
    example_text
)

print("\nOriginal text:")
print(
    example_text[:500]
)

print("\nFirst 50 tokenizer tokens:")

for index, token in enumerate(
    example_tokens[:50]
):

    print(
        f"{index:3d} -> {token}"
    )


# ============================================================
# FUNCTION:
# CHARACTER SPAN → TOKEN BIO LABELS
# ============================================================

def create_token_labels(
    text,
    entities,
    tokenizer
):

    """
    Convert character-level entity annotations
    into token-level BIO labels.

    Uses tokenizer offset mappings so that
    labels align with actual Transformer tokens.
    """

    encoding = tokenizer(
        text,
        truncation=False,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    input_ids = encoding["input_ids"]

    offset_mapping = encoding[
        "offset_mapping"
    ]

    token_labels = []

    for token_index, (
        token_start,
        token_end
    ) in enumerate(offset_mapping):

        # Special tokens such as [CLS] and [SEP]
        if token_start == token_end:

            token_labels.append(
                -100
            )

            continue

        label = "O"

        # ----------------------------------------------------
        # Find entity overlapping this token
        # ----------------------------------------------------

        for entity in entities:

            entity_start = int(
                entity["start"]
            )

            entity_end = int(
                entity["end"]
            )

            entity_label = entity[
                "label"
            ]

            # No overlap
            if (
                token_end <= entity_start
                or
                token_start >= entity_end
            ):

                continue

            # ------------------------------------------------
            # Beginning vs inside
            # ------------------------------------------------

            if (
                token_start
                <= entity_start
            ):

                label = (
                    f"B-{entity_label}"
                )

            else:

                label = (
                    f"I-{entity_label}"
                )

            break

        token_labels.append(
            label2id[label]
        )

    return (
        input_ids,
        offset_mapping,
        token_labels
    )


# ============================================================
# CREATE TOKENIZED RECORDS
# ============================================================

print("\n" + "=" * 70)
print("CREATING TOKEN-LEVEL LABELS")
print("=" * 70)

tokenized_records = []

total_records = len(records)

for index, record in enumerate(
    records
):

    text = record["text"]

    entities = record[
        "entities"
    ]

    (
        input_ids,
        offsets,
        labels
    ) = create_token_labels(
        text,
        entities,
        tokenizer
    )

    tokenized_records.append({

        "record_id": index,

        "input_ids": input_ids,

        "attention_mask": [
            1
            for _ in input_ids
        ],

        "labels": labels,

        "offset_mapping": offsets

    })

    if (
        (index + 1) % 50 == 0
        or
        index == total_records - 1
    ):

        print(
            f"Processed "
            f"{index + 1}/{total_records}"
        )


print(
    "\n✓ Token-level labels created."
)


# ============================================================
# TOKEN LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TOKEN LENGTH ANALYSIS")
print("=" * 70)

token_lengths = [
    len(record["input_ids"])
    for record
    in tokenized_records
]

length_series = pd.Series(
    token_lengths
)

print(
    length_series.describe()
)

print(
    "\nMaximum tokens:",
    max(token_lengths)
)

print(
    "Minimum tokens:",
    min(token_lengths)
)

print(
    "Average tokens:",
    round(
        length_series.mean(),
        2
    )
)

long_records = sum(
    length > MAX_LENGTH
    for length
    in token_lengths
)

print(
    f"\nRecords longer than "
    f"{MAX_LENGTH} tokens:",
    long_records
)


# ============================================================
# LABEL ALIGNMENT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LABEL ALIGNMENT VALIDATION")
print("=" * 70)

alignment_errors = 0

for record in tokenized_records:

    if not (
        len(record["input_ids"])
        ==
        len(record["labels"])
    ):

        alignment_errors += 1

    if not (
        len(record["input_ids"])
        ==
        len(record["attention_mask"])
    ):

        alignment_errors += 1


print(
    "Alignment errors:",
    alignment_errors
)

if alignment_errors == 0:

    print(
        "✓ Token/label alignment passed."
    )

else:

    print(
        "⚠ Alignment errors detected."
    )


# ============================================================
# CREATE HUMAN-READABLE SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("CREATING TOKEN-LABEL PREVIEW")
print("=" * 70)

sample = tokenized_records[0]

sample_tokens = tokenizer.convert_ids_to_tokens(
    sample["input_ids"]
)

sample_labels = [
    id2label[label_id]
    if label_id != -100
    else "SPECIAL"
    for label_id
    in sample["labels"]
]

preview_rows = []

for token, label in zip(
    sample_tokens,
    sample_labels
):

    preview_rows.append({

        "Token": token,

        "Label": label

    })


preview_df = pd.DataFrame(
    preview_rows
)


print(
    "\nFirst 40 token-label pairs:"
)

print(
    preview_df.head(40).to_string(
        index=False
    )
)


# ============================================================
# SAVE TOKEN-LABEL PREVIEW
# ============================================================

preview_path = os.path.join(
    OUTPUT_DIR,
    "biobert_token_label_preview.csv"
)

preview_df.to_csv(
    preview_path,
    index=False
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / VALIDATION / TEST SPLITS")
print("=" * 70)

indices = list(
    range(
        len(tokenized_records)
    )
)

random.shuffle(indices)

total = len(indices)

train_end = int(
    total * 0.70
)

validation_end = int(
    total * 0.85
)

train_indices = indices[
    :train_end
]

validation_indices = indices[
    train_end:validation_end
]

test_indices = indices[
    validation_end:
]


train_records = [
    tokenized_records[i]
    for i in train_indices
]

validation_records = [
    tokenized_records[i]
    for i in validation_indices
]

test_records = [
    tokenized_records[i]
    for i in test_indices
]


print(
    "\nTraining records:",
    len(train_records)
)

print(
    "Validation records:",
    len(validation_records)
)

print(
    "Testing records:",
    len(test_records)
)


# ============================================================
# SAVE SPLITS
# ============================================================

print("\n" + "=" * 70)
print("SAVING TRANSFORMER DATA")
print("=" * 70)


def save_json(
    data,
    path
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False
        )


train_path = os.path.join(
    OUTPUT_DIR,
    "train.json"
)

validation_path = os.path.join(
    OUTPUT_DIR,
    "validation.json"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.json"
)


save_json(
    train_records,
    train_path
)

save_json(
    validation_records,
    validation_path
)

save_json(
    test_records,
    test_path
)


# ============================================================
# SAVE DATASET CONFIGURATION
# ============================================================

configuration = {

    "dataset": DATASET_NAME
    if "DATASET_NAME" in globals()
    else "E3-JSI/synthetic-multi-med-notes-ner-v1",

    "language": "english",

    "tokenizer": BIOBERT_TOKENIZER,

    "max_length": MAX_LENGTH,

    "num_records": len(records),

    "train_records": len(
        train_records
    ),

    "validation_records": len(
        validation_records
    ),

    "test_records": len(
        test_records
    ),

    "num_entity_types": len(
        label_data[
            "entity_labels"
        ]
    ),

    "num_bio_labels": len(
        label2id
    ),

    "label2id": label2id,

    "id2label": {
        str(key): value
        for key, value
        in id2label.items()
    }

}


configuration_path = os.path.join(
    OUTPUT_DIR,
    "transformer_dataset_config.json"
)


with open(
    configuration_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        configuration,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE TOKEN LENGTH STATISTICS
# ============================================================

length_statistics = pd.DataFrame({

    "Metric": [

        "Total Records",
        "Minimum Tokens",
        "Maximum Tokens",
        "Average Tokens",
        "Median Tokens",
        "Records Over 512 Tokens"

    ],

    "Value": [

        len(token_lengths),

        min(token_lengths),

        max(token_lengths),

        round(
            float(
                length_series.mean()
            ),
            2
        ),

        float(
            length_series.median()
        ),

        long_records

    ]

})


length_statistics_path = os.path.join(
    OUTPUT_DIR,
    "transformer_token_length_statistics.csv"
)


length_statistics.to_csv(
    length_statistics_path,
    index=False
)


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRANSFORMER DATA QUALITY CHECK")
print("=" * 70)

quality_passed = True


if len(records) != 400:

    quality_passed = False

    print(
        "✗ Expected 400 English records."
    )


if alignment_errors != 0:

    quality_passed = False

    print(
        "✗ Token/label alignment failed."
    )


if len(train_records) == 0:

    quality_passed = False

    print(
        "✗ Training set is empty."
    )


if len(validation_records) == 0:

    quality_passed = False

    print(
        "✗ Validation set is empty."
    )


if len(test_records) == 0:

    quality_passed = False

    print(
        "✗ Test set is empty."
    )


if quality_passed:

    print(
        "\n✓ TRANSFORMER DATA PREPARATION PASSED"
    )

else:

    print(
        "\n⚠ TRANSFORMER DATA PREPARATION "
        "REQUIRES REVIEW"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRANSFORMER PREPARATION SUMMARY")
print("=" * 70)

print(
    "\nEnglish records:",
    len(records)
)

print(
    "Entity types:",
    len(
        label_data[
            "entity_labels"
        ]
    )
)

print(
    "BIO labels:",
    len(label2id)
)

print(
    "Tokenizer:",
    BIOBERT_TOKENIZER
)

print(
    "Training records:",
    len(train_records)
)

print(
    "Validation records:",
    len(validation_records)
)

print(
    "Testing records:",
    len(test_records)
)

print(
    "Maximum token length:",
    max(token_lengths)
)

print(
    "Records > 512 tokens:",
    long_records
)

print(
    "\nFiles created:"
)

print(train_path)
print(validation_path)
print(test_path)
print(configuration_path)
print(preview_path)
print(length_statistics_path)


print("\n" + "=" * 70)
print("TRANSFORMER NER DATA PREPARATION COMPLETED")
print("=" * 70)