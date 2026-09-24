import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# HEALTHAI - LONG DOCUMENT NER CHUNKING
# ============================================================

print("=" * 70)
print("HEALTHAI - LONG DOCUMENT NER CHUNKING")
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

TOKENIZER_NAME = "dmis-lab/biobert-v1.1"

MAX_LENGTH = 512

# Leave room for [CLS] and [SEP]
CONTENT_LENGTH = 510

# Small overlap helps preserve entities/context near boundaries
STRIDE = 64

RANDOM_SEED = 42

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed English NER data...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    records = json.load(file)

print(
    "✓ Records loaded:",
    len(records)
)


# ============================================================
# LOAD LABEL MAP
# ============================================================

print("\nLoading label map...")

with open(
    LABEL_MAP_FILE,
    "r",
    encoding="utf-8"
) as file:

    label_data = json.load(file)


label2id = {
    str(key): int(value)
    for key, value
    in label_data["label2id"].items()
}

id2label = {
    int(key): value
    for key, value
    in label_data["id2label"].items()
}


print(
    "✓ Labels loaded:",
    len(label2id)
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING BIOBERT TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

print(
    "✓ Tokenizer loaded."
)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


# ============================================================
# TOKENIZE ONE DOCUMENT WITHOUT TRUNCATION
# ============================================================

def tokenize_document(text):

    """
    Tokenize the complete document.

    We keep offset mappings so that every token can
    be connected back to its original character span.
    """

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True
    )

    return encoding


# ============================================================
# FIND TOKEN LABEL
# ============================================================

def get_token_label(
    token_start,
    token_end,
    entities
):

    """
    Assign BIO label to one token based on
    character-level entity spans.
    """

    overlapping_entity = None

    for entity in entities:

        entity_start = int(
            entity["start"]
        )

        entity_end = int(
            entity["end"]
        )

        # No overlap
        if (
            token_end <= entity_start
            or
            token_start >= entity_end
        ):
            continue

        overlapping_entity = entity
        break

    if overlapping_entity is None:

        return "O"


    entity_start = int(
        overlapping_entity["start"]
    )

    entity_end = int(
        overlapping_entity["end"]
    )

    entity_label = (
        overlapping_entity["label"]
    )


    # Token begins at entity start
    if token_start <= entity_start:

        return (
            f"B-{entity_label}"
        )


    # Token is inside entity
    return (
        f"I-{entity_label}"
    )


# ============================================================
# CREATE CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("CREATING 512-TOKEN CHUNKS")
print("=" * 70)


all_chunks = []

document_statistics = []

chunk_id = 0


for document_id, record in enumerate(
    records
):

    text = record["text"]

    entities = record["entities"]


    # --------------------------------------------------------
    # Tokenize complete document
    # --------------------------------------------------------

    encoding = tokenize_document(
        text
    )

    input_ids = encoding[
        "input_ids"
    ]

    offsets = encoding[
        "offset_mapping"
    ]


    total_tokens = len(
        input_ids
    )


    # --------------------------------------------------------
    # Create token-level labels
    # --------------------------------------------------------

    token_labels = []

    for (
        token_start,
        token_end
    ) in offsets:

        label = get_token_label(
            token_start,
            token_end,
            entities
        )

        token_labels.append(
            label2id[label]
        )


    # --------------------------------------------------------
    # Determine chunk positions
    # --------------------------------------------------------

    if total_tokens <= CONTENT_LENGTH:

        starts = [0]

    else:

        starts = list(
            range(
                0,
                total_tokens,
                CONTENT_LENGTH - STRIDE
            )
        )


    document_chunk_count = 0


    # --------------------------------------------------------
    # Create each chunk
    # --------------------------------------------------------

    for start in starts:

        end = min(
            start + CONTENT_LENGTH,
            total_tokens
        )


        chunk_input_ids = (
            input_ids[start:end]
        )

        chunk_labels = (
            token_labels[start:end]
        )

        chunk_offsets = (
            offsets[start:end]
        )


        # ----------------------------------------------------
        # Add special tokens
        # ----------------------------------------------------

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id


        final_input_ids = (
            [cls_id]
            + chunk_input_ids
            + [sep_id]
        )


        final_labels = (
            [-100]
            + chunk_labels
            + [-100]
        )


        attention_mask = [
            1
            for _ in final_input_ids
        ]


        # ----------------------------------------------------
        # Verify maximum length
        # ----------------------------------------------------

        if len(final_input_ids) > MAX_LENGTH:

            raise ValueError(
                "Chunk exceeds MAX_LENGTH: "
                f"{len(final_input_ids)}"
            )


        # ----------------------------------------------------
        # Count entities represented in chunk
        # ----------------------------------------------------

        chunk_entity_labels = []

        for label_id in chunk_labels:

            if label_id != label2id["O"]:

                chunk_entity_labels.append(
                    id2label[label_id]
                )


        # ----------------------------------------------------
        # Save chunk
        # ----------------------------------------------------

        all_chunks.append({

            "chunk_id": chunk_id,

            "document_id": document_id,

            "input_ids": final_input_ids,

            "attention_mask": attention_mask,

            "labels": final_labels,

            "offset_mapping": (
                [(0, 0)]
                + chunk_offsets
                + [(0, 0)]
            )

        })


        chunk_id += 1

        document_chunk_count += 1


        # ----------------------------------------------------
        # Stop once final chunk reached
        # ----------------------------------------------------

        if end >= total_tokens:

            break


    document_statistics.append({

        "document_id": document_id,

        "original_tokens": total_tokens,

        "chunks_created": document_chunk_count

    })


    if (
        (document_id + 1) % 50 == 0
        or
        document_id == len(records) - 1
    ):

        print(
            f"Processed documents: "
            f"{document_id + 1}/{len(records)}"
        )


# ============================================================
# CHUNK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHUNKING SUMMARY")
print("=" * 70)

print(
    "\nOriginal documents:",
    len(records)
)

print(
    "Total chunks:",
    len(all_chunks)
)


chunk_lengths = [
    len(chunk["input_ids"])
    for chunk in all_chunks
]

chunk_length_series = pd.Series(
    chunk_lengths
)


print(
    "\nChunk length statistics:"
)

print(
    chunk_length_series.describe()
)

print(
    "\nMaximum chunk length:",
    max(chunk_lengths)
)

print(
    "Minimum chunk length:",
    min(chunk_lengths)
)

print(
    "Average chunk length:",
    round(
        chunk_length_series.mean(),
        2
    )
)


# ============================================================
# VERIFY ALL CHUNKS <= 512
# ============================================================

print("\n" + "=" * 70)
print("MAXIMUM LENGTH VALIDATION")
print("=" * 70)

oversized_chunks = sum(
    length > MAX_LENGTH
    for length in chunk_lengths
)

print(
    "Chunks over 512 tokens:",
    oversized_chunks
)

if oversized_chunks == 0:

    print(
        "✓ All chunks fit within 512 tokens."
    )

else:

    print(
        "✗ Some chunks exceed 512 tokens."
    )


# ============================================================
# LABEL ALIGNMENT CHECK
# ============================================================

print("\n" + "=" * 70)
print("LABEL ALIGNMENT CHECK")
print("=" * 70)

alignment_errors = 0

for chunk in all_chunks:

    if len(
        chunk["input_ids"]
    ) != len(
        chunk["labels"]
    ):

        alignment_errors += 1


    if len(
        chunk["input_ids"]
    ) != len(
        chunk["attention_mask"]
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
        "✗ Token/label alignment failed."
    )


# ============================================================
# DOCUMENT → CHUNK STATISTICS
# ============================================================

statistics_df = pd.DataFrame(
    document_statistics
)


print("\n" + "=" * 70)
print("DOCUMENT CHUNK STATISTICS")
print("=" * 70)

print(
    statistics_df[
        "chunks_created"
    ].describe()
)


# ============================================================
# SHOW EXAMPLE CHUNK
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE CHUNK")
print("=" * 70)

if len(all_chunks) > 0:

    example_chunk = all_chunks[0]

    example_tokens = (
        tokenizer.convert_ids_to_tokens(
            example_chunk[
                "input_ids"
            ]
        )
    )

    example_labels = []

    for label_id in (
        example_chunk["labels"]
    ):

        if label_id == -100:

            example_labels.append(
                "SPECIAL"
            )

        else:

            example_labels.append(
                id2label[label_id]
            )


    example_df = pd.DataFrame({

        "Token": example_tokens,

        "Label": example_labels

    })


    print(
        example_df.head(50).to_string(
            index=False
        )
    )


# ============================================================
# SAVE CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("SAVING CHUNKED DATA")
print("=" * 70)


chunks_path = os.path.join(
    OUTPUT_DIR,
    "all_ner_chunks.json"
)


with open(
    chunks_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False
    )


# ============================================================
# SAVE DOCUMENT STATISTICS
# ============================================================

statistics_path = os.path.join(
    OUTPUT_DIR,
    "document_chunk_statistics.csv"
)


statistics_df.to_csv(
    statistics_path,
    index=False
)


# ============================================================
# SAVE CHUNK LENGTH STATISTICS
# ============================================================

length_statistics = pd.DataFrame({

    "Metric": [

        "Original Documents",
        "Total Chunks",
        "Minimum Chunk Length",
        "Maximum Chunk Length",
        "Average Chunk Length",
        "Chunks Over 512",
        "Alignment Errors"

    ],

    "Value": [

        len(records),
        len(all_chunks),
        min(chunk_lengths),
        max(chunk_lengths),
        round(
            float(
                chunk_length_series.mean()
            ),
            2
        ),
        oversized_chunks,
        alignment_errors

    ]

})


length_statistics_path = os.path.join(
    OUTPUT_DIR,
    "chunking_quality_report.csv"
)


length_statistics.to_csv(
    length_statistics_path,
    index=False
)


# ============================================================
# SAVE EXAMPLE
# ============================================================

example_path = os.path.join(
    OUTPUT_DIR,
    "chunk_token_label_preview.csv"
)


if len(all_chunks) > 0:

    example_df.to_csv(
        example_path,
        index=False
    )


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHUNKING QUALITY CHECK")
print("=" * 70)

quality_passed = True


if oversized_chunks != 0:

    quality_passed = False


if alignment_errors != 0:

    quality_passed = False


if len(all_chunks) == 0:

    quality_passed = False


if quality_passed:

    print(
        "\n✓ LONG DOCUMENT CHUNKING PASSED"
    )

else:

    print(
        "\n✗ LONG DOCUMENT CHUNKING FAILED"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL LONG DOCUMENT CHUNKING SUMMARY")
print("=" * 70)

print(
    "\nOriginal documents:",
    len(records)
)

print(
    "Total chunks:",
    len(all_chunks)
)

print(
    "Maximum chunk:",
    max(chunk_lengths)
)

print(
    "Chunks >512:",
    oversized_chunks
)

print(
    "Alignment errors:",
    alignment_errors
)

print(
    "\nFiles created:"
)

print(
    chunks_path
)

print(
    statistics_path
)

print(
    length_statistics_path
)

print(
    example_path
)


print("\n" + "=" * 70)
print("LONG DOCUMENT NER CHUNKING COMPLETED")
print("=" * 70)