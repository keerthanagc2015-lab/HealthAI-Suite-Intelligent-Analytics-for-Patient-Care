"""
HEALTHAI - RAG DOCUMENT CHUNKING

Step 38D

Splits validated healthcare documents into overlapping chunks
while preserving source metadata.

Chunk configuration:
- Chunk size: 800 characters
- Overlap: 150 characters

Output:
- Individual chunk JSON files
- Combined chunk dataset
- Chunk statistics
- Chunk preview
"""

import json
import os
import re

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = (
    "data/processed/rag/documents/"
    "normalized"
)

QUALITY_SUMMARY = (
    "data/processed/rag/documents/"
    "quality/rag_document_quality_summary.json"
)

OUTPUT_DIR = (
    "data/processed/rag/chunks"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("HEALTHAI - RAG DOCUMENT CHUNKING")
print("=" * 75)


# ============================================================
# VALIDATE QUALITY STATUS
# ============================================================

if not os.path.exists(
    QUALITY_SUMMARY
):

    raise FileNotFoundError(
        f"""
Quality summary not found:

{os.path.abspath(QUALITY_SUMMARY)}

Run Step 38C first.
"""
    )


with open(
    QUALITY_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    quality_summary = json.load(f)


if not quality_summary.get(
    "ready_for_chunking",
    False
):

    raise RuntimeError(
        """
RAG documents are not marked as ready for chunking.

Review Step 38C quality validation first.
"""
    )


print()
print(
    "✓ Document quality validation confirmed."
)


# ============================================================
# CHUNKING FUNCTION
# ============================================================

def normalize_text(text):
    """
    Normalize whitespace before chunking.
    """

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Create overlapping character-based chunks.

    Example:

    chunk 1: characters 0-800
    chunk 2: characters 650-1450
    chunk 3: characters 1300-2100

    Therefore each adjacent chunk shares 150 characters.
    """

    if overlap >= chunk_size:

        raise ValueError(
            "Chunk overlap must be smaller "
            "than chunk size."
        )


    text = normalize_text(
        text
    )


    if not text:

        return []


    chunks = []

    start = 0

    text_length = len(text)


    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk_text = text[
            start:end
        ].strip()


        if chunk_text:

            chunks.append(
                {
                    "text": chunk_text,

                    "start_char": start,

                    "end_char": end
                }
            )


        if end >= text_length:

            break


        start = (
            end - overlap
        )


    return chunks


# ============================================================
# LOAD DOCUMENTS
# ============================================================

document_files = sorted(
    [
        filename
        for filename in os.listdir(
            INPUT_DIR
        )
        if filename.endswith(".json")
        and filename != "rag_ingestion_report.json"
    ]
)


print()
print(
    f"Documents discovered: "
    f"{len(document_files)}"
)

print(
    f"Chunk size: "
    f"{CHUNK_SIZE} characters"
)

print(
    f"Chunk overlap: "
    f"{CHUNK_OVERLAP} characters"
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

all_chunks = []

document_statistics = []

total_documents = 0
total_chunks = 0


for filename in document_files:

    filepath = os.path.join(
        INPUT_DIR,
        filename
    )


    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        document = json.load(f)


    document_id = document[
        "document_id"
    ]

    text = document[
        "text"
    ]

    metadata = document[
        "metadata"
    ]


    chunks = create_chunks(
        text
    )


    total_documents += 1
    total_chunks += len(
        chunks
    )


    print()
    print("-" * 75)

    print(
        f"Document: {document_id}"
    )

    print(
        f"Original characters: "
        f"{len(text):,}"
    )

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )


    document_chunk_ids = []


    # --------------------------------------------------------
    # CREATE CHUNK RECORDS
    # --------------------------------------------------------

    for chunk_index, chunk in enumerate(
        chunks
    ):

        chunk_id = (
            f"{document_id}_"
            f"chunk_{chunk_index:04d}"
        )


        chunk_record = {

            "chunk_id": chunk_id,

            "document_id": document_id,

            "chunk_index": chunk_index,

            "text": chunk["text"],

            "start_char": chunk[
                "start_char"
            ],

            "end_char": chunk[
                "end_char"
            ],

            "chunk_size": len(
                chunk["text"]
            ),

            "metadata": {

                "source_id": metadata[
                    "source_id"
                ],

                "organization": metadata[
                    "organization"
                ],

                "country": metadata[
                    "country"
                ],

                "topic": metadata[
                    "topic"
                ],

                "source_type": metadata[
                    "source_type"
                ],

                "authority": metadata[
                    "authority"
                ],

                "source_url": metadata[
                    "source_url"
                ]
            }
        }


        all_chunks.append(
            chunk_record
        )


        document_chunk_ids.append(
            chunk_id
        )


    # --------------------------------------------------------
    # DOCUMENT STATISTICS
    # --------------------------------------------------------

    chunk_lengths = [
        len(chunk["text"])
        for chunk in chunks
    ]


    document_statistics.append(
        {
            "document_id": document_id,

            "topic": metadata[
                "topic"
            ],

            "organization": metadata[
                "organization"
            ],

            "original_characters": len(
                text
            ),

            "num_chunks": len(
                chunks
            ),

            "min_chunk_size": (
                min(chunk_lengths)
                if chunk_lengths
                else 0
            ),

            "max_chunk_size": (
                max(chunk_lengths)
                if chunk_lengths
                else 0
            ),

            "mean_chunk_size": (
                sum(chunk_lengths)
                / len(chunk_lengths)
                if chunk_lengths
                else 0
            )
        }
    )


# ============================================================
# SAVE COMBINED CHUNKS
# ============================================================

combined_json = os.path.join(
    OUTPUT_DIR,
    "all_rag_chunks.json"
)


with open(
    combined_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print("=" * 75)

print(
    "✓ Combined chunk dataset saved:"
)

print(
    os.path.abspath(
        combined_json
    )
)


# ============================================================
# SAVE INDIVIDUAL CHUNK FILES
# ============================================================

individual_dir = os.path.join(
    OUTPUT_DIR,
    "individual"
)

os.makedirs(
    individual_dir,
    exist_ok=True
)


for chunk in all_chunks:

    chunk_path = os.path.join(
        individual_dir,
        f"{chunk['chunk_id']}.json"
    )


    with open(
        chunk_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunk,
            f,
            indent=4,
            ensure_ascii=False
        )


print()
print(
    f"✓ Individual chunk files saved: "
    f"{len(all_chunks)}"
)


# ============================================================
# CHUNK STATISTICS
# ============================================================

chunk_sizes = [
    chunk["chunk_size"]
    for chunk in all_chunks
]


statistics = {

    "num_documents": total_documents,

    "num_chunks": total_chunks,

    "chunk_size": CHUNK_SIZE,

    "chunk_overlap": CHUNK_OVERLAP,

    "total_characters": sum(
        chunk_sizes
    ),

    "min_chunk_size": (
        min(chunk_sizes)
        if chunk_sizes
        else 0
    ),

    "max_chunk_size": (
        max(chunk_sizes)
        if chunk_sizes
        else 0
    ),

    "mean_chunk_size": (
        sum(chunk_sizes)
        / len(chunk_sizes)
        if chunk_sizes
        else 0
    )
}


statistics_path = os.path.join(
    OUTPUT_DIR,
    "rag_chunk_statistics.json"
)


with open(
    statistics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )


# ============================================================
# SAVE DOCUMENT STATISTICS
# ============================================================

document_stats_df = pd.DataFrame(
    document_statistics
)


document_stats_path = os.path.join(
    OUTPUT_DIR,
    "rag_document_chunk_statistics.csv"
)


document_stats_df.to_csv(
    document_stats_path,
    index=False
)


# ============================================================
# CHUNK PREVIEW
# ============================================================

preview_rows = []


for chunk in all_chunks[:20]:

    preview_rows.append(
        {
            "chunk_id": chunk[
                "chunk_id"
            ],

            "document_id": chunk[
                "document_id"
            ],

            "topic": chunk[
                "metadata"
            ]["topic"],

            "chunk_size": chunk[
                "chunk_size"
            ],

            "text_preview": chunk[
                "text"
            ][:300]
        }
    )


preview_df = pd.DataFrame(
    preview_rows
)


preview_path = os.path.join(
    OUTPUT_DIR,
    "rag_chunk_preview.csv"
)


preview_df.to_csv(
    preview_path,
    index=False
)


# ============================================================
# VALIDATION CHECKS
# ============================================================

empty_chunks = sum(
    1
    for chunk in all_chunks
    if not chunk["text"].strip()
)


oversized_chunks = sum(
    1
    for chunk in all_chunks
    if chunk["chunk_size"]
    > CHUNK_SIZE
)


missing_metadata = sum(
    1
    for chunk in all_chunks
    if not chunk["metadata"].get(
        "source_id"
    )
)


unique_chunk_ids = len(
    set(
        chunk["chunk_id"]
        for chunk in all_chunks
    )
)


duplicate_chunk_ids = (
    total_chunks
    -
    unique_chunk_ids
)


# ============================================================
# FINAL REPORT
# ============================================================

validation = {

    "total_documents": total_documents,

    "total_chunks": total_chunks,

    "empty_chunks": empty_chunks,

    "oversized_chunks": oversized_chunks,

    "missing_metadata": missing_metadata,

    "duplicate_chunk_ids": (
        duplicate_chunk_ids
    ),

    "chunk_size": CHUNK_SIZE,

    "chunk_overlap": CHUNK_OVERLAP,

    "ready_for_embedding": (
        total_chunks > 0
        and empty_chunks == 0
        and oversized_chunks == 0
        and missing_metadata == 0
        and duplicate_chunk_ids == 0
    )
}


validation_path = os.path.join(
    OUTPUT_DIR,
    "rag_chunk_validation.json"
)


with open(
    validation_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        validation,
        f,
        indent=4
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print()
print("=" * 75)
print("CHUNKING SUMMARY")
print("=" * 75)

print()

print(
    f"Documents processed       : "
    f"{total_documents}"
)

print(
    f"Total chunks              : "
    f"{total_chunks}"
)

print(
    f"Chunk size                : "
    f"{CHUNK_SIZE}"
)

print(
    f"Chunk overlap             : "
    f"{CHUNK_OVERLAP}"
)

print(
    f"Minimum chunk size        : "
    f"{statistics['min_chunk_size']}"
)

print(
    f"Maximum chunk size        : "
    f"{statistics['max_chunk_size']}"
)

print(
    f"Mean chunk size           : "
    f"{statistics['mean_chunk_size']:.2f}"
)


print()
print("=" * 75)
print("VALIDATION")
print("=" * 75)

print()

print(
    f"Empty chunks              : "
    f"{empty_chunks}"
)

print(
    f"Oversized chunks          : "
    f"{oversized_chunks}"
)

print(
    f"Missing metadata          : "
    f"{missing_metadata}"
)

print(
    f"Duplicate chunk IDs       : "
    f"{duplicate_chunk_ids}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 75)

if validation["ready_for_embedding"]:

    print(
        "STEP 38D - CHUNKING COMPLETED"
    )

    print()
    print(
        "✓ All chunks passed validation."
    )

    print()
    print(
        "✓ RAG corpus is ready for embeddings."
    )

else:

    print(
        "STEP 38D - CHUNKING REQUIRES REVIEW"
    )

print("=" * 75)


print()
print("Saved files:")

print(
    os.path.abspath(
        combined_json
    )
)

print(
    os.path.abspath(
        statistics_path
    )
)

print(
    os.path.abspath(
        document_stats_path
    )
)

print(
    os.path.abspath(
        preview_path
    )
)

print(
    os.path.abspath(
        validation_path
    )
)


print()
print("Next:")
print(
    "38E - RAG Embedding Generation"
)