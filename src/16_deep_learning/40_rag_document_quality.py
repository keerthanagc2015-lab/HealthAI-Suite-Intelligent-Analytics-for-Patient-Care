"""
HEALTHAI - RAG DOCUMENT QUALITY VALIDATION

Step 38C

Validates normalized healthcare documents before chunking
and embedding.

Checks:
- File existence
- JSON validity
- Required fields
- Metadata completeness
- Text length
- Duplicate documents
- Empty documents
- Basic content quality
"""

import hashlib
import json
import os

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_REGISTRY = (
    "data/processed/rag/documents/"
    "rag_source_registry.json"
)

DOCUMENT_DIR = (
    "data/processed/rag/documents/"
    "normalized"
)

OUTPUT_DIR = (
    "data/processed/rag/documents/"
    "quality"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# REQUIRED FIELDS
# ============================================================

REQUIRED_DOCUMENT_FIELDS = [
    "document_id",
    "title",
    "text",
    "metadata"
]

REQUIRED_METADATA_FIELDS = [
    "source_id",
    "organization",
    "country",
    "topic",
    "source_type",
    "authority",
    "source_url"
]


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("HEALTHAI - RAG DOCUMENT QUALITY VALIDATION")
print("=" * 75)


# ============================================================
# LOAD REGISTRY
# ============================================================

if not os.path.exists(SOURCE_REGISTRY):

    raise FileNotFoundError(
        f"""
Source registry not found:

{os.path.abspath(SOURCE_REGISTRY)}
"""
    )


with open(
    SOURCE_REGISTRY,
    "r",
    encoding="utf-8"
) as f:

    sources = json.load(f)


print()
print(
    f"Registered sources: {len(sources)}"
)


# ============================================================
# VALIDATION STORAGE
# ============================================================

results = []

text_hashes = {}

total_characters = 0

valid_documents = 0
invalid_documents = 0
duplicate_documents = 0
short_documents = 0


# ============================================================
# VALIDATE EACH DOCUMENT
# ============================================================

for source in sources:

    source_id = source["source_id"]

    filename = (
        f"{source_id}.json"
    )

    filepath = os.path.join(
        DOCUMENT_DIR,
        filename
    )

    print()
    print("-" * 75)

    print(
        f"Validating: {source_id}"
    )

    record = {

        "source_id": source_id,

        "file": filename,

        "file_exists": False,

        "valid_json": False,

        "required_fields_present": False,

        "metadata_complete": False,

        "text_non_empty": False,

        "text_length": 0,

        "duplicate_text": False,

        "quality_status": "FAILED",

        "issues": []
    }


    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    if not os.path.exists(filepath):

        record["issues"].append(
            "Document file does not exist."
        )

        results.append(record)

        invalid_documents += 1

        print(
            "✗ File not found"
        )

        continue


    record["file_exists"] = True


    # --------------------------------------------------------
    # JSON LOADING
    # --------------------------------------------------------

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            document = json.load(f)

        record["valid_json"] = True

    except Exception as error:

        record["issues"].append(
            f"Invalid JSON: {error}"
        )

        results.append(record)

        invalid_documents += 1

        print(
            f"✗ Invalid JSON: {error}"
        )

        continue


    # --------------------------------------------------------
    # REQUIRED DOCUMENT FIELDS
    # --------------------------------------------------------

    missing_document_fields = [
        field
        for field in REQUIRED_DOCUMENT_FIELDS
        if field not in document
    ]


    if len(missing_document_fields) == 0:

        record[
            "required_fields_present"
        ] = True

    else:

        record["issues"].append(
            "Missing document fields: "
            +
            ", ".join(
                missing_document_fields
            )
        )


    # --------------------------------------------------------
    # METADATA VALIDATION
    # --------------------------------------------------------

    metadata = document.get(
        "metadata",
        {}
    )


    missing_metadata_fields = [
        field
        for field in REQUIRED_METADATA_FIELDS
        if field not in metadata
    ]


    if len(missing_metadata_fields) == 0:

        record[
            "metadata_complete"
        ] = True

    else:

        record["issues"].append(
            "Missing metadata fields: "
            +
            ", ".join(
                missing_metadata_fields
            )
        )


    # --------------------------------------------------------
    # TEXT VALIDATION
    # --------------------------------------------------------

    text = document.get(
        "text",
        ""
    )


    if isinstance(
        text,
        str
    ):

        text_length = len(
            text.strip()
        )

    else:

        text_length = 0


    record[
        "text_length"
    ] = text_length


    total_characters += text_length


    if text_length > 0:

        record[
            "text_non_empty"
        ] = True

    else:

        record["issues"].append(
            "Document text is empty."
        )


    # --------------------------------------------------------
    # SHORT DOCUMENT CHECK
    # --------------------------------------------------------

    if (
        text_length > 0
        and
        text_length < 500
    ):

        record["issues"].append(
            "Document is unusually short "
            f"({text_length} characters)."
        )

        short_documents += 1


    # --------------------------------------------------------
    # DUPLICATE TEXT CHECK
    # --------------------------------------------------------

    if text_length > 0:

        text_hash = hashlib.sha256(
            text.encode(
                "utf-8"
            )
        ).hexdigest()


        if text_hash in text_hashes:

            record[
                "duplicate_text"
            ] = True

            record["issues"].append(
                "Duplicate document text detected. "
                f"Same content as {text_hashes[text_hash]}."
            )

            duplicate_documents += 1

        else:

            text_hashes[
                text_hash
            ] = source_id


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    critical_checks = [

        record["file_exists"],

        record["valid_json"],

        record["required_fields_present"],

        record["metadata_complete"],

        record["text_non_empty"]
    ]


    if all(critical_checks):

        record[
            "quality_status"
        ] = "PASSED"

        valid_documents += 1

        print(
            "✓ Quality checks passed"
        )

    else:

        invalid_documents += 1

        print(
            "✗ Quality checks failed"
        )


    print(
        f"  Text length: "
        f"{text_length:,} characters"
    )

    print(
        f"  Topic: "
        f"{metadata.get('topic', 'N/A')}"
    )

    print(
        f"  Organization: "
        f"{metadata.get('organization', 'N/A')}"
    )


    results.append(record)


# ============================================================
# CREATE QUALITY DATAFRAME
# ============================================================

quality_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE QUALITY REPORT
# ============================================================

quality_csv = os.path.join(
    OUTPUT_DIR,
    "rag_document_quality_report.csv"
)

quality_df.to_csv(
    quality_csv,
    index=False
)


print()
print("=" * 75)
print("QUALITY SUMMARY")
print("=" * 75)

print()

print(
    f"Registered sources       : "
    f"{len(sources)}"
)

print(
    f"Valid documents          : "
    f"{valid_documents}"
)

print(
    f"Invalid documents        : "
    f"{invalid_documents}"
)

print(
    f"Duplicate documents     : "
    f"{duplicate_documents}"
)

print(
    f"Short documents (<500)  : "
    f"{short_documents}"
)

print(
    f"Total extracted chars   : "
    f"{total_characters:,}"
)


# ============================================================
# TOPIC SUMMARY
# ============================================================

topic_rows = []

for source in sources:

    source_id = source[
        "source_id"
    ]

    matching = quality_df[
        quality_df[
            "source_id"
        ]
        ==
        source_id
    ]

    if len(matching) == 0:
        continue

    row = matching.iloc[0]

    topic_rows.append({

        "source_id": source_id,

        "topic": source[
            "topic"
        ],

        "organization": source[
            "organization"
        ],

        "country": source[
            "country"
        ],

        "characters": int(
            row["text_length"]
        ),

        "status": row[
            "quality_status"
        ]
    })


topic_df = pd.DataFrame(
    topic_rows
)


topic_csv = os.path.join(
    OUTPUT_DIR,
    "rag_document_topic_summary.csv"
)


topic_df.to_csv(
    topic_csv,
    index=False
)


# ============================================================
# TEXT PREVIEW
# ============================================================

preview_rows = []


for source in sources:

    source_id = source[
        "source_id"
    ]

    filepath = os.path.join(
        DOCUMENT_DIR,
        f"{source_id}.json"
    )

    if not os.path.exists(
        filepath
    ):
        continue


    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            document = json.load(f)


        text = document.get(
            "text",
            ""
        )


        preview_rows.append({

            "source_id": source_id,

            "topic": source[
                "topic"
            ],

            "preview": text[:500]
        })


    except Exception:
        pass


preview_df = pd.DataFrame(
    preview_rows
)


preview_csv = os.path.join(
    OUTPUT_DIR,
    "rag_document_text_preview.csv"
)


preview_df.to_csv(
    preview_csv,
    index=False
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary = {

    "registered_sources": len(
        sources
    ),

    "valid_documents": valid_documents,

    "invalid_documents": invalid_documents,

    "duplicate_documents": (
        duplicate_documents
    ),

    "short_documents": (
        short_documents
    ),

    "total_extracted_characters": (
        total_characters
    ),

    "ready_for_chunking": (
        valid_documents == len(sources)
        and duplicate_documents == 0
    )
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "rag_document_quality_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 75)

if summary["ready_for_chunking"]:

    print(
        "STEP 38C - QUALITY VALIDATION PASSED"
    )

    print()
    print(
        "All registered documents are ready "
        "for chunking."
    )

else:

    print(
        "STEP 38C - QUALITY VALIDATION "
        "REQUIRES REVIEW"
    )

    print()
    print(
        "Review the quality report before "
        "continuing."
    )


print("=" * 75)

print()
print("Saved reports:")

print(
    os.path.abspath(
        quality_csv
    )
)

print(
    os.path.abspath(
        topic_csv
    )
)

print(
    os.path.abspath(
        preview_csv
    )
)

print(
    os.path.abspath(
        summary_path
    )
)

print()
print("Next:")
print(
    "38D - RAG Document Chunking"
)