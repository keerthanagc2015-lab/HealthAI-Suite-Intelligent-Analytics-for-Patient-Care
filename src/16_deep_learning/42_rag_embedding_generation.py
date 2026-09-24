"""
HealthAI - RAG Embedding Generation

Step 38E:
Generate dense semantic embeddings for the validated RAG chunks.

Input:
    data/processed/rag/chunks/all_rag_chunks.json

Output:
    data/processed/rag/embeddings/
        rag_embeddings.npy
        rag_embedding_metadata.json
        rag_embedding_statistics.json
        rag_embedding_validation.json
"""

from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "chunks"
    / "all_rag_chunks.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "embeddings"
)

EMBEDDING_FILE = OUTPUT_DIR / "rag_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "rag_embedding_metadata.json"
STATISTICS_FILE = OUTPUT_DIR / "rag_embedding_statistics.json"
VALIDATION_FILE = OUTPUT_DIR / "rag_embedding_validation.json"


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 16

NORMALIZE_EMBEDDINGS = True


# ============================================================
# HELPERS
# ============================================================

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - RAG EMBEDDING GENERATION")
print("=" * 75)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"RAG chunk file not found:\n{INPUT_FILE}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CHUNKS
# ============================================================

print("\nLoading RAG chunks...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

if not isinstance(chunks, list):
    raise ValueError(
        "Expected all_rag_chunks.json to contain a list."
    )

print(f"✓ Chunks loaded: {len(chunks)}")


# ============================================================
# VALIDATE AND PREPARE CHUNKS
# ============================================================

texts = []
metadata = []

required_chunk_fields = [
    "chunk_id",
    "document_id",
    "chunk_index",
    "text",
    "start_char",
    "end_char",
    "chunk_size",
    "metadata",
]

required_metadata_fields = [
    "source_id",
    "organization",
    "country",
    "topic",
    "source_type",
    "authority",
    "source_url",
]


for index, chunk in enumerate(chunks):

    # --------------------------------------------------------
    # Validate top-level chunk fields
    # --------------------------------------------------------

    for field in required_chunk_fields:
        if field not in chunk:
            raise ValueError(
                f"Chunk {index} is missing field: {field}"
            )

    # --------------------------------------------------------
    # Validate text
    # --------------------------------------------------------

    text = str(chunk["text"]).strip()

    if not text:
        raise ValueError(
            f"Chunk {index} contains empty text."
        )

    # --------------------------------------------------------
    # Validate nested metadata
    # --------------------------------------------------------

    chunk_metadata = chunk["metadata"]

    if not isinstance(chunk_metadata, dict):
        raise ValueError(
            f"Chunk {index} metadata is not a dictionary."
        )

    for field in required_metadata_fields:
        if field not in chunk_metadata:
            raise ValueError(
                f"Chunk {index} metadata is missing: {field}"
            )

    texts.append(text)

    # --------------------------------------------------------
    # Preserve complete retrieval metadata
    # --------------------------------------------------------

    metadata.append(
        {
            "embedding_index": index,

            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],

            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
            "chunk_size": chunk["chunk_size"],

            "source_id": chunk_metadata["source_id"],
            "organization": chunk_metadata["organization"],
            "country": chunk_metadata["country"],
            "topic": chunk_metadata["topic"],
            "source_type": chunk_metadata["source_type"],
            "authority": chunk_metadata["authority"],
            "source_url": chunk_metadata["source_url"],

            "text_length": len(text),
            "text": text,
        }
    )


print("✓ Chunk structure validation passed.")
print(f"✓ Texts prepared: {len(texts)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")
print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

embedding_dimension = model.get_sentence_embedding_dimension()

print("✓ Embedding model loaded.")
print(f"Embedding dimension: {embedding_dimension}")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Normalize embeddings: {NORMALIZE_EMBEDDINGS}")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=NORMALIZE_EMBEDDINGS,
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ============================================================
# SHAPE VALIDATION
# ============================================================

print("\nEmbedding shape:")
print(embeddings.shape)

expected_shape = (
    len(texts),
    embedding_dimension
)

if embeddings.shape != expected_shape:
    raise ValueError(
        f"Unexpected embedding shape.\n"
        f"Expected: {expected_shape}\n"
        f"Actual: {embeddings.shape}"
    )

print("✓ Embedding shape validated.")


# ============================================================
# NUMERICAL VALIDATION
# ============================================================

if not np.isfinite(embeddings).all():
    raise ValueError(
        "Embeddings contain NaN or infinite values."
    )

if np.allclose(embeddings, 0):
    raise ValueError(
        "All embeddings are zero vectors."
    )

print("✓ No NaN or infinite values.")
print("✓ Embeddings are non-zero.")


# ============================================================
# NORMALIZATION VALIDATION
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1
)

if NORMALIZE_EMBEDDINGS:

    normalization_error = float(
        np.max(np.abs(norms - 1.0))
    )

    print(
        f"Maximum normalization error: "
        f"{normalization_error:.8f}"
    )

    if normalization_error > 1e-4:
        raise ValueError(
            "Some embeddings are not unit-normalized."
        )

    print("✓ Embeddings are L2-normalized.")


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    EMBEDDING_FILE,
    embeddings
)

print("\n✓ Embeddings saved:")
print(EMBEDDING_FILE)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_payload = {
    "embedding_model": MODEL_NAME,
    "embedding_dimension": embedding_dimension,
    "normalized": NORMALIZE_EMBEDDINGS,
    "number_of_embeddings": len(embeddings),
    "metadata": metadata,
}

save_json(
    metadata_payload,
    METADATA_FILE
)

print("✓ Embedding metadata saved:")
print(METADATA_FILE)


# ============================================================
# STATISTICS
# ============================================================

statistics = {
    "embedding_model": MODEL_NAME,
    "number_of_embeddings": int(len(embeddings)),
    "embedding_dimension": int(embedding_dimension),
    "batch_size": BATCH_SIZE,
    "normalized": NORMALIZE_EMBEDDINGS,
    "embedding_dtype": str(embeddings.dtype),

    "minimum_value": float(np.min(embeddings)),
    "maximum_value": float(np.max(embeddings)),
    "mean_value": float(np.mean(embeddings)),
    "std_value": float(np.std(embeddings)),

    "mean_vector_norm": float(np.mean(norms)),
    "minimum_vector_norm": float(np.min(norms)),
    "maximum_vector_norm": float(np.max(norms)),
}

save_json(
    statistics,
    STATISTICS_FILE
)

print("✓ Embedding statistics saved:")
print(STATISTICS_FILE)


# ============================================================
# VALIDATION REPORT
# ============================================================

unique_chunk_ids = len(
    set(
        item["chunk_id"]
        for item in metadata
    )
)

duplicate_chunk_ids = (
    len(chunks) - unique_chunk_ids
)

validation = {
    "input_chunks": len(chunks),

    "generated_embeddings": int(
        len(embeddings)
    ),

    "embedding_dimension": int(
        embedding_dimension
    ),

    "embedding_shape": list(
        embeddings.shape
    ),

    "shape_valid": bool(
        embeddings.shape == expected_shape
    ),

    "finite_values": bool(
        np.isfinite(embeddings).all()
    ),

    "non_zero_embeddings": bool(
        not np.allclose(embeddings, 0)
    ),

    "unique_chunk_ids": unique_chunk_ids,

    "duplicate_chunk_ids": duplicate_chunk_ids,

    "normalized": NORMALIZE_EMBEDDINGS,

    "metadata_count_matches": bool(
        len(metadata) == len(embeddings)
    ),

    "ready_for_vector_store": bool(
        len(chunks) == len(embeddings)
        and embeddings.shape == expected_shape
        and np.isfinite(embeddings).all()
        and not np.allclose(embeddings, 0)
        and unique_chunk_ids == len(chunks)
        and len(metadata) == len(embeddings)
    ),
}

save_json(
    validation,
    VALIDATION_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("STEP 38E - EMBEDDING GENERATION COMPLETED")
print("=" * 75)

print(
    f"\nEmbedding model          : {MODEL_NAME}"
)

print(
    f"Chunks                   : {len(chunks)}"
)

print(
    f"Embeddings               : {len(embeddings)}"
)

print(
    f"Embedding dimension      : "
    f"{embedding_dimension}"
)

print(
    f"Embedding shape          : "
    f"{embeddings.shape}"
)

print(
    f"Normalized               : "
    f"{NORMALIZE_EMBEDDINGS}"
)

print(
    f"Duplicate chunk IDs      : "
    f"{duplicate_chunk_ids}"
)

print(
    f"Ready for vector store   : "
    f"{validation['ready_for_vector_store']}"
)

print("\nSaved files:")

print(EMBEDDING_FILE)
print(METADATA_FILE)
print(STATISTICS_FILE)
print(VALIDATION_FILE)

print(
    "\n✓ RAG chunks have been converted "
    "into semantic vectors."
)

print(
    "✓ Ready for Step 38F - "
    "Vector Store + Semantic Retrieval."
)