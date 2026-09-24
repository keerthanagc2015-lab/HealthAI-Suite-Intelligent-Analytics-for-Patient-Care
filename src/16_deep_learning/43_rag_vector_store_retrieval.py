"""
HealthAI - RAG Vector Store + Semantic Retrieval

Step 38F:
Build a lightweight local vector store over the RAG embeddings
and perform semantic retrieval using cosine similarity.

Input:
    data/processed/rag/embeddings/rag_embeddings.npy
    data/processed/rag/embeddings/rag_embedding_metadata.json

Output:
    data/processed/rag/vector_store/
        rag_vector_store.json
        rag_retrieval_test_results.json
        rag_retrieval_evaluation.csv
        rag_vector_store_validation.json
"""

from pathlib import Path
import json
import csv

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "embeddings"
)

VECTOR_STORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "vector_store"
)

EMBEDDING_FILE = (
    EMBEDDING_DIR
    / "rag_embeddings.npy"
)

METADATA_FILE = (
    EMBEDDING_DIR
    / "rag_embedding_metadata.json"
)

VECTOR_STORE_FILE = (
    VECTOR_STORE_DIR
    / "rag_vector_store.json"
)

TEST_RESULTS_FILE = (
    VECTOR_STORE_DIR
    / "rag_retrieval_test_results.json"
)

EVALUATION_FILE = (
    VECTOR_STORE_DIR
    / "rag_retrieval_evaluation.csv"
)

VALIDATION_FILE = (
    VECTOR_STORE_DIR
    / "rag_vector_store_validation.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [
    {
        "query": "What are the symptoms of diabetes?",
        "expected_topics": ["Diabetes"],
    },
    {
        "query": "How can diabetes be managed and controlled?",
        "expected_topics": [
            "Diabetes",
            "Type 2 Diabetes Self-Care",
        ],
    },
    {
        "query": "What is high blood pressure and how can it be prevented?",
        "expected_topics": ["Hypertension"],
    },
    {
        "query": "What can I do to manage high blood pressure?",
        "expected_topics": ["Hypertension"],
    },
    {
        "query": "What are noncommunicable diseases?",
        "expected_topics": [
            "Noncommunicable Diseases",
        ],
    },
    {
        "query": "How can someone live with a chronic illness?",
        "expected_topics": [
            "Living With Chronic Illness",
        ],
    },
]


# ============================================================
# HELPERS
# ============================================================

def save_json(data, path):
    """Save Python object as formatted JSON."""

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def cosine_similarity(
    query_vector,
    document_vectors
):
    """
    Calculate cosine similarity between one query vector
    and all document vectors.

    Because our embeddings are normalized, cosine similarity
    is effectively the dot product.
    """

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:
        raise ValueError(
            "Query embedding is a zero vector."
        )

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    if np.any(document_norms == 0):
        raise ValueError(
            "Vector store contains zero vectors."
        )

    query_vector = (
        query_vector / query_norm
    )

    document_vectors = (
        document_vectors
        / document_norms[:, None]
    )

    return np.dot(
        document_vectors,
        query_vector
    )


def retrieve(
    query,
    model,
    embeddings,
    metadata,
    top_k=3
):
    """
    Convert the user query into an embedding
    and retrieve the top-k most similar chunks.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )

    top_k = min(
        top_k,
        len(similarities)
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        item = metadata[int(index)]

        results.append(
            {
                "rank": rank,

                "embedding_index": int(
                    index
                ),

                "similarity_score": float(
                    similarities[index]
                ),

                "chunk_id": item[
                    "chunk_id"
                ],

                "document_id": item[
                    "document_id"
                ],

                "chunk_index": item[
                    "chunk_index"
                ],

                "topic": item[
                    "topic"
                ],

                "organization": item[
                    "organization"
                ],

                "authority": item[
                    "authority"
                ],

                "source_type": item[
                    "source_type"
                ],

                "source_url": item[
                    "source_url"
                ],

                "text": item[
                    "text"
                ],
            }
        )

    return results


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - RAG VECTOR STORE + SEMANTIC RETRIEVAL")
print("=" * 75)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not EMBEDDING_FILE.exists():

    raise FileNotFoundError(
        f"Embedding file not found:\n"
        f"{EMBEDDING_FILE}"
    )


if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n"
        f"{METADATA_FILE}"
    )


VECTOR_STORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"✓ Embeddings loaded: "
    f"{embeddings.shape}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading embedding metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata_payload = json.load(f)


metadata = metadata_payload[
    "metadata"
]

print(
    f"✓ Metadata records loaded: "
    f"{len(metadata)}"
)


# ============================================================
# VALIDATE VECTOR STORE
# ============================================================

print("\nValidating vector store...")


# Number of embeddings must match
# number of metadata records.

if len(embeddings) != len(metadata):

    raise ValueError(
        "Embedding count does not match "
        "metadata count."
    )


# Embeddings must be a 2D matrix.

if embeddings.ndim != 2:

    raise ValueError(
        "Embeddings must be a "
        "2-dimensional matrix."
    )


# No invalid numerical values.

if not np.isfinite(
    embeddings
).all():

    raise ValueError(
        "Embeddings contain NaN "
        "or infinite values."
    )


# Embeddings must not all be zero.

if np.allclose(
    embeddings,
    0
):

    raise ValueError(
        "Embeddings are all zero."
    )


# Check unique chunk IDs.

chunk_ids = [
    item["chunk_id"]
    for item in metadata
]

unique_chunk_ids = len(
    set(chunk_ids)
)

duplicate_chunk_ids = (
    len(chunk_ids)
    - unique_chunk_ids
)


if duplicate_chunk_ids != 0:

    raise ValueError(
        "Duplicate chunk IDs found."
    )


embedding_dimensions = (
    embeddings.shape[1]
)


print(
    "✓ Count validation passed."
)

print(
    "✓ Shape validation passed."
)

print(
    "✓ Numerical validation passed."
)

print(
    "✓ Chunk ID validation passed."
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

vector_store = {

    "vector_store_type":
        "numpy_dense_vector_store",

    "embedding_model":
        MODEL_NAME,

    "embedding_dimension":
        int(embedding_dimensions),

    "number_of_vectors":
        int(len(embeddings)),

    "similarity_metric":
        "cosine_similarity",

    "normalized_embeddings":
        True,

    "records":
        metadata,
}


save_json(
    vector_store,
    VECTOR_STORE_FILE
)


print(
    "\n✓ Vector store metadata saved:"
)

print(
    VECTOR_STORE_FILE
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "\nLoading embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "✓ Embedding model loaded."
)


# ============================================================
# RUN RETRIEVAL TESTS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "SEMANTIC RETRIEVAL TESTS"
)

print(
    "=" * 75
)


all_test_results = []

evaluation_rows = []


for test_number, test_case in enumerate(
    TEST_QUERIES,
    start=1
):

    query = test_case[
        "query"
    ]

    expected_topics = test_case[
        "expected_topics"
    ]


    results = retrieve(
        query=query,
        model=model,
        embeddings=embeddings,
        metadata=metadata,
        top_k=TOP_K
    )


    retrieved_topics = [
        result["topic"]
        for result in results
    ]


    expected_topic_found = any(
        topic in expected_topics
        for topic in retrieved_topics
    )


    print(
        "\n" + "-" * 75
    )

    print(
        f"Query {test_number}: "
        f"{query}"
    )

    print(
        f"Expected topics: "
        f"{expected_topics}"
    )


    print(
        "\nTop results:"
    )


    for result in results:

        print(
            f"\nRank {result['rank']}"
        )

        print(
            f"Similarity : "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Topic      : "
            f"{result['topic']}"
        )

        print(
            f"Document   : "
            f"{result['document_id']}"
        )

        print(
            f"Chunk      : "
            f"{result['chunk_id']}"
        )


        preview = (
            result["text"]
            .replace("\n", " ")
            .strip()
        )


        if len(preview) > 220:

            preview = (
                preview[:220]
                + "..."
            )


        print(
            f"Text       : "
            f"{preview}"
        )


    print(
        f"\nExpected topic retrieved: "
        f"{expected_topic_found}"
    )


    # IMPORTANT:
    # This key name is used consistently
    # throughout the evaluation section.

    all_test_results.append(
        {
            "query_number":
                test_number,

            "query":
                query,

            "expected_topics":
                expected_topics,

            "retrieved_topics":
                retrieved_topics,

            "expected_topic_found":
                expected_topic_found,

            "results":
                results,
        }
    )


    evaluation_rows.append(
        {
            "query_number":
                test_number,

            "query":
                query,

            "expected_topics":
                "|".join(
                    expected_topics
                ),

            "top_1_topic":
                (
                    results[0]["topic"]
                    if results
                    else ""
                ),

            "top_1_similarity":
                (
                    results[0][
                        "similarity_score"
                    ]
                    if results
                    else 0.0
                ),

            "top_3_topics":
                "|".join(
                    retrieved_topics
                ),

            "expected_topic_found_in_top3":
                expected_topic_found,
        }
    )


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

successful_queries = sum(
    row["expected_topic_found"]
    for row in all_test_results
)


total_queries = len(
    all_test_results
)


if total_queries > 0:

    top3_topic_hit_rate = (
        successful_queries
        / total_queries
    )

else:

    top3_topic_hit_rate = 0.0


top1_similarities = [
    row["results"][0][
        "similarity_score"
    ]
    for row in all_test_results
    if row["results"]
]


if top1_similarities:

    mean_top1_similarity = float(
        np.mean(
            top1_similarities
        )
    )

else:

    mean_top1_similarity = 0.0


# ============================================================
# SAVE RETRIEVAL TEST RESULTS
# ============================================================

test_summary = {

    "embedding_model":
        MODEL_NAME,

    "number_of_vectors":
        int(len(embeddings)),

    "embedding_dimension":
        int(embedding_dimensions),

    "top_k":
        TOP_K,

    "number_of_test_queries":
        total_queries,

    "successful_queries":
        successful_queries,

    "top3_topic_hit_rate":
        top3_topic_hit_rate,

    "mean_top1_similarity":
        mean_top1_similarity,

    "tests":
        all_test_results,
}


save_json(
    test_summary,
    TEST_RESULTS_FILE
)


print(
    "\n✓ Retrieval test results saved:"
)

print(
    TEST_RESULTS_FILE
)


# ============================================================
# SAVE EVALUATION CSV
# ============================================================

with open(
    EVALUATION_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "query_number",
            "query",
            "expected_topics",
            "top_1_topic",
            "top_1_similarity",
            "top_3_topics",
            "expected_topic_found_in_top3",
        ]
    )


    writer.writeheader()


    writer.writerows(
        evaluation_rows
    )


print(
    "✓ Retrieval evaluation saved:"
)

print(
    EVALUATION_FILE
)


# ============================================================
# FINAL VALIDATION
# ============================================================

validation = {

    "vector_store_exists":
        VECTOR_STORE_FILE.exists(),

    "embedding_count":
        int(len(embeddings)),

    "metadata_count":
        int(len(metadata)),

    "embedding_dimension":
        int(embedding_dimensions),

    "duplicate_chunk_ids":
        int(duplicate_chunk_ids),

    "finite_embeddings":
        bool(
            np.isfinite(
                embeddings
            ).all()
        ),

    "non_zero_embeddings":
        bool(
            not np.allclose(
                embeddings,
                0
            )
        ),

    "retrieval_tests":
        total_queries,

    "successful_queries":
        successful_queries,

    "top3_topic_hit_rate":
        top3_topic_hit_rate,

    "ready_for_rag_generation":
        bool(
            len(embeddings)
            == len(metadata)

            and duplicate_chunk_ids
            == 0

            and np.isfinite(
                embeddings
            ).all()

            and not np.allclose(
                embeddings,
                0
            )

            and total_queries > 0
        ),
}


save_json(
    validation,
    VALIDATION_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 38F - VECTOR STORE + "
    "RETRIEVAL COMPLETED"
)

print(
    "=" * 75
)


print(
    f"\nVectors                   : "
    f"{len(embeddings)}"
)


print(
    f"Vector dimension          : "
    f"{embedding_dimensions}"
)


print(
    f"Similarity metric         : "
    f"Cosine similarity"
)


print(
    f"Top-K                     : "
    f"{TOP_K}"
)


print(
    f"Test queries              : "
    f"{total_queries}"
)


print(
    f"Successful queries       : "
    f"{successful_queries}"
)


print(
    f"Top-3 topic hit rate      : "
    f"{top3_topic_hit_rate:.2%}"
)


print(
    f"Mean Top-1 similarity     : "
    f"{mean_top1_similarity:.4f}"
)


print(
    f"Duplicate chunk IDs      : "
    f"{duplicate_chunk_ids}"
)


print(
    f"Ready for RAG generation : "
    f"{validation['ready_for_rag_generation']}"
)


print(
    "\nSaved files:"
)


print(
    VECTOR_STORE_FILE
)

print(
    TEST_RESULTS_FILE
)

print(
    EVALUATION_FILE
)

print(
    VALIDATION_FILE
)


print(
    "\n✓ Semantic retrieval is operational."
)

print(
    "✓ Ready for the next RAG generation stage."
)