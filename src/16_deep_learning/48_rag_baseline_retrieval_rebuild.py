"""
HealthAI - Baseline RAG Retrieval Rebuild

Purpose:
Recreate the Step 38F baseline retrieval results
using the existing vector store.

This does NOT rebuild embeddings.

It simply performs pure cosine-similarity retrieval
and saves the baseline results needed for comparison
with Step 38I.

Input:
    data/processed/rag/embeddings/rag_embeddings.npy
    data/processed/rag/embeddings/rag_embedding_metadata.json

Output:
    data/processed/rag/retrieval/
        rag_baseline_retrieval_results.json
        rag_baseline_retrieval_evaluation.csv
"""


from pathlib import Path
import json
import csv

import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "embeddings"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
)

EMBEDDING_FILE = (
    EMBEDDING_DIR
    / "rag_embeddings.npy"
)

METADATA_FILE = (
    EMBEDDING_DIR
    / "rag_embedding_metadata.json"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "rag_baseline_retrieval_results.json"
)

EVALUATION_FILE = (
    OUTPUT_DIR
    / "rag_baseline_retrieval_evaluation.csv"
)


# ============================================================
# MODEL
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 3


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    {
        "question":
            "What are the symptoms of diabetes?",

        "expected_topics":
            [
                "Diabetes",
                "Type 2 Diabetes Self-Care"
            ],

        "answerable":
            True,
    },

    {
        "question":
            "How can diabetes be managed?",

        "expected_topics":
            [
                "Diabetes",
                "Type 2 Diabetes Self-Care"
            ],

        "answerable":
            True,
    },

    {
        "question":
            "What is high blood pressure?",

        "expected_topics":
            [
                "Hypertension"
            ],

        "answerable":
            True,
    },

    {
        "question":
            "How can high blood pressure be prevented?",

        "expected_topics":
            [
                "Hypertension"
            ],

        "answerable":
            True,
    },

    {
        "question":
            "What are noncommunicable diseases?",

        "expected_topics":
            [
                "Noncommunicable Diseases"
            ],

        "answerable":
            True,
    },

    {
        "question":
            "How can someone cope with a chronic illness?",

        "expected_topics":
            [
                "Living With Chronic Illness"
            ],

        "answerable":
            True,
    },

    {
        "question":
            "What is the recommended treatment for appendicitis?",

        "expected_topics":
            [],

        "answerable":
            False,
    },
]


# ============================================================
# HELPERS
# ============================================================

def save_json(
    data,
    path
):

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

    query_norm = np.linalg.norm(
        query_vector
    )

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    if query_norm == 0:

        raise ValueError(
            "Query embedding is zero."
        )

    if np.any(
        document_norms == 0
    ):

        raise ValueError(
            "Vector store contains zero vectors."
        )

    query_vector = (
        query_vector
        / query_norm
    )

    document_vectors = (
        document_vectors
        / document_norms[:, None]
    )

    return np.dot(
        document_vectors,
        query_vector
    )


# ============================================================
# START
# ============================================================

print("=" * 75)

print(
    "HEALTHAI - BASELINE RAG RETRIEVAL REBUILD"
)

print("=" * 75)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print(
    "\nLoading embeddings..."
)

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

print(
    "\nLoading metadata..."
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    payload = json.load(f)


metadata = payload[
    "metadata"
]

print(
    f"✓ Metadata loaded: "
    f"{len(metadata)}"
)


# ============================================================
# VALIDATE
# ============================================================

if len(embeddings) != len(metadata):

    raise ValueError(
        "Embedding count does not match "
        "metadata count."
    )


if not np.isfinite(
    embeddings
).all():

    raise ValueError(
        "Embeddings contain NaN or infinity."
    )


print(
    "✓ Vector store validation passed."
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading embedding model..."
)

embedding_model = (
    SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )
)

print(
    "✓ Embedding model loaded."
)


# ============================================================
# RUN BASELINE
# ============================================================

results = []

evaluation_rows = []


print(
    "\n" + "=" * 75
)

print(
    "BASELINE SEMANTIC RETRIEVAL"
)

print(
    "=" * 75
)


for number, test_case in enumerate(
    TEST_QUESTIONS,
    start=1
):

    question = test_case[
        "question"
    ]

    expected_topics = test_case[
        "expected_topics"
    ]

    answerable = test_case[
        "answerable"
    ]


    print(
        "\n" + "-" * 75
    )

    print(
        f"Question {number}: "
        f"{question}"
    )


    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )


    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    indices = np.argsort(
        scores
    )[::-1][:TOP_K]


    retrieved_chunks = []


    for rank, index in enumerate(
        indices,
        start=1
    ):

        item = metadata[
            int(index)
        ]

        retrieved_chunks.append(
            {
                "rank":
                    rank,

                "embedding_index":
                    int(index),

                "chunk_id":
                    item["chunk_id"],

                "document_id":
                    item["document_id"],

                "topic":
                    item["topic"],

                "organization":
                    item["organization"],

                "authority":
                    item["authority"],

                "source_type":
                    item["source_type"],

                "source_url":
                    item["source_url"],

                "text":
                    item["text"],

                "similarity_score":
                    float(
                        scores[index]
                    ),
            }
        )


    retrieved_topics = [
        chunk["topic"]
        for chunk in retrieved_chunks
    ]


    expected_topic_found = any(
        topic in expected_topics
        for topic in retrieved_topics
    )


    top_similarity = (
        retrieved_chunks[0]
        ["similarity_score"]
        if retrieved_chunks
        else 0.0
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nTop similarity: "
        f"{top_similarity:.4f}"
    )

    print(
        f"Retrieved topics: "
        f"{retrieved_topics}"
    )

    print(
        f"Expected topic found: "
        f"{expected_topic_found}"
    )


    for chunk in retrieved_chunks:

        print(
            f"  Rank {chunk['rank']}: "
            f"{chunk['topic']} | "
            f"{chunk['similarity_score']:.4f}"
        )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "question_number":
                number,

            "question":
                question,

            "expected_topics":
                expected_topics,

            "answerable":
                answerable,

            "top_similarity":
                top_similarity,

            "retrieved_topics":
                retrieved_topics,

            "expected_topic_found":
                expected_topic_found,

            "retrieved_chunks":
                retrieved_chunks,
        }
    )


    evaluation_rows.append(
        {
            "question_number":
                number,

            "question":
                question,

            "answerable":
                answerable,

            "expected_topics":
                "|".join(
                    expected_topics
                ),

            "top_similarity":
                top_similarity,

            "top_1_topic":
                (
                    retrieved_chunks[0]["topic"]
                    if retrieved_chunks
                    else ""
                ),

            "top_3_topics":
                "|".join(
                    retrieved_topics
                ),

            "expected_topic_found":
                expected_topic_found,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

answerable_results = [
    result
    for result in results
    if result["answerable"]
]


top3_hits = sum(
    result["expected_topic_found"]
    for result in answerable_results
)


if answerable_results:

    top3_hit_rate = (
        top3_hits
        / len(answerable_results)
    )

else:

    top3_hit_rate = 0.0


mean_top1_similarity = float(
    np.mean(
        [
            result["top_similarity"]
            for result in answerable_results
        ]
    )
)


# ============================================================
# SAVE JSON
# ============================================================

save_json(
    {
        "method":
            "baseline_semantic_retrieval",

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "top_k":
            TOP_K,

        "number_of_questions":
            len(results),

        "answerable_questions":
            len(answerable_results),

        "top_3_topic_hit_rate":
            top3_hit_rate,

        "mean_top1_similarity":
            mean_top1_similarity,

        "results":
            results,
    },
    RESULTS_FILE
)


# ============================================================
# SAVE CSV
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
            "question_number",
            "question",
            "answerable",
            "expected_topics",
            "top_similarity",
            "top_1_topic",
            "top_3_topics",
            "expected_topic_found",
        ]
    )

    writer.writeheader()

    writer.writerows(
        evaluation_rows
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "BASELINE RETRIEVAL REBUILD COMPLETED"
)

print(
    "=" * 75
)

print(
    f"\nTop-3 topic hit rate:"
    f" {top3_hit_rate:.2%}"
)

print(
    f"Mean Top-1 similarity:"
    f" {mean_top1_similarity:.4f}"
)

print(
    "\nSaved:"
)

print(
    RESULTS_FILE
)

print(
    EVALUATION_FILE
)

print(
    "\n✓ Baseline artifact recreated."
)

print(
    "✓ Ready for Step 38J comparison."
)