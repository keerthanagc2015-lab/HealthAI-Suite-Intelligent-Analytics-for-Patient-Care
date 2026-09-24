"""
HealthAI - Question-Aware RAG Retrieval

Step 38I

Improves the baseline semantic retrieval by adding:

1. Semantic similarity
2. Query keyword relevance
3. Topic consistency
4. Evidence reranking
5. Duplicate/near-duplicate topic control

This script evaluates retrieval independently from generation.

Input:
    data/processed/rag/embeddings/rag_embeddings.npy
    data/processed/rag/embeddings/rag_embedding_metadata.json

Output:
    data/processed/rag/retrieval/improved/
        rag_improved_retrieval_results.json
        rag_improved_retrieval_evaluation.csv
        rag_improved_retrieval_validation.json
"""

from pathlib import Path
import json
import csv
import re
from collections import Counter

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
    / "improved"
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
    / "rag_improved_retrieval_results.json"
)

EVALUATION_FILE = (
    OUTPUT_DIR
    / "rag_improved_retrieval_evaluation.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "rag_improved_retrieval_validation.json"
)


# ============================================================
# MODEL
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

INITIAL_TOP_K = 10

FINAL_TOP_K = 3

MIN_SEMANTIC_SCORE = 0.35

MIN_FINAL_SCORE = 0.25


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    {
        "question":
            "What are the symptoms of diabetes?",

        "expected_topics":
            ["Diabetes", "Type 2 Diabetes Self-Care"],

        "answerable":
            True,
    },

    {
        "question":
            "How can diabetes be managed?",

        "expected_topics":
            ["Diabetes", "Type 2 Diabetes Self-Care"],

        "answerable":
            True,
    },

    {
        "question":
            "What is high blood pressure?",

        "expected_topics":
            ["Hypertension"],

        "answerable":
            True,
    },

    {
        "question":
            "How can high blood pressure be prevented?",

        "expected_topics":
            ["Hypertension"],

        "answerable":
            True,
    },

    {
        "question":
            "What are noncommunicable diseases?",

        "expected_topics":
            ["Noncommunicable Diseases"],

        "answerable":
            True,
    },

    {
        "question":
            "How can someone cope with a chronic illness?",

        "expected_topics":
            ["Living With Chronic Illness"],

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
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "are",
    "is",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "how",
    "can",
    "be",
    "with",
    "and",
    "in",
    "on",
    "someone",
    "their",
    "does",
    "do",
    "about",
    "what's",
}


# ============================================================
# HELPERS
# ============================================================

def save_json(data, path):

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


def tokenize_text(text):

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 2
    ]


def keyword_overlap(
    question,
    text
):
    """
    Calculate lexical overlap between
    the question and retrieved chunk.
    """

    question_tokens = set(
        tokenize_text(question)
    )

    text_tokens = set(
        tokenize_text(text)
    )

    if not question_tokens:

        return 0.0

    overlap = (
        question_tokens
        & text_tokens
    )

    return (
        len(overlap)
        / len(question_tokens)
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
            "Query vector is zero."
        )

    if np.any(
        document_norms == 0
    ):

        raise ValueError(
            "Vector store contains "
            "zero vectors."
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
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(
    question,
    embedding_model,
    embeddings,
    metadata
):

    query_embedding = (
        embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )

    indices = np.argsort(
        scores
    )[::-1][:INITIAL_TOP_K]

    results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        item = metadata[
            int(index)
        ]

        semantic_score = float(
            scores[index]
        )

        lexical_score = keyword_overlap(
            question,
            item["text"]
        )

        results.append(
            {
                "initial_rank":
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

                "semantic_score":
                    semantic_score,

                "keyword_score":
                    lexical_score,
            }
        )

    return results


# ============================================================
# TOPIC INFERENCE
# ============================================================

def infer_query_topic(
    question,
    retrieved_results
):
    """
    Infer the dominant topic from the highest-quality
    semantic retrieval candidates.

    We deliberately don't hard-code topic names.
    """

    eligible = [
        result
        for result in retrieved_results
        if result["semantic_score"]
        >= MIN_SEMANTIC_SCORE
    ]

    if not eligible:

        return None

    topic_scores = {}

    for result in eligible:

        topic = result["topic"]

        score = (
            0.7
            * result["semantic_score"]
            +
            0.3
            * result["keyword_score"]
        )

        topic_scores[
            topic
        ] = (
            topic_scores.get(
                topic,
                0.0
            )
            + score
        )

    if not topic_scores:

        return None

    return max(
        topic_scores,
        key=topic_scores.get
    )


# ============================================================
# RERANK
# ============================================================

def rerank_results(
    question,
    retrieved_results,
    inferred_topic
):
    """
    Combine semantic and lexical relevance.

    Score:

        70% semantic relevance
        30% keyword relevance

    Topic match provides an additional
    consistency bonus.
    """

    reranked = []

    for result in retrieved_results:

        semantic_score = result[
            "semantic_score"
        ]

        keyword_score = result[
            "keyword_score"
        ]

        combined_score = (
            0.70
            * semantic_score
            +
            0.30
            * keyword_score
        )

        topic_match = (
            inferred_topic is not None
            and result["topic"]
            == inferred_topic
        )

        if topic_match:

            combined_score += 0.05

        result = dict(
            result
        )

        result[
            "topic_match"
        ] = topic_match

        result[
            "reranked_score"
        ] = min(
            combined_score,
            1.0
        )

        reranked.append(
            result
        )

    reranked.sort(
        key=lambda x:
            x["reranked_score"],
        reverse=True
    )

    return reranked


# ============================================================
# DIVERSIFIED TOP-K
# ============================================================

def select_final_evidence(
    reranked_results
):
    """
    Select final evidence.

    Prefer highly relevant chunks while
    preventing all results from being
    identical copies of the same chunk.
    """

    selected = []

    seen_chunk_ids = set()

    seen_texts = set()

    for result in reranked_results:

        if result[
            "reranked_score"
        ] < MIN_FINAL_SCORE:

            continue

        chunk_id = result[
            "chunk_id"
        ]

        normalized_text = re.sub(
            r"\s+",
            " ",
            result["text"].lower()
        ).strip()

        if chunk_id in seen_chunk_ids:

            continue

        if normalized_text in seen_texts:

            continue

        selected.append(
            result
        )

        seen_chunk_ids.add(
            chunk_id
        )

        seen_texts.add(
            normalized_text
        )

        if len(selected) >= FINAL_TOP_K:

            break

    return selected


# ============================================================
# START
# ============================================================

print("=" * 75)

print(
    "HEALTHAI - QUESTION-AWARE RAG RETRIEVAL"
)

print("=" * 75)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD VECTOR STORE
# ============================================================

print(
    "\nLoading embeddings..."
)

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"✓ Embeddings: "
    f"{embeddings.shape}"
)


print(
    "\nLoading metadata..."
)

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
    f"✓ Metadata records: "
    f"{len(metadata)}"
)


if len(embeddings) != len(metadata):

    raise ValueError(
        "Embedding and metadata counts "
        "do not match."
    )


if not np.isfinite(
    embeddings
).all():

    raise ValueError(
        "Embedding matrix contains "
        "invalid values."
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
# RUN TESTS
# ============================================================

all_results = []

evaluation_rows = []


print(
    "\n" + "=" * 75
)

print(
    "QUESTION-AWARE RETRIEVAL TESTS"
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
    # Initial semantic retrieval
    # --------------------------------------------------------

    initial_results = semantic_retrieve(
        question,
        embedding_model,
        embeddings,
        metadata
    )


    # --------------------------------------------------------
    # Infer dominant topic
    # --------------------------------------------------------

    inferred_topic = infer_query_topic(
        question,
        initial_results
    )


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    reranked_results = rerank_results(
        question,
        initial_results,
        inferred_topic
    )


    # --------------------------------------------------------
    # Final evidence
    # --------------------------------------------------------

    final_results = select_final_evidence(
        reranked_results
    )


    final_topics = [
        result["topic"]
        for result in final_results
    ]


    initial_topics = [
        result["topic"]
        for result in initial_results[:5]
    ]


    expected_topic_found = any(
        topic in expected_topics
        for topic in final_topics
    )


    top_initial_score = (
        initial_results[0]
        ["semantic_score"]
        if initial_results
        else 0.0
    )


    top_final_score = (
        final_results[0]
        ["reranked_score"]
        if final_results
        else 0.0
    )


    # --------------------------------------------------------
    # Topic consistency
    # --------------------------------------------------------

    if final_topics:

        topic_counts = Counter(
            final_topics
        )

        dominant_final_topic = (
            topic_counts.most_common(1)[0][0]
        )

        dominant_topic_ratio = (
            topic_counts[
                dominant_final_topic
            ]
            / len(final_topics)
        )

    else:

        dominant_final_topic = None

        dominant_topic_ratio = 0.0


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nInferred topic: "
        f"{inferred_topic}"
    )

    print(
        f"Initial Top-5 topics: "
        f"{initial_topics}"
    )

    print(
        f"Final topics: "
        f"{final_topics}"
    )

    print(
        f"Top semantic score: "
        f"{top_initial_score:.4f}"
    )

    print(
        f"Top reranked score: "
        f"{top_final_score:.4f}"
    )

    print(
        f"Expected topic found: "
        f"{expected_topic_found}"
    )


    print(
        "\nFinal evidence:"
    )


    for rank, result in enumerate(
        final_results,
        start=1
    ):

        print(
            f"  {rank}. "
            f"{result['topic']} | "
            f"semantic="
            f"{result['semantic_score']:.4f} | "
            f"keyword="
            f"{result['keyword_score']:.4f} | "
            f"final="
            f"{result['reranked_score']:.4f}"
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    all_results.append(
        {
            "question_number":
                number,

            "question":
                question,

            "expected_topics":
                expected_topics,

            "answerable":
                answerable,

            "inferred_topic":
                inferred_topic,

            "initial_top_5_topics":
                initial_topics,

            "final_topics":
                final_topics,

            "expected_topic_found":
                expected_topic_found,

            "top_initial_similarity":
                top_initial_score,

            "top_final_score":
                top_final_score,

            "dominant_final_topic":
                dominant_final_topic,

            "dominant_topic_ratio":
                dominant_topic_ratio,

            "final_evidence":
                final_results,

            "initial_results":
                initial_results,
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

            "inferred_topic":
                inferred_topic
                or "",

            "initial_top_1_similarity":
                top_initial_score,

            "final_top_1_score":
                top_final_score,

            "expected_topic_found":
                expected_topic_found,

            "final_evidence_count":
                len(final_results),

            "dominant_topic":
                dominant_final_topic
                or "",

            "dominant_topic_ratio":
                dominant_topic_ratio,
        }
    )


# ============================================================
# METRICS
# ============================================================

answerable_results = [
    result
    for result in all_results
    if result["answerable"]
]


unanswerable_results = [
    result
    for result in all_results
    if not result["answerable"]
]


topic_hits = sum(
    result["expected_topic_found"]
    for result in answerable_results
)


if answerable_results:

    final_topic_hit_rate = (
        topic_hits
        / len(answerable_results)
    )

else:

    final_topic_hit_rate = 0.0


# ------------------------------------------------------------
# Topic consistency
# ------------------------------------------------------------

if answerable_results:

    mean_topic_consistency = np.mean(
        [
            result[
                "dominant_topic_ratio"
            ]
            for result in answerable_results
        ]
    )

else:

    mean_topic_consistency = 0.0


# ------------------------------------------------------------
# Out-of-domain
# ------------------------------------------------------------

out_of_domain_empty_count = sum(
    len(result["final_evidence"]) == 0
    for result in unanswerable_results
)


if unanswerable_results:

    out_of_domain_empty_rate = (
        out_of_domain_empty_count
        / len(unanswerable_results)
    )

else:

    out_of_domain_empty_rate = 0.0


# ============================================================
# SAVE RESULTS
# ============================================================

save_json(
    {
        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "initial_top_k":
            INITIAL_TOP_K,

        "final_top_k":
            FINAL_TOP_K,

        "semantic_weight":
            0.70,

        "keyword_weight":
            0.30,

        "topic_bonus":
            0.05,

        "final_topic_hit_rate":
            final_topic_hit_rate,

        "mean_topic_consistency":
            float(
                mean_topic_consistency
            ),

        "out_of_domain_empty_rate":
            out_of_domain_empty_rate,

        "results":
            all_results,
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
            "inferred_topic",
            "initial_top_1_similarity",
            "final_top_1_score",
            "expected_topic_found",
            "final_evidence_count",
            "dominant_topic",
            "dominant_topic_ratio",
        ]
    )

    writer.writeheader()

    writer.writerows(
        evaluation_rows
    )


# ============================================================
# VALIDATION
# ============================================================

validation = {

    "vector_store_valid":
        True,

    "embedding_count":
        int(len(embeddings)),

    "embedding_dimension":
        int(embeddings.shape[1]),

    "metadata_count":
        int(len(metadata)),

    "test_question_count":
        len(all_results),

    "answerable_question_count":
        len(answerable_results),

    "final_topic_hit_rate":
        final_topic_hit_rate,

    "mean_topic_consistency":
        float(
            mean_topic_consistency
        ),

    "out_of_domain_empty_rate":
        out_of_domain_empty_rate,

    "retrieval_pipeline_ready":
        len(all_results)
        == len(TEST_QUESTIONS),
}


save_json(
    validation,
    VALIDATION_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 38I - QUESTION-AWARE RETRIEVAL COMPLETED"
)

print(
    "=" * 75
)

print(
    f"\nInitial Top-K:"
    f" {INITIAL_TOP_K}"
)

print(
    f"Final Top-K:"
    f" {FINAL_TOP_K}"
)

print(
    f"Final topic hit rate:"
    f" {final_topic_hit_rate:.2%}"
)

print(
    f"Mean topic consistency:"
    f" {mean_topic_consistency:.2%}"
)

print(
    f"Out-of-domain empty evidence rate:"
    f" {out_of_domain_empty_rate:.2%}"
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
    VALIDATION_FILE
)

print(
    "\n✓ Question-aware retrieval completed."
)